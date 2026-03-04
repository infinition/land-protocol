//! Swarm Intelligence - automatic mesh and tensor sharding between LaRuche nodes.
//!
//! When multiple LaRuche nodes are on the same network, they can
//! automatically pool their VRAM and distribute model layers
//! across nodes (tensor sharding over Ethernet).

use crate::manifest::PartialManifest;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Describes the state of a LaRuche swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmState {
    /// Unique swarm identifier
    pub swarm_id: Uuid,

    /// All peer nodes in the swarm
    pub peers: HashMap<Uuid, SwarmPeer>,

    /// The coordinator node ID
    pub coordinator_id: Uuid,

    /// Combined VRAM across all peers in MB
    pub total_vram_mb: u64,

    /// Combined RAM across all peers in MB
    pub total_ram_mb: u64,

    /// Current sharding configuration
    pub sharding: Option<ShardingConfig>,

    /// Swarm creation time
    pub formed_at: DateTime<Utc>,
}

/// A peer node in the swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmPeer {
    pub node_id: Uuid,
    pub node_name: String,
    pub host: String,
    pub port: u16,
    pub vram_mb: u64,
    pub ram_mb: u64,
    pub last_heartbeat: DateTime<Utc>,
    pub status: PeerStatus,
    /// Which layers this peer is responsible for
    pub assigned_layers: Option<LayerRange>,
}

/// Health status of a swarm peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PeerStatus {
    /// Online and healthy
    Active,
    /// Responding but under heavy load
    Busy,
    /// Missed recent heartbeats
    Suspect,
    /// Confirmed offline
    Down,
    /// Syncing model data
    Syncing,
}

/// Configuration for tensor sharding across the swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardingConfig {
    /// Model being sharded
    pub model_name: String,

    /// Total model size in MB
    pub model_size_mb: u64,

    /// Total number of layers in the model
    pub total_layers: u32,

    /// Layer assignment per peer
    pub assignments: HashMap<Uuid, LayerRange>,

    /// Whether a redundancy mirror is active
    pub mirror_enabled: bool,
}

/// A range of model layers assigned to a peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerRange {
    pub start: u32,
    pub end: u32,
}

impl LayerRange {
    pub fn count(&self) -> u32 {
        self.end - self.start + 1
    }
}

impl SwarmState {
    /// Create a new swarm with this node as coordinator.
    pub fn new_as_coordinator(node_id: Uuid, node_name: String, host: String, port: u16, vram_mb: u64, ram_mb: u64) -> Self {
        let mut peers = HashMap::new();
        peers.insert(node_id, SwarmPeer {
            node_id,
            node_name,
            host,
            port,
            vram_mb,
            ram_mb,
            last_heartbeat: Utc::now(),
            status: PeerStatus::Active,
            assigned_layers: None,
        });

        Self {
            swarm_id: Uuid::new_v4(),
            peers,
            coordinator_id: node_id,
            total_vram_mb: vram_mb,
            total_ram_mb: ram_mb,
            sharding: None,
            formed_at: Utc::now(),
        }
    }

    /// Add a peer to the swarm.
    pub fn add_peer(&mut self, manifest: &PartialManifest) -> Option<Uuid> {
        let node_id = manifest.node_id?;
        let peer = SwarmPeer {
            node_id,
            node_name: manifest.node_name.clone().unwrap_or_default(),
            host: manifest.host.clone(),
            port: manifest.port.unwrap_or(crate::DEFAULT_API_PORT),
            vram_mb: 0, // Will be updated from full manifest
            ram_mb: 0,
            last_heartbeat: Utc::now(),
            status: PeerStatus::Syncing,
            assigned_layers: None,
        };

        self.peers.insert(node_id, peer);
        self.recalculate_totals();
        Some(node_id)
    }

    /// Remove a peer from the swarm.
    pub fn remove_peer(&mut self, node_id: &Uuid) -> bool {
        let removed = self.peers.remove(node_id).is_some();
        if removed {
            self.recalculate_totals();
        }
        removed
    }

    /// Process a heartbeat from a peer.
    pub fn heartbeat(&mut self, node_id: &Uuid) -> bool {
        if let Some(peer) = self.peers.get_mut(node_id) {
            peer.last_heartbeat = Utc::now();
            if matches!(
                peer.status,
                PeerStatus::Suspect | PeerStatus::Syncing | PeerStatus::Down
            ) {
                peer.status = PeerStatus::Active;
            }
            true
        } else {
            false
        }
    }

    /// Check for stale peers (missed heartbeats).
    /// Returns list of peers marked as suspect or down.
    pub fn check_health(&mut self, timeout_secs: i64) -> Vec<Uuid> {
        let now = Utc::now();
        let mut unhealthy = Vec::new();

        for (id, peer) in &mut self.peers {
            let elapsed = (now - peer.last_heartbeat).num_seconds();
            if elapsed > timeout_secs * 3 && peer.status != PeerStatus::Down {
                peer.status = PeerStatus::Down;
                unhealthy.push(*id);
            } else if elapsed > timeout_secs
                && matches!(
                    peer.status,
                    PeerStatus::Active | PeerStatus::Busy | PeerStatus::Syncing
                )
            {
                peer.status = PeerStatus::Suspect;
                unhealthy.push(*id);
            }
        }

        unhealthy
    }

    /// Calculate a sharding configuration for a model across available peers.
    pub fn plan_sharding(&self, model_name: &str, total_layers: u32, model_size_mb: u64) -> ShardingConfig {
        let active_peers: Vec<&SwarmPeer> = self.peers.values()
            .filter(|p| matches!(p.status, PeerStatus::Active | PeerStatus::Busy))
            .collect();

        let peer_count = active_peers.len() as u32;
        let layers_per_peer = total_layers / peer_count.max(1);
        let mut assignments = HashMap::new();
        let mut current_layer = 0;

        for (i, peer) in active_peers.iter().enumerate() {
            let end_layer = if i as u32 == peer_count - 1 {
                total_layers - 1 // Last peer gets remaining layers
            } else {
                current_layer + layers_per_peer - 1
            };

            assignments.insert(peer.node_id, LayerRange {
                start: current_layer,
                end: end_layer,
            });

            current_layer = end_layer + 1;
        }

        ShardingConfig {
            model_name: model_name.to_string(),
            model_size_mb,
            total_layers,
            assignments,
            mirror_enabled: peer_count >= 3, // Enable redundancy with 3+ nodes
        }
    }

    /// Get active peer count.
    pub fn active_peer_count(&self) -> usize {
        self.peers.values()
            .filter(|p| matches!(p.status, PeerStatus::Active | PeerStatus::Busy))
            .count()
    }

    fn recalculate_totals(&mut self) {
        self.total_vram_mb = self.peers.values().map(|p| p.vram_mb).sum();
        self.total_ram_mb = self.peers.values().map(|p| p.ram_mb).sum();
    }
}

/// Resilience configuration for the LaRuche Resilience system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceConfig {
    /// Enable automatic model mirroring (RAID-1 for AI)
    pub mirror_enabled: bool,

    /// Heartbeat interval in seconds
    pub heartbeat_interval_secs: u64,

    /// Time before marking a peer as suspect
    pub suspect_timeout_secs: u64,

    /// Time before marking a peer as down
    pub down_timeout_secs: u64,

    /// Maximum failover time in ms
    pub max_failover_ms: u64,

    /// Enable distributed checkpointing for long tasks
    pub checkpoint_enabled: bool,

    /// Checkpoint interval in seconds
    pub checkpoint_interval_secs: u64,
}

impl Default for ResilienceConfig {
    fn default() -> Self {
        Self {
            mirror_enabled: true,
            heartbeat_interval_secs: 1,
            suspect_timeout_secs: 3,
            down_timeout_secs: 10,
            max_failover_ms: 500,
            checkpoint_enabled: true,
            checkpoint_interval_secs: 30,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_formation() {
        let node_a = Uuid::new_v4();
        let mut swarm = SwarmState::new_as_coordinator(
            node_a, "laruche-a".into(), "192.168.1.10".into(), 8419, 8192, 16384,
        );

        assert_eq!(swarm.active_peer_count(), 1);
        assert_eq!(swarm.coordinator_id, node_a);

        // Add peer
        let manifest = PartialManifest {
            node_id: Some(Uuid::new_v4()),
            node_name: Some("laruche-b".into()),
            host: "192.168.1.11".into(),
            port: Some(8419),
            ..default_partial()
        };
        swarm.add_peer(&manifest);

        assert_eq!(swarm.peers.len(), 2);
    }

    #[test]
    fn test_sharding_plan() {
        let node_a = Uuid::new_v4();
        let mut swarm = SwarmState::new_as_coordinator(
            node_a, "laruche-a".into(), "192.168.1.10".into(), 8419, 4096, 8192,
        );

        let node_b = Uuid::new_v4();
        swarm.peers.insert(node_b, SwarmPeer {
            node_id: node_b,
            node_name: "laruche-b".into(),
            host: "192.168.1.11".into(),
            port: 8419,
            vram_mb: 4096,
            ram_mb: 8192,
            last_heartbeat: Utc::now(),
            status: PeerStatus::Active,
            assigned_layers: None,
        });

        let config = swarm.plan_sharding("llama-70b", 80, 40000);
        assert_eq!(config.assignments.len(), 2);

        // Total layers should cover all 80
        let total: u32 = config.assignments.values()
            .map(|r| r.count())
            .sum();
        assert_eq!(total, 80);
    }

    #[test]
    fn test_syncing_becomes_active_on_heartbeat() {
        let node_a = Uuid::new_v4();
        let mut swarm = SwarmState::new_as_coordinator(
            node_a, "laruche-a".into(), "192.168.1.10".into(), 8419, 8192, 16384,
        );
        let peer_id = Uuid::new_v4();
        let manifest = PartialManifest {
            node_id: Some(peer_id),
            node_name: Some("laruche-b".into()),
            host: "192.168.1.11".into(),
            port: Some(8419),
            ..default_partial()
        };
        swarm.add_peer(&manifest);
        assert_eq!(swarm.peers.get(&peer_id).unwrap().status, PeerStatus::Syncing);

        assert!(swarm.heartbeat(&peer_id));
        assert_eq!(swarm.peers.get(&peer_id).unwrap().status, PeerStatus::Active);
    }

    fn default_partial() -> PartialManifest {
        PartialManifest {
            protocol_version: None,
            node_id: None,
            node_name: None,
            tier: None,
            tokens_per_sec: None,
            memory_usage_pct: None,
            queue_depth: None,
            port: None,
            dashboard_port: None,
            capabilities: Vec::new(),
            temperature_c: None,
            in_swarm: false,
            peer_count: 0,
            is_coordinator: false,
            model: None,
            host: String::new(),
        }
    }
}
