//! # LAND Protocol (Local AI Network Discovery)
//!
//! Core library implementing the LAND protocol for automatic discovery
//! and communication between LaRuche nodes on a local network.
//!
//! ## Architecture
//!
//! ```text
//!  ┌──────────────┐     mDNS multicast      ┌──────────────┐
//!  │ LaRuche     │ ◄──────────────────────► │ LaRuche     │
//!  │  (LLM+RAG)   │    _ai-inference._tcp    │  (VLM+Audio) │
//!  └──────┬───────┘                          └──────┬───────┘
//!         │              LAND Protocol              │
//!         ▼                                         ▼
//!  ┌──────────────┐                          ┌──────────────┐
//!  │   Manifest    │                          │   Manifest    │
//!  │  Cognitif     │                          │  Cognitif     │
//!  └──────────────┘                          └──────────────┘
//! ```

pub mod capabilities;
pub mod discovery;
pub mod manifest;
pub mod auth;
pub mod qos;
pub mod swarm;
pub mod error;

pub use capabilities::Capability;
pub use discovery::{LandBroadcaster, LandListener};
pub use manifest::CognitiveManifest;
pub use auth::{ProximityAuth, TrustCircle, AuthToken};
pub use qos::{QosLevel, QosPolicy};
pub use swarm::SwarmState;
pub use error::LandError;

/// LAND protocol version
pub const PROTOCOL_VERSION: &str = "0.1.0";

/// mDNS service type for LAND
pub const SERVICE_TYPE: &str = "_ai-inference._tcp.local.";

/// Default multicast interval in seconds
pub const BROADCAST_INTERVAL_SECS: u64 = 2;

/// Default API port for inference
pub const DEFAULT_API_PORT: u16 = 8419;

/// Default dashboard port
pub const DEFAULT_DASHBOARD_PORT: u16 = 8420;

/// Get the local IP address of the machine.
/// Falls back to 127.0.0.1 if discovery fails.
pub fn get_local_ip() -> String {
    local_ip_address::local_ip()
        .map(|ip| ip.to_string())
        .unwrap_or_else(|_| "127.0.0.1".to_string())
}
