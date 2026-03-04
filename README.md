<div align="center">
<img height="300" alt="image" src="https://github.com/user-attachments/assets/32aae849-064a-40eb-b83e-170713664034" />
</div>

# LAND Protocol (Local AI Network Discovery)

> **"The HDMI of AI"**

A lightweight, local-first protocol for automatic discovery and communication between AI nodes on a local network.

## Why LAND?

In the current AI landscape, hardware is often siloed or cloud-dependent. **LAND** is an open standard that allows any device (from a tiny ESP32 to a multi-GPU server) to join a local collective intelligence with no API keys and no internet requirement.

- **Discovery:** mDNS on UDP port **5353**.
- **Transport:** TCP for the cognitive API (default port **8419**).

LAND separates:

- **LAND (the standard):** an open protocol for local AI node discovery and communication.
- **LaRuche (the product):** a concrete implementation of that protocol.

## Core features

- **Zero-config discovery:** mDNS/DNS-SD peer discovery on LAN.
- **Cognitive manifest:** each node broadcasts identity, capabilities, and load.
- **Proof of proximity:** local trust/auth primitives.
- **Swarm intelligence:** peer state and resilience helpers.
- **Priority QoS:** request prioritization primitives.

## How it works

Nodes advertise using `_ai-inference._tcp.local.` and expose key metadata through TXT records.
Clients and peers listen to these announcements and build a live map of available intelligence.

## Current implementation notes

- Listener stale timeout is `45s`.
- `ServiceRemoved` events are treated as transient hints; eviction is timeout-based.
- `PartialManifest::api_url()` and `dashboard_url()` are IPv6-safe.
- URL helpers:
  - `land_protocol::format_host_for_url(host)`
  - `land_protocol::endpoint_url(host, port, tls)`
- `qos::RequestQueue::dequeue()` marks requests as active; call `complete(qos)` when processing ends.
- `swarm::heartbeat()` promotes `Syncing`, `Suspect`, and `Down` peers back to `Active`.

## Usage

### Add as dependency

```toml
[dependencies]
land-protocol = { git = "https://github.com/infinition/land-protocol" }
```

### 1. Discovering nodes (client side)

```rust
use land_protocol::discovery::LandListener;

#[tokio::main]
async fn main() {
    let mut listener = LandListener::new().unwrap();
    let nodes = listener.start().unwrap();

    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    let snapshot = nodes.read().await;
    for (_id, node) in snapshot.iter() {
        println!(
            "{} @ {}:{}",
            node.manifest.node_name.as_deref().unwrap_or("unknown"),
            node.manifest.host,
            node.manifest.port.unwrap_or(8419)
        );
    }
}
```

### 2. Broadcasting a node (server side)

```rust
use land_protocol::discovery::LandBroadcaster;
use land_protocol::manifest::{CognitiveManifest, HardwareTier};

#[tokio::main]
async fn main() {
    let mut manifest = CognitiveManifest::new("node-a".into(), HardwareTier::Core);
    let mut broadcaster = LandBroadcaster::new().unwrap();
    broadcaster.register(&manifest).unwrap();

    let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(2));
    loop {
        interval.tick().await;
        manifest.performance.queue_depth = manifest.performance.queue_depth.saturating_add(1);
        broadcaster.update(&manifest).unwrap();
    }
}
```

## Notes for integrators

- Prefer `PartialManifest::api_url()` / `dashboard_url()` over manual string formatting.
- Use URL helpers for direct endpoint building, especially with IPv6 hosts.
- If you use `RequestQueue`, pair `dequeue()` with `complete(qos)`.
