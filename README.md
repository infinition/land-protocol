# LAND Protocol

Local AI Network Discovery protocol used by LaRuche nodes and clients.

## Purpose

LAND provides:

- Zero-config node discovery on LAN via mDNS/DNS-SD
- Compact capability and load metadata broadcast (TXT records)
- Swarm state helpers (peers, health, sharding)
- QoS primitives for request prioritization
- Proximity-auth primitives for local trust flows

## Transport and defaults

- Service type: `_ai-inference._tcp.local.`
- mDNS multicast: UDP `5353`
- Default API port: `8419`
- Default dashboard port: `8420`

## Crate modules

- `capabilities`: typed capability model (`llm`, `vlm`, `code`, ...)
- `manifest`: cognitive manifest + TXT encode/decode
- `discovery`: `LandBroadcaster` and `LandListener`
- `swarm`: peer health, swarm state, sharding planning
- `qos`: queue + priority policy primitives
- `auth`: proximity-based auth helpers
- `error`: protocol error types

## Discovery behavior (current)

- Listener stale timeout: `45s`
- REMOVE mDNS events are treated as transient hints; eviction occurs on stale timeout
- Broadcaster can be re-announced by calling `update(&manifest)` periodically

## Usage

### Listener

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

### Broadcaster

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

## URL helpers

- `land_protocol::format_host_for_url(host)`
- `land_protocol::endpoint_url(host, port, tls)`

These helpers bracket raw IPv6 hosts automatically.

## Notes for integrators

- `PartialManifest::api_url()` and `dashboard_url()` use IPv6-safe URL formatting.
- `qos::RequestQueue::dequeue()` now marks requests as active; call `complete(qos)` when done.
- `swarm::heartbeat()` promotes `Syncing/Suspect/Down` peers back to `Active`.
