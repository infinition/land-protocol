#  LAND Protocol (Local AI Network Discovery)

> **"The HDMI of AI"**

A lightweight, local-first protocol for automatic discovery and communication between AI nodes on a local network.

##  Why LAND?

In the current AI landscape, hardware is often siloed or cloud-dependent. **LAND** (Local AI Network Discovery) is an open standard that allows any device-from a tiny ESP32 to a multi-GPU server-to join a local "Collective Intelligence" without configuration, API keys, or internet access.

It separates the **Standard** from the **Product**:
- **LAND (The Standard):** An open language for AI nodes to find and talk to each other.
- **LaRuche (The Product):** A premium implementation of this protocol.

##  Core Features

- **Zero-Config Discovery:** Uses mDNS/DNS-SD to find peers instantly on the local network.
- **Cognitive Manifest:** Every node broadcasts a rich profile:
    - **Physical Identity:** Node name, hardware tier (Nano, Core, Pro, Max).
    - **Intelligence Menu:** Available models (LLM, VLM, RAG, Audio, Code).
    - **Real-time Load:** Queue depth and tokens per second throughput.
- **Proof of Proximity:** Secure physical-based authentication (NFC/Button press).
- **Swarm Intelligence:** Built-in logic for resource sharing and resilience.
- **Priority QoS:** Dedicated lanes for critical inference tasks.

##  How it works

The protocol identifies nodes via the mDNS service type `_ai-inference._tcp.local.`. Each node broadcasts a Gzipped JSON payload (the **Cognitive Manifest**) containing its current state. 

Clients (apps, SDKs) listen for these broadcasts to build a real-time map of available local intelligence.

##  Usage

### Add as a dependency
```toml
[dependencies]
land-protocol = { git = "https://github.com/infinition/land-protocol" }
```

### 1. Discovering Nodes (Client side)
Use this if you are building an app or service that wants to use available AI nodes.

```rust
use land_protocol::discovery::LandListener;

#[tokio::main]
async fn main() {
    let mut listener = LandListener::new().unwrap();
    let nodes_map = listener.start().unwrap();

    println!("Scanning for AI nodes...");
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;

    let nodes = nodes_map.read().await;
    for (id, node) in nodes.iter() {
        println!("Found: {} ({}) at {}:{}", 
            node.manifest.node_name.as_deref().unwrap_or("unnamed"),
            node.manifest.tier.as_deref().unwrap_or("unknown"),
            node.manifest.host,
            node.manifest.port.unwrap_or(8419)
        );
    }
}
```

### 2. Broadcasting a Node (Server side)
Use this if you are building a new AI-enabled device or bridge.

```rust
use land_protocol::discovery::LandBroadcaster;
use land_protocol::manifest::{CognitiveManifest, HardwareTier};

#[tokio::main]
async fn main() {
    let mut manifest = CognitiveManifest::new("my-custom-node".into(), HardwareTier::Core);
    // Add your models and capacities...
    
    let mut broadcaster = LandBroadcaster::new().unwrap();
    broadcaster.register(&manifest).unwrap();
    
    println!("Node is now visible on the network as a LAND peer.");
    loop { tokio::time::sleep(tokio::time::Duration::from_secs(60)).await; }
}
```

##  License

Licensed under either of:
- Apache License, Version 2.0
- MIT license

---
*Maintained by [infinition](https://github.com/infinition). Part of the LaRuche ecosystem.*
