# Getting Started — spectral-graph-agent-rs

> *Estimated time: 5 minutes*

## Prerequisites

- **Rust 1.75+** (MSRV)
- Cargo (included with Rust)

## Installation

```toml
[dependencies]
spectral_graph_agent_rs = "0.1.0"
```

Or from source:

```bash
git clone https://github.com/SuperInstance/spectral-graph-agent-rs.git
cd spectral-graph-agent-rs
cargo build --release
cargo test
```

## Core Concept

This crate provides core functionality for the SuperInstance ecosystem.

## Quick Example

```
use spectral_graph_agent_rs::SpectralGraph;
let instance = SpectralGraph::new();
```

## Running Tests

```bash
cargo test
```

## Next Steps

- [ARCHITECTURE.md](./ARCHITECTURE.md) — Internal design
- [PLUG_AND_PLAY.md](./PLUG_AND_PLAY.md) — Integration
- [CONTRIBUTING.md](./CONTRIBUTING.md) — Contributing
