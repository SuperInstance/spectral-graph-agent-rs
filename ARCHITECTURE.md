# Architecture — spectral-graph-agent-rs

> *Internal design and data flow.*

## Overview

This crate provides functionality for the SuperInstance ecosystem.

## Core Types

- **`SpectralGraph`**

## Key Functions

- `new()`
- `add_edge()`
- `finalize()`
- `build_complete()`
- `build_path()`
- `build_cycle()`
- `build_star()`
- `build_grid2d()`

## Source Structure

1 Rust source file(s) in `src/`.
Language: Rust

## Cross-Repo References

- [ternary-core](https://github.com/SuperInstance/ternary-core) — shared Z₃ traits
- [ternary-types](https://github.com/SuperInstance/ternary-types) — type-level encodings
- [Full SuperInstance fleet](https://github.com/orgs/SuperInstance/repositories?q=ternary)
