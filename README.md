# Sumeru

<p align="center">
  <strong>In-Memory Columnar State &amp; Heterogeneous Compute Engine</strong>
</p>

<p align="center">
  <a href="./README.md">English</a> · <a href="./README_zh-CN.md">简体中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-2024_Edition-orange?logo=rust" alt="Rust 2024" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License" />
  <img src="https://img.shields.io/badge/Engine-Heterogeneous_Columnar_Compute-10B981" alt="Architecture" />
</p>

**Sumeru** is an in-memory **columnar state and heterogeneous compute engine** built for high-frequency state evolution, large-scale multi-agent simulations, and real-time digital twins. Your state lives in 16KB page-aligned contiguous columns, read/write conflicts are resolved statically at build time, and multi-core CPU SIMD, GPU compute kernels, and Apache Arrow analytical pipelines operate on the exact same physical memory with zero copy.

## Why Sumeru?

Analytical columnar databases (like DuckDB) incur per-query SQL planning and operator tree overhead that cannot sustain microsecond-level ticks, while row-oriented object graphs suffer from pointer chasing and cannot feed SIMD or GPU pipelines directly. Sumeru bridges this gap:

- **16KB page-aligned contiguous columns**: State is clustered by schema into dense typed columns and copy-on-write blocks, eliminating pointer chasing while aligning naturally to CPU cache lines and GPU page tables.
- **Static lock-free scheduling**: Operators declare read/write sets up front; conflicts are proven statically into a deterministic dependency graph (DAG) with zero runtime lock overhead.
- **Bitmask-jump & 64-row dirty tracking**: 64-bit hardware word bitmasks, 8-way in-place bit-planes, and 64-row dirty epochs skip untouched blocks in one step.
- **First-class relational topology**: Model entity pairs, hierarchies, and zero-sized tags directly inside columnar tables with lazy target bitmaps.
- **Arrow-native & time-travel ready**: Two-phase atomic batch commits with zero-copy Apache Arrow `RecordBatch` export, whole-column snapshots, and delta replay.

### When to use Sumeru?

- Build high-frequency simulations, physics/agent steppers, or real-time digital twins requiring microsecond ticks over millions of state rows.
- Run fine-grained state rules (CPU SIMD), batch compute kernels (GPU UMA / CUDA), and real-time columnar analytics (Apache Arrow) over a single zero-copy memory layout.
- Embed a deterministic in-memory state layer with time-travel snapshots, delta replication, and two-phase transactional safety.

## Quick Start

```rust
use serde::{Deserialize, Serialize};
use sumeru::{EntityId, Pair, Store};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct Position { x: f32, y: f32 }

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct Velocity { vx: f32, vy: f32 }

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct Active;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct Frozen;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct ChildOf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut store = Store::new();
    let root = EntityId::from_u128(1);
    let entity = EntityId::from_u128(2);

    store.insert((
        entity,
        (
            Position { x: 0.0, y: 0.0 },
            Velocity { vx: 1.5, vy: 2.0 },
            Active,
            Pair::<ChildOf>::new(ChildOf, root),
        ),
    ))?;

    let baseline_epoch = store.epoch();

    // Set status flag
    store.set_flag::<0>(entity, true)?;

    // Filter and update in place
    for (_id, (pos, vel)) in store
        .query_mut::<(&mut Position, &Velocity)>()?
        .with::<Active>()
        .without::<Frozen>()
        .with_pair::<ChildOf>(root)
        .with_flag::<0>()
        .iter()
    {
        pos.x += vel.vx;
        pos.y += vel.vy;
    }

    assert_eq!(store.get::<Position>(entity), Some(&Position { x: 1.5, y: 2.0 }));

    // Query data modified since baseline_epoch
    let dirty_pos = store
        .query::<&Position>()
        .changed_since::<Position>(baseline_epoch)
        .iter()
        .count();
    let dirty_vel = store
        .query::<&Velocity>()
        .changed_since::<Velocity>(baseline_epoch)
        .iter()
        .count();

    assert_eq!(dirty_pos, 1);
    assert_eq!(dirty_vel, 0);
    Ok(())
}
```

## Installation

Add `sumeru` to your `Cargo.toml`:

```toml
[dependencies]
sumeru = { git = "https://github.com/XYCaptain/Sumeru.git" }
```

## Examples & Benchmarks

Run the built-in example suites with `cargo run --example <name>`:

- `flecs` (85 scenarios): Dynamic column bindings, relational pairs, multi-phase pipelines, observers, and reflection
- `gpu` (13 scenarios): 16KB page-aligned zero-copy GPU compute, typed Shader IR, and three-tier parallel dispatch

```bash
cargo test                                    # Run unit, integration, and security gate tests
cargo run --example flecs                     # Run state & relational modeling examples (85 scenarios)
cargo run --example gpu --features gpu        # Run 16KB zero-copy GPU compute examples (13 scenarios)
cargo bench --bench store_benchmark           # Run contiguous columnar throughput benchmarks
```

## Architecture

<p align="center">
  <img src="./assets/pipeline.svg" alt="Sumeru Architecture" width="840" />
</p>

Sumeru implements a unified-memory columnar pipeline: writes commit in two phases into 16KB page-aligned columns, and a 3-level cached scheduler dispatches work lock-free across heterogeneous targets:

- **Page-aligned contiguous store**: State is clustered by schema in 16KB page-aligned columns with 8-way in-place bit-planes and 64-row dirty epochs—eliminating pointer chasing and table migration.
- **3-level cached lock-free schedule**: Read/write bitsets statically resolve conflicts at build time (L1), while pre-bound column slots (L2) and 32–64KB cache slices (L3) dispatch work with zero runtime locks.
- **Zero-copy heterogeneous dispatch**: The same contiguous column buffers feed multi-core CPU SIMD slices, map zero-copy into GPU unified memory (Metal UMA / wgpu), or export as Apache Arrow `RecordBatch`es.

## Features & Roadmap

**Storage & Schema**
- [x] **Unified-Memory Page-Aligned Columnar Store**: 16KB hardware page-aligned contiguous columns with block-level copy-on-write (CoW), generational row indexing, and flat column pointer caches for zero-copy CPU SIMD and GPU co-processing
- [x] **Relational ECS & In-Place State Bit-Planes**: Native archetype tables, `(Relation, Target)` entity pairs, zero-sized tags, global singletons, and 8-way non-migrating row bit-planes that eliminate table migration on high-frequency state toggles
- [x] **Two-Phase Deterministic Atomic Commit**: Pre-validated batch transactions with per-row patch folding and structural topology batching, avoiding runtime lock contention and per-row commit overhead
- [ ] **Prototype Inheritance & Parameterized Prefabs (`IsA` Hierarchy)**: Hierarchical component sharing and copy-on-write overrides via `IsA` relations for zero-redundancy entity instantiation

**Query & Filtering**
- [x] **Multi-Level Cached Compiled Query Engine**: Inverted column-to-table bitmaps with incremental topology caching (L1), pre-bound column slots (L2), and multi-column row cursors that reduce runtime matching to $O(1)$ sequential scans
- [x] **Word-Aligned Bitmask Filtering & Dirty Epochs**: Standalone stateless bitmap primitives, 64-bit word jump filtering, contiguous run coalescing, relational target bitmaps, and 64-row dirty epochs for incremental fixed-point evaluation
- [ ] **Transitive Graph Queries & Worst-Case Optimal Joins (WCOJ)**: Relational transitive closure derivation, hierarchy traversal caches, and multi-hop graph join operators
- [ ] **Vectorized Columnar Analytics & Spatial Indexing**: Native SIMD vectorized aggregations, group-by analytics, and spatial proximity indexing pushed down directly onto contiguous column chunks

**Schedule & Heterogeneous Compute**
- [x] **Static Disjoint-Access Parallel DAG Scheduler**: Compile/build-time read/write bitset conflict resolution and static ambiguity detection with lock-free topological wave dispatch
- [x] **Cache-Conscious Morsel-Driven Parallelism**: Adaptive L1/L2 cache-resident subchunk planner (32–64KB) paired with work-stealing zero-copy slice borrowing for high-throughput SIMD iteration
- [x] **Zero-Bubble Multi-Kernel GPU Compute Pass**: Rust AST to typed Shader IR compilation pipeline (with Naga static validation and dead-write elimination), 16KB hardware page dirty bitmap folding, and explicit memory barriers within a single command buffer
- [ ] **Cross-Platform GPU Backends & Hybrid CPU-GPU DAG Orchestration**: Unified heterogeneous backends across Metal, Vulkan, DirectX 12, and WebGPU with automated CPU-GPU pipeline scheduling and JIT kernel compilation

**Snapshots & Ecosystem**
- [x] **Zero-Copy Time-Travel Snapshots & Column-Pruned Replay**: Whole-column zero-copy `rkyv` archives, frame commit receipts, and column-pruned delta replay for sub-millisecond deterministic state rollback
- [x] **Zero-Copy Apache Arrow Bridge**: In-situ export of physical columns into Apache Arrow `RecordBatch`es without memory reallocation or serialization
- [ ] **Streaming Columnar Exchange & Lakehouse Interop**: Zero-copy streaming I/O for Apache Arrow Flight, Parquet, and random-access columnar formats (Vortex / Lance)
- [ ] **Cross-Language Zero-Copy ABI & WebAssembly Runtime**: Standard C-ABI, zero-copy Python / C++ bindings via the Arrow C Data Interface, and portable WebAssembly / WebGPU execution

## License

MIT License. See [LICENSE](./LICENSE) for details.
