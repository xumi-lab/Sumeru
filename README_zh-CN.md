# Sumeru

<p align="center">
  <strong>面向实时状态演进与仿真的内存列式异构状态计算引擎</strong>
</p>

<p align="center">
  <a href="./README.md">English</a> · <a href="./README_zh-CN.md">简体中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-2024_Edition-orange?logo=rust" alt="Rust 2024" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License" />
  <img src="https://img.shields.io/badge/Engine-Heterogeneous_Columnar_Compute-10B981" alt="Architecture" />
</p>

**Sumeru** 是一个面向高频状态演进、大规模多智能体仿真与数字孪生的**内存列式异构状态计算引擎**。你的状态数据常驻于 16KB 页对齐的连续列内存中，查询与算子调度在构建期静态消解读写冲突，让多核 CPU SIMD、GPU 加速管线与 Apache Arrow 分析引擎在同一份物理内存上零拷贝协同。

## Why Sumeru?

传统分析型列存（如 DuckDB）受限于单次查询的 SQL 解析与火山/向量化算子树开销，难以支撑微秒级时间步（Tick）的高频状态读写；而行存对象图又无法直接交付给 SIMD 与 GPU。Sumeru 弥合了这一断层：

- **16KB 页对齐连续列存**：按列组合自动聚簇为紧凑连续列与写时复制块，消除指针追逐，天然对齐 CPU 缓存行与 GPU 页表。
- **静态无锁并发调度**：算子显式声明读写集，在构建期静态消解冲突并生成确定性依赖图（DAG），运行期零锁开销。
- **位图跳跃与 64 行块脏追踪**：内置 64 位硬件字位图跳跃过滤、8 路原位状态位平面与 64 行块级脏追踪，整块跳过未修改数据。
- **一等公民关系拓扑**：在连续列存表内原生表达实体关系对、层级树与零大小标签，配备惰性目标位图索引。
- **Arrow 原生与时间回溯**：两阶段原子事务提交，支持零拷贝导出 Apache Arrow `RecordBatch`、整列极速快照与增量重放。

### When to use Sumeru?

- 构建需要微秒级单步迭代、百万级状态行并发更新的多智能体仿真、物理步进或实时数字孪生系统。
- 在同一份内存状态上同时运行高频规则更新（CPU SIMD）、批量加速核函数（GPU UMA / CUDA）与实时列式分析（Apache Arrow）。
- 需要确定性时间回溯（Time-Travel）、帧间增量复制与两阶段原子事务保护的内存状态底座。

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

    // 设置状态位
    store.set_flag::<0>(entity, true)?;

    // 链式过滤并批量更新
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

    // 查询 baseline_epoch 之后发生变更的数据
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

在 `Cargo.toml` 中添加依赖：

```toml
[dependencies]
sumeru = { git = "https://github.com/XYCaptain/Sumeru.git" }
```

## Examples & Benchmarks

使用 `cargo run --example <name>` 运行内置示例与测试套件：

- `flecs`（85 个场景）：动态列绑定、实体关系对、多阶段流水线、事件监听与反射序列化
- `gpu`（13 个场景）：16KB 页对齐零拷贝 GPU 计算、强类型 Shader IR 与三层并行调度
- `gpu_multi_kernel_pass`：统一 `#[kernel]` 算子、`Vec3` 向量运算、直接列视图元组解构与零气泡多 Kernel 计算通道

```bash
cargo test                                              # 运行核心单元与开源安全边界测试
cargo run --example flecs                               # 运行状态与关系建模示例集（85 个场景）
cargo run --example gpu --features gpu                  # 运行 16KB 零拷贝 GPU 异构计算示例集（13 个场景）
cargo run --example gpu_multi_kernel_pass --features gpu # 运行多 Kernel 零气泡 GPU 计算通道与 #[kernel] 演示
cargo bench --bench store_benchmark                     # 运行连续列存与查询吞吐基准测试
```

## Architecture

<p align="center">
  <img src="./assets/pipeline_zh.svg" alt="Sumeru Architecture" width="840" />
</p>

Sumeru 采用面向统一内存的列式流水线：写入经两阶段事务提交至 16KB 页对齐连续列，再由三级缓存调度器无锁分发至异构计算目标：

- **页对齐连续列存**：状态按列组合聚簇于 16KB 页对齐内存中，配合 8 路原位位平面与 64 行脏纪元，消除指针追逐与跨表迁移。
- **三级缓存无锁调度**：构建期基于读写集位图静态消解冲突（L1），运行期通过列槽位预绑定（L2）与 32–64KB 缓存切片（L3）无锁派发任务。
- **零拷贝异构计算**：同一份连续列内存既可由多核 CPU SIMD 原位步进，也能零拷贝映射至 GPU（Metal UMA / wgpu）或导出为 Apache Arrow `RecordBatch`。

## Features & Roadmap

**Storage & Schema**
- [x] **统一内存页对齐列存与写时复制**：基于 16KB 硬件页对齐连续列缓冲与块级写时复制（CoW），结合代际实体寻址与平坦列指针缓存，实现 CPU SIMD 与 GPU 统一内存的零拷贝物理共址
- [x] **关系型 ECS 与原位多维状态建模**：原生支持表结构组件、`(Relation, Target)` 实体关系对、零大小标签、全局单例及 8 路非迁移原位状态位平面，消除高频状态切换引发的跨表数据搬移
- [x] **两阶段确定性原子事务提交**：支持构建期预校验、同行多补丁自动折叠与结构性拓扑变更的批量原子提交，免除运行期锁竞争与逐行事务开销
- [ ] **原型继承体系与参数化组件模板（`IsA` Hierarchy & Prefabs）**：支持基于 `IsA` 关系的层次化原型共享与写时覆盖，实现大规模同构实体的零冗余内存复用与动态派生

**Query & Filtering**
- [x] **多级缓存编译期查询引擎**：融合列倒排位图索引与增量拓扑缓存（L1）、二级列槽位预绑定（L2）及多列联合游标，将运行期模式匹配收敛为常数级顺序扫描
- [x] **硬件字位图跳跃过滤与脏纪元追踪**：集成独立无状态位图原语库，结合 64 位硬件字位图跳跃、连续跑程折叠、关系目标索引与 64 行块级写纪元，实现增量定点计算
- [ ] **传递性关系图查询与最坏情况最优连接（Transitive Graph & WCOJ）**：面向实体关系图提供传递闭包推导、层级树深度遍历缓存与多跳关系连接算子
- [ ] **原生向量化列式聚合与空间索引**：在连续列块上直接下推 SIMD 向量化聚合、分组统计与多维空间邻近查询，弥合实时状态仿真与内存 OLAP 分析边界

**Schedule & Heterogeneous Compute**
- [x] **静态无锁并行依赖图调度器（Disjoint-Access Parallel DAG）**：构建期基于读写集位图静态完成冲突消解与歧义检测，运行期按拓扑波次无锁派发多核任务
- [x] **缓存感知的自适应分块并行（Morsel-Driven Parallelism）**：根据单行字节跨度自适应规划 L1/L2 缓存驻留切片（32–64KB），通过多核工作窃取直借连续列内存实现高吞吐 SIMD 遍历
- [x] **零气泡多 Kernel GPU 异构计算通道**：内置 Rust AST 到强类型 Shader IR 的编译管线（含 Naga 静态验证与死写消除），通过 16KB 硬件脏页位图与显存屏障在单命令缓冲区内流式编码多算子
- [ ] **跨平台异构后端与 CPU-GPU 混合图编排**：拓展 Metal / Vulkan / DirectX 12 / WebGPU 统一异构后端，支持 CPU 与 GPU 算子在全局依赖图中的混合流水线编排与内核即时编译（JIT）

**Snapshots & Ecosystem**
- [x] **零拷贝时间旅行快照与按列裁剪重放**：基于整列零拷贝归档、帧间变更收据与按列裁剪增量重放，实现亚毫秒级确定性状态回溯
- [x] **Apache Arrow 零拷贝内存桥接**：支持物理连续列向 Apache Arrow `RecordBatch` 的原位零拷贝导出，无缝对接现代数据科学分析栈
- [ ] **流式列存交换与云原生开放格式**：支持 Apache Arrow Flight、Parquet 及新一代随机访问列存格式（Vortex / Lance）的零拷贝流式读写
- [ ] **跨语言零拷贝 ABI 与沙箱运行时**：提供标准化 C-ABI、基于 Arrow C Data Interface 的 Python / C++ 零拷贝绑定及 WebAssembly / WebGPU 运行时

## License

MIT License. 详见 [LICENSE](./LICENSE)。
