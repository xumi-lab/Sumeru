//! 16KB 硬件页对齐 GPU 零拷贝、Naga 着色器编译校验与三层遍历体系单元测试 (M0 ~ M5)

use crate::storage::column::page_aligned::{PageAlignedBuffer, HARDWARE_PAGE_SIZE};
use crate::storage::gpu::{
    distance, dot, length, normalize, vec2, vec3, vec4, GpuAssignOp, GpuBinaryOp, GpuBlock,
    GpuComponentSchema, GpuExpr, GpuKernelBindingSpec, GpuKernelDescriptor, GpuLit, GpuLValue,
    GpuMathIntrinsic, GpuScalarKind, GpuStmt, GpuValueType, GpuVectorExt,
};
use crate::storage::gpu::metal::compile_and_validate_pipeline;
use crate::{
    gpu, gpu_component, gpu_kernel, EntityId, GpuKernel, GpuShaderStruct, LayoutOptimization,
    MetalBufferCache, MetalComputeEngine, MetalNoCopyBufferDescriptor, MetalThreadgroupDispatch,
    Store,
};

#[gpu]
pub struct SimPosition {
    pub x: f32,
    pub y: f32,
}

#[gpu]
pub struct SimVelocity {
    pub vx: f32,
    pub vy: f32,
}

#[gpu]
pub struct SimMass {
    pub inv_mass: f32,
}

#[gpu]
pub fn integrate_sim_motion(pos: &mut SimPosition, vel: &SimVelocity) {
    pos.x += vel.vx;
    pos.y += vel.vy;
}

#[gpu]
pub fn damp_velocity(vel: &mut SimVelocity) {
    vel.vx *= 0.5;
    vel.vy *= 0.5;
}

#[gpu]
pub fn integrate_weighted_motion(pos: &mut SimPosition, vel: &SimVelocity, mass: &SimMass) {
    pos.x += vel.vx * mass.inv_mass;
    pos.y += vel.vy * mass.inv_mass;
}

#[gpu]
pub fn clamp_and_early_return(pos: &mut SimPosition, vel: &SimVelocity) {
    let mut step_x: f32 = vel.vx.abs().clamp(0.0, 10.0);
    let scale = 2.0_f32 * 0.5;
    step_x *= scale;
    pos.x += step_x;
    if pos.x > 100.0 {
        return;
    }
    pos.y += vel.vy;
}

#[gpu_component]
pub struct LegacyAttrPoint {
    pub px: f32,
    pub py: f32,
}

#[allow(dead_code)]
#[gpu_kernel(LegacyPointStepKernel)]
pub fn step_legacy_point(pt: &mut LegacyAttrPoint) {
    pt.px += 1.0;
    pt.py += 2.0;
}

#[test]
fn m0_page_aligned_buffer_16kb_invariants_and_zeroed_tail() {
    let mut buf = PageAlignedBuffer::<SimPosition>::with_capacity(10);
    assert_eq!(buf.as_ptr() as usize % HARDWARE_PAGE_SIZE, 0);
    assert_eq!(buf.allocated_bytes(), HARDWARE_PAGE_SIZE);
    assert_eq!(buf.allocated_page_count(), 1);
    assert_eq!(buf.page_count(), 0);
    let init_gen = buf.generation();

    for i in 0..100 {
        buf.push(SimPosition {
            x: i as f32,
            y: (i * 2) as f32,
        });
    }
    assert_eq!(buf.len(), 100);
    assert_eq!(buf.active_bytes(), 100 * 8);
    assert_eq!(buf.page_count(), 1);
    assert_eq!(buf.generation(), init_gen);

    // 验证 16KB 页内活跃数据尾部均为 0（防止 GPU Threadgroup 尾页越界读取未初始化内存）
    let full_page = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.allocated_bytes())
    };
    assert!(full_page[buf.active_bytes()..].iter().all(|&b| b == 0));

    // 验证 pop、swap_remove 与 drain 后被腾出的尾部槽位严格归零 (I6)
    let _ = buf.pop();
    assert_eq!(buf.len(), 99);
    let full_page_after_pop = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.allocated_bytes())
    };
    assert!(full_page_after_pop[buf.active_bytes()..].iter().all(|&b| b == 0));

    let _ = buf.swap_remove(10);
    assert_eq!(buf.len(), 98);
    let full_page_after_swap = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.allocated_bytes())
    };
    assert!(full_page_after_swap[buf.active_bytes()..].iter().all(|&b| b == 0));

    // 扩容超过 1 个 16KB 页（2048 个 SimPosition = 16384B），触发跨页重分配与代际递增
    for i in 98..2500 {
        buf.push(SimPosition {
            x: i as f32,
            y: i as f32,
        });
    }
    assert_eq!(buf.as_ptr() as usize % HARDWARE_PAGE_SIZE, 0);
    assert_eq!(buf.allocated_bytes() % HARDWARE_PAGE_SIZE, 0);
    assert!(buf.allocated_bytes() >= 2 * HARDWARE_PAGE_SIZE);
    assert_eq!(buf.page_count(), 2);
    assert!(buf.generation() > init_gen);

    let drained = buf.drain(0..2500);
    assert_eq!(drained.len(), 2500);
    assert_eq!(buf.len(), 0);
    let full_after_drain = unsafe {
        std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.allocated_bytes())
    };
    assert!(full_after_drain.iter().all(|&b| b == 0));
}

#[test]
fn m1_zero_copy_gpu_views_dirty_bitmap_and_swap_remove_tracking() {
    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();

    // 插入 2200 个实体，跨越 2 个 16KB 物理页（每页 2048 个 8B 的 SimPosition）
    let mut entities = Vec::with_capacity(2200);
    for i in 0..2200 {
        let e = EntityId::from_u128((i + 1) as u128);
        store
            .insert((
                e,
                (
                    SimPosition {
                        x: i as f32,
                        y: 0.0,
                    },
                    SimVelocity { vx: 1.0, vy: 2.0 },
                ),
            ))
            .unwrap();
        entities.push(e);
    }

    let baseline_epoch = store.epoch();
    let views = store.gpu_column_views::<SimPosition>(baseline_epoch).unwrap();
    assert_eq!(views.len(), 1);
    let (_, view) = &views[0];
    assert!(view.is_zero_copy_aligned());
    assert_eq!(view.geometry().page_count, 2);
    assert!(!view.dirty_bitmap().has_any_dirty());

    // 修改第 10 行（位于第 0 个 16KB 页），第 1 页应保持干净
    store
        .modify::<SimPosition, _>(entities[10], |p| p.x += 100.0)
        .unwrap();
    let views_after_mod = store.gpu_column_views::<SimPosition>(baseline_epoch).unwrap();
    let dirty = views_after_mod[0].1.dirty_bitmap();
    assert!(dirty.is_page_dirty(0));
    assert!(!dirty.is_page_dirty(1));
    let runs = dirty.coalesced_runs();
    assert_eq!(runs.len(), 1);
    assert_eq!(runs[0].first_page, 0);
    assert_eq!(runs[0].byte_offset, 0);
    assert_eq!(runs[0].byte_len, HARDWARE_PAGE_SIZE);

    // 记录新 epoch，然后删除位于第 0 页的实体 entities[20]：
    // swap_remove 会将尾部（第 1 页）实体搬移回填到 row 20（第 0 页），必须标记第 0 页物理脏！
    let pre_delete_epoch = store.epoch();
    store.delete_entity(entities[20]).unwrap();
    let views_after_del = store
        .gpu_column_views::<SimPosition>(pre_delete_epoch)
        .unwrap();
    let dirty_del = views_after_del[0].1.dirty_bitmap();
    assert!(
        dirty_del.is_page_dirty(0),
        "swap_remove must mark destination row's 16KB physical page dirty"
    );
}

#[test]
fn m1_gpu_column_pin_cow_safety_and_page_table() {
    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    for i in 0..128 {
        let e = EntityId::from_u128((i + 100) as u128);
        store
            .insert((
                e,
                (SimPosition {
                    x: i as f32,
                    y: 1.0,
                },),
            ))
            .unwrap();
    }

    let pins = store.pin_gpu_columns::<SimPosition>().unwrap();
    assert_eq!(pins.len(), 1);
    let pinned_ptr = pins[0].base_ptr();
    assert_eq!(pinned_ptr as usize % HARDWARE_PAGE_SIZE, 0);
    assert_eq!(pins[0].as_slice()[0].x, 0.0);

    // 在 Pin 存活期间对 Store 发起 CPU 修改：触发 COW 复制，Pin 指向的原 16KB 物理页保持不变！
    let first_entity = EntityId::from_u128(100);
    store
        .modify::<SimPosition, _>(first_entity, |p| p.x = 999.0)
        .unwrap();
    assert_eq!(
        pins[0].as_slice()[0].x,
        0.0,
        "Pinned GPU buffer must remain immutable via COW while async GPU ticket holds it"
    );
    assert_eq!(store.get::<SimPosition>(first_entity).unwrap().x, 999.0);

    let page_table = store.build_gpu_page_table::<SimPosition>().unwrap();
    assert_eq!(page_table.total_active_rows(), 128);
    assert_eq!(page_table.total_pages(), 1);
}

#[test]
fn m2_metal_nocopy_descriptor_cache_and_simd32_grid() {
    let grid_empty = MetalThreadgroupDispatch::compute_1d(0, 256);
    assert_eq!(grid_empty.threadgroup_count, 0);

    let grid_small = MetalThreadgroupDispatch::compute_1d(50, 256);
    assert_eq!(grid_small.threads_per_threadgroup, 64); // 向上对齐到 32 的倍数
    assert_eq!(grid_small.threadgroup_count, 1);

    let grid_large = MetalThreadgroupDispatch::compute_1d(1000, 256);
    assert_eq!(grid_large.threads_per_threadgroup, 256);
    assert_eq!(grid_large.threadgroup_count, 4);
    assert_eq!(grid_large.total_grid_threads, 1024);

    // 极端大行数饱和保护（不发生 u32 算术溢出 panic）
    let grid_huge = MetalThreadgroupDispatch::compute_1d(u32::MAX as usize, 256);
    assert!(grid_huge.threadgroup_count > 0);

    // 非 16KB 对齐指针应当被 MetalNoCopyBufferDescriptor 拒绝
    let unaligned_ptr = (HARDWARE_PAGE_SIZE + 64) as *mut u8;
    assert!(
        MetalNoCopyBufferDescriptor::validate_raw(unaligned_ptr, HARDWARE_PAGE_SIZE, 64, 1)
            .is_err()
    );

    let aligned_ptr = HARDWARE_PAGE_SIZE as *mut u8;
    // active_bytes 超过 allocated_bytes 应当被拒绝
    assert!(
        MetalNoCopyBufferDescriptor::validate_raw(
            aligned_ptr,
            HARDWARE_PAGE_SIZE,
            HARDWARE_PAGE_SIZE + 8,
            1
        )
        .is_err()
    );

    let desc =
        MetalNoCopyBufferDescriptor::validate_raw(aligned_ptr, HARDWARE_PAGE_SIZE, 64, 1).unwrap();
    let mut cache = MetalBufferCache::new();
    let arch_id = crate::id::ArchetypeId(1);
    let cid = crate::id::ComponentTypeId::of::<SimPosition>();

    let (h1, hit1) = cache.get_or_insert(arch_id, cid, &desc);
    assert!(!hit1);
    let (h2, hit2) = cache.get_or_insert(arch_id, cid, &desc);
    assert!(hit2);
    assert_eq!(h1.handle_id, h2.handle_id);

    // 当 generation 递增（重分配）时，缓存自动失效并原地替换新句柄（entries.len() 保持为 1）
    let desc_reallocated =
        MetalNoCopyBufferDescriptor::validate_raw(aligned_ptr, HARDWARE_PAGE_SIZE * 2, 64, 2)
            .unwrap();
    let (h3, hit3) = cache.get_or_insert(arch_id, cid, &desc_reallocated);
    assert!(!hit3);
    assert_ne!(h1.handle_id, h3.handle_id);
    assert_eq!(cache.len(), 1);
}

#[test]
fn m3_multiplatform_shader_generation_and_real_naga_compilation() {
    let msl_struct = SimPosition::msl_struct_definition();
    assert!(msl_struct.contains("struct SimPosition"));
    assert!(msl_struct.contains("float x;"));

    let wgsl_struct = SimPosition::wgsl_struct_definition();
    assert!(wgsl_struct.contains("struct SimPosition"));
    assert!(wgsl_struct.contains("x: f32,"));

    let desc = integrate_sim_motion.descriptor();
    assert_eq!(desc.name(), "integrate_sim_motion");
    assert_eq!(desc.bindings().len(), 2);
    assert!(desc.bindings()[0].mutable);
    assert!(!desc.bindings()[1].mutable);

    assert!(desc.msl_source().contains("kernel void integrate_sim_motion"));
    assert!(desc
        .msl_source()
        .contains("device SimPosition* sumeru_buf_pos [[buffer(0)]]"));
    assert!(desc
        .msl_source()
        .contains("const device SimVelocity* sumeru_buf_vel [[buffer(1)]]"));
    assert!(desc
        .msl_source()
        .contains("if (gid >= uniforms.active_rows)"));

    assert!(desc.wgsl_source().contains("@compute @workgroup_size(256)"));
    assert!(desc
        .wgsl_source()
        .contains("var<storage, read_write> sumeru_buf_pos: array<SimPosition>;"));
    assert!(desc
        .wgsl_source()
        .contains("var<storage, read> sumeru_buf_vel: array<SimVelocity>;"));

    // 真实 Naga WGSL 解析 + IR 类型验证 + MSL 后端代码生成校验
    for kernel_desc in [
        integrate_sim_motion.descriptor(),
        damp_velocity.descriptor(),
        integrate_weighted_motion.descriptor(),
        clamp_and_early_return.descriptor(),
        step_legacy_point.descriptor(),
        scale_velocity_with_lifetime.descriptor(),
    ] {
        let compiled = compile_and_validate_pipeline(kernel_desc)
            .expect("Generated WGSL/MSL shader must pass Naga frontend, IR validator, and MSL backend");
        assert!(compiled.shader_validated);
        assert!(compiled.naga_translated_msl.contains("kernel void"));
    }
}

#[test]
fn m3_three_tier_explicit_traversal_cpu_par_and_gpu() {
    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);

    for i in 0..256 {
        let e = EntityId::from_u128((i + 1) as u128);
        store
            .insert((
                e,
                (
                    SimPosition {
                        x: i as f32,
                        y: (i * 10) as f32,
                    },
                    SimVelocity { vx: 1.0, vy: 2.0 },
                    SimMass { inv_mass: 0.5 },
                ),
            ))
            .unwrap();
    }

    // =========================================================================
    // ① CPU 单线程遍历：无类型标注闭包 `.for_each(|(pos, vel)| ...)` 与算子直传 `.run(op)`
    // =========================================================================
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .for_each(|(pos, vel)| {
            pos.x += vel.vx;
            pos.y += vel.vy;
        });

    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .run(integrate_sim_motion);

    let e0 = EntityId::from_u128(1);
    assert_eq!(
        *store.get::<SimPosition>(e0).unwrap(),
        SimPosition { x: 2.0, y: 4.0 }
    );

    // =========================================================================
    // ② CPU 多核并行遍历 (Rayon)：无类型标注闭包 `.par_for_each` 与算子直传 `.par_run` / `.par().run`
    // =========================================================================
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .par_for_each(|(pos, vel)| {
            pos.x += vel.vx;
            pos.y += vel.vy;
        });

    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .par_run(integrate_sim_motion);

    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .par()
        .run(integrate_sim_motion);

    assert_eq!(
        *store.get::<SimPosition>(e0).unwrap(),
        SimPosition { x: 5.0, y: 10.0 }
    );

    // =========================================================================
    // ③ GPU 硬件并行遍历：显式 `.gpu().for_each(op)?` / `.gpu().run(op)?`
    // =========================================================================
    let stats1 = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each(integrate_sim_motion)
        .unwrap();
    assert_eq!(stats1.dispatched_archetypes, 1);
    assert_eq!(stats1.total_rows, 256);

    let mut engine = MetalComputeEngine::new();
    let stats2 = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, integrate_sim_motion)
        .unwrap();
    assert_eq!(stats2.buffer_cache_misses, 2);
    assert_eq!(stats2.buffer_cache_hits, 0);

    // 第二帧在同一个 MetalComputeEngine 上派发：100% 命中 MTLBuffer 零拷贝缓存！
    let stats3 = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .run_on(&mut engine, integrate_sim_motion)
        .unwrap();
    assert_eq!(stats3.buffer_cache_hits, 2);
    assert_eq!(stats3.buffer_cache_misses, 0);

    assert_eq!(
        *store.get::<SimPosition>(e0).unwrap(),
        SimPosition { x: 8.0, y: 16.0 }
    );

    // 单参数与三参数 #[gpu] 算子测试
    store
        .query_mut::<&mut SimVelocity>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, damp_velocity)
        .unwrap();
    assert_eq!(
        *store.get::<SimVelocity>(e0).unwrap(),
        SimVelocity { vx: 0.5, vy: 1.0 }
    );

    store
        .query_mut::<(&mut SimPosition, &SimVelocity, &SimMass)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, integrate_weighted_motion)
        .unwrap();
    assert_eq!(
        *store.get::<SimPosition>(e0).unwrap(),
        SimPosition { x: 8.25, y: 16.5 }
    );
}

#[gpu]
pub fn scale_velocity_with_lifetime<'a>(vel: &'a mut SimVelocity) {
    vel.vx *= 2.0;
    vel.vy *= 2.0;
}

#[test]
fn m4_completeness_audit_cow_gpu_dispatch_snapshot_merge_and_ro_par() {
    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store
        .register_gpu_query::<(&mut SimPosition, &mut SimVelocity)>()
        .unwrap();

    for i in 0..64 {
        let e = EntityId::from_u128((i + 500) as u128);
        store
            .insert((
                e,
                (
                    SimPosition {
                        x: (i * 2) as f32,
                        y: 1.0,
                    },
                    SimVelocity { vx: 10.0, vy: 20.0 },
                ),
            ))
            .unwrap();
    }

    let mut engine = MetalComputeEngine::new();
    // 1. 首次 GPU 派发缓存 16KB 句柄
    let stats_init = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, integrate_sim_motion)
        .unwrap();
    assert_eq!(stats_init.buffer_cache_misses, 2);

    // 2. 锁定 SimPosition 列供异步 GPU 票据持有，然后再次执行可变 GPU 派发：
    //    可变列 SimPosition 必须自动触发 COW 独占分离（生成新 16KB 页缓冲与新 generation），
    //    绝不覆盖已 Pin 的旧 16KB 缓冲！而只读列 SimVelocity 则直接命中原零拷贝缓存！
    let pins = store.pin_gpu_columns::<SimPosition>().unwrap();
    assert_eq!(pins[0].as_slice()[0].x, 10.0);

    let stats_cow = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, integrate_sim_motion)
        .unwrap();
    assert_eq!(
        pins[0].as_slice()[0].x,
        10.0,
        "Pinned 16KB buffer must not be overwritten by subsequent GPU mutable dispatch"
    );
    assert_eq!(
        store.get::<SimPosition>(EntityId::from_u128(500)).unwrap().x,
        20.0
    );
    // SimPosition 因 COW 分配了新 generation（1 miss），SimVelocity 保持只读命中（1 hit）
    assert_eq!(stats_cow.buffer_cache_misses, 1);
    assert_eq!(stats_cow.buffer_cache_hits, 1);
    drop(pins);

    // 3. 显式生命周期标注的 #[gpu] 算子测试
    store
        .query_mut::<&mut SimVelocity>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, scale_velocity_with_lifetime)
        .unwrap();
    assert_eq!(
        *store.get::<SimVelocity>(EntityId::from_u128(500)).unwrap(),
        SimVelocity { vx: 20.0, vy: 40.0 }
    );

    // 4. 只读查询 (store.query) 的无类型标注闭包 `.for_each` 与 `.par_for_each` 对称性测试 (C8)
    let mut sum_x = 0.0f32;
    store
        .query::<(&SimPosition, &SimVelocity)>()
        .for_each(|(pos, _vel)| {
            sum_x += pos.x;
        });
    assert!(sum_x > 0.0);

    let par_count = std::sync::atomic::AtomicUsize::new(0);
    store
        .query::<(&SimPosition, &SimVelocity)>()
        .par_for_each(|(pos, vel)| {
            if pos.x >= 0.0 && vel.vx >= 0.0 {
                par_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });
    assert_eq!(par_count.load(std::sync::atomic::Ordering::Relaxed), 64);

    // 5. 快照恢复与分区合并后 16KB 页对齐不变量保持验证
    let snap = store.snapshot().unwrap();
    let mut restored_store =
        Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    restored_store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();
    restored_store.restore_snapshot(&snap).unwrap();
    let restored_views = restored_store.gpu_column_views::<SimPosition>(0).unwrap();
    assert!(restored_views[0].1.is_zero_copy_aligned());

    let mut partition = Store::new(); // 默认 Compact 布局的独立分区
    partition
        .insert((
            EntityId::from_u128(9999),
            (
                SimPosition { x: 1.0, y: 2.0 },
                SimMass { inv_mass: 1.0 },
            ),
        ))
        .unwrap();
    restored_store.merge_partition(partition).unwrap();
    let merged_views = restored_store.gpu_column_views::<SimPosition>(0).unwrap();
    assert_eq!(merged_views.len(), 2);
    assert!(
        merged_views.iter().all(|(_, v)| v.is_zero_copy_aligned()),
        "All archetypes after merge_partition into a PageAlignedGpu store must be 16KB aligned"
    );
}

#[test]
fn m5_mechanism_audit_regressions_c1_c3_c4_c5_c7() {
    // -------------------------------------------------------------------------
    // C1: `gpu_column_views_mut` 必须在 `GpuColumnPin` 存活时触发 COW 独占分离
    // -------------------------------------------------------------------------
    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store.register_gpu_component::<SimPosition>().unwrap();
    let e1 = EntityId::from_u128(1);
    store
        .insert((e1, (SimPosition { x: 42.0, y: 7.0 },)))
        .unwrap();

    let pins = store.pin_gpu_columns::<SimPosition>().unwrap();
    assert_eq!(pins[0].as_slice()[0].x, 42.0);
    let pinned_addr = pins[0].base_ptr() as usize;

    {
        let mut mut_views = store.gpu_column_views_mut::<SimPosition>().unwrap();
        assert_eq!(mut_views.len(), 1);
        let new_addr = mut_views[0].1.base_mut_ptr() as usize;
        assert_ne!(
            new_addr, pinned_addr,
            "gpu_column_views_mut must allocate a fresh COW 16KB buffer when column is pinned"
        );
        mut_views[0].1.as_mut_slice()[0].x = 777.0;
    }
    assert_eq!(
        pins[0].as_slice()[0].x,
        42.0,
        "Pinned buffer must remain untouched after mutation through gpu_column_views_mut"
    );
    assert_eq!(store.get::<SimPosition>(e1).unwrap().x, 777.0);
    drop(pins);

    // -------------------------------------------------------------------------
    // C3: 冷表 (Cold Archetype) 在主 Store 经过多次其他表写入后发生 `swap_remove`，
    //     必须使用当前 Store 的全局 `write_epoch` 标记被覆盖行，不能被增量同步漏掉
    // -------------------------------------------------------------------------
    let mut cold_hot_store =
        Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    cold_hot_store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();

    // 表 A（冷表）：仅含 SimPosition，包含 3 个实体
    let cold_e0 = EntityId::from_u128(10);
    let cold_e1 = EntityId::from_u128(11);
    let cold_e2 = EntityId::from_u128(12);
    for (idx, e) in [cold_e0, cold_e1, cold_e2].into_iter().enumerate() {
        cold_hot_store
            .insert((
                e,
                (SimPosition {
                    x: (idx + 1) as f32,
                    y: 0.0,
                },),
            ))
            .unwrap();
    }

    // 表 B（热表）：含 (SimPosition, SimVelocity)，连续发生 20 次修改推进 Store 全局 epoch
    let hot_e = EntityId::from_u128(20);
    cold_hot_store
        .insert((
            hot_e,
            (
                SimPosition { x: 0.0, y: 0.0 },
                SimVelocity { vx: 1.0, vy: 1.0 },
            ),
        ))
        .unwrap();
    for _ in 0..20 {
        cold_hot_store
            .modify::<SimVelocity, _>(hot_e, |v| v.vx += 1.0)
            .unwrap();
    }

    // 记录删除前的全局快照纪元（此时 global epoch 远大于冷表局部的旧 epoch）
    let before_cold_delete_epoch = cold_hot_store.epoch();

    // 删除冷表第 0 行实体 cold_e0：cold_e2 被 swap_remove 回填到第 0 行
    cold_hot_store.delete_entity(cold_e0).unwrap();

    // 增量脏页视图与 ChangedSince 查询都必须捕获到冷表第 0 页的变更！
    let cold_views = cold_hot_store
        .gpu_column_views::<SimPosition>(before_cold_delete_epoch)
        .unwrap();
    let cold_arch_view = cold_views
        .iter()
        .find(|(_, v)| v.len() == 2)
        .expect("cold archetype must have 2 remaining entities");
    assert!(
        cold_arch_view.1.dirty_bitmap().is_page_dirty(0),
        "Cold archetype swap_remove must stamp global store epoch so dirty page is detected"
    );

    let changed_entities: Vec<EntityId> = cold_hot_store
        .query::<&SimPosition>()
        .changed_since::<SimPosition>(before_cold_delete_epoch)
        .iter()
        .map(|(e, _)| e)
        .collect();
    assert!(
        changed_entities.contains(&cold_e2),
        "ChangedSince must detect entity moved by swap_remove in a cold archetype"
    );

    // -------------------------------------------------------------------------
    // C4 & C5: 空的 PageAlignedGpu 主库先 `register_gpu_query`，再走 `merge_partition` 快路径
    //          合并 Compact 分区，随后执行 `snapshot()` + `restore_snapshot()`
    // -------------------------------------------------------------------------
    let mut empty_target =
        Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    empty_target
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();

    let mut worker_partition = Store::new();
    for i in 0..32 {
        worker_partition
            .insert((
                EntityId::from_u128((i + 1000) as u128),
                (
                    SimPosition {
                        x: i as f32,
                        y: 1.0,
                    },
                    SimVelocity { vx: 2.0, vy: 3.0 },
                ),
            ))
            .unwrap();
    }

    // 走 merge_disjoint_from 的 empty fast-path
    empty_target.merge_partition(worker_partition).unwrap();
    let fast_merged_views = empty_target.gpu_column_views::<SimPosition>(0).unwrap();
    assert_eq!(fast_merged_views.len(), 1);
    assert!(
        fast_merged_views[0].1.is_zero_copy_aligned(),
        "Fast-path merge into empty PageAlignedGpu store must promote columns to 16KB page alignment"
    );

    // 合并后做 snapshot 与 restore_snapshot 必须成功（Rkyv codec 完整保留）
    let snap = empty_target.snapshot().unwrap();
    let mut restored = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    restored
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();
    restored.restore_snapshot(&snap).unwrap();
    assert_eq!(restored.len(), 32);
    assert!(
        restored.gpu_column_views::<SimPosition>(0).unwrap()[0]
            .1
            .is_zero_copy_aligned()
    );

    // -------------------------------------------------------------------------
    // C7: 含提前 `return;` 的 `#[gpu]` 内核在 WGSL 与 CPU 中均不得丢失提前返回前的可变写入
    // -------------------------------------------------------------------------
    let early_ret_desc = clamp_and_early_return.descriptor();
    assert!(
        early_ret_desc
            .wgsl_source()
            .contains("if (pos.x > 100.0) { sumeru_buf_pos[gid] = pos; return; }"),
        "WGSL early return must write back mutable buffers before returning: {}",
        early_ret_desc.wgsl_source()
    );

    let mut ret_store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    let e_high = EntityId::from_u128(9001);
    let e_low = EntityId::from_u128(9002);
    ret_store
        .insert((
            e_high,
            (
                SimPosition { x: 98.0, y: 10.0 },
                SimVelocity { vx: 5.0, vy: 50.0 },
            ),
        ))
        .unwrap();
    ret_store
        .insert((
            e_low,
            (
                SimPosition { x: 10.0, y: 10.0 },
                SimVelocity { vx: 5.0, vy: 50.0 },
            ),
        ))
        .unwrap();

    ret_store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each(clamp_and_early_return)
        .unwrap();

    // e_high: pos.x 先加 5.0 变为 103.0 (> 100.0)，触发提前 return，pos.y 保持 10.0，但 pos.x 的修改保留为 103.0！
    assert_eq!(
        *ret_store.get::<SimPosition>(e_high).unwrap(),
        SimPosition { x: 103.0, y: 10.0 }
    );
    // e_low: pos.x 变为 15.0 (<= 100.0)，继续执行 pos.y += 50.0 变为 60.0
    assert_eq!(
        *ret_store.get::<SimPosition>(e_low).unwrap(),
        SimPosition { x: 15.0, y: 60.0 }
    );
}

#[test]
fn m6_runtime_gpu_detection_driver_dispatch_and_morsel_par() {
    use crate::storage::gpu::metal::{
        probe_system_gpu_driver, GpuExecutionMode, GpuHardwareDriver,
    };
    use crate::storage::gpu::{GpuDispatchUniforms, GpuKernelDescriptor};
    use std::sync::{Arc, Mutex};

    // 1. 运行时探测：根据宿主机真实硬件决定 HardwareGpu 或 CpuFallback（绝不硬编码死分支）
    let probed = probe_system_gpu_driver();
    let auto_engine = MetalComputeEngine::new();
    if probed.is_some() {
        assert_eq!(auto_engine.execution_mode(), GpuExecutionMode::HardwareGpu);
        assert!(auto_engine.has_hardware_gpu());
    } else {
        assert_eq!(auto_engine.execution_mode(), GpuExecutionMode::CpuFallback);
        assert!(!auto_engine.has_hardware_gpu());
    }

    // 2. 硬件驱动分支验证：注入自定义 GpuHardwareDriver，验证 dispatch_archetype_raw 走真实驱动路径，
    //    且当缓存发生 COW 世代替换或驱逐时自动调用 unregister_buffer 回收旧 GPU 句柄。
    #[derive(Debug, Default)]
    struct RecordedDriverState {
        compiled_kernels: Vec<String>,
        registered_handles: Vec<u64>,
        unregistered_handles: Vec<u64>,
        dispatched_rows: Vec<usize>,
    }

    #[derive(Debug)]
    struct MockHardwareGpuDriver {
        state: Arc<Mutex<RecordedDriverState>>,
    }

    impl GpuHardwareDriver for MockHardwareGpuDriver {
        fn device_name(&self) -> &str {
            "Mock-UMA-Hardware-GPU"
        }

        fn has_unified_memory(&self) -> bool {
            true
        }

        fn compile_pipeline(
            &mut self,
            _descriptor: &GpuKernelDescriptor,
            compiled: &crate::MetalCompiledPipeline,
        ) -> Result<u32, crate::StateError> {
            self.state
                .lock()
                .unwrap()
                .compiled_kernels
                .push(compiled.entry_point.to_string());
            Ok(256)
        }

        unsafe fn register_no_copy_buffer(
            &mut self,
            handle_id: u64,
            desc: &MetalNoCopyBufferDescriptor,
        ) -> Result<(), crate::StateError> {
            assert!(
                (desc.base_ptr as usize).is_multiple_of(HARDWARE_PAGE_SIZE)
                    && desc.allocated_bytes.is_multiple_of(HARDWARE_PAGE_SIZE),
                "Hardware driver must only receive 16KB page-aligned zero-copy buffers"
            );
            self.state
                .lock()
                .unwrap()
                .registered_handles
                .push(handle_id);
            Ok(())
        }

        fn unregister_buffer(&mut self, handle_id: u64) {
            self.state
                .lock()
                .unwrap()
                .unregistered_handles
                .push(handle_id);
        }

        unsafe fn dispatch_compute(
            &mut self,
            descriptor: &GpuKernelDescriptor,
            buffer_handles: &[u64],
            raw_ptrs: &[*mut u8],
            active_rows: usize,
            _grid: &MetalThreadgroupDispatch,
            uniforms: &GpuDispatchUniforms,
        ) -> Result<(), crate::StateError> {
            assert_eq!(buffer_handles.len(), raw_ptrs.len());
            self.state
                .lock()
                .unwrap()
                .dispatched_rows
                .push(active_rows);
            // 模拟 GPU 硬件在 UMA 零拷贝页上直接执行着色器逻辑
            unsafe {
                (descriptor.cpu_runner())(raw_ptrs, active_rows, uniforms);
            }
            Ok(())
        }
    }

    let recorded = Arc::new(Mutex::new(RecordedDriverState::default()));
    let mut hw_engine = MetalComputeEngine::with_driver(Box::new(MockHardwareGpuDriver {
        state: Arc::clone(&recorded),
    }));
    assert_eq!(hw_engine.execution_mode(), GpuExecutionMode::HardwareGpu);
    assert_eq!(hw_engine.device_name(), "Mock-UMA-Hardware-GPU");

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();
    for i in 0..128u128 {
        store
            .insert((
                EntityId::from_u128(i + 1),
                (
                    SimPosition {
                        x: i as f32,
                        y: 1.0,
                    },
                    SimVelocity { vx: 2.0, vy: 3.0 },
                ),
            ))
            .unwrap();
    }

    let stats1 = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut hw_engine, integrate_sim_motion)
        .unwrap();
    assert_eq!(stats1.execution_mode, GpuExecutionMode::HardwareGpu);
    assert_eq!(stats1.total_rows, 128);
    assert_eq!(stats1.buffer_cache_misses, 2);
    assert_eq!(
        *store.get::<SimPosition>(EntityId::from_u128(1)).unwrap(),
        SimPosition { x: 2.0, y: 4.0 }
    );

    // 持有零拷贝列锁定票据 (pin_gpu_columns) 期间发起第二次 GPU 可变写入：
    // Phase 1b 触发 Arc::make_mut COW 克隆并自增 buffer_generation，
    // MetalBufferCache 按 (archetype_id, component_id) 替换旧条目并通知硬件驱动 unregister_buffer
    let pins = store.pin_gpu_columns::<SimPosition>().unwrap();
    let stats2 = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut hw_engine, integrate_sim_motion)
        .unwrap();
    drop(pins);
    assert_eq!(stats2.execution_mode, GpuExecutionMode::HardwareGpu);
    assert_eq!(stats2.buffer_cache_hits, 1); // 只读列 SimVelocity 保持原世代命中
    assert_eq!(stats2.buffer_cache_misses, 1); // 可写列 SimPosition 因 COW 世代递增重新注册

    {
        let guard = recorded.lock().unwrap();
        assert_eq!(guard.compiled_kernels, vec!["integrate_sim_motion"]);
        assert_eq!(guard.dispatched_rows, vec![128, 128]);
        assert_eq!(guard.registered_handles.len(), 3);
        assert_eq!(
            guard.unregistered_handles.len(),
            1,
            "COW generation bump must unregister stale MTLBuffer handle from hardware driver"
        );
    }

    // 3. Morsel-Driven 零物化并行迭代验证（> 2500 行跨多个 1024 行 Morsel 分片）
    const ACTIVE_FLAG: u8 = 1;
    let mut par_store = Store::new();
    for i in 0..2560u128 {
        let e = EntityId::from_u128(i + 10_000);
        par_store
            .insert((
                e,
                (
                    SimPosition {
                        x: i as f32,
                        y: (i * 2) as f32,
                    },
                    SimVelocity { vx: 1.5, vy: -0.5 },
                ),
            ))
            .unwrap();
        if i % 2 == 0 {
            par_store.set_flag::<ACTIVE_FLAG>(e, true).unwrap();
        }
    }

    // 3a. 带 with_flag 过滤的跨 Morsel `par_for_each` 可变并行更新
    par_store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .with_flag::<ACTIVE_FLAG>()
        .par_for_each(|(pos, vel)| {
            pos.x += vel.vx;
            pos.y += vel.vy;
        });

    // 验证偶数行（有 ACTIVE_FLAG）已更新，奇数行未变
    assert_eq!(
        *par_store
            .get::<SimPosition>(EntityId::from_u128(10_000))
            .unwrap(),
        SimPosition { x: 1.5, y: -0.5 }
    );
    assert_eq!(
        *par_store
            .get::<SimPosition>(EntityId::from_u128(10_001))
            .unwrap(),
        SimPosition { x: 1.0, y: 2.0 }
    );
    assert_eq!(
        *par_store
            .get::<SimPosition>(EntityId::from_u128(12_558))
            .unwrap(),
        SimPosition {
            x: 2558.0 + 1.5,
            y: 5116.0 - 0.5
        }
    );

    // 3b. 只读 SelectQuery 跨 Morsel `par().for_each` 并行聚合
    use std::sync::atomic::{AtomicUsize, Ordering};
    let visited = AtomicUsize::new(0);
    par_store
        .query::<(&SimPosition, &SimVelocity)>()
        .with_flag::<ACTIVE_FLAG>()
        .par()
        .for_each(|(_pos, _vel)| {
            visited.fetch_add(1, Ordering::Relaxed);
        });
    assert_eq!(visited.load(Ordering::Relaxed), 1280);

    // 4. M7-1 & M7-2 机制完备性回归：
    //    - M7-1: 不匹配查询 Q 的其它 Archetype（例如仅含 SimPosition 不含 SimVelocity）绝不能在 query_mut 遍历中被误标脏页；
    //    - M7-2: 被 pin_gpu_columns 锁定的只读列（SimVelocity）在 par_for_each 中绝不触发多余的 COW 克隆。
    let pos_only_entity = EntityId::from_u128(999_001);
    store
        .insert((pos_only_entity, (SimPosition { x: 42.0, y: 42.0 },)))
        .unwrap();
    let epoch_before_mut = store.epoch();
    let vel_pins = store.pin_gpu_columns::<SimVelocity>().unwrap();
    let vel_gen_before = vel_pins[0].generation();

    // CPU 顺序 query_mut::<(&mut SimPosition, &SimVelocity)> 与并行 par_for_each
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .for_each(|(pos, vel)| {
            pos.x += vel.vx;
        });
    store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .par_for_each(|(pos, vel)| {
            pos.y += vel.vy;
        });

    // 验证仅含 SimPosition 的不匹配 Archetype 未被误标脏
    let pos_views = store.gpu_column_views::<SimPosition>(epoch_before_mut).unwrap();
    let pos_only_view = pos_views
        .iter()
        .find(|(_, v)| v.len() == 1)
        .expect("pos-only archetype must exist");
    assert!(
        !pos_only_view.1.dirty_bitmap().has_any_dirty(),
        "Unmatched archetype must not be marked dirty by query_mut::<(&mut SimPosition, &SimVelocity)>"
    );

    // 验证只读列 SimVelocity 在持有 Pin 期间执行 par_for_each 未触发 COW 重分配（generation 不变）
    let vel_views = store.gpu_column_views::<SimVelocity>(0).unwrap();
    assert_eq!(
        vel_views[0].1.generation(),
        vel_gen_before,
        "Read-only column in par_for_each must not trigger COW clone while pinned"
    );
    drop(vel_pins);

    // 5. M8-1..M8-5 深度机制回归：
    //    - M8-1: restore_snapshot_owned 必须与 restore_snapshot 一致，推进并烙印 next_epoch 且归一化 16KB 页对齐；
    //    - M8-3 & M8-4: merge_disjoint_from 合并持有 GpuColumnPin 的源分区时不得触发源列多余 COW，且自动升级目标 Store 的 16KB 页对齐原型；
    //    - M8-5: Store::clone 将 epoch 重置为基线 1 时，Archetype 块纪元同步重置为 1，确保 changed_since(cloned.epoch()) 不会误报历史块。
    let snap_owned = store.snapshot().unwrap();
    let pre_owned_restore_epoch = store.epoch();
    store.restore_snapshot_owned(snap_owned).unwrap();
    assert!(
        store.epoch() > pre_owned_restore_epoch,
        "restore_snapshot_owned must advance epoch"
    );
    let owned_restored_views = store
        .gpu_column_views::<SimPosition>(pre_owned_restore_epoch)
        .unwrap();
    assert!(
        owned_restored_views
            .iter()
            .all(|(_, v)| v.is_zero_copy_aligned() && v.dirty_bitmap().has_any_dirty()),
        "restore_snapshot_owned must preserve 16KB page alignment and stamp dirty epoch > pre_owned_restore_epoch"
    );

    let mut cloned_store = store.clone();
    assert_eq!(cloned_store.epoch(), 1);
    assert_eq!(
        cloned_store
            .query::<&SimPosition>()
            .changed_since::<SimPosition>(cloned_store.epoch())
            .iter()
            .count(),
        0,
        "Freshly cloned Store must not report pre-clone rows as changed_since(cloned.epoch())"
    );
    cloned_store
        .get_mut::<SimPosition>(EntityId::from_u128(1))
        .unwrap()
        .x += 1.0;
    assert_eq!(
        cloned_store
            .query::<&SimPosition>()
            .changed_since::<SimPosition>(1)
            .iter()
            .count(),
        64,
        "Cloned Store mutation on row 0 must only mark chunk 0 (64 rows out of 128) as changed_since(1)"
    );

    // 6. M9-1..M9-4 深度机制回归：
    //    - M9-2: update_batch 仅修改第 0 块（row 0）时，绝不能把第 1 块（rows 64..128）误标脏；
    //    - M9-3: CommandBuffer 原位单组件/多组件更新仅修改第 0 块时，绝不能把第 1 块误标脏；
    //    - M9-1: EntityBundle 快路径插入已有 Archetype 时必须烙印 next_epoch，使 changed_since 精确捕获新行。
    let epoch_before_direct = cloned_store.epoch();
    cloned_store
        .update_batch_discard(std::iter::once((
            EntityId::from_u128(1),
            SimPosition { x: 77.0, y: 88.0 },
        )))
        .unwrap();
    assert_eq!(
        cloned_store
            .query::<&SimPosition>()
            .changed_since::<SimPosition>(epoch_before_direct)
            .iter()
            .count(),
        64,
        "update_batch_discard on row 0 must only mark chunk 0 (64 rows out of 128) dirty"
    );

    let epoch_before_cmd = cloned_store.epoch();
    let mut cmd = cloned_store.commands();
    cmd.update_batch(std::iter::once((
        EntityId::from_u128(1),
        SimPosition { x: 99.0, y: 111.0 },
    )));
    cmd.apply_discard(&mut cloned_store).unwrap();
    assert_eq!(
        cloned_store
            .query::<&SimPosition>()
            .changed_since::<SimPosition>(epoch_before_cmd)
            .iter()
            .count(),
        64,
        "CommandBuffer native update on row 0 must only mark chunk 0 (64 rows out of 128) dirty"
    );

    let epoch_before_bundle = cloned_store.epoch();
    let mut bundle = crate::EntityBundle::new();
    bundle.insert(SimPosition { x: 5.0, y: 6.0 }).unwrap();
    cloned_store
        .create_entity(EntityId::from_u128(999_002), bundle)
        .unwrap();
    assert!(
        cloned_store
            .query::<&SimPosition>()
            .changed_since::<SimPosition>(epoch_before_bundle)
            .iter()
            .any(|(e, _)| e == EntityId::from_u128(999_002)),
        "EntityBundle fast-path insertion into existing archetype must stamp next_epoch"
    );

    // 7. M10-1..M10-6 深度机制回归：
    //    - M10-1: StateSpace (WorldEngine) 的 for_each_chunk_mut* / par_for_each_chunk_mut* 必须同步烙印写入列的块纪元；
    //    - M10-2: Store::modify_tuple 仅标记元组内被借用修改的列，不误标同表内未触碰的其它列；
    //    - M10-4: chunks_mut() 配合行级过滤器返回整表连续切片时，全表所有 64 行块均须标记写纪元，且只读 chunks() 跳过零匹配表；
    //    - M10-5: merge_disjoint_from 向非空 Store 合并时（无论追加至已有原型或新建原型）必须完整保留非迁移行级状态位平面 (flag_planes)；
    //    - M10-6: try_get_mut / get_mut 在缺失组件失败时绝不污染块纪元。
    {
        use crate::al::{All as AlAll, StateSpace};
        let ep_before_al = cloned_store.epoch();
        StateSpace::for_each_chunk_mut2::<SimPosition, SimVelocity, AlAll>(
            &mut cloned_store,
            |_ents, pos, vel| {
                for (p, v) in pos.iter_mut().zip(vel.iter()) {
                    p.x += v.vx;
                }
            },
        );
        assert_eq!(
            cloned_store
                .query::<(&SimPosition, &SimVelocity)>()
                .changed_since::<SimPosition>(ep_before_al)
                .iter()
                .count(),
            128,
            "M10-1: StateSpace::for_each_chunk_mut2 must mark written column SimPosition chunk epochs"
        );
        assert_eq!(
            cloned_store
                .query::<(&SimPosition, &SimVelocity)>()
                .changed_since::<SimVelocity>(ep_before_al)
                .iter()
                .count(),
            0,
            "M10-1: StateSpace::for_each_chunk_mut2 must not mark read-only column SimVelocity dirty"
        );

        let ep_before_al_par = cloned_store.epoch();
        StateSpace::par_for_each_chunk_mut2::<SimPosition, SimVelocity, AlAll>(
            &mut cloned_store,
            |_ents, pos, vel| {
                for (p, v) in pos.iter_mut().zip(vel.iter()) {
                    p.y += v.vy;
                }
            },
        );
        assert_eq!(
            cloned_store
                .query::<(&SimPosition, &SimVelocity)>()
                .changed_since::<SimPosition>(ep_before_al_par)
                .iter()
                .count(),
            128,
            "M10-1: StateSpace::par_for_each_chunk_mut2 must mark written column SimPosition chunk epochs"
        );
    }

    // M10-2: modify_tuple column isolation
    let ep_before_mod_tuple = cloned_store.epoch();
    cloned_store
        .modify_tuple::<(SimPosition,), _>(EntityId::from_u128(1), |(pos,)| {
            pos.x += 10.0;
        })
        .unwrap();
    assert_eq!(
        cloned_store
            .query::<(&SimPosition, &SimVelocity)>()
            .changed_since::<SimPosition>(ep_before_mod_tuple)
            .iter()
            .count(),
        64,
        "M10-2: modify_tuple::<(SimPosition,)> on row 0 must mark SimPosition chunk 0 dirty"
    );
    assert_eq!(
        cloned_store
            .query::<(&SimPosition, &SimVelocity)>()
            .changed_since::<SimVelocity>(ep_before_mod_tuple)
            .iter()
            .count(),
        0,
        "M10-2: modify_tuple::<(SimPosition,)> must NOT mark untouched column SimVelocity dirty"
    );

    // M10-4: chunks() & chunks_mut() with row-level filter
    let ep_after_row0_mod = ep_before_mod_tuple; // only chunk 0 of SimPosition in (SimPosition, SimVelocity) is dirty since ep_before_mod_tuple
    assert_eq!(
        cloned_store
            .query::<&SimPosition>()
            .changed_since::<SimPosition>( cloned_store.epoch())
            .chunks()
            .count(),
        0,
        "M10-4: read-only chunks() must skip archetypes where row filter matches 0 rows"
    );
    let ep_before_chunks_mut = cloned_store.epoch();
    for (_ents, pos_slice) in cloned_store
        .query_mut::<&mut SimPosition>()
        .unwrap()
        .changed_since::<SimPosition>(ep_after_row0_mod)
        .chunks_mut()
    {
        // Mutate a row in chunk 1 (row 100) of the yielded whole-archetype slice
        if pos_slice.len() > 100 {
            pos_slice[100].x += 5.0;
        }
    }
    assert_eq!(
        cloned_store
            .query::<(&SimPosition, &SimVelocity)>()
            .changed_since::<SimPosition>(ep_before_chunks_mut)
            .iter()
            .count(),
        128,
        "M10-4: chunks_mut() yields the entire archetype slice and must mark all chunks of that archetype dirty"
    );

    // M10-5: merge_disjoint_from into non-empty Store preserves flag_planes
    let mut flag_src = Store::new();
    let e_existing_arch = EntityId::from_u128(888_001);
    let e_new_arch = EntityId::from_u128(888_002);
    flag_src
        .insert((
            e_existing_arch,
            (SimPosition { x: 1.0, y: 2.0 }, SimVelocity { vx: 0.1, vy: 0.2 }),
        ))
        .unwrap();
    flag_src
        .insert((e_new_arch, (SimVelocity { vx: 9.0, vy: 9.0 },)))
        .unwrap();
    flag_src.set_flag::<7>(e_existing_arch, true).unwrap();
    flag_src.set_flag::<7>(e_new_arch, true).unwrap();
    cloned_store.merge_disjoint_from(flag_src).unwrap();
    assert!(
        cloned_store.has_flag::<7>(e_existing_arch),
        "M10-5: merge_disjoint_from must preserve row flags when appending to existing archetype"
    );
    assert!(
        cloned_store.has_flag::<7>(e_new_arch),
        "M10-5: merge_disjoint_from must preserve row flags when moving new archetype"
    );

    // M10-6: failed try_get_mut does not advance epoch or stamp chunk epochs
    let ep_before_failed_get_mut = cloned_store.epoch();
    assert!(
        cloned_store
            .try_get_mut::<SimVelocity>(pos_only_entity)
            .is_err()
    );
    assert_eq!(cloned_store.epoch(), ep_before_failed_get_mut);
}

#[gpu]
pub struct SimVec3State {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub mask: [u32; 2],
    pub steps: i32,
}

#[gpu]
pub fn integrate_vec3_with_loops_and_bits(state: &mut SimVec3State) {
    for axis in 0usize..3usize {
        if (state.mask[0] & (1u32 << (axis as u32))) == 0u32 {
            continue;
        }
        state.pos[axis] += state.vel[axis];
    }
    let mut rem: i32 = state.steps;
    let mut acc: f32 = 0.0;
    while rem > 0 {
        rem -= 1;
        acc += 0.5;
        if acc >= 2.0 {
            break;
        }
    }
    state.pos[0] += acc;
    state.mask[1] |= 0x80u32;
}

#[test]
fn m7_shader_vectors_loops_and_persistent_query_gpu_dispatch() {
    use crate::QueryBuilder;

    let desc = integrate_vec3_with_loops_and_bits.descriptor();
    let compiled = compile_and_validate_pipeline(desc)
        .expect("Vector/array + for/while/continue/break kernel must pass Naga validation");
    assert!(compiled.shader_validated);

    let mut store_hw = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    for i in 0..64u128 {
        store_hw
            .insert((
                EntityId::from_u128(i + 1),
                (SimVec3State {
                    pos: [1.0, 2.0, 3.0],
                    vel: [10.0, 20.0, 30.0],
                    // Enable axis 0 and axis 2 (binary 0b101 = 5), skip axis 1
                    mask: [5, 0],
                    steps: 6,
                },),
            ))
            .unwrap();
    }
    let mut store_cpu = store_hw.clone();

    let mut hw_engine = MetalComputeEngine::new();
    let mut cpu_engine = MetalComputeEngine::new_cpu_fallback();

    let q_persistent_hw = QueryBuilder::<&mut SimVec3State>::new()
        .build(&store_hw)
        .unwrap();
    let q_persistent_cpu = QueryBuilder::<&mut SimVec3State>::new()
        .build(&store_cpu)
        .unwrap();

    // Cross-store dispatch on a persistent Query handle must fail with StoreMismatch
    assert!(matches!(
        q_persistent_hw.gpu_for_each_on(&mut store_cpu, &mut cpu_engine, integrate_vec3_with_loops_and_bits),
        Err(crate::error::StateError::StoreMismatch { .. })
    ));

    let stats_hw = q_persistent_hw
        .gpu_for_each_on(&mut store_hw, &mut hw_engine, integrate_vec3_with_loops_and_bits)
        .unwrap();
    let stats_cpu = q_persistent_cpu
        .gpu_for_each_on(&mut store_cpu, &mut cpu_engine, integrate_vec3_with_loops_and_bits)
        .unwrap();
    assert_eq!(stats_hw.total_rows, 64);
    assert_eq!(stats_cpu.total_rows, 64);

    // Axis 0: 1.0 + 10.0 + 2.0 (while loop broken at acc=2.0) = 13.0
    // Axis 1: skipped via `continue` -> remains 2.0
    // Axis 2: 3.0 + 30.0 = 33.0
    for i in 0..64u128 {
        let e = EntityId::from_u128(i + 1);
        let s_hw = *store_hw.get::<SimVec3State>(e).unwrap();
        let s_cpu = *store_cpu.get::<SimVec3State>(e).unwrap();
        assert_eq!(s_hw, s_cpu, "Hardware GPU and CPU fallback must be bit-exact on row {i}");
        assert_eq!(s_hw.pos, [13.0, 2.0, 33.0]);
        assert_eq!(s_hw.mask, [5, 0x80]);
    }

    // Row-filtered persistent Query must be rejected by gpu_for_each_on with zero side effects
    let epoch_before = store_hw.epoch();
    let q_row_filtered = QueryBuilder::<&mut SimVec3State>::new()
        .with_flag::<0>()
        .build(&store_hw)
        .unwrap();
    assert!(
        q_row_filtered
            .gpu_for_each_on(&mut store_hw, &mut hw_engine, integrate_vec3_with_loops_and_bits)
            .is_err()
    );
    assert_eq!(store_hw.epoch(), epoch_before);
}

const TEST_SUBSTEP_DT: f32 = 0.25;
const TEST_MAX_SPEED: f32 = 8.0;

#[gpu]
pub fn helper_clamp_symmetric(val: f32, limit: f32) -> f32 {
    if val > limit {
        limit
    } else if val < -limit {
        -limit
    } else {
        val
    }
}

#[gpu]
pub fn helper_damped_delta(v: f32, damping: f32) -> f32 {
    let bounded = helper_clamp_symmetric(v, TEST_MAX_SPEED);
    bounded * TEST_SUBSTEP_DT * damping
}

#[gpu]
pub fn integrate_with_ir_consts_and_helpers(state: &mut SimVec3State) {
    const INLINE_DAMPING: f32 = 0.5;
    // Notice `0..state.mask[0]` uses an unsuffixed `0`! `ShaderTypeEnv` automatically infers
    // `state.mask[0]` is `u32` from `SimVec3State::FIELDS` and promotes `0` -> `0u`, step -> `1u`.
    for _step in 0..state.mask[0] {
        state.pos[0] += helper_damped_delta(state.vel[0], INLINE_DAMPING);
        state.pos[1] += helper_damped_delta(state.vel[1], std::f32::consts::FRAC_1_SQRT_2 * 0.0 + 1.0);
    }
}

#[test]
fn m8_typed_shader_ir_external_consts_and_composable_helpers() {
    let desc = integrate_with_ir_consts_and_helpers.descriptor();
    // Topological helper order: `helper_clamp_symmetric` must appear before `helper_damped_delta`
    let msl = desc.msl_source();
    let wgsl = desc.wgsl_source();
    let msl_pos_clamp = msl.find("inline float helper_clamp_symmetric").expect("MSL helper 1");
    let msl_pos_damped = msl.find("inline float helper_damped_delta").expect("MSL helper 2");
    assert!(
        msl_pos_clamp < msl_pos_damped,
        "Callee helper must be emitted before caller helper in MSL"
    );

    let wgsl_pos_clamp = wgsl.find("fn helper_clamp_symmetric").expect("WGSL helper 1");
    let wgsl_pos_damped = wgsl.find("fn helper_damped_delta").expect("WGSL helper 2");
    assert!(
        wgsl_pos_clamp < wgsl_pos_damped,
        "Callee helper must be emitted before caller helper in WGSL"
    );
    assert!(
        wgsl.contains("for (var _step = 0u; _step < state.mask[0]; _step += 1u)"),
        "ShaderTypeEnv must infer `state.mask[0]` as u32 and promote `0` to `0u`: {wgsl}"
    );

    let compiled = compile_and_validate_pipeline(desc)
        .expect("Typed IR with external consts and composable helpers must pass Naga validation");
    assert!(compiled.shader_validated);

    let mut store_gpu = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    let e1 = EntityId::from_u128(1);
    store_gpu
        .insert((
            e1,
            (SimVec3State {
                pos: [2.0, 5.0, 0.0],
                vel: [20.0, -4.0, 0.0], // 20.0 clamped to 8.0; -4.0 stays -4.0
                mask: [4, 0],
                steps: 4,
            },),
        ))
        .unwrap();
    let mut store_cpu = store_gpu.clone();

    store_gpu
        .query_mut::<&mut SimVec3State>()
        .unwrap()
        .gpu()
        .for_each(integrate_with_ir_consts_and_helpers)
        .unwrap();

    store_cpu
        .query_mut::<&mut SimVec3State>()
        .unwrap()
        .run(integrate_with_ir_consts_and_helpers);

    // 4 steps:
    // pos[0]: 2.0 + 4 * (8.0 * 0.25 * 0.5) = 2.0 + 4.0 = 6.0
    // pos[1]: 5.0 + 4 * (-4.0 * 0.25 * 1.0) = 5.0 - 4.0 = 1.0
    let s_gpu = *store_gpu.get::<SimVec3State>(e1).unwrap();
    let s_cpu = *store_cpu.get::<SimVec3State>(e1).unwrap();
    assert_eq!(s_gpu, s_cpu);
    assert_eq!(s_gpu.pos, [6.0, 1.0, 0.0]);
}

#[gpu]
pub struct SimDrag {
    pub coeff: f32,
}

#[gpu]
pub struct SimArchetypeMarkerA {
    pub tag: u32,
}

#[gpu]
pub struct SimArchetypeMarkerB {
    pub tag: u32,
}

#[gpu]
pub fn helper_add_vec3_in_place(dst: &mut [f32; 3], delta: [f32; 3]) {
    dst[0] += delta[0];
    dst[1] += delta[1];
    dst[2] += delta[2];
}

#[gpu]
pub fn helper_translate_state_with_drag(
    state: &mut SimVec3State,
    vel: &SimVelocity,
    drag: &SimDrag,
    dt: f32,
) {
    let factor = 1.0 - drag.coeff * dt;
    let delta: [f32; 3] = [vel.vx * dt * factor, vel.vy * dt * factor, 1.5];
    helper_add_vec3_in_place(&mut state.pos, delta);
}

#[gpu]
pub fn integrate_single_uniform(pos: &mut SimPosition, vel: &SimVelocity, dt: f32) {
    pos.x += vel.vx * dt;
    pos.y += vel.vy * dt;
}

#[gpu]
pub fn integrate_5comp_with_uniforms_and_helpers(
    pos: &mut SimPosition,
    vel: &SimVelocity,
    state: &mut SimVec3State,
    mass: &SimMass,
    drag: &SimDrag,
    dt: f32,
    step_inc: u32,
) {
    // 1. Bidirectional u32 literal coercion in assignment, comparison, bitwise, shift, and min/max
    state.mask[0] += 1;
    state.mask[0] += step_inc;
    state.mask[1] = 3;
    if state.mask[0] > 0 && (state.mask[1] & 3) == 3 {
        state.mask[1] = (state.mask[1] << 1) | 1;
    }
    let capped_mask = state.mask[0].min(100);
    state.mask[0] = capped_mask;

    // 2. Composable void helper taking `&mut SimVec3State`, `&SimVelocity`, `&SimDrag`, and `dt: f32`
    helper_translate_state_with_drag(state, vel, drag, dt);

    // 3. Local array repeat `[val; N]` + `loop { ... break; }`
    let mut bonus: [f32; 3] = [0.5; 3];
    let mut iter_count = 0;
    loop {
        bonus[0] += 0.25;
        iter_count += 1;
        if iter_count >= 2 {
            break;
        }
    }
    pos.x += vel.vx * mass.inv_mass * dt + bonus[0];
    pos.y += vel.vy * mass.inv_mass * dt + bonus[1];
}

#[test]
fn m9_dynamic_uniforms_bidirectional_u32_struct_helpers_and_batch_dispatch() {
    use crate::QueryBuilder;

    // 1. Validate shader generation & Naga WGSL/MSL compilation for 5-component + 2-uniform kernel
    let bound = integrate_5comp_with_uniforms_and_helpers.with_args((0.5f32, 2u32));
    let desc = bound.descriptor();
    let wgsl = desc.wgsl_source();
    let msl = desc.msl_source();

    assert!(
        wgsl.contains("state.mask[0] += 1u;"),
        "Bidirectional u32 inference must suffix `1` in `state.mask[0] += 1`: {wgsl}"
    );
    assert!(
        wgsl.contains("min(state.mask[0], 100u)"),
        "Bidirectional u32 inference must suffix `100` in `min(state.mask[0], 100)`: {wgsl}"
    );
    assert!(
        wgsl.contains("let dt: f32 = bitcast<f32>(uniforms.scalar_arg_0);"),
        "Dynamic f32 scalar uniform must be unpacked via bitcast<f32>: {wgsl}"
    );
    assert!(
        wgsl.contains("let step_inc: u32 = uniforms.scalar_arg_1;"),
        "Dynamic u32 scalar uniform must be unpacked from scalar_arg_1: {wgsl}"
    );
    assert!(
        msl.contains("const float dt = as_type<float>(uniforms.scalar_arg_0);"),
        "MSL dynamic f32 scalar uniform must be unpacked via as_type<float>: {msl}"
    );

    let compiled = compile_and_validate_pipeline(desc)
        .expect("5-component + 2-uniform + struct/array-ref helper kernel must pass Naga validation");
    assert!(compiled.shader_validated);

    // Also validate single-uniform kernel ergonomics (`with_args(0.25f32)` without tuple)
    let single_bound = integrate_single_uniform.with_args(0.25f32);
    let single_compiled = compile_and_validate_pipeline(single_bound.descriptor())
        .expect("Single scalar uniform kernel must pass Naga validation");
    assert!(single_compiled.shader_validated);

    // 2. Test 5-tuple RegisterGpuQuery and populate 3 distinct Archetypes to verify
    //    single-MTLCommandBuffer multi-archetype batch dispatch
    let mut store_hw = Store::new();
    store_hw
        .register_gpu_query::<(
            &mut SimPosition,
            &SimVelocity,
            &mut SimVec3State,
            &SimMass,
            &SimDrag,
        )>()
        .unwrap();

    let base_comps = (
        SimPosition { x: 1.0, y: 2.0 },
        SimVelocity { vx: 4.0, vy: 8.0 },
        SimVec3State {
            pos: [10.0, 20.0, 30.0],
            vel: [1.0, 2.0, 3.0],
            mask: [4, 0],
            steps: 1,
        },
        SimMass { inv_mass: 0.5 },
        SimDrag { coeff: 0.2 },
    );

    // Archetype 1: exact 5 components (16 entities)
    for i in 0..16u128 {
        store_hw
            .insert((EntityId::from_u128(100 + i), base_comps))
            .unwrap();
    }
    // Archetype 2: 5 components + SimArchetypeMarkerA (16 entities)
    for i in 0..16u128 {
        store_hw
            .insert((
                EntityId::from_u128(200 + i),
                (
                    base_comps.0,
                    base_comps.1,
                    base_comps.2,
                    base_comps.3,
                    base_comps.4,
                    SimArchetypeMarkerA { tag: 1 },
                ),
            ))
            .unwrap();
    }
    // Archetype 3: 5 components + SimArchetypeMarkerB (16 entities)
    for i in 0..16u128 {
        store_hw
            .insert((
                EntityId::from_u128(300 + i),
                (
                    base_comps.0,
                    base_comps.1,
                    base_comps.2,
                    base_comps.3,
                    base_comps.4,
                    SimArchetypeMarkerB { tag: 2 },
                ),
            ))
            .unwrap();
    }

    let mut store_cpu_fallback = store_hw.clone();
    let mut store_seq = store_hw.clone();
    let mut store_par = store_hw.clone();
    let mut store_persistent = store_hw.clone();

    let mut hw_engine = MetalComputeEngine::new();
    let mut cpu_engine = MetalComputeEngine::new_cpu_fallback();

    type Q5<'a> = (
        &'a mut SimPosition,
        &'a SimVelocity,
        &'a mut SimVec3State,
        &'a SimMass,
        &'a SimDrag,
    );

    // Hardware GPU batch dispatch across 3 archetypes
    let stats_hw = store_hw
        .query_mut::<Q5>()
        .unwrap()
        .gpu()
        .for_each_on(
            &mut hw_engine,
            integrate_5comp_with_uniforms_and_helpers.with_args((0.5f32, 2u32)),
        )
        .unwrap();
    assert_eq!(stats_hw.dispatched_archetypes, 3);
    assert_eq!(stats_hw.total_rows, 48);
    if hw_engine.has_hardware_gpu() {
        assert_eq!(
            stats_hw.command_buffer_submissions, 1,
            "Multi-archetype GPU dispatch must encode all 3 archetypes into a single MTLCommandBuffer submission"
        );
    } else {
        assert_eq!(stats_hw.command_buffer_submissions, 0);
    }

    // CPU Fallback engine batch dispatch
    let stats_cpu = store_cpu_fallback
        .query_mut::<Q5>()
        .unwrap()
        .gpu()
        .for_each_on(
            &mut cpu_engine,
            integrate_5comp_with_uniforms_and_helpers.with_args((0.5f32, 2u32)),
        )
        .unwrap();
    assert_eq!(stats_cpu.dispatched_archetypes, 3);
    assert_eq!(stats_cpu.total_rows, 48);
    assert_eq!(stats_cpu.command_buffer_submissions, 0);

    // CPU sequential `.run(...)` with bound dynamic uniform args
    store_seq
        .query_mut::<Q5>()
        .unwrap()
        .run(integrate_5comp_with_uniforms_and_helpers.with_args((0.5f32, 2u32)));

    // CPU parallel `.par_run(...)` with bound dynamic uniform args
    store_par
        .query_mut::<Q5>()
        .unwrap()
        .par_run(integrate_5comp_with_uniforms_and_helpers.with_args((0.5f32, 2u32)));

    // Persistent `Query::gpu_for_each_on` across 3 archetypes
    let q_persistent = QueryBuilder::<Q5<'static>>::new()
        .build(&store_persistent)
        .unwrap();
    let stats_persistent = q_persistent
        .gpu_for_each_on(
            &mut store_persistent,
            &mut hw_engine,
            integrate_5comp_with_uniforms_and_helpers.with_args((0.5f32, 2u32)),
        )
        .unwrap();
    assert_eq!(stats_persistent.dispatched_archetypes, 3);
    assert_eq!(stats_persistent.total_rows, 48);
    if hw_engine.has_hardware_gpu() {
        assert_eq!(stats_persistent.command_buffer_submissions, 1);
    }

    // Verify all 48 entities across all 3 archetypes match identically across all 5 execution paths
    // Expected values for dt=0.5, step_inc=2:
    // - state.mask[0]: 4 + 1 + 2 = 7
    // - state.mask[1]: (3 << 1) | 1 = 7
    // - factor = 1.0 - 0.2 * 0.5 = 0.9
    // - delta = [4.0 * 0.5 * 0.9, 8.0 * 0.5 * 0.9, 1.5] = [1.8, 3.6, 1.5]
    // - state.pos = [11.8, 23.6, 31.5]
    // - bonus = [0.5 + 0.25*2, 0.5, 0.5] = [1.0, 0.5, 0.5]
    // - pos.x = 1.0 + 4.0 * 0.5 * 0.5 + 1.0 = 3.0
    // - pos.y = 2.0 + 8.0 * 0.5 * 0.5 + 0.5 = 4.5
    for prefix in [100u128, 200, 300] {
        for i in 0..16u128 {
            let e = EntityId::from_u128(prefix + i);
            let p_hw = *store_hw.get::<SimPosition>(e).unwrap();
            let p_cpu = *store_cpu_fallback.get::<SimPosition>(e).unwrap();
            let p_seq = *store_seq.get::<SimPosition>(e).unwrap();
            let p_par = *store_par.get::<SimPosition>(e).unwrap();
            let p_per = *store_persistent.get::<SimPosition>(e).unwrap();
            assert_eq!(p_hw, p_cpu);
            assert_eq!(p_hw, p_seq);
            assert_eq!(p_hw, p_par);
            assert_eq!(p_hw, p_per);
            assert!((p_hw.x - 3.0).abs() < 1e-5 && (p_hw.y - 4.5).abs() < 1e-5);

            let s_hw = *store_hw.get::<SimVec3State>(e).unwrap();
            let s_cpu = *store_cpu_fallback.get::<SimVec3State>(e).unwrap();
            let s_seq = *store_seq.get::<SimVec3State>(e).unwrap();
            let s_par = *store_par.get::<SimVec3State>(e).unwrap();
            let s_per = *store_persistent.get::<SimVec3State>(e).unwrap();
            assert_eq!(s_hw, s_cpu);
            assert_eq!(s_hw, s_seq);
            assert_eq!(s_hw, s_par);
            assert_eq!(s_hw, s_per);
            assert_eq!(s_hw.mask, [7, 7]);
            assert!((s_hw.pos[0] - 11.8).abs() < 1e-5);
            assert!((s_hw.pos[1] - 23.6).abs() < 1e-5);
            assert!((s_hw.pos[2] - 31.5).abs() < 1e-5);
        }
    }
}

#[test]
fn m10_gpu_compute_pass_multi_kernel_bubble_elimination() {
    let mut engine = MetalComputeEngine::new();
    let has_gpu = engine.has_hardware_gpu();

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();

    let old_epoch = store.epoch();
    const NUM_ENTITIES: usize = 256;
    for i in 0..NUM_ENTITIES {
        store
            .insert((
                EntityId::from_u128(i as u128 + 1),
                (
                    SimPosition { x: 10.0, y: 20.0 },
                    SimVelocity { vx: 4.0, vy: 6.0 },
                ),
            ))
            .unwrap();
    }

    // 开启 GpuComputePass：连续调度多 Kernel 并插入显存屏障，单次 commit + wait 消除气泡
    let mut pass = store.gpu_compute_pass(&mut engine).unwrap();
    assert_eq!(pass.dispatches(), 0);

    // Kernel 1: 积分位置 (pos += vel) -> pos 变为 (14.0, 26.0)
    pass.dispatch(integrate_sim_motion).unwrap();
    assert_eq!(pass.dispatches(), 1);

    // 显存屏障: 确保 Kernel 1 的写操作对后续 Kernel 可见
    pass.barrier().unwrap();

    // Kernel 2: 速度阻尼 (vel *= 0.5) -> vel 变为 (2.0, 3.0)
    pass.dispatch(damp_velocity).unwrap();
    assert_eq!(pass.dispatches(), 2);

    // 显存屏障
    pass.barrier().unwrap();

    // Kernel 3: 再次积分位置 (pos += vel) -> pos 变为 (16.0, 29.0)
    pass.dispatch(integrate_sim_motion).unwrap();
    assert_eq!(pass.dispatches(), 3);

    // 提交命令缓冲区并统一等待完成，批量推进写纪元与脏页位图
    let stats = pass.finish().unwrap();

    if has_gpu {
        // 多 Kernel 在单个命令缓冲区中统一提交，仅产生 1 次 hardware command buffer submission！
        assert_eq!(stats.command_buffer_submissions, 1);
    }
    assert_eq!(stats.dispatched_archetypes, 3);
    assert_eq!(stats.total_rows, NUM_ENTITIES * 3);

    // 校验 Store 全局纪元已推进
    assert!(store.epoch() > old_epoch);

    // 校验数据执行结果正确性
    for i in 0..NUM_ENTITIES {
        let e = EntityId::from_u128(i as u128 + 1);
        let pos = *store.get::<SimPosition>(e).unwrap();
        let vel = *store.get::<SimVelocity>(e).unwrap();
        assert!((pos.x - 16.0).abs() < 1e-5, "pos.x was {}", pos.x);
        assert!((pos.y - 29.0).abs() < 1e-5, "pos.y was {}", pos.y);
        assert!((vel.vx - 2.0).abs() < 1e-5, "vel.vx was {}", vel.vx);
        assert!((vel.vy - 3.0).abs() < 1e-5, "vel.vy was {}", vel.vy);
    }

    // 校验 CPU 脏页位图已感知到 GPU 批量写回
    let pos_views = store.gpu_column_views::<SimPosition>(old_epoch).unwrap();
    assert!(!pos_views.is_empty());
    assert!(pos_views[0].1.dirty_bitmap().dirty_page_count() > 0);

    let vel_views = store.gpu_column_views::<SimVelocity>(old_epoch).unwrap();
    assert!(!vel_views.is_empty());
    assert!(vel_views[0].1.dirty_bitmap().dirty_page_count() > 0);
}

#[test]
fn m10_gpu_compute_pass_cpu_fallback_and_backward_compatibility() {
    let mut fallback_engine = MetalComputeEngine::new_cpu_fallback();
    assert_eq!(fallback_engine.execution_mode(), crate::GpuExecutionMode::CpuFallback);

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity)>()
        .unwrap();

    const NUM_ENTITIES: usize = 64;
    for i in 0..NUM_ENTITIES {
        store
            .insert((
                EntityId::from_u128(i as u128 + 1),
                (
                    SimPosition { x: 0.0, y: 0.0 },
                    SimVelocity { vx: 1.0, vy: 2.0 },
                ),
            ))
            .unwrap();
    }

    // 测试链式调用与 execute 别名
    let mut pass = store.gpu_compute_pass(&mut fallback_engine).unwrap();
    pass.for_each(integrate_sim_motion)
        .unwrap()
        .barrier()
        .unwrap()
        .for_each(damp_velocity)
        .unwrap();
    pass.execute().unwrap();

    for i in 0..NUM_ENTITIES {
        let e = EntityId::from_u128(i as u128 + 1);
        let pos = *store.get::<SimPosition>(e).unwrap();
        let vel = *store.get::<SimVelocity>(e).unwrap();
        assert!((pos.x - 1.0).abs() < 1e-5);
        assert!((pos.y - 2.0).abs() < 1e-5);
        assert!((vel.vx - 0.5).abs() < 1e-5);
        assert!((vel.vy - 1.0).abs() < 1e-5);
    }

    // 验证既有单 Query .gpu().for_each(...) API 保持 100% 向后兼容
    let stats = store
        .query_mut::<(&mut SimPosition, &SimVelocity)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut fallback_engine, integrate_sim_motion)
        .unwrap();
    assert_eq!(stats.total_rows, NUM_ENTITIES);

    for i in 0..NUM_ENTITIES {
        let e = EntityId::from_u128(i as u128 + 1);
        let pos = *store.get::<SimPosition>(e).unwrap();
        assert!((pos.x - 1.5).abs() < 1e-5);
        assert!((pos.y - 3.0).abs() < 1e-5);
    }
}

#[test]
fn m10_gpu_compute_pass_encode_query_with_filters() {
    let mut engine = MetalComputeEngine::new();

    let mut store = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    store
        .register_gpu_query::<(&mut SimPosition, &SimVelocity, &SimMass)>()
        .unwrap();

    // Archetype 1: 带 SimMass (inv_mass: 2.0)
    for i in 0..32u128 {
        store
            .insert((
                EntityId::from_u128(100 + i),
                (
                    SimPosition { x: 0.0, y: 0.0 },
                    SimVelocity { vx: 2.0, vy: 3.0 },
                    SimMass { inv_mass: 2.0 },
                ),
            ))
            .unwrap();
    }

    // Archetype 2: 不带 SimMass
    for i in 0..32u128 {
        store
            .insert((
                EntityId::from_u128(200 + i),
                (
                    SimPosition { x: 0.0, y: 0.0 },
                    SimVelocity { vx: 2.0, vy: 3.0 },
                ),
            ))
            .unwrap();
    }

    // 构造预编译长期 Query 并添加原型级过滤：仅匹配带 SimMass 的表
    let query_with_mass = crate::storage::query::QueryBuilder::<(&'static mut SimPosition, &'static SimVelocity)>::new()
        .with::<SimMass>()
        .build(&store)
        .unwrap();

    let mut pass = store.gpu_compute_pass(&mut engine).unwrap();
    pass.encode_query(&query_with_mass, integrate_sim_motion).unwrap();
    let stats = pass.finish().unwrap();

    // 仅 32 行匹配并分派，不带 SimMass 的 32 行未被分派
    assert_eq!(stats.total_rows, 32);

    for i in 0..32u128 {
        let e_with = EntityId::from_u128(100 + i);
        let pos_with = *store.get::<SimPosition>(e_with).unwrap();
        assert!((pos_with.x - 2.0).abs() < 1e-5);
        assert!((pos_with.y - 3.0).abs() < 1e-5);

        let e_without = EntityId::from_u128(200 + i);
        let pos_without = *store.get::<SimPosition>(e_without).unwrap();
        assert_eq!(pos_without.x, 0.0);
        assert_eq!(pos_without.y, 0.0);
    }
}

#[gpu]
pub struct SimVecParticle {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub speed: f32,
}

#[gpu]
pub fn update_vec_particle_m11(p: &mut SimVecParticle, mass: &SimMass) {
    if p.speed < 0.0 {
        return;
    }
    let v = vec3(p.vel[0], p.vel[1], p.vel[2]);
    let len = length(v);
    let dir = v.normalize();
    let dist = v.distance(vec3(0.0, 0.0, 0.0));
    let d = dir.dot(v);
    p.speed = len * mass.inv_mass + (dist - d) * 0.0;
    p.pos[0] += dir[0] * p.speed;
    p.pos[1] += dir[1] * p.speed;
    p.pos[2] += dir[2] * p.speed;
}

#[gpu]
pub fn read_only_inspector_kernel(p: &mut SimPosition, vel: &SimVelocity) {
    let _s = vel.vx + p.x;
    if p.x > 50.0 {
        return;
    }
}

#[test]
fn m11_shader_vector_ir_and_naga_validation() {
    let block = GpuBlock {
        stmts: vec![
            GpuStmt::Let {
                name: "v1",
                mutable: false,
                explicit_ty: Some(GpuValueType::Vector(GpuScalarKind::F32, 3)),
                init: GpuExpr::vec3(
                    GpuExpr::Lit(GpuLit::F32(1.0)),
                    GpuExpr::Lit(GpuLit::F32(2.0)),
                    GpuExpr::Lit(GpuLit::F32(3.0)),
                ),
            },
            GpuStmt::Let {
                name: "v2",
                mutable: false,
                explicit_ty: Some(GpuValueType::Vector(GpuScalarKind::F32, 3)),
                init: GpuExpr::vec3(
                    GpuExpr::Lit(GpuLit::F32(4.0)),
                    GpuExpr::Lit(GpuLit::F32(5.0)),
                    GpuExpr::Lit(GpuLit::F32(6.0)),
                ),
            },
            GpuStmt::Let {
                name: "v_add",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::Binary {
                    op: GpuBinaryOp::Add,
                    lhs: Box::new(GpuExpr::Var("v1")),
                    rhs: Box::new(GpuExpr::Var("v2")),
                },
            },
            GpuStmt::Let {
                name: "v_sub",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::Binary {
                    op: GpuBinaryOp::Sub,
                    lhs: Box::new(GpuExpr::Var("v2")),
                    rhs: Box::new(GpuExpr::Var("v1")),
                },
            },
            GpuStmt::Let {
                name: "v_mul",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::Binary {
                    op: GpuBinaryOp::Mul,
                    lhs: Box::new(GpuExpr::Var("v_add")),
                    rhs: Box::new(GpuExpr::Lit(GpuLit::F32(0.5))),
                },
            },
            GpuStmt::Let {
                name: "d",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::MathCall {
                    receiver: Box::new(GpuExpr::Var("v1")),
                    intrinsic: GpuMathIntrinsic::Dot,
                    args: vec![GpuExpr::Var("v2")],
                },
            },
            GpuStmt::Let {
                name: "len",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::MathCall {
                    receiver: Box::new(GpuExpr::Var("v_mul")),
                    intrinsic: GpuMathIntrinsic::Length,
                    args: vec![],
                },
            },
            GpuStmt::Let {
                name: "norm",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::MathCall {
                    receiver: Box::new(GpuExpr::Var("v1")),
                    intrinsic: GpuMathIntrinsic::Normalize,
                    args: vec![],
                },
            },
            GpuStmt::Let {
                name: "dist",
                mutable: false,
                explicit_ty: None,
                init: GpuExpr::MathCall {
                    receiver: Box::new(GpuExpr::Var("v1")),
                    intrinsic: GpuMathIntrinsic::Distance,
                    args: vec![GpuExpr::Var("v2")],
                },
            },
            GpuStmt::Assign {
                op: GpuAssignOp::AddAssign,
                target: GpuLValue::Field {
                    base: Box::new(GpuLValue::Var("pos")),
                    field: "x",
                },
                value: GpuExpr::Binary {
                    op: GpuBinaryOp::Add,
                    lhs: Box::new(GpuExpr::Var("d")),
                    rhs: Box::new(GpuExpr::Binary {
                        op: GpuBinaryOp::Add,
                        lhs: Box::new(GpuExpr::Var("len")),
                        rhs: Box::new(GpuExpr::Var("dist")),
                    }),
                },
            },
            GpuStmt::Assign {
                op: GpuAssignOp::AddAssign,
                target: GpuLValue::Field {
                    base: Box::new(GpuLValue::Var("pos")),
                    field: "y",
                },
                value: GpuExpr::Index {
                    base: Box::new(GpuExpr::Var("norm")),
                    index: Box::new(GpuExpr::Lit(GpuLit::UntypedInt(1))),
                },
            },
        ],
    };

    let comp = GpuComponentSchema {
        param_name: "pos",
        struct_name: <SimPosition as GpuShaderStruct>::STRUCT_NAME,
        fields: <SimPosition as GpuShaderStruct>::FIELDS,
        mutable: true,
    };
    let binding = GpuKernelBindingSpec {
        param_name: "pos",
        struct_name: <SimPosition as GpuShaderStruct>::STRUCT_NAME,
        type_id: crate::ComponentTypeId::of::<SimPosition>(),
        elem_size: std::mem::size_of::<SimPosition>(),
        mutable: true,
    };

    let (msl, wgsl) = crate::storage::gpu::ir::compile_kernel_ir_sources(
        "test_vector_ir_pipeline",
        &[comp],
        &[<SimPosition as GpuShaderStruct>::msl_struct_definition()],
        &[<SimPosition as GpuShaderStruct>::wgsl_struct_definition()],
        &[binding],
        &block,
    );

    // 验证 WGSL 源码结构与向量运算
    assert!(wgsl.contains("vec3<f32>(1.0, 2.0, 3.0)"));
    assert!(wgsl.contains("dot(v1, v2)"));
    assert!(wgsl.contains("length(v_mul)"));
    assert!(wgsl.contains("normalize(v1)"));
    assert!(wgsl.contains("distance(v1, v2)"));

    // 验证 MSL 源码结构与向量运算
    assert!(msl.contains("float3(1.0, 2.0, 3.0)"));
    assert!(msl.contains("dot(v1, v2)"));
    assert!(msl.contains("length(v_mul)"));
    assert!(msl.contains("normalize(v1)"));
    assert!(msl.contains("distance(v1, v2)"));

    unsafe fn cpu_dummy(_: &[*mut u8], _: usize, _: &crate::storage::gpu::GpuDispatchUniforms) {}

    let desc = unsafe {
        GpuKernelDescriptor::new(
            "test_vector_ir_pipeline",
            vec![binding],
            msl,
            wgsl,
            cpu_dummy,
        )
    };

    let compiled = compile_and_validate_pipeline(&desc)
        .expect("Generated vector IR WGSL must pass Naga validation");
    assert!(compiled.shader_validated);
    assert!(compiled.naga_translated_msl.contains("kernel void"));
}

#[test]
fn m11_redundant_writeback_elimination() {
    let desc_particle = update_vec_particle_m11.descriptor();
    let msl_part = desc_particle.msl_source();
    let wgsl_part = desc_particle.wgsl_source();

    // 1. mass 是只读引用 (&SimMass)，确保 writeback 被完全消除
    assert!(
        !msl_part.contains("sumeru_buf_mass[gid] = mass;"),
        "Redundant writeback for read-only `mass` must be eliminated from MSL: {msl_part}"
    );
    assert!(
        !wgsl_part.contains("sumeru_buf_mass[gid] = mass;"),
        "Redundant writeback for read-only `mass` must be eliminated from WGSL: {wgsl_part}"
    );

    // 2. mass 声明必须使用只读限定符
    assert!(
        msl_part.contains("const device SimMass* sumeru_buf_mass [[buffer(1)]]"),
        "MSL buffer for read-only `mass` must be `const device`: {msl_part}"
    );
    assert!(
        wgsl_part.contains("var<storage, read> sumeru_buf_mass"),
        "WGSL buffer for read-only `mass` must be `var<storage, read>`: {wgsl_part}"
    );

    // 3. p 是可写参数且在函数体内被写入，必须保留 writeback
    assert!(
        msl_part.contains("sumeru_buf_p[gid] = p;"),
        "Writeback for mutable `p` must be preserved in MSL: {msl_part}"
    );
    assert!(
        wgsl_part.contains("sumeru_buf_p[gid] = p;"),
        "Writeback for mutable `p` must be preserved in WGSL: {wgsl_part}"
    );

    // 4. 针对声明为 `&mut` 但函数体内从未写入的参数，验证写入消除与只读限定符优化
    let desc_inspector = read_only_inspector_kernel.descriptor();
    let msl_insp = desc_inspector.msl_source();
    let wgsl_insp = desc_inspector.wgsl_source();

    assert!(
        !msl_insp.contains("sumeru_buf_p[gid] = p;"),
        "Unwritten `&mut p` must have its writeback eliminated in MSL: {msl_insp}"
    );
    assert!(
        !wgsl_insp.contains("sumeru_buf_p[gid] = p;"),
        "Unwritten `&mut p` must have its writeback eliminated in WGSL: {wgsl_insp}"
    );
    assert!(
        msl_insp.contains("const device SimPosition* sumeru_buf_p [[buffer(0)]]"),
        "Unwritten `&mut p` must be qualified as `const device` in MSL: {msl_insp}"
    );
    assert!(
        wgsl_insp.contains("var<storage, read> sumeru_buf_p"),
        "Unwritten `&mut p` must be qualified as `var<storage, read>` in WGSL: {wgsl_insp}"
    );

    // 5. 编译并通过 Naga 验证（严格校验只读管线资源绑定）
    let compiled_part = compile_and_validate_pipeline(desc_particle)
        .expect("Kernel with read-only elimination must pass Naga validation");
    assert!(compiled_part.shader_validated);

    let compiled_insp = compile_and_validate_pipeline(desc_inspector)
        .expect("Inspector kernel with unwritten &mut elimination must pass Naga validation");
    assert!(compiled_insp.shader_validated);
}

#[test]
fn m11_three_tier_vector_math_traversal() {
    let mut store_seq = Store::new().with_layout_optimization(LayoutOptimization::PageAlignedGpu);
    for i in 0..128u128 {
        let e = EntityId::from_u128(i + 1);
        store_seq
            .insert((
                e,
                (
                    SimVecParticle {
                        pos: [i as f32, (i * 2) as f32, 0.0],
                        vel: [3.0, 4.0, 0.0],
                        speed: 0.0,
                    },
                    SimMass { inv_mass: 0.5 },
                ),
            ))
            .unwrap();
    }
    let mut store_par = store_seq.clone();
    let mut store_gpu = store_seq.clone();

    // 验证 CPU 向量工具函数与 Trait 一致性
    let v2 = vec2(3.0, 4.0);
    assert_eq!(length(v2), 5.0);
    assert_eq!(dot(v2, vec2(1.0, 0.0)), 3.0);
    assert_eq!(distance(v2, vec2(0.0, 0.0)), 5.0);
    let n2 = normalize(v2);
    assert!((n2[0] - 0.6).abs() < 1e-5);
    assert!((n2[1] - 0.8).abs() < 1e-5);

    let v4 = vec4(1.0, 2.0, 3.0, 4.0);
    assert_eq!(dot(v4, v4), 30.0);

    // Tier 1: CPU Sequential
    store_seq
        .query_mut::<(&mut SimVecParticle, &SimMass)>()
        .unwrap()
        .run(update_vec_particle_m11);

    // Tier 2: CPU Rayon Morsel Parallel
    store_par
        .query_mut::<(&mut SimVecParticle, &SimMass)>()
        .unwrap()
        .par_run(update_vec_particle_m11);

    // Tier 3: GPU (CPU fallback or hardware Metal)
    let mut engine = MetalComputeEngine::new();
    store_gpu
        .query_mut::<(&mut SimVecParticle, &SimMass)>()
        .unwrap()
        .gpu()
        .for_each_on(&mut engine, update_vec_particle_m11)
        .unwrap();

    // 验证三层遍历结果 100% 严格一致
    for i in 0..128u128 {
        let e = EntityId::from_u128(i + 1);
        let p_seq = *store_seq.get::<SimVecParticle>(e).unwrap();
        let p_par = *store_par.get::<SimVecParticle>(e).unwrap();
        let p_gpu = *store_gpu.get::<SimVecParticle>(e).unwrap();

        assert!((p_seq.pos[0] - p_par.pos[0]).abs() < 1e-4, "Seq vs Par pos.x mismatch at {i}");
        assert!((p_seq.pos[1] - p_par.pos[1]).abs() < 1e-4, "Seq vs Par pos.y mismatch at {i}");
        assert!((p_seq.pos[2] - p_par.pos[2]).abs() < 1e-4, "Seq vs Par pos.z mismatch at {i}");
        assert!((p_seq.speed - p_par.speed).abs() < 1e-4, "Seq vs Par speed mismatch at {i}");

        assert!((p_seq.pos[0] - p_gpu.pos[0]).abs() < 1e-4, "Seq vs GPU pos.x mismatch at {i}");
        assert!((p_seq.pos[1] - p_gpu.pos[1]).abs() < 1e-4, "Seq vs GPU pos.y mismatch at {i}");
        assert!((p_seq.pos[2] - p_gpu.pos[2]).abs() < 1e-4, "Seq vs GPU pos.z mismatch at {i}");
        assert!((p_seq.speed - p_gpu.speed).abs() < 1e-4, "Seq vs GPU speed mismatch at {i}");
    }
}

// ============================================================================
// M12: 面向算子直调与元组管线的对称计算模型 (`#[kernel]`, `Vec2/3/4`, `queries_mut`, `ComputeTask`)
// ============================================================================

#[crate::kernel]
pub struct ErgoPosition {
    pub coords: crate::Vec3,
    pub uv: crate::Vec2,
}

#[crate::kernel]
pub struct ErgoVelocity {
    pub velocity: crate::Vec3,
}

#[crate::kernel]
pub struct ErgoDynamics {
    pub force: crate::Vec3,
    pub tint: crate::Vec4,
    pub drag: f32,
}

#[crate::kernel]
pub struct ErgoActiveTag {
    pub active: u32,
}

#[crate::kernel]
pub struct ErgoFrozenTag {
    pub frozen: u32,
}

#[crate::kernel]
pub fn ergo_integrate(pos: &mut ErgoPosition, vel: &ErgoVelocity, dt: f32) {
    pos.coords += vel.velocity * dt;
    pos.uv.x += dt;
    pos.uv.y = pos.coords.x * 0.5;
}

#[crate::kernel]
pub fn ergo_damp_dynamics(vel: &mut ErgoVelocity, dyns: &mut ErgoDynamics, dt: f32) {
    let v_acc = vel.velocity + dyns.force * dt;
    let damp_factor = (1.0 - dyns.drag * dt).max(0.0);
    let len_sq = v_acc.length_squared();
    vel.velocity = v_acc * damp_factor + crate::Vec3::splat(len_sq * 0.0);
    dyns.force *= 0.5;
    dyns.tint.w = v_acc.dot(dyns.force);
}

#[crate::kernel]
pub fn ergo_step_fused(
    pos: &mut ErgoPosition,
    vel: &mut ErgoVelocity,
    dyns: &mut ErgoDynamics,
    dt: f32,
) {
    ergo_integrate(pos, vel, dt);
    ergo_damp_dynamics(vel, dyns, dt);
    ergo_integrate(pos, vel, dt);
}

#[crate::kernel]
pub fn ergo_touch_pos_only(pos: &mut ErgoPosition, offset: f32) {
    pos.coords.z += offset;
}

#[crate::kernel]
pub fn ergo_touch_dyns_only(dyns: &mut ErgoDynamics, scale: f32) {
    dyns.drag *= scale;
}

#[test]
fn m12_vec2_vec3_vec4_operator_overloading_and_kernel_parity() {
    use crate::{Vec2, Vec3, Vec4};

    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);
    assert_eq!(a + b, Vec3::new(5.0, 7.0, 9.0));
    assert_eq!(b - a, Vec3::new(3.0, 3.0, 3.0));
    assert_eq!(a * 2.0, Vec3::new(2.0, 4.0, 6.0));
    assert_eq!(2.0 * a, Vec3::new(2.0, 4.0, 6.0));
    assert_eq!(b / 2.0, Vec3::new(2.0, 2.5, 3.0));
    assert_eq!(-a, Vec3::new(-1.0, -2.0, -3.0));
    assert_eq!(a.dot(b), 32.0);
    assert_eq!(Vec2::new(3.0, 4.0).length(), 5.0);
    assert_eq!(Vec2::new(3.0, 4.0).length_squared(), 25.0);
    assert!((Vec4::splat(2.0).normalize().length() - 1.0).abs() < 1e-5);

    let mut store_par =
        Store::new().with_layout_optimization(crate::LayoutOptimization::PageAlignedGpu);
    for i in 0..1500usize {
        store_par
            .insert((
                EntityId::from_u128(i as u128 + 1),
                (
                    ErgoPosition {
                        coords: Vec3::new(1.0, 2.0, 3.0),
                        uv: Vec2::new(0.0, 0.0),
                    },
                    ErgoVelocity {
                        velocity: Vec3::new(10.0, 20.0, 30.0),
                    },
                    ErgoDynamics {
                        force: Vec3::new(2.0, 4.0, 6.0),
                        tint: Vec4::new(1.0, 1.0, 1.0, 0.0),
                        drag: 0.5,
                    },
                    ErgoActiveTag { active: 1 },
                ),
            ))
            .unwrap();
    }
    // 插入冻结实体（验证 `.without::<ErgoFrozenTag>()` 过滤隔离）
    let frozen_id = EntityId::from_u128(999_999);
    store_par
        .insert((
            frozen_id,
            (
                ErgoPosition {
                    coords: Vec3::splat(50.0),
                    uv: Vec2::ZERO,
                },
                ErgoVelocity {
                    velocity: Vec3::splat(10.0),
                },
                ErgoDynamics {
                    force: Vec3::splat(1.0),
                    tint: Vec4::ZERO,
                    drag: 0.1,
                },
                ErgoActiveTag { active: 1 },
                ErgoFrozenTag { frozen: 1 },
            ),
        ))
        .unwrap();

    let mut store_gpu = store_par.clone();
    let mut store_fused = store_par.clone();
    let mut store_rayon = store_par.clone();

    // 1. `.par()` 多算子元组管线（>= 1024 行自动走 Rayon 1024-row Morsel 多核并行）
    {
        use crate::{ComputeTask, QueryTupleFilterExt};
        let (mut pos, mut vel, mut dyns) = store_par
            .queries_mut::<(&mut ErgoPosition, &mut ErgoVelocity, &mut ErgoDynamics)>()
            .unwrap()
            .with::<ErgoActiveTag>()
            .without::<ErgoFrozenTag>();

        let par_stats = (
            ergo_integrate.bind(&mut pos, &vel, 0.1),
            ergo_damp_dynamics.bind(&mut vel, &mut dyns, 0.1),
            ergo_integrate.bind(&mut pos, &vel, 0.1),
        )
            .par()
            .unwrap();
        assert_eq!(par_stats.total_rows, 1500 * 3);
        assert_eq!(pos.cache_misses(), 1);
        assert!(pos.cache_hits() >= 2);
    }

    // 2. `.gpu()` 多算子元组管线（自动检测 RAW/WAW/WAR 依赖插入 2 道显存屏障，单次 CommandBuffer 提交）
    {
        use crate::{ComputeTask, QueryTupleFilterExt};
        let (mut pos, mut vel, mut dyns) = store_gpu
            .queries_mut::<(&mut ErgoPosition, &mut ErgoVelocity, &mut ErgoDynamics)>()
            .unwrap()
            .with::<ErgoActiveTag>()
            .without::<ErgoFrozenTag>();

        let gpu_stats = (
            ergo_integrate.bind(&mut pos, &vel, 0.1),
            ergo_damp_dynamics.bind(&mut vel, &mut dyns, 0.1),
            ergo_integrate.bind(&mut pos, &vel, 0.1),
        )
            .gpu()
            .unwrap();
        assert_eq!(gpu_stats.total_rows, 1500 * 3);
        assert_eq!(gpu_stats.barriers_inserted, 2);

        // 验证无读写交集的两阶段管线自动 0 屏障插入
        let disjoint_stats = (
            ergo_touch_pos_only.bind(&mut pos, 0.0),
            ergo_touch_dyns_only.bind(&mut dyns, 1.0),
        )
            .gpu()
            .unwrap();
        assert_eq!(disjoint_stats.barriers_inserted, 0);
    }

    // 3. 单 Kernel 融合算子 `ergo_step_fused.run(...)`
    {
        use crate::QueryTupleFilterExt;
        let (mut pos, mut vel, mut dyns) = store_fused
            .queries_mut::<(&mut ErgoPosition, &mut ErgoVelocity, &mut ErgoDynamics)>()
            .unwrap()
            .with::<ErgoActiveTag>()
            .without::<ErgoFrozenTag>();
        let fused_stats = ergo_step_fused
            .run(&mut pos, &mut vel, &mut dyns, 0.1)
            .unwrap();
        assert_eq!(fused_stats.total_rows, 1500);
        assert_eq!(fused_stats.barriers_inserted, 0);
    }

    // 4. 用户手写 Rayon `.into_par_iter()` 循环直接调用同一 `#[kernel]` 函数
    {
        use crate::QueryTupleFilterExt;
        use rayon::prelude::*;
        let (mut pos, mut vel, mut dyns) = store_rayon
            .queries_mut::<(&mut ErgoPosition, &mut ErgoVelocity, &mut ErgoDynamics)>()
            .unwrap()
            .with::<ErgoActiveTag>()
            .without::<ErgoFrozenTag>();

        (&mut pos, &mut vel, &mut dyns)
            .into_par_iter()
            .for_each(|(p, v, d)| ergo_step_fused(p, v, d, 0.1));
        assert_eq!(pos.cache_misses(), 1);
        assert!(pos.cache_hits() >= 2);
    }

    // 验证四种执行形态结果 100% 严格一致，且 Frozen 实体未被修改
    for i in [1u128, 512, 1024, 1500] {
        let e = EntityId::from_u128(i);
        let p_par = *store_par.get::<ErgoPosition>(e).unwrap();
        let p_gpu = *store_gpu.get::<ErgoPosition>(e).unwrap();
        let p_fused = *store_fused.get::<ErgoPosition>(e).unwrap();
        let p_rayon = *store_rayon.get::<ErgoPosition>(e).unwrap();

        assert!((p_par.coords - p_gpu.coords).length() < 1e-4);
        assert!((p_par.coords - p_fused.coords).length() < 1e-4);
        assert!((p_par.coords - p_rayon.coords).length() < 1e-4);
        assert!((p_par.uv - p_gpu.uv).length() < 1e-4);
    }
    assert_eq!(
        store_gpu.get::<ErgoPosition>(frozen_id).unwrap().coords,
        Vec3::splat(50.0)
    );
}

#[test]
fn m12_shared_col_cache_zero_recompute_and_incremental_cr_refresh() {
    use crate::{QueryTupleFilterExt, Vec2, Vec3};

    let mut store =
        Store::new().with_layout_optimization(crate::LayoutOptimization::PageAlignedGpu);
    for i in 0..64usize {
        store
            .insert((
                EntityId::from_u128(i as u128 + 1),
                (
                    ErgoPosition {
                        coords: Vec3::new(0.0, 0.0, 0.0),
                        uv: Vec2::ZERO,
                    },
                    ErgoVelocity {
                        velocity: Vec3::new(1.0, 2.0, 3.0),
                    },
                    ErgoActiveTag { active: 1 },
                ),
            ))
            .unwrap();
    }

    let shared_cache_clone = {
        let (mut pos, vel) = store
            .queries_mut::<(&mut ErgoPosition, &ErgoVelocity)>()
            .unwrap()
            .with::<ErgoActiveTag>();

        // 首次执行（< 1024 行自适应走当前线程顺序执行）：冷启动扫描 1 次
        ergo_integrate.par(&mut pos, &vel, 1.0).unwrap();
        assert_eq!(pos.cache_misses(), 1);
        assert_eq!(pos.cache_hits(), 0);

        // 无 CR 变更连续执行 3 次：必须 100% 命中缓存，misses 保持为 1
        ergo_integrate.par(&mut pos, &vel, 1.0).unwrap();
        ergo_integrate.gpu(&mut pos, &vel, 1.0).unwrap();
        ergo_integrate.run(&mut pos, &vel, 1.0).unwrap();
        assert_eq!(pos.cache_misses(), 1);
        assert_eq!(pos.cache_hits(), 3);

        std::sync::Arc::clone(&pos.shared_cache)
    };

    // 触发 CR 结构变更：新增含额外组件 `ErgoDynamics` 的新 Archetype
    let new_e = EntityId::from_u128(10_000);
    store
        .insert((
            new_e,
            (
                ErgoPosition {
                    coords: Vec3::ZERO,
                    uv: Vec2::ZERO,
                },
                ErgoVelocity {
                    velocity: Vec3::new(5.0, 5.0, 5.0),
                },
                ErgoDynamics {
                    force: Vec3::ZERO,
                    tint: crate::Vec4::ZERO,
                    drag: 0.0,
                },
                ErgoActiveTag { active: 1 },
            ),
        ))
        .unwrap();

    // 验证 SharedColCache::resolve 执行 O(\Delta A) 增量刷新并纳入新 Archetype
    let filter = crate::storage::query::With::<ErgoActiveTag, crate::storage::query::All> {
        previous: crate::storage::query::All,
        type_id: crate::ComponentTypeId::of::<ErgoActiveTag>(),
        marker: std::marker::PhantomData,
    };
    let refreshed = crate::SharedColCache::resolve(&shared_cache_clone, &store, &filter);
    assert_eq!(refreshed.len(), 2);
}







