use super::*;

#[must_use = "a query does no work until iterated"]
pub struct QueryMutBuilder<'a, Q: QueryMutData<'a>, F: QueryFilter = All> {
    pub(crate) archetypes: &'a mut [Archetype],
    pub(crate) row_upper_bound: usize,
    pub(crate) write_epoch: u64,
    pub(crate) filter: F,
    pub(crate) marker: PhantomData<Q>,
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> std::fmt::Debug for QueryMutBuilder<'a, Q, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryMutBuilder")
            .field("row_upper_bound", &self.row_upper_bound)
            .finish()
    }
}

impl<'a, Q: QueryMutData<'a>> QueryMutBuilder<'a, Q> {
    pub(crate) fn new(store: &'a mut crate::storage::store::Store) -> Self {
        let row_upper_bound = store.len();
        let write_epoch = store.epoch();
        Self {
            archetypes: &mut store.archetypes,
            row_upper_bound,
            write_epoch,
            filter: All,
            marker: PhantomData,
        }
    }
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> QueryMutBuilder<'a, Q, F> {
    /// Require component T without examining or borrowing its values.
    #[inline]
    pub fn with<T: 'static>(self) -> QueryMutBuilder<'a, Q, With<T, F>> {
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: With {
                previous: self.filter,
                type_id: ComponentTypeId::of::<T>(),
                marker: PhantomData,
            },
            marker: PhantomData,
        }
    }

    /// Skip entire tables containing component T.
    #[inline]
    pub fn without<T: 'static>(self) -> QueryMutBuilder<'a, Q, Without<T, F>> {
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: Without {
                previous: self.filter,
                type_id: ComponentTypeId::of::<T>(),
                marker: PhantomData,
            },
            marker: PhantomData,
        }
    }

    /// Require relationship pair `(R, target)` on the entity (accelerated via per-table `PairTargetIndex`).
    #[inline]
    pub fn with_pair<R: ModelComponent + Serialize + DeserializeOwned + 'static>(
        self,
        target: EntityId,
    ) -> QueryMutBuilder<'a, Q, WithPairTarget<R, F>> {
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: WithPairTarget::new(self.filter, target),
            marker: PhantomData,
        }
    }

    /// Skip entire tables containing relationship pair `Pair<R>`.
    #[inline]
    pub fn without_pair<R: 'static>(
        self,
    ) -> QueryMutBuilder<'a, Q, Without<crate::al::Pair<R>, F>> {
        self.without::<crate::al::Pair<R>>()
    }

    /// Require zero-sized compile-time slot tag `Tag<N>`.
    #[inline]
    pub fn with_tag<const N: u8>(self) -> QueryMutBuilder<'a, Q, With<crate::al::Tag<N>, F>> {
        self.with::<crate::al::Tag<N>>()
    }

    /// Skip entire tables containing zero-sized compile-time slot tag `Tag<N>`.
    #[inline]
    pub fn without_tag<const N: u8>(
        self,
    ) -> QueryMutBuilder<'a, Q, Without<crate::al::Tag<N>, F>> {
        self.without::<crate::al::Tag<N>>()
    }

    /// Require non-migrating row status bit `FLAG` (`0..8`) to be set (`1`).
    #[inline]
    pub fn with_flag<const FLAG: u8>(self) -> QueryMutBuilder<'a, Q, WithFlag<FLAG, F>> {
        const { assert!(FLAG < 8, "RowFlag index must be in 0..8") };
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: WithFlag::new(self.filter),
            marker: PhantomData,
        }
    }

    /// Require non-migrating row status bit `FLAG` (`0..8`) to be unset (`0`).
    #[inline]
    pub fn without_flag<const FLAG: u8>(self) -> QueryMutBuilder<'a, Q, WithoutFlag<FLAG, F>> {
        const { assert!(FLAG < 8, "RowFlag index must be in 0..8") };
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: WithoutFlag::new(self.filter),
            marker: PhantomData,
        }
    }

    /// Filter to 64-row chunks of component `T` modified strictly after `since_epoch`.
    #[inline]
    pub fn changed_since<T: 'static>(
        self,
        since_epoch: u64,
    ) -> QueryMutBuilder<'a, Q, ChangedSince<T, F>> {
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: ChangedSince::new(self.filter, since_epoch),
            marker: PhantomData,
        }
    }

    /// Add a row-level component predicate filter.
    #[inline]
    pub fn filter<C: ModelComponent + Serialize + DeserializeOwned>(
        self,
        predicate: impl Fn(&C) -> bool + Send + Sync + 'static,
    ) -> QueryMutBuilder<'a, Q, BusinessFilter<F, C>> {
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: BusinessFilter::new(self.filter, predicate),
            marker: PhantomData,
        }
    }

    /// Add a row-level component field equality filter.
    #[inline]
    pub fn filter_eq<C: ModelComponent + Serialize + DeserializeOwned, Field: ComponentField<C>>(
        self,
        expected: Field::Value,
    ) -> QueryMutBuilder<'a, Q, FilterEq<F, C, Field>> {
        QueryMutBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: FilterEq::new(self.filter, expected),
            marker: PhantomData,
        }
    }

    /// Consume the builder into a lazy, forward-only row iterator.
    #[inline]
    pub fn iter_mut(self) -> QueryMutIter<'a, Q, F> {
        QueryMutIter {
            archetypes: self.archetypes.iter_mut(),
            entities: [].iter(),
            segment_start_row: 0,
            segment_offset: 0,
            filtered_mask: None,
            state: None,
            current: None,
            remaining_upper_bound: self.row_upper_bound,
            write_epoch: self.write_epoch,
            filter: self.filter,
        }
    }

    /// Alias for `iter_mut()`.
    #[inline]
    pub fn iter(self) -> QueryMutIter<'a, Q, F> {
        self.iter_mut()
    }

    /// Consume the builder into an archetype-level contiguous slice iterator.
    #[inline]
    pub fn chunks_mut(self) -> QueryMutChunksIter<'a, Q, F> {
        QueryMutChunksIter {
            archetypes: self.archetypes.iter_mut(),
            current: None,
            write_epoch: self.write_epoch,
            filter: self.filter,
            marker: PhantomData,
        }
    }

    /// Alias for `chunks_mut()`.
    #[inline]
    pub fn chunks(self) -> QueryMutChunksIter<'a, Q, F> {
        self.chunks_mut()
    }

    /// **① CPU 单线程顺序遍历（闭包）**：对所有匹配实体逐行执行闭包，完整支持 `|(pos, vel)| { pos.x += vel.vx; }` 自动类型推断。
    ///
    /// 若需直接传入 `#[gpu]` 算子函数，请使用 [`Self::run`]（如 `.run(integrate_sim_motion)`）或在闭包内调用。
    #[inline]
    pub fn for_each<Op>(self, mut op: Op)
    where
        Q::Item: super::QueryRowItem,
        Op: FnMut(<Q::Item as super::QueryRowItem>::Comps),
    {
        use super::QueryRowItem;
        for item in self.iter_mut() {
            op(item.into_comps());
        }
    }

    /// **① CPU 单线程顺序遍历（带 `EntityId`）**：对所有匹配实体逐行执行 `|(entity, comps)| { ... }` 闭包。
    #[inline]
    pub fn for_each_with_entity<Op: FnMut(Q::Item)>(self, mut op: Op) {
        for item in self.iter_mut() {
            op(item);
        }
    }

    /// **① CPU 单线程算子直传执行**：直接传入 `#[gpu]` 算子或实现 [`super::QueryForEachOp`] 的算子逐行执行。
    #[inline]
    pub fn run<M, Op: super::QueryForEachOp<Q::Item, M>>(self, mut op: Op) {
        for item in self.iter_mut() {
            op.call_row(item);
        }
    }

    /// 进入 **② CPU 多核并行 (Rayon)** 执行模式，通过 `.par().for_each(...)` 或 `.par().run(op)` 并行处理匹配实体。
    #[inline]
    pub fn par(self) -> QueryParMutRunner<'a, Q, F> {
        QueryParMutRunner { builder: self }
    }

    /// **② CPU 多核并行遍历（闭包）**：在多核线程池上并行执行闭包（`self.par().for_each(op)` 的简写），完整支持闭包参数类型自动推断。
    #[inline]
    pub fn par_for_each<Op>(self, op: Op)
    where
        Q::Item: Send + super::QueryRowItem,
        Op: Fn(<Q::Item as super::QueryRowItem>::Comps) + Send + Sync,
    {
        self.par().for_each(op);
    }

    /// **② CPU 多核并行算子直传执行**：在多核线程池上并行执行 `#[gpu]` 算子（`self.par().run(op)` 的简写）。
    #[inline]
    pub fn par_run<M, Op: super::QueryParForEachOp<Q::Item, M>>(self, op: Op)
    where
        Q::Item: Send,
    {
        self.par().run(op);
    }

    /// 进入 **③ GPU 硬件并行** 执行模式，通过 `.gpu().for_each(kernel)?` 将 16KB 页对齐列零拷贝派发至 GPU。
    #[cfg(feature = "gpu")]
    #[inline]
    pub fn gpu(self) -> QueryGpuRunner<'a, Q, F> {
        QueryGpuRunner { builder: self }
    }
}

/// CPU 多核并行查询执行器（由 [`QueryMutBuilder::par`] 创建，采用 [`SubchunkPlanner`](crate::storage::primitives::SubchunkPlanner) 自适应缓存驻留切块，零行级堆分配）。
#[must_use = "a parallel query runner does no work until .for_each(...) or .run(...) is called"]
pub struct QueryParMutRunner<'a, Q: QueryMutData<'a>, F: QueryFilter = All> {
    builder: QueryMutBuilder<'a, Q, F>,
}

struct ParMutTableSegment<'a> {
    entities: &'a [EntityId],
    columns: smallvec::SmallVec<[(ComponentTypeId, ColumnDescriptor); 4]>,
    mask: Option<crate::storage::archetype::RowBitMask>,
    seg_range: Range<usize>,
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> QueryParMutRunner<'a, Q, F> {
    fn for_each_item_par<V>(self, visit: V)
    where
        Q::Item: Send,
        V: Fn(Q::Item) + Send + Sync,
    {
        use rayon::prelude::*;
        Q::validate_query();
        let morsel_rows = crate::storage::primitives::SubchunkPlanner::default()
            .subchunk_rows(Q::total_row_width());
        let mut segments = Vec::new();

        for archetype in self.builder.archetypes.iter_mut() {
            let arch_len = archetype.len();
            if arch_len == 0
                || !self.builder.filter.matches_archetype(archetype)
                || !archetype_matches_query_mut::<Q>(archetype)
            {
                continue;
            }
            let mask = if F::IS_ROW_FILTERED {
                let m = self.builder.filter.collect_matched_row_mask(archetype);
                if m.as_ref().is_some_and(|b| b.is_all_zero()) {
                    continue;
                }
                m
            } else {
                None
            };
            Q::mark_written_columns(archetype, mask.as_ref(), self.builder.write_epoch);
            let Some(mut table) = MutableTable::new::<Q>(archetype) else {
                continue;
            };
            while let Some(seg_range) = table.next_range() {
                segments.push(ParMutTableSegment {
                    entities: table.entities,
                    columns: table.columns.clone(),
                    mask: mask.clone(),
                    seg_range,
                });
            }
        }

        segments.par_iter().for_each(|seg| {
            let seg_len = seg.seg_range.len();
            let num_morsels = (seg_len + morsel_rows - 1) / morsel_rows;
            (0..num_morsels).into_par_iter().for_each(|morsel_idx| {
                let start = seg.seg_range.start + morsel_idx * morsel_rows;
                let end = (start + morsel_rows).min(seg.seg_range.end);
                if start >= end {
                    return;
                }
                if let Some(mask) = &seg.mask {
                    let first_word = start >> 6;
                    let last_word = (end - 1) >> 6;
                    if (first_word..=last_word)
                        .all(|w| mask.words().get(w).copied().unwrap_or(0) == 0)
                    {
                        return;
                    }
                    let Some(mut state) = Q::prepare(&seg.columns, start..end) else {
                        return;
                    };
                    let mut cur = start;
                    while cur < end {
                        let w_idx = cur >> 6;
                        let bit_idx = cur & 63;
                        let word = mask.words().get(w_idx).copied().unwrap_or(0) >> bit_idx;
                        if word == 0 {
                            let next_cur = ((w_idx + 1) << 6).min(end);
                            let skip = next_cur - cur;
                            if skip > 0 {
                                Q::skip_rows(&mut state, skip);
                            }
                            cur = next_cur;
                            continue;
                        }
                        let tz = word.trailing_zeros() as usize;
                        let target = (cur + tz).min(end);
                        let skip = target - cur;
                        if skip > 0 {
                            Q::skip_rows(&mut state, skip);
                        }
                        if target >= end {
                            break;
                        }
                        let item = Q::next(&mut state, seg.entities[target]);
                        visit(item);
                        cur = target + 1;
                    }
                } else {
                    let Some(mut state) = Q::prepare(&seg.columns, start..end) else {
                        return;
                    };
                    for &entity in &seg.entities[start..end] {
                        let item = Q::next(&mut state, entity);
                        visit(item);
                    }
                }
            });
        });
    }

    /// 在 Rayon 多核线程池上按 1024 行 Morsel 连续切片并行执行闭包（零行级堆分配，完整支持闭包参数类型自动推断）。
    pub fn for_each<Op>(self, op: Op)
    where
        Q::Item: Send + super::QueryRowItem,
        Op: Fn(<Q::Item as super::QueryRowItem>::Comps) + Send + Sync,
    {
        use super::QueryRowItem;
        self.for_each_item_par(|item| {
            op(item.into_comps());
        });
    }

    /// 在 Rayon 多核线程池上按 1024 行 Morsel 连续切片并行执行 `|(entity, comps)|` 闭包。
    pub fn for_each_with_entity<Op>(self, op: Op)
    where
        Q::Item: Send,
        Op: Fn(Q::Item) + Send + Sync,
    {
        self.for_each_item_par(op);
    }

    /// 在 Rayon 多核线程池上按 1024 行 Morsel 连续切片并行执行 `#[gpu]` 算子。
    pub fn run<M, Op: super::QueryParForEachOp<Q::Item, M>>(self, op: Op)
    where
        Q::Item: Send,
    {
        self.for_each_item_par(|item| {
            op.call_row_par(item);
        });
    }
}

/// GPU 硬件并行查询执行器（由 [`QueryMutBuilder::gpu`] 创建）。
#[cfg(feature = "gpu")]
#[must_use = "a GPU query runner does no work until .for_each(...) is called"]
pub struct QueryGpuRunner<'a, Q: QueryMutData<'a>, F: QueryFilter = All> {
    builder: QueryMutBuilder<'a, Q, F>,
}

#[cfg(feature = "gpu")]
impl<'a, Q: QueryMutData<'a>, F: QueryFilter> QueryGpuRunner<'a, Q, F> {
    /// 使用默认线程本地 `MetalComputeEngine` 执行 16KB 页对齐零拷贝 GPU 内核派发。
    #[inline]
    pub fn for_each<K: crate::storage::gpu::GpuKernel<'a, Query = Q>>(
        self,
        kernel: K,
    ) -> Result<crate::storage::gpu::GpuDispatchStats, StateError> {
        thread_local! {
            static DEFAULT_METAL_ENGINE: std::cell::RefCell<crate::storage::gpu::MetalComputeEngine> =
                std::cell::RefCell::new(crate::storage::gpu::MetalComputeEngine::new());
        }
        DEFAULT_METAL_ENGINE.with(|cell| {
            let mut engine = cell.borrow_mut();
            self.for_each_on(&mut engine, kernel)
        })
    }

    /// 在指定的 [`MetalComputeEngine`] 实例上执行两阶段原子 16KB 页对齐零拷贝 GPU 内核派发。
    ///
    /// - **阶段 0**：校验不存在行级 CPU 谓词过滤器，并预编译/校验内核着色器管线；
    /// - **阶段 1a**：对所有匹配 Archetype 执行纯只读列完整性与容量溢出校验（失败时零物理列变更）；
    /// - **阶段 1b**：对匹配 Archetype 执行 16KB 页对齐提升、可变列 COW 独占分离与零拷贝描述符构造；
    /// - **阶段 2**：全部预检通过后执行硬件/回退派发并统一标记写纪元。
    pub fn for_each_on<K: crate::storage::gpu::GpuKernel<'a, Query = Q>>(
        self,
        engine: &mut crate::storage::gpu::MetalComputeEngine,
        kernel: K,
    ) -> Result<crate::storage::gpu::GpuDispatchStats, StateError> {
        if F::IS_ROW_FILTERED {
            return Err(StateError::AccessDenied(
                "GPU whole-column dispatch does not support row-level CPU predicate filters; use archetype filters (.with / .without)".into(),
            ));
        }
        Q::validate_query();

        let desc = kernel.descriptor();
        engine.ensure_pipeline_compiled(desc)?;

        struct PreparedArchetypeDispatch {
            arch_idx: usize,
            arch_id: crate::id::ArchetypeId,
            active_rows: usize,
            pages_16k: usize,
            raw_bindings: Vec<(ComponentTypeId, crate::storage::gpu::MetalNoCopyBufferDescriptor)>,
        }

        // 阶段 1a：纯只读预校验（确保任意 Archetype 异常时绝不半途解冻或提升其它 Archetype 的列）
        let mut matched_arch_indices = Vec::new();
        for (arch_idx, archetype) in self.builder.archetypes.iter().enumerate() {
            let active_rows = archetype.len();
            if active_rows == 0 || !self.builder.filter.matches_archetype(archetype) {
                continue;
            }
            let comp_slice = archetype.component_types_slice();
            if !desc
                .bindings()
                .iter()
                .all(|b| comp_slice.binary_search(&b.type_id).is_ok())
            {
                continue;
            }
            let arch_id = archetype.id();
            let cols = archetype.columns();
            for b in desc.bindings() {
                let col = cols.get(&b.type_id).ok_or_else(|| {
                    StateError::ComponentNotFound(format!("{}", b.type_id))
                })?;
                if col.len() != active_rows {
                    return Err(StateError::AccessDenied(format!(
                        "Column {} length {} does not match archetype {} active_rows {}",
                        b.struct_name,
                        col.len(),
                        arch_id.0,
                        active_rows
                    )));
                }
                let _active_bytes = active_rows.checked_mul(b.elem_size).ok_or_else(|| {
                    StateError::CapacityExhausted("GPU column active byte size overflow".into())
                })?;
            }
            matched_arch_indices.push(arch_idx);
        }

        // 阶段 1b：16KB 页对齐提升、COW 独占分离与零拷贝描述符构造
        let mut prepared = Vec::with_capacity(matched_arch_indices.len());
        for arch_idx in matched_arch_indices {
            let archetype = &mut self.builder.archetypes[arch_idx];
            let active_rows = archetype.len();
            let arch_id = archetype.id();
            let mut changed_any = false;
            for b in desc.bindings() {
                let needs_promote = archetype
                    .columns()
                    .get(&b.type_id)
                    .is_some_and(|col| !col.is_page_aligned());
                if needs_promote || b.mutable {
                    let col = archetype
                        .get_erased_column_mut(b.type_id)
                        .expect("verified in Phase 1a");
                    if needs_promote {
                        col.promote_to_page_aligned();
                    }
                    if b.mutable {
                        col.ensure_contiguous()?;
                    }
                    changed_any = true;
                }
            }
            if changed_any {
                archetype.rebuild_flat_columns();
            }

            let mut raw_bindings = Vec::with_capacity(desc.bindings().len());
            let mut arch_pages = 0usize;
            {
                let cols = archetype.columns();
                for b in desc.bindings() {
                    let col = cols.get(&b.type_id).expect("verified in Phase 1a");
                    let (ptr, allocated_bytes) = col.metal_raw_parts().ok_or_else(|| {
                        StateError::AccessDenied(format!(
                            "Column {} in archetype {} is not backed by a 16KB PageAlignedBuffer",
                            b.struct_name, arch_id.0
                        ))
                    })?;
                    let generation = col.physical_version();
                    let active_bytes = active_rows * b.elem_size;
                    let buf_desc =
                        crate::storage::gpu::MetalNoCopyBufferDescriptor::validate_raw(
                            ptr,
                            allocated_bytes,
                            active_bytes,
                            generation,
                        )?;
                    let geom = crate::storage::gpu::PageGeometry::compute(
                        b.elem_size,
                        active_rows,
                        allocated_bytes,
                    );
                    arch_pages += geom.page_count;
                    raw_bindings.push((b.type_id, buf_desc));
                }
            }

            prepared.push(PreparedArchetypeDispatch {
                arch_idx,
                arch_id,
                active_rows,
                pages_16k: arch_pages,
                raw_bindings,
            });
        }

        // 阶段 2：单 CommandBuffer 批量合并派发并提交列写纪元
        let scalar_args = kernel.scalar_uniform_words();
        let batch_inputs: Vec<crate::storage::gpu::GpuArchetypeBatchInput> = prepared
            .iter()
            .map(|item| crate::storage::gpu::GpuArchetypeBatchInput {
                archetype_id: item.arch_id,
                active_rows: item.active_rows,
                bindings: item.raw_bindings.clone(),
            })
            .collect();
        let (hits, misses, submissions) = unsafe {
            engine.dispatch_archetypes_batch_raw(desc, &batch_inputs, scalar_args)?
        };

        let mut stats = crate::storage::gpu::GpuDispatchStats {
            execution_mode: engine.execution_mode(),
            buffer_cache_hits: hits,
            buffer_cache_misses: misses,
            command_buffer_submissions: submissions,
            ..Default::default()
        };
        for item in prepared {
            let archetype = &mut self.builder.archetypes[item.arch_idx];
            Q::mark_written_columns(archetype, None, self.builder.write_epoch);
            stats.dispatched_archetypes += 1;
            stats.total_rows += item.active_rows;
            stats.total_pages_16k += item.pages_16k;
        }

        Ok(stats)
    }

    /// [`QueryGpuRunner::for_each`] 的算子直传别名。
    #[inline]
    pub fn run<K: crate::storage::gpu::GpuKernel<'a, Query = Q>>(
        self,
        kernel: K,
    ) -> Result<crate::storage::gpu::GpuDispatchStats, StateError> {
        self.for_each(kernel)
    }

    /// [`QueryGpuRunner::for_each_on`] 的算子直传别名。
    #[inline]
    pub fn run_on<K: crate::storage::gpu::GpuKernel<'a, Query = Q>>(
        self,
        engine: &mut crate::storage::gpu::MetalComputeEngine,
        kernel: K,
    ) -> Result<crate::storage::gpu::GpuDispatchStats, StateError> {
        self.for_each_on(engine, kernel)
    }

    /// [`QueryGpuRunner::for_each`] 的显式派发别名。
    #[inline]
    pub fn dispatch<K: crate::storage::gpu::GpuKernel<'a, Query = Q>>(
        self,
        kernel: K,
    ) -> Result<crate::storage::gpu::GpuDispatchStats, StateError> {
        self.for_each(kernel)
    }

    /// [`QueryGpuRunner::for_each_on`] 的显式派发别名。
    #[inline]
    pub fn dispatch_on<K: crate::storage::gpu::GpuKernel<'a, Query = Q>>(
        self,
        engine: &mut crate::storage::gpu::MetalComputeEngine,
        kernel: K,
    ) -> Result<crate::storage::gpu::GpuDispatchStats, StateError> {
        self.for_each_on(engine, kernel)
    }
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> IntoIterator for QueryMutBuilder<'a, Q, F> {
    type Item = Q::Item;
    type IntoIter = QueryMutIter<'a, Q, F>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[inline]
fn archetype_matches_query_mut<'a, Q: QueryMutData<'a>>(archetype: &Archetype) -> bool {
    let arch_len = archetype.len();
    let wanted = Q::access_descriptors();
    let req_mask = Q::required_mask();
    let comp_slice = archetype.component_types_slice();
    for (idx, (cid, _, _)) in wanted.iter().enumerate() {
        let present = comp_slice.binary_search(cid).is_ok();
        if !present {
            if (req_mask & (1u16 << idx)) != 0 {
                return false;
            }
            continue;
        }
        let Some(col) = archetype.columns().get(cid) else {
            return false;
        };
        if col.len() != arch_len {
            return false;
        }
    }
    true
}

/// Descriptors are captured before lending any reference, then only raw
/// disjoint ranges are derived. Values remain owned by the exclusively borrowed Store.
pub(crate) struct MutableTable<'a> {
    entities: &'a [EntityId],
    columns: smallvec::SmallVec<[(ComponentTypeId, ColumnDescriptor); 4]>,
    row: usize,
}
impl<'a> MutableTable<'a> {
    #[inline]
    fn new<Q: QueryMutData<'a>>(archetype: &'a mut Archetype) -> Option<Self> {
        let wanted = Q::access_descriptors();
        let req_mask = Q::required_mask();
        let (entities, comp_slice, flat_cols, may_be_segmented, cols) =
            archetype.split_for_query_mut();
        let mut columns = smallvec::SmallVec::with_capacity(wanted.len());
        if !may_be_segmented && flat_cols.len() == comp_slice.len() {
            for (idx, (id, _, is_write)) in wanted.into_iter().enumerate() {
                let is_req = (req_mask & (1u16 << idx)) != 0;
                if let Some(pos) = comp_slice.iter().position(|&cid| cid == id) {
                    let entry = unsafe { *flat_cols.get_unchecked(pos) };
                    let desc = if is_write {
                        ColumnDescriptor::from_column(unsafe { &mut *entry.col_ptr })
                    } else {
                        ColumnDescriptor::from_column_ref(unsafe { &*entry.col_ptr })
                    };
                    if desc.len != entities.len() {
                        return None;
                    }
                    columns.push((id, desc));
                } else if is_req {
                    return None;
                }
            }
            return Some(Self {
                entities,
                columns,
                row: 0,
            });
        }
        for (idx, (id, _, _)) in wanted.iter().enumerate() {
            let is_req = (req_mask & (1u16 << idx)) != 0;
            if is_req && !comp_slice.contains(id) {
                return None;
            }
        }
        for (idx, (id, _, is_write)) in wanted.into_iter().enumerate() {
            let is_req = (req_mask & (1u16 << idx)) != 0;
            if let Some(col) = cols.get_mut(&id) {
                let desc = if is_write {
                    ColumnDescriptor::from_column(&mut **col)
                } else {
                    ColumnDescriptor::from_column_ref(&**col)
                };
                if desc.len != entities.len() {
                    return None;
                }
                columns.push((id, desc));
            } else if is_req || comp_slice.contains(&id) {
                return None;
            }
        }
        Some(Self {
            entities,
            columns,
            row: 0,
        })
    }
    fn next_range(&mut self) -> Option<Range<usize>> {
        if self.row >= self.entities.len() { return None; }
        let end = self.columns.iter().fold(self.entities.len(), |end, (_, col)| end.min(col.locate(self.row).2));
        assert!(end > self.row);
        let range = self.row..end;
        self.row = end;
        Some(range)
    }
}

/// Forward-only row iterator over mutable component slices.
pub struct QueryMutIter<'a, Q: QueryMutData<'a>, F: QueryFilter = All> {
    archetypes: std::slice::IterMut<'a, Archetype>,
    entities: std::slice::Iter<'a, EntityId>,
    segment_start_row: usize,
    segment_offset: usize,
    filtered_mask: Option<crate::storage::archetype::RowBitMask>,
    state: Option<Q::State>,
    current: Option<MutableTable<'a>>,
    remaining_upper_bound: usize,
    write_epoch: u64,
    filter: F,
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> Iterator for QueryMutIter<'a, Q, F> {
    type Item = Q::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if !self.entities.as_slice().is_empty() {
                if let Some(mask) = &self.filtered_mask {
                    let rem_len = self.entities.len();
                    let start_row = self.segment_start_row + self.segment_offset;
                    let seg_end = start_row + rem_len;
                    let mut cur = start_row;
                    while cur < seg_end {
                        let w_idx = cur >> 6;
                        let bit_idx = cur & 63;
                        let word = mask.words().get(w_idx).copied().unwrap_or(0) >> bit_idx;
                        if word != 0 {
                            cur += word.trailing_zeros() as usize;
                            break;
                        }
                        cur = (w_idx + 1) << 6;
                    }
                    let skip = cur.min(seg_end) - start_row;
                    if skip > 0 {
                        self.segment_offset += skip;
                        self.remaining_upper_bound -= skip;
                        if skip >= rem_len {
                            self.entities = [].iter();
                            continue;
                        }
                        self.entities = self.entities.as_slice()[skip..].iter();
                        Q::skip_rows(
                            self.state.as_mut().expect("prepared component slices"),
                            skip,
                        );
                    }
                }
                if let Some(&entity) = self.entities.next() {
                    self.segment_offset += 1;
                    self.remaining_upper_bound -= 1;
                    let item = Q::next(
                        self.state.as_mut().expect("prepared component slices"),
                        entity,
                    );
                    return Some(item);
                }
            }
            self.state = None;
            if let Some(table) = &mut self.current {
                if let Some(range) = table.next_range() {
                    self.segment_start_row = range.start;
                    self.segment_offset = 0;
                    self.state = Q::prepare(&table.columns, range.clone());
                    self.entities = table.entities[range].iter();
                    continue;
                }
            }
            self.current = None;
            self.filtered_mask = None;
            let archetype = self.archetypes.next()?;
            let arch_len = archetype.len();
            if arch_len != 0
                && self.filter.matches_archetype(archetype)
                && archetype_matches_query_mut::<Q>(archetype)
            {
                if F::IS_ROW_FILTERED {
                    let mask = self.filter.collect_matched_row_mask(archetype);
                    if mask.as_ref().is_some_and(|m| m.is_all_zero()) {
                        self.remaining_upper_bound -= arch_len;
                        continue;
                    }
                    self.filtered_mask = mask;
                }
                Q::mark_written_columns(
                    archetype,
                    self.filtered_mask.as_ref(),
                    self.write_epoch,
                );
                if !archetype.may_be_segmented_or_unaligned()
                    && archetype.flat_columns().len() == archetype.component_types_slice().len()
                {
                    if let Some((state, entities)) = Q::fast_prepare_archetype(archetype) {
                        self.segment_start_row = 0;
                        self.segment_offset = 0;
                        self.state = Some(state);
                        self.entities = entities.iter();
                        continue;
                    }
                    self.remaining_upper_bound -= arch_len;
                    continue;
                }
                self.current = MutableTable::new::<Q>(archetype);
                if self.current.is_some() { continue; }
            }
            self.remaining_upper_bound -= arch_len;
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let lower = if self.filtered_mask.is_some() {
            0
        } else {
            self.entities.len()
        };
        (lower, Some(self.remaining_upper_bound))
    }
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> std::iter::FusedIterator for QueryMutIter<'a, Q, F> {}

/// Archetype-level iterator over mutable contiguous component slices.
pub struct QueryMutChunksIter<'a, Q: QueryMutData<'a>, F: QueryFilter = All> {
    archetypes: std::slice::IterMut<'a, Archetype>,
    current: Option<MutableTable<'a>>,
    write_epoch: u64,
    filter: F,
    marker: PhantomData<Q>,
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> Iterator for QueryMutChunksIter<'a, Q, F> {
    type Item = Q::Chunk;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(table) = &mut self.current {
                if let Some(range) = table.next_range() {
                    return Q::chunk(&table.entities[range.clone()], &table.columns, range);
                }
            }
            self.current = None;
            let archetype = self.archetypes.next()?;
            if !archetype.is_empty()
                && self.filter.matches_archetype(archetype)
                && archetype_matches_query_mut::<Q>(archetype)
            {
                if F::IS_ROW_FILTERED {
                    let m = self.filter.collect_matched_row_mask(archetype);
                    if m.as_ref().is_some_and(|b| b.is_all_zero()) {
                        continue;
                    }
                }
                // M10-4：QueryMutChunksIter 向调用方暴露整张原型（或整段）的可变切片，因此必须标记全部块为已写
                Q::mark_written_columns(archetype, None, self.write_epoch);
                if !archetype.may_be_segmented_or_unaligned()
                    && archetype.flat_columns().len() == archetype.component_types_slice().len()
                {
                    if let Some(chunk) = Q::fast_chunk_archetype(archetype) {
                        return Some(chunk);
                    }
                    continue;
                }
                self.current = MutableTable::new::<Q>(archetype);
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, None)
    }
}

impl<'a, Q: QueryMutData<'a>, F: QueryFilter> std::iter::FusedIterator
    for QueryMutChunksIter<'a, Q, F>
{
}

impl Store {
    /// 导出匹配类型的连续列切片块迭代器
    #[inline]
    pub fn query_chunks<T: ModelComponent + Serialize + DeserializeOwned>(
        &self,
    ) -> QueryChunks<'_, T> {
        QueryChunks::new(&self.archetypes)
    }

    /// 开启只读元组集合查询构建器
    #[inline]
    pub fn query<'a, Q: QueryData<'a>>(
        &'a self,
    ) -> SelectBuilder<'a, Q> {
        SelectBuilder::new(self)
    }

    /// 开启排他连续批量直接写访问构建器
    #[inline]
    pub fn query_mut<'a, Q: QueryMutData<'a>>(
        &'a mut self,
    ) -> Result<QueryMutBuilder<'a, Q>, StateError> {
        self.check_poisoned()?;
        Q::validate_query();
        self.next_epoch()?;
        self.bump_epoch();
        Ok(QueryMutBuilder::new(self))
    }


}

