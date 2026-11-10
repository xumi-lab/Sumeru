use super::*;

#[must_use = "a query does no work until iterated"]
pub struct SelectBuilder<'a, Q: QueryData<'a>, F: QueryFilter = All> {
    pub(crate) archetypes: &'a [Archetype],
    pub(crate) row_upper_bound: usize,
    pub(crate) filter: F,
    pub(crate) marker: PhantomData<Q>,
}

impl<'a, Q: QueryData<'a>, F: QueryFilter> std::fmt::Debug for SelectBuilder<'a, Q, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectBuilder")
            .field("row_upper_bound", &self.row_upper_bound)
            .finish()
    }
}

impl<'a, Q: QueryData<'a>> SelectBuilder<'a, Q> {
    pub(crate) fn new(store: &'a crate::storage::store::Store) -> Self {
        Self {
            archetypes: &store.archetypes,
            row_upper_bound: store.len(),
            filter: All,
            marker: PhantomData,
        }
    }
}

impl<'a, Q: QueryData<'a>, F: QueryFilter> SelectBuilder<'a, Q, F> {
    /// Require component T without borrowing or examining its values.
    #[inline]
    pub fn with<T: 'static>(self) -> SelectBuilder<'a, Q, With<T, F>> {
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: With {
                previous: self.filter,
                type_id: ComponentTypeId::of::<T>(),
                marker: PhantomData,
            },
            marker: PhantomData,
        }
    }

    /// Skip entire tables containing component T, before extracting any slices.
    #[inline]
    pub fn without<T: 'static>(self) -> SelectBuilder<'a, Q, Without<T, F>> {
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
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
    ) -> SelectBuilder<'a, Q, WithPairTarget<R, F>> {
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: WithPairTarget::new(self.filter, target),
            marker: PhantomData,
        }
    }

    /// Skip entire tables containing relationship pair `Pair<R>`.
    #[inline]
    pub fn without_pair<R: 'static>(self) -> SelectBuilder<'a, Q, Without<crate::al::Pair<R>, F>> {
        self.without::<crate::al::Pair<R>>()
    }

    /// Require zero-sized compile-time slot tag `Tag<N>`.
    #[inline]
    pub fn with_tag<const N: u8>(self) -> SelectBuilder<'a, Q, With<crate::al::Tag<N>, F>> {
        self.with::<crate::al::Tag<N>>()
    }

    /// Skip entire tables containing zero-sized compile-time slot tag `Tag<N>`.
    #[inline]
    pub fn without_tag<const N: u8>(self) -> SelectBuilder<'a, Q, Without<crate::al::Tag<N>, F>> {
        self.without::<crate::al::Tag<N>>()
    }

    /// Require non-migrating row status bit `FLAG` (`0..8`) to be set (`1`).
    #[inline]
    pub fn with_flag<const FLAG: u8>(self) -> SelectBuilder<'a, Q, WithFlag<FLAG, F>> {
        const { assert!(FLAG < 8, "RowFlag index must be in 0..8") };
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: WithFlag::new(self.filter),
            marker: PhantomData,
        }
    }

    /// Require non-migrating row status bit `FLAG` (`0..8`) to be unset (`0`).
    #[inline]
    pub fn without_flag<const FLAG: u8>(self) -> SelectBuilder<'a, Q, WithoutFlag<FLAG, F>> {
        const { assert!(FLAG < 8, "RowFlag index must be in 0..8") };
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: WithoutFlag::new(self.filter),
            marker: PhantomData,
        }
    }

    /// Filter to 64-row chunks of component `T` modified strictly after `since_epoch`.
    #[inline]
    pub fn changed_since<T: 'static>(
        self,
        since_epoch: u64,
    ) -> SelectBuilder<'a, Q, ChangedSince<T, F>> {
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: ChangedSince::new(self.filter, since_epoch),
            marker: PhantomData,
        }
    }

    /// Add a row-level component predicate filter.
    #[inline]
    pub fn filter<C: ModelComponent + Serialize + DeserializeOwned>(
        self,
        predicate: impl Fn(&C) -> bool + Send + Sync + 'static,
    ) -> SelectBuilder<'a, Q, BusinessFilter<F, C>> {
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: BusinessFilter::new(self.filter, predicate),
            marker: PhantomData,
        }
    }

    /// Add a row-level component field equality filter.
    #[inline]
    pub fn filter_eq<C: ModelComponent + Serialize + DeserializeOwned, Field: ComponentField<C>>(
        self,
        expected: Field::Value,
    ) -> SelectBuilder<'a, Q, FilterEq<F, C, Field>> {
        SelectBuilder {
            archetypes: self.archetypes,
            row_upper_bound: self.row_upper_bound,
            filter: FilterEq::new(self.filter, expected),
            marker: PhantomData,
        }
    }

    /// Consume the builder into a lazy, forward-only row iterator.
    #[inline]
    pub fn iter(self) -> QueryIter<'a, Q, F> {
        QueryIter {
            archetypes: self.archetypes.iter(),
            current: None,
            row: 0,
            segment_start_row: 0,
            segment_offset: 0,
            entities: [].iter(),
            filtered_mask: None,
            state: None,
            remaining_upper_bound: self.row_upper_bound,
            filter: self.filter,
        }
    }

    /// Consume the builder into an archetype-level contiguous slice iterator.
    ///
    /// Iterates over every table matching the filter predicates, yielding
    /// borrow tuples of contiguous slices `(&[EntityId], (&[A], &[B]))`.
    #[inline]
    pub fn chunks(self) -> QueryChunksIter<'a, Q, F> {
        QueryChunksIter {
            archetypes: self.archetypes.iter(),
            current: None,
            row: 0,
            filter: self.filter,
            marker: PhantomData,
        }
    }

    /// **① CPU 单线程只读顺序遍历（闭包）**：对所有匹配实体逐行执行闭包，完整支持 `|(pos, vel)| { ... }` 自动类型推断。
    #[inline]
    pub fn for_each<Op>(self, mut op: Op)
    where
        Q::Item: super::QueryRowItem,
        Op: FnMut(<Q::Item as super::QueryRowItem>::Comps),
    {
        use super::QueryRowItem;
        for item in self.iter() {
            op(item.into_comps());
        }
    }

    /// **① CPU 单线程只读顺序遍历（带 `EntityId`）**：对所有匹配实体逐行执行 `|(entity, comps)|` 闭包。
    #[inline]
    pub fn for_each_with_entity<Op: FnMut(Q::Item)>(self, mut op: Op) {
        for item in self.iter() {
            op(item);
        }
    }

    /// **① CPU 单线程只读算子直传执行**：直接传入实现 [`super::QueryForEachOp`] 的算子逐行执行。
    #[inline]
    pub fn run<M, Op: super::QueryForEachOp<Q::Item, M>>(self, mut op: Op) {
        for item in self.iter() {
            op.call_row(item);
        }
    }

    /// 进入 **② CPU 多核并行 (Rayon)** 只读执行模式。
    #[inline]
    pub fn par(self) -> SelectParRunner<'a, Q, F> {
        SelectParRunner { builder: self }
    }

    /// **② CPU 多核只读并行遍历（闭包）**：在多核线程池上并行执行只读闭包（完整支持闭包参数类型自动推断）。
    #[inline]
    pub fn par_for_each<Op>(self, op: Op)
    where
        Q::Item: Send + super::QueryRowItem,
        Op: Fn(<Q::Item as super::QueryRowItem>::Comps) + Send + Sync,
    {
        self.par().for_each(op);
    }

    /// **② CPU 多核只读并行算子直传执行**：在多核线程池上并行执行算子。
    #[inline]
    pub fn par_run<M, Op: super::QueryParForEachOp<Q::Item, M>>(self, op: Op)
    where
        Q::Item: Send,
    {
        self.par().run(op);
    }
}

/// CPU 多核只读并行查询执行器（由 [`SelectBuilder::par`] 创建，采用 [`SubchunkPlanner`](crate::storage::primitives::SubchunkPlanner) 自适应缓存驻留切块，零行级堆分配）。
#[must_use = "a parallel query runner does no work until .for_each(...) or .run(...) is called"]
pub struct SelectParRunner<'a, Q: QueryData<'a>, F: QueryFilter = All> {
    builder: SelectBuilder<'a, Q, F>,
}

struct ParReadTable<'a> {
    archetype: &'a Archetype,
    mask: Option<crate::storage::archetype::RowBitMask>,
}

impl<'a, Q: QueryData<'a>, F: QueryFilter> SelectParRunner<'a, Q, F> {
    fn for_each_item_par<V>(self, visit: V)
    where
        Q::Item: Send,
        V: Fn(Q::Item) + Send + Sync,
    {
        use rayon::prelude::*;
        let morsel_rows = crate::storage::primitives::SubchunkPlanner::default()
            .subchunk_rows(Q::total_row_width());
        let mut tables = Vec::new();
        for archetype in self.builder.archetypes.iter() {
            let arch_len = archetype.len();
            if arch_len == 0 || !self.builder.filter.matches_archetype(archetype) {
                continue;
            }
            if Q::prepare(archetype, 0).is_none() {
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
            tables.push(ParReadTable { archetype, mask });
        }

        tables.par_iter().for_each(|table| {
            let arch_len = table.archetype.len();
            let num_morsels = (arch_len + morsel_rows - 1) / morsel_rows;
            (0..num_morsels).into_par_iter().for_each(|morsel_idx| {
                let morsel_start = morsel_idx * morsel_rows;
                let morsel_end = (morsel_start + morsel_rows).min(arch_len);
                if morsel_start >= morsel_end {
                    return;
                }
                if let Some(mask) = &table.mask {
                    let first_word = morsel_start >> 6;
                    let last_word = (morsel_end - 1) >> 6;
                    if (first_word..=last_word)
                        .all(|w| mask.words().get(w).copied().unwrap_or(0) == 0)
                    {
                        return;
                    }
                }
                let entities_all = table.archetype.entities();
                let mut seg_start = morsel_start;
                while seg_start < morsel_end {
                    let Some((mut state, avail_end)) = Q::prepare(table.archetype, seg_start)
                    else {
                        break;
                    };
                    let seg_end = avail_end.min(morsel_end);
                    if seg_end <= seg_start {
                        break;
                    }
                    if let Some(mask) = &table.mask {
                        let mut cur = seg_start;
                        while cur < seg_end {
                            let w_idx = cur >> 6;
                            let bit_idx = cur & 63;
                            let word = mask.words().get(w_idx).copied().unwrap_or(0) >> bit_idx;
                            if word == 0 {
                                let next_cur = ((w_idx + 1) << 6).min(seg_end);
                                let skip = next_cur - cur;
                                if skip > 0 {
                                    Q::skip_rows(&mut state, skip);
                                }
                                cur = next_cur;
                                continue;
                            }
                            let tz = word.trailing_zeros() as usize;
                            let target = (cur + tz).min(seg_end);
                            let skip = target - cur;
                            if skip > 0 {
                                Q::skip_rows(&mut state, skip);
                            }
                            if target >= seg_end {
                                break;
                            }
                            let item = Q::next(&mut state, entities_all[target]);
                            visit(item);
                            cur = target + 1;
                        }
                    } else {
                        for &entity in &entities_all[seg_start..seg_end] {
                            let item = Q::next(&mut state, entity);
                            visit(item);
                        }
                    }
                    seg_start = seg_end;
                }
            });
        });
    }

    /// 在 Rayon 多核线程池上按 1024 行 Morsel 连续切片并行对所有匹配实体执行只读闭包（完整支持闭包参数类型自动推断）。
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

    /// 在 Rayon 多核线程池上按 1024 行 Morsel 连续切片并行对所有匹配实体执行 `|(entity, comps)|` 闭包。
    pub fn for_each_with_entity<Op>(self, op: Op)
    where
        Q::Item: Send,
        Op: Fn(Q::Item) + Send + Sync,
    {
        self.for_each_item_par(op);
    }

    /// 在 Rayon 多核线程池上按 1024 行 Morsel 连续切片并行对所有匹配实体直接执行算子。
    pub fn run<M, Op: super::QueryParForEachOp<Q::Item, M>>(self, op: Op)
    where
        Q::Item: Send,
    {
        self.for_each_item_par(|item| {
            op.call_row_par(item);
        });
    }
}

impl<'a, Q: QueryData<'a>, F: QueryFilter> IntoIterator for SelectBuilder<'a, Q, F> {
    type Item = Q::Item;
    type IntoIter = QueryIter<'a, Q, F>;
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Lock-step iteration over borrowed entity and component slices.
///
/// No per-row identity lookup, downcast, allocation, clone or encoding is done.
/// Order follows current table/row order; it is not stable across structural
/// writes or snapshot restoration. Borrowing excludes writes during iteration.
pub struct QueryIter<'a, Q: QueryData<'a>, F: QueryFilter = All> {
    archetypes: std::slice::Iter<'a, Archetype>,
    current: Option<&'a Archetype>,
    row: usize,
    segment_start_row: usize,
    segment_offset: usize,
    entities: std::slice::Iter<'a, EntityId>,
    filtered_mask: Option<crate::storage::archetype::RowBitMask>,
    state: Option<Q::State>,
    remaining_upper_bound: usize,
    filter: F,
}

impl<'a, Q: QueryData<'a>, F: QueryFilter> Iterator for QueryIter<'a, Q, F> {
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
            if let Some(archetype) = self.current {
                if self.row < archetype.len() {
                    if let Some((state, end)) = Q::prepare(archetype, self.row) {
                        assert!(end > self.row && end <= archetype.len(), "query segment must advance");
                        self.segment_start_row = self.row;
                        self.segment_offset = 0;
                        self.entities = archetype.entities()[self.row..end].iter();
                        self.row = end;
                        self.state = Some(state);
                        continue;
                    }
                    self.remaining_upper_bound -= archetype.len() - self.row;
                }
                self.current = None;
                self.filtered_mask = None;
            }
            let archetype = self.archetypes.next()?;
            if archetype.is_empty() {
                continue;
            }
            if self.filter.matches_archetype(archetype) {
                if F::IS_ROW_FILTERED {
                    let mask = self.filter.collect_matched_row_mask(archetype);
                    if mask.as_ref().is_some_and(|m| m.is_all_zero()) {
                        self.remaining_upper_bound -= archetype.len();
                        continue;
                    }
                    self.filtered_mask = mask;
                }
                self.current = Some(archetype);
                self.row = 0;
                continue;
            }
            self.remaining_upper_bound -= archetype.len();
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

impl<'a, Q: QueryData<'a>, F: QueryFilter> std::iter::FusedIterator for QueryIter<'a, Q, F> {}

/// Archetype-level iteration over contiguous entity and component slices.
///
/// Yields `Q::Chunk` (e.g., `(&'a [EntityId], (&'a [A], &'a [B]))`) for each matching table.
/// Slices are contiguous, equal-length, zero-copy, and suitable for SIMD or batch processing.
pub struct QueryChunksIter<'a, Q: QueryData<'a>, F: QueryFilter = All> {
    archetypes: std::slice::Iter<'a, Archetype>,
    current: Option<&'a Archetype>,
    row: usize,
    filter: F,
    marker: PhantomData<Q>,
}

impl<'a, Q: QueryData<'a>, F: QueryFilter> Iterator for QueryChunksIter<'a, Q, F> {
    type Item = Q::Chunk;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(archetype) = self.current {
                if self.row < archetype.len() {
                    if let Some((chunk, end)) = Q::chunk(archetype, self.row) {
                        assert!(end > self.row && end <= archetype.len(), "query segment must advance");
                        if end < archetype.len() {
                            self.row = end;
                        } else {
                            self.current = None;
                        }
                        return Some(chunk);
                    }
                }
                self.current = None;
            }
            let archetype = self.archetypes.next()?;
            if archetype.is_empty() {
                continue;
            }
            if self.filter.matches_archetype(archetype) {
                if F::IS_ROW_FILTERED {
                    let m = self.filter.collect_matched_row_mask(archetype);
                    if m.as_ref().is_some_and(|b| b.is_all_zero()) {
                        continue;
                    }
                }
                if let Some((chunk, end)) = Q::chunk(archetype, 0) {
                    assert!(end > 0 && end <= archetype.len(), "query segment must advance");
                    if end < archetype.len() {
                        self.current = Some(archetype);
                        self.row = end;
                    }
                    return Some(chunk);
                }
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        // One table can now yield several physical segments.
        (0, None)
    }
}

impl<'a, Q: QueryData<'a>, F: QueryFilter> std::iter::FusedIterator for QueryChunksIter<'a, Q, F> {}

/// 强类型分块查询迭代器。
///
/// 遍历包含组件 `T` 的所有 Archetype，直接借用连续切片 `(&[EntityId], &[T])`，实现完全零堆分配与零拷贝遍历。
pub struct QueryChunks<'a, T: ModelComponent + Serialize + DeserializeOwned> {
    inner: QueryChunksIter<'a, &'a T>,
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> QueryChunks<'a, T> {
    pub(crate) fn new(archetypes: &'a [Archetype]) -> Self {
        Self {
            inner: QueryChunksIter {
                archetypes: archetypes.iter(),
                current: None,
                row: 0,
                filter: All,
                marker: PhantomData,
            },
        }
    }
}

impl<'a, T: ModelComponent + Serialize + DeserializeOwned> Iterator for QueryChunks<'a, T> {
    type Item = (&'a [EntityId], &'a [T]);

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

