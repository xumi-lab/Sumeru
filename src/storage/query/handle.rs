//! 长期可复用查询句柄与流式构建器 (Query & QueryBuilder)
//!
//! 提供基于强类型访问形态 `Q: QueryMutShape` 与过滤链 `F: QueryFilter` 的长期查询能力：
//! 1. **Query**：持有不可变过滤硬边界与匹配成功的 ArchetypeId 缓存，随代际戳自适应增量刷新；
//! 2. **QueryBuilder**：提供链式 `.with()`、`.without()`、`.filter_eq()`、`.filter()` 构造 API；
//! 3. **QueryEntityRequest**：封装面向确定性实体 ID 列表的查询请求契约。

use std::marker::PhantomData;

use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::id::ArchetypeId;
use crate::id::ComponentTypeId;
use super::spec::ComponentField;
use super::spec::{
    All, BusinessFilter, ChangedSince, FilterEq, QueryFilter, QueryMutShape, With, WithFlag,
    WithPairTarget, Without, WithoutFlag,
};
use crate::storage::store::Store;
use crate::binding::EntityId;
use crate::context::ModelComponent;
use crate::error::StateError;

/// 预绑定原型表与组件物理列槽位偏移项（Level-2 预绑定缓存，对标 Flecs `ecs_query_t`）
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatchedTableBinding {
    pub archetype_id: ArchetypeId,
    /// 按查询形态 `Q` 的参数声明顺序预绑定的 `Archetype::flat_columns` 物理列下标
    pub flat_col_indices: smallvec::SmallVec<[u16; 8]>,
}

/// 长期可复用查询句柄 (Query)。
///
/// 保存查询类型签名 `Q`（组件读写形态）、过滤器 `F` 以及基于 `archetype_set_generation` 缓存的物理原型表列表。
/// 杜绝在每次查询构建时全表重新扫描，并在底层存储发生结构修改（Archetype 增删）时支持 $O(\Delta A)$ 自适应增量刷新。
#[derive(Debug, Clone)]
pub(crate) struct QueryCache {
    pub(crate) archetype_set_generation: u64,
    pub(crate) topology_epoch: u64,
    pub(crate) validated_archetype_len: usize,
    pub(crate) matched_archetypes: std::sync::Arc<[ArchetypeId]>,
    pub(crate) matched_bindings: std::sync::Arc<[MatchedTableBinding]>,
    pub(crate) shared_template: Option<std::sync::Arc<dyn std::any::Any + Send + Sync>>,
    pub(crate) template_limit_bytes: usize,
    #[cfg(feature = "test-internals")]
    pub(crate) template_rebuilds: u64,
}

impl QueryCache {
    #[inline]
    pub(crate) fn is_up_to_date(&self, store: &Store) -> bool {
        self.archetype_set_generation == store.archetype_set_generation()
            && self.topology_epoch == store.archetype_topology_epoch()
            && self.validated_archetype_len == store.archetypes.len()
            && self.matched_bindings.len() == self.matched_archetypes.len()
    }

    pub(crate) fn refresh_for_store<Q: QueryMutShape, F: QueryFilter>(
        &mut self,
        store: &Store,
        filter: &F,
    ) {
        if self.is_up_to_date(store) {
            return;
        }
        let current_gen = store.archetype_set_generation();
        let current_topo = store.archetype_topology_epoch();
        let current_len = store.archetypes.len();

        if self.topology_epoch == current_topo
            && self.validated_archetype_len <= current_len
            && self.matched_bindings.len() == self.matched_archetypes.len()
        {
            if self.validated_archetype_len < current_len {
                let mut delta_matched = Vec::new();
                let mut delta_bindings = Vec::new();
                if super::spec::append_bindings_from_slice::<
                    Q,
                    F,
                    crate::storage::archetype::Archetype,
                >(
                    &store.archetypes[self.validated_archetype_len..current_len],
                    filter,
                    &mut delta_matched,
                    &mut delta_bindings,
                )
                .is_ok()
                    && !delta_matched.is_empty()
                {
                    let mut all_matched =
                        Vec::with_capacity(self.matched_archetypes.len() + delta_matched.len());
                    all_matched.extend_from_slice(&self.matched_archetypes);
                    all_matched.extend(delta_matched);

                    let mut all_bindings =
                        Vec::with_capacity(self.matched_bindings.len() + delta_bindings.len());
                    all_bindings.extend_from_slice(&self.matched_bindings);
                    all_bindings.extend(delta_bindings);

                    self.matched_archetypes = std::sync::Arc::from(all_matched.into_boxed_slice());
                    self.matched_bindings = std::sync::Arc::from(all_bindings.into_boxed_slice());
                }
            }
            self.validated_archetype_len = current_len;
            self.archetype_set_generation = current_gen;
            return;
        }

        let (matched, bindings) = super::spec::rebuild_matched_bindings::<Q, F>(store, filter);
        self.matched_archetypes = std::sync::Arc::from(matched.into_boxed_slice());
        self.matched_bindings = std::sync::Arc::from(bindings.into_boxed_slice());
        self.archetype_set_generation = current_gen;
        self.topology_epoch = current_topo;
        self.validated_archetype_len = current_len;
    }
}

/// 长期可复用查询句柄 (Query)。
///
/// 保存查询类型签名 `Q`（组件读写形态）、预编译的访问描述符、过滤器 `F` 以及基于 `archetype_set_generation` 缓存的物理原型表列表。
/// 杜绝在每次查询构建时全表重新扫描，并在底层存储发生结构修改（Archetype 增删）时支持自适应增量刷新。
#[derive(Debug)]
pub struct Query<Q: QueryMutShape, F: QueryFilter + Clone = All> {
    pub(crate) store_id: u64,
    pub(crate) session_closed: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub(crate) cache: std::sync::RwLock<QueryCache>,
    pub(crate) descriptors: std::sync::Arc<[(ComponentTypeId, &'static str, bool)]>,
    pub(crate) filter: F,
    pub(crate) entities: Option<std::sync::Arc<[EntityId]>>,
    pub(crate) marker: PhantomData<fn() -> Q>,
}

impl<Q: QueryMutShape, F: QueryFilter + Clone> Clone for Query<Q, F> {
    fn clone(&self) -> Self {
        let cache_val = self
            .cache
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|p| p.into_inner().clone());
        Self {
            store_id: self.store_id,
            session_closed: self.session_closed.clone(),
            cache: std::sync::RwLock::new(cache_val),
            descriptors: std::sync::Arc::clone(&self.descriptors),
            filter: self.filter.clone(),
            entities: self.entities.clone(),
            marker: PhantomData,
        }
    }
}

impl<Q: QueryMutShape> Query<Q, All> {
    /// 从底层存储快速创建无过滤的长期查询对象
    #[inline]
    pub fn from_store(store: &Store) -> Result<Self, StateError> {
        QueryBuilder::<Q>::new().build(store)
    }
}

impl<Q: QueryMutShape, F: QueryFilter + Clone> Query<Q, F> {
    pub(crate) fn check_session(&self) -> Result<(), StateError> {
        if self.session_closed.as_ref().is_some_and(|flag| flag.load(std::sync::atomic::Ordering::Acquire)) {
            return Err(StateError::AccessDenied("SharedSession is closed".into()));
        }
        Ok(())
    }

    /// Explicitly transfer this declaration out of its application session.
    pub fn into_shared(mut self) -> Self {
        self.session_closed = None;
        self
    }
    /// 创建新的长期查询对象（仅限内部由 QueryBuilder::build 构建，杜绝外部传入伪造的匹配原型列表）。
    #[inline]
    pub(crate) fn new(
        store_id: u64,
        archetype_set_generation: u64,
        matched_archetypes: std::sync::Arc<[ArchetypeId]>,
        descriptors: std::sync::Arc<[(ComponentTypeId, &'static str, bool)]>,
        filter: F,
    ) -> Self {
        Self::new_with_bindings(
            store_id,
            archetype_set_generation,
            0,
            0,
            matched_archetypes,
            std::sync::Arc::from([]),
            descriptors,
            filter,
        )
    }

    /// 创建包含 Level-2 预绑定槽位的长期查询对象。
    #[inline]
    pub(crate) fn new_with_bindings(
        store_id: u64,
        archetype_set_generation: u64,
        topology_epoch: u64,
        validated_archetype_len: usize,
        matched_archetypes: std::sync::Arc<[ArchetypeId]>,
        matched_bindings: std::sync::Arc<[MatchedTableBinding]>,
        descriptors: std::sync::Arc<[(ComponentTypeId, &'static str, bool)]>,
        filter: F,
    ) -> Self {
        Self {
            store_id,
            session_closed: None,
            cache: std::sync::RwLock::new(QueryCache {
                archetype_set_generation,
                topology_epoch,
                validated_archetype_len,
                matched_archetypes,
                matched_bindings,
                shared_template: None,
                template_limit_bytes: 64 * 1024 * 1024,
                #[cfg(feature = "test-internals")]
                template_rebuilds: 0,
            }),
            descriptors,
            filter,
            entities: None,
            marker: PhantomData,
        }
    }

    /// 基于给定过滤器从底层存储构建 Query
    #[inline]
    pub fn with_filter(store: &Store, filter: F) -> Result<Self, StateError> {
        QueryBuilder {
            filter,
            marker: PhantomData,
        }
        .build(store)
    }

    /// 绑定的存储内核实例 ID。
    #[inline]
    pub fn store_id(&self) -> u64 {
        self.store_id
    }

    /// 匹配的原型表集合代际版本。
    #[inline]
    pub fn archetype_set_generation(&self) -> u64 {
        self.cache
            .read()
            .map(|g| g.archetype_set_generation)
            .unwrap_or_else(|p| p.into_inner().archetype_set_generation)
    }

    /// 获取预编译的组件访问描述符切片引用（0 堆分配、0 锁保护、编译期确定不可变）。
    #[inline]
    pub fn descriptors(&self) -> &[(ComponentTypeId, &'static str, bool)] {
        &self.descriptors
    }

    /// 获取显式实体列表切片引用（若指定了实体限制）。
    #[inline]
    pub fn entities(&self) -> Option<&[EntityId]> {
        self.entities.as_deref()
    }

    /// Bound retained shared identity/request templates (default: 64 MiB).
    /// Zero disables this optimization; over-limit queries use the original
    /// protected path. This is not a process-wide or live-View memory budget.
    /// Changing the bound drops this Query's template, not any active View.
    pub fn set_template_cache_limit_bytes(&mut self, limit: usize) -> Result<(), StateError> {
        let cache = self.cache.get_mut().map_err(|_| StateError::SessionPoisoned)?;
        cache.template_limit_bytes = limit;
        cache.shared_template = None;
        Ok(())
    }

    /// 匹配的原型表 ID 列表快照（克隆 Arc<[ArchetypeId]>，零切片元素深拷贝）。
    #[inline]
    pub fn matched_archetypes_arc(&self) -> std::sync::Arc<[ArchetypeId]> {
        self.cache
            .read()
            .map(|g| std::sync::Arc::clone(&g.matched_archetypes))
            .unwrap_or_else(|p| std::sync::Arc::clone(&p.into_inner().matched_archetypes))
    }

    /// 匹配的原型表 ID 列表快照。
    #[inline]
    pub fn matched_archetypes(&self) -> Vec<ArchetypeId> {
        self.matched_archetypes_arc().to_vec()
    }

    /// 匹配的原型表数量。
    #[inline]
    pub fn matched_count(&self) -> usize {
        self.cache
            .read()
            .map(|g| g.matched_archetypes.len())
            .unwrap_or_else(|p| p.into_inner().matched_archetypes.len())
    }

    /// 当前配置的过滤器引用。
    #[inline]
    pub fn filter(&self) -> &F {
        &self.filter
    }

    /// 动态更新过滤器链中所有 [`ChangedSince`](crate::storage::query::spec::ChangedSince) 节点的 `since_epoch` 游标（不改变原型拓扑匹配缓存）。
    #[inline]
    pub fn set_filter_since_epoch(&mut self, epoch: u64) {
        self.filter.update_since_epoch(epoch);
    }

    /// 独占刷新缓存：持有 &mut self 时直接通过 get_mut 绕过 RwLock，彻底消除原子锁争用。
    #[inline]
    pub fn ensure_cache_current_mut(&mut self, store: &Store) -> Result<(), StateError> {
        self.check_session()?;
        if store.store_id() != self.store_id {
            return Err(StateError::StoreMismatch {
                expected_store: self.store_id,
                actual_store: store.store_id(),
            });
        }

        let cache = self.cache.get_mut().map_err(|_| StateError::SessionPoisoned)?;
        cache.refresh_for_store::<Q, F>(store, &self.filter);
        Ok(())
    }

    /// 独占刷新缓存的同义方法。
    #[inline]
    pub fn ensure_cache_current(&mut self, store: &Store) -> Result<(), StateError> {
        self.ensure_cache_current_mut(store)
    }

    /// Refresh/check the cache under its lock, then invoke the callback WITHOUT
    /// that lock. The Arc snapshot keeps the slice alive (no element copy or
    /// allocation on a cache hit). Callback reentry/panic cannot poison the cache.
    pub(crate) fn with_matched_slice<R>(
        &self,
        store: &Store,
        f: impl FnOnce(&[ArchetypeId], &[(ComponentTypeId, &'static str, bool)]) -> Result<R, StateError>,
    ) -> Result<R, StateError> {
        self.check_session()?;
        if store.store_id() != self.store_id {
            return Err(StateError::StoreMismatch {
                expected_store: self.store_id,
                actual_store: store.store_id(),
            });
        }
        {
            let guard = self.cache.read().map_err(|_| StateError::SessionPoisoned)?;
            if guard.is_up_to_date(store) {
                let matched = std::sync::Arc::clone(&guard.matched_archetypes);
                drop(guard);
                return f(&matched, &self.descriptors);
            }
        }
        let mut guard = self.cache.write().map_err(|_| StateError::SessionPoisoned)?;
        guard.refresh_for_store::<Q, F>(store, &self.filter);
        let matched = std::sync::Arc::clone(&guard.matched_archetypes);
        drop(guard);
        f(&matched, &self.descriptors)
    }

    /// 刷新并获取预绑定物理列槽位 `Arc<[MatchedTableBinding]>` 快照（缓存命中时零堆分配）。
    pub(crate) fn resolved_bindings_arc(
        &self,
        store: &Store,
    ) -> Result<std::sync::Arc<[MatchedTableBinding]>, StateError> {
        self.check_session()?;
        if store.store_id() != self.store_id {
            return Err(StateError::StoreMismatch {
                expected_store: self.store_id,
                actual_store: store.store_id(),
            });
        }
        {
            let guard = self.cache.read().map_err(|_| StateError::SessionPoisoned)?;
            if guard.is_up_to_date(store) {
                return Ok(std::sync::Arc::clone(&guard.matched_bindings));
            }
        }
        let mut guard = self.cache.write().map_err(|_| StateError::SessionPoisoned)?;
        guard.refresh_for_store::<Q, F>(store, &self.filter);
        Ok(std::sync::Arc::clone(&guard.matched_bindings))
    }

    /// 匹配的原型表与预绑定物理列槽位快照（克隆 Arc<[MatchedTableBinding]>，零切片元素深拷贝）。
    #[inline]
    pub fn matched_bindings_arc(&self) -> std::sync::Arc<[MatchedTableBinding]> {
        self.cache
            .read()
            .map(|g| std::sync::Arc::clone(&g.matched_bindings))
            .unwrap_or_else(|p| std::sync::Arc::clone(&p.into_inner().matched_bindings))
    }

    /// 独占闭包借用预绑定原型表槽位切片与过滤器引用：0 锁争用、0 堆分配、0 次克隆。
    #[inline]
    pub fn with_matched_bindings_mut<R>(
        &mut self,
        store: &Store,
        f: impl FnOnce(&[MatchedTableBinding], &F) -> Result<R, StateError>,
    ) -> Result<R, StateError> {
        self.ensure_cache_current_mut(store)?;
        let cache = self.cache.get_mut().map_err(|_| StateError::SessionPoisoned)?;
        f(&cache.matched_bindings, &self.filter)
    }

    /// 共享闭包借用匹配原型切片：消除任何全量克隆 Vec<ArchetypeId> 的开销。
    #[inline]
    pub fn with_matched_archetypes<R>(
        &self,
        store: &Store,
        f: impl FnOnce(&[ArchetypeId]) -> Result<R, StateError>,
    ) -> Result<R, StateError> {
        self.with_matched_slice(store, |matched, _| f(matched))
    }

    /// 独占闭包借用匹配原型切片与过滤器引用：0 锁争用、0 堆分配、0 次克隆。
    #[inline]
    pub fn with_matched_archetypes_mut<R>(
        &mut self,
        store: &Store,
        f: impl FnOnce(&[ArchetypeId], &F) -> Result<R, StateError>,
    ) -> Result<R, StateError> {
        self.ensure_cache_current_mut(store)?;
        let cache = self.cache.get_mut().map_err(|_| StateError::SessionPoisoned)?;
        f(&cache.matched_archetypes, &self.filter)
    }

    /// 构造指定实体 ID 列表的查询请求对象。
    #[inline]
    pub fn entity_request<'a>(&'a self, ids: &'a [EntityId]) -> QueryEntityRequest<'a, Q, F> {
        QueryEntityRequest::new(self, ids)
    }
}

/// 实体级查询请求描述对象。
#[derive(Clone, Copy, Debug)]
pub struct QueryEntityRequest<'a, Q: QueryMutShape, F: QueryFilter + Clone> {
    query: &'a Query<Q, F>,
    ids: &'a [EntityId],
}

impl<'a, Q: QueryMutShape, F: QueryFilter + Clone> QueryEntityRequest<'a, Q, F> {
    #[inline]
    pub fn new(query: &'a Query<Q, F>, ids: &'a [EntityId]) -> Self {
        Self { query, ids }
    }

    #[inline]
    pub fn query(&self) -> &'a Query<Q, F> {
        self.query
    }

    #[inline]
    pub fn ids(&self) -> &'a [EntityId] {
        self.ids
    }
}

/// 强类型查询构建器。
#[derive(Clone, Debug)]
pub struct QueryBuilder<Q: QueryMutShape, F = All> {
    pub(crate) filter: F,
    pub(crate) marker: PhantomData<fn() -> Q>,
}

impl<Q: QueryMutShape> QueryBuilder<Q, All> {
    /// 创建无过滤初始状态的查询构建器。
    #[inline]
    pub fn new() -> Self {
        Self {
            filter: All,
            marker: PhantomData,
        }
    }
}

impl<Q: QueryMutShape> Default for QueryBuilder<Q, All> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<Q: QueryMutShape, F: QueryFilter + Clone> QueryBuilder<Q, F> {
    /// 使用指定的过滤器创建查询构建器。
    #[inline]
    pub fn with_filter(filter: F) -> Self {
        Self {
            filter,
            marker: PhantomData,
        }
    }

    /// 增加包含组件 T 的硬边界过滤条件（仅校验表级原型元数据）。
    #[inline]
    pub fn with<T: 'static>(self) -> QueryBuilder<Q, With<T, F>> {
        QueryBuilder {
            filter: With::new(self.filter),
            marker: PhantomData,
        }
    }

    /// 增加排除组件 T 的硬边界过滤条件（仅校验表级原型元数据）。
    #[inline]
    pub fn without<T: 'static>(self) -> QueryBuilder<Q, Without<T, F>> {
        QueryBuilder {
            filter: Without::new(self.filter),
            marker: PhantomData,
        }
    }

    /// 增加包含特定目标实体关系对 `(R, target)` 的行级硬边界过滤条件（基于原型内嵌二级位图索引加速）。
    #[inline]
    pub fn with_pair<R: ModelComponent + Serialize + DeserializeOwned + 'static>(
        self,
        target: EntityId,
    ) -> QueryBuilder<Q, WithPairTarget<R, F>> {
        QueryBuilder {
            filter: WithPairTarget::new(self.filter, target),
            marker: PhantomData,
        }
    }

    /// 增加排除关系对 `Pair<R>` 的表级硬边界过滤条件。
    #[inline]
    pub fn without_pair<R: 'static>(self) -> QueryBuilder<Q, Without<crate::al::Pair<R>, F>> {
        self.without::<crate::al::Pair<R>>()
    }

    /// 增加包含零大小静态标签 `Tag<N>` 的表级硬边界过滤条件。
    #[inline]
    pub fn with_tag<const N: u8>(self) -> QueryBuilder<Q, With<crate::al::Tag<N>, F>> {
        self.with::<crate::al::Tag<N>>()
    }

    /// 增加排除零大小静态标签 `Tag<N>` 的表级硬边界过滤条件。
    #[inline]
    pub fn without_tag<const N: u8>(self) -> QueryBuilder<Q, Without<crate::al::Tag<N>, F>> {
        self.without::<crate::al::Tag<N>>()
    }

    /// 增加包含非迁移行级位平面状态标记 `FLAG` 的过滤条件（O(1) 64-bit word 位与）。
    #[inline]
    pub fn with_flag<const FLAG: u8>(self) -> QueryBuilder<Q, WithFlag<FLAG, F>> {
        QueryBuilder {
            filter: WithFlag::new(self.filter),
            marker: PhantomData,
        }
    }

    /// 增加排除非迁移行级位平面状态标记 `FLAG` 的过滤条件（O(1) 64-bit word 位差）。
    #[inline]
    pub fn without_flag<const FLAG: u8>(self) -> QueryBuilder<Q, WithoutFlag<FLAG, F>> {
        QueryBuilder {
            filter: WithoutFlag::new(self.filter),
            marker: PhantomData,
        }
    }

    /// 增加基于 64 行分块写代际（Chunk Write Epoch）的增量脏数据过滤条件。
    #[inline]
    pub fn changed_since<T: 'static>(
        self,
        since_epoch: u64,
    ) -> QueryBuilder<Q, ChangedSince<T, F>> {
        QueryBuilder {
            filter: ChangedSince::new(self.filter, since_epoch),
            marker: PhantomData,
        }
    }

    /// 增加基于组件特定字段相等性的硬边界过滤条件。
    #[inline]
    pub fn filter_eq<C: ModelComponent + Serialize + DeserializeOwned, Field: ComponentField<C>>(
        self,
        expected: Field::Value,
    ) -> QueryBuilder<Q, FilterEq<F, C, Field>> {
        QueryBuilder {
            filter: FilterEq::new(self.filter, expected),
            marker: PhantomData,
        }
    }

    /// 增加基于受信任业务谓词闭包的行级过滤条件。
    #[inline]
    pub fn filter<C: ModelComponent + Serialize + DeserializeOwned>(
        self,
        predicate: impl Fn(&C) -> bool + Send + Sync + 'static,
    ) -> QueryBuilder<Q, BusinessFilter<F, C>> {
        QueryBuilder {
            filter: BusinessFilter::new(self.filter, predicate),
            marker: PhantomData,
        }
    }

    /// 构建长期可复用的查询句柄，利用组件倒排位图索引预计算匹配成功的 ArchetypeId 集合及物理列槽位偏移。
    pub fn build(self, store: &Store) -> Result<Query<Q, F>, StateError> {
        let (matched, bindings) = super::spec::match_bindings_for_store::<Q, F>(store, &self.filter)?;
        let descriptors = std::sync::Arc::from(
            super::spec::merged_access_descriptors::<Q, F>(&self.filter).into_boxed_slice(),
        );
        Ok(Query::new_with_bindings(
            store.store_id(),
            store.archetype_set_generation(),
            store.archetype_topology_epoch(),
            store.archetypes.len(),
            std::sync::Arc::from(matched.into_boxed_slice()),
            std::sync::Arc::from(bindings.into_boxed_slice()),
            descriptors,
            self.filter,
        ))
    }

    /// 构建长期可复用的查询计划句柄（`build` 的同义方法）。
    #[inline]
    pub fn plan(self, store: &Store) -> Result<Query<Q, F>, StateError> {
        self.build(store)
    }
}

impl Store {
    /// 创建面向持久化查询计划 `Query<Q, F>` 的流式构建器。
    #[inline]
    pub fn query_builder<Q: QueryMutShape>(&self) -> QueryBuilder<Q, All> {
        QueryBuilder::new()
    }
}

impl<Q: QueryMutShape, F: QueryFilter + Clone> Query<Q, F> {
    /// 返回当前查询基于查询形状 `Q` 与过滤器 `F` 完整推导出的调度器访问描述符（包含读写集、原型互斥过滤与行级标志依赖）。
    #[inline]
    pub fn access_descriptor(&self) -> crate::schedule::AccessDescriptor {
        crate::schedule::system::deduce_access::<Q, F>(&self.filter)
    }

    /// 检测当前查询计划与另一查询计划是否存在组件级或标志位读写冲突（自动考虑 `With<T>`/`Without<T>` 原型不相交剪枝）。
    #[inline]
    pub fn conflicts_with<Q2: QueryMutShape, F2: QueryFilter + Clone>(
        &self,
        other: &Query<Q2, F2>,
    ) -> bool {
        self.access_descriptor().conflicts_with(&other.access_descriptor())
    }

    /// 判定当前查询计划与另一查询计划是否兼容（可无冲突并发执行）。
    #[inline]
    pub fn is_compatible_with<Q2: QueryMutShape, F2: QueryFilter + Clone>(
        &self,
        other: &Query<Q2, F2>,
    ) -> bool {
        !self.conflicts_with(other)
    }

    /// 基于预编译的匹配原型表与 Level-2 槽位绑定缓存遍历只读元素。
    pub fn for_each<'store>(
        &self,
        store: &'store Store,
        f: impl FnMut(Q::Item<'store>),
    ) -> Result<(), StateError>
    where
        for<'a> Q::View<'a>: super::spec::QueryData<'a, Item = Q::Item<'a>>,
    {
        let bindings = self.resolved_bindings_arc(store)?;
        super::spec::iter::for_each_matched_ro::<Q, F>(store, &bindings, &self.filter, f);
        Ok(())
    }

    /// 基于预编译的匹配原型表与 Level-2 槽位绑定缓存遍历可变元素（零堆分配 Arc 快照）。
    pub fn for_each_mut<'store>(
        &self,
        store: &'store mut Store,
        f: impl FnMut(Q::Item<'store>),
    ) -> Result<(), StateError> {
        store.check_poisoned()?;
        Q::validate_shape()?;
        store.next_epoch()?;
        let bindings = self.resolved_bindings_arc(store)?;
        store.bump_epoch();
        super::spec::iter::for_each_matched_mut::<Q, F>(store, &bindings, &self.filter, f);
        Ok(())
    }

    /// 基于预编译的匹配原型表缓存遍历只读连续切片块（Level-2 预绑定极速路径）。
    pub fn for_each_chunk<'store>(
        &self,
        store: &'store Store,
        f: impl FnMut(Q::Chunk<'store>),
    ) -> Result<(), StateError>
    where
        for<'a> Q::View<'a>: super::spec::QueryData<'a, Chunk = Q::Chunk<'a>>,
    {
        let bindings = self.resolved_bindings_arc(store)?;
        super::spec::iter::for_each_matched_chunk_ro::<Q, F>(store, &bindings, &self.filter, f);
        Ok(())
    }

    /// 基于预编译的匹配原型表缓存遍历可变连续切片块（Level-2 预绑定极速路径，零堆分配 Arc 快照）。
    pub fn for_each_chunk_mut<'store>(
        &self,
        store: &'store mut Store,
        f: impl FnMut(Q::Chunk<'store>),
    ) -> Result<(), StateError> {
        store.check_poisoned()?;
        Q::validate_shape()?;
        store.next_epoch()?;
        let bindings = self.resolved_bindings_arc(store)?;
        store.bump_epoch();
        super::spec::iter::for_each_matched_chunk_mut::<Q, F>(store, &bindings, &self.filter, f);
        Ok(())
    }

    /// 基于 Rayon 的多核并行只读行遍历（按 4096 行 = 64 个掩码字对齐子块切分，支持 `RowBitMask` 字级跳跃过滤）。
    pub fn par_for_each<'store>(
        &self,
        store: &'store Store,
        f: impl Fn(Q::Item<'store>) + Send + Sync,
    ) -> Result<(), StateError>
    where
        for<'a> Q::View<'a>: super::spec::QueryData<'a, Item = Q::Item<'a>>,
    {
        let bindings = self.resolved_bindings_arc(store)?;
        super::spec::iter::par_for_each_matched_ro::<Q, F>(store, &bindings, &self.filter, f);
        Ok(())
    }

    /// 基于 Rayon 的多核并行可变行遍历（主线程统一完成 COW 解耦与 64 行脏块纪元标记，工作线程无锁并行写互不重叠的 4096 行子块）。
    pub fn par_for_each_mut<'store>(
        &self,
        store: &'store mut Store,
        f: impl Fn(Q::Item<'store>) + Send + Sync,
    ) -> Result<(), StateError> {
        store.check_poisoned()?;
        Q::validate_shape()?;
        store.next_epoch()?;
        let bindings = self.resolved_bindings_arc(store)?;
        store.bump_epoch();
        super::spec::iter::par_for_each_matched_mut::<Q, F>(store, &bindings, &self.filter, f);
        Ok(())
    }

    /// 基于 Rayon 的多核并行只读连续切片块遍历（按 4096 行子块切分，支持 `ChangedSince` 与行级掩码连续段）。
    pub fn par_for_each_chunk<'store>(
        &self,
        store: &'store Store,
        f: impl Fn(Q::Chunk<'store>) + Send + Sync,
    ) -> Result<(), StateError>
    where
        for<'a> Q::View<'a>: super::spec::QueryData<'a, Chunk = Q::Chunk<'a>>,
    {
        let bindings = self.resolved_bindings_arc(store)?;
        super::spec::iter::par_for_each_matched_chunk_ro::<Q, F>(
            store,
            &bindings,
            &self.filter,
            f,
        );
        Ok(())
    }

    /// 基于 Rayon 的多核并行可变连续切片块遍历（按 4096 行子块切分，支持 `ChangedSince` 与行级掩码连续段）。
    pub fn par_for_each_chunk_mut<'store>(
        &self,
        store: &'store mut Store,
        f: impl Fn(Q::Chunk<'store>) + Send + Sync,
    ) -> Result<(), StateError> {
        store.check_poisoned()?;
        Q::validate_shape()?;
        store.next_epoch()?;
        let bindings = self.resolved_bindings_arc(store)?;
        store.bump_epoch();
        super::spec::iter::par_for_each_matched_chunk_mut::<Q, F>(
            store,
            &bindings,
            &self.filter,
            f,
        );
        Ok(())
    }

    /// 基于预编译的匹配原型缓存执行两阶段原子 16KB 页对齐零拷贝 GPU 内核派发（使用线程本地默认计算引擎）。
    #[cfg(feature = "gpu")]
    #[inline]
    pub fn gpu_for_each<K>(&self, store: &mut Store, kernel: K) -> Result<crate::storage::gpu::GpuDispatchStats, StateError>
    where
        Q: super::spec::QueryMutData<'static>,
        K: crate::storage::gpu::GpuKernel<'static, Query = Q>,
    {
        thread_local! {
            static DEFAULT_QUERY_METAL_ENGINE: std::cell::RefCell<crate::storage::gpu::MetalComputeEngine> =
                std::cell::RefCell::new(crate::storage::gpu::MetalComputeEngine::new());
        }
        DEFAULT_QUERY_METAL_ENGINE.with(|cell| {
            let mut engine = cell.borrow_mut();
            self.gpu_for_each_on(store, &mut engine, kernel)
        })
    }

    /// 基于预编译的匹配原型缓存在指定 [`MetalComputeEngine`](crate::storage::gpu::MetalComputeEngine) 上执行两阶段原子 16KB 页对齐零拷贝 GPU 派发。
    ///
    /// - **阶段 0**：拦截行级 CPU 谓词过滤器，校验 `Store` 归属与内核管线编译；
    /// - **阶段 1a**：对缓存命中的非空 Archetype 执行纯只读列完整性与容量溢出校验（失败时零副作用）；
    /// - **阶段 1b**：对匹配 Archetype 按需提升为 16KB 页对齐、执行可变列 COW 独占分离并生成零拷贝描述符；
    /// - **阶段 2**：统一提交硬件/回退计算派发并推进 Store 与写入列的 64 行块纪元。
    #[cfg(feature = "gpu")]
    pub fn gpu_for_each_on<K>(
        &self,
        store: &mut Store,
        engine: &mut crate::storage::gpu::MetalComputeEngine,
        kernel: K,
    ) -> Result<crate::storage::gpu::GpuDispatchStats, StateError>
    where
        Q: super::spec::QueryMutData<'static>,
        K: crate::storage::gpu::GpuKernel<'static, Query = Q>,
    {
        if F::IS_ROW_FILTERED {
            return Err(StateError::AccessDenied(
                "GPU whole-column dispatch does not support row-level CPU predicate filters; use archetype filters (.with / .without)".into(),
            ));
        }
        store.check_poisoned()?;
        Q::validate_shape()?;
        <Q as super::spec::QueryMutData<'static>>::validate_query();
        let write_epoch = store.next_epoch()?;
        let matched = self.with_matched_slice(store, |_, _| Ok(self.matched_archetypes_arc()))?;

        let desc = kernel.descriptor();
        engine.ensure_pipeline_compiled(desc)?;

        struct PreparedQueryArchDispatch {
            arch_idx: usize,
            arch_id: ArchetypeId,
            active_rows: usize,
            pages_16k: usize,
            raw_bindings: Vec<(ComponentTypeId, crate::storage::gpu::MetalNoCopyBufferDescriptor)>,
        }

        // 阶段 1a：纯只读预校验（确保任意 Archetype 异常时绝不产生半途提升、COW 解冻或纪元推进副作用）
        let mut valid_arch_indices = Vec::with_capacity(matched.len());
        for &arch_id in matched.iter() {
            let arch_idx = arch_id.0 as usize;
            let Some(archetype) = store.archetypes.get(arch_idx) else {
                continue;
            };
            let active_rows = archetype.len();
            if active_rows == 0 || !self.filter.matches_archetype(archetype) {
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
            valid_arch_indices.push(arch_idx);
        }

        // 阶段 1b：16KB 页对齐按需提升、COW 独占分离与零拷贝描述符构造
        let mut prepared = Vec::with_capacity(valid_arch_indices.len());
        for arch_idx in valid_arch_indices {
            let archetype = &mut store.archetypes[arch_idx];
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

            prepared.push(PreparedQueryArchDispatch {
                arch_idx,
                arch_id,
                active_rows,
                pages_16k: arch_pages,
                raw_bindings,
            });
        }

        // 阶段 2：推进 Store 全局纪元，执行单 CommandBuffer 批量合并派发并标记写入列的 64 行块纪元
        let scalar_args = kernel.scalar_uniform_words();
        let batch_inputs: Vec<crate::storage::gpu::GpuArchetypeBatchInput> = prepared
            .iter()
            .map(|item| crate::storage::gpu::GpuArchetypeBatchInput {
                archetype_id: item.arch_id,
                active_rows: item.active_rows,
                bindings: item.raw_bindings.clone(),
            })
            .collect();
        store.bump_epoch();
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
            let archetype = &mut store.archetypes[item.arch_idx];
            <Q as super::spec::sealed::FetchMut<'static>>::mark_written_columns(
                archetype,
                None,
                write_epoch,
            );
            stats.dispatched_archetypes += 1;
            stats.total_rows += item.active_rows;
            stats.total_pages_16k += item.pages_16k;
        }

        Ok(stats)
    }
}

#[cfg(test)]
#[path = "handle_tests.rs"]
mod tests;
