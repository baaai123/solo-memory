---
title: feat: 角色记忆后端改造（Character Store）
type: feat
status: active
date: 2026-09-01
origin: /home/pc/projects/memory/memory-ui/docs/brainstorms/2026-09-01-character-memory-requirements.md
---

# 角色记忆后端改造（Character Store）

**Target repo:** memory for solo（memory-skill 后端）

## Overview

为 memory-skill 后端新增"角色"能力：角色是全局记忆的**引用集合**（不复制数据），存于独立 SQLite 表；角色会话生成的记忆自动双写（全局 + 角色引用 + 来源标记）；agent 绑定角色后 weave/检索只返回该角色的引用记忆（全局其余不可见）；删除全局记忆时级联清理所有角色引用。

## Problem Frame

当前记忆是单层结构，weave 注入全部记忆导致上下文污染。后端需要新增角色引用存储、双写钩子、检索白名单过滤、agent 绑定查询，才能支撑前端角色 UI 与 agent 角色隔离。本规划仅覆盖后端（memory_skill 包内），前端 memory-ui 配套另行规划。

## Requirements Trace

- R1-R4. 角色 CRUD（创建/查看/编辑/删除角色引用集合）
- R5-R6. 手动加入/移除角色引用
- R7. 角色会话记忆自动双写 + 来源角色标记
- R8. 删除全局记忆级联清理角色引用
- R9-R11. agent 绑定角色 + weave 白名单过滤 + 未绑定退化现状

## Scope Boundaries

- 仅改 `memory_skill/` 后端包与配套测试；不改前端 memory-ui、不改 dsh/opencode 插件仓库
- 角色引用存独立 SQLite 表（用户已确认），不塞进 MemoryEntry.metadata
- 不做角色间继承/合并、多用户、公网化
- 仓库 main 上已有的未提交 WIP（metadata 合并修复、get_entry、tools.py 扩充）**不动**，角色改造叠加在其上

### Deferred to Separate Tasks

- 前端 memory-ui 角色 UI（角色 CRUD 页、全局/角色双视图、接入页绑定）——后续独立规划
- 后端 API 端点供前端调用（角色列表/详情/创建/编辑/删除/绑定）——与前端配套实现

## Context & Research

### Relevant Code and Patterns

- `memory_skill/learned_store.py`：`LearnedStore`（ChromaDB 向量 + SQLite FTS5）。`delete(entry_id)` 是级联清理的挂点；`update()` 已有 `_merge_update_metadata` 处理 metadata 路由（未提交 WIP）
- `memory_skill/retriever.py`：`Retriever.retrieve()` 是 3 信号 RRF 融合（语义 ChromaDB + BM25 FTS5 + 时间衰减），`filters` 参数目前只到语义 leg，BM25 leg 手动应用 category 过滤——角色 ID 白名单过滤在此函数内实现（语义 leg 传入 + BM25 leg 手动过滤）
- `memory_skill/ingestor.py`：`ingest_dialogue()` 已支持 `extra_metadata` 参数（可注入 `source_role`）；双写钩子加在 learned_store 写入后
- `memory_skill/weaver.py`：`weave()` → `_apply_standard_blocks()` 中所有 `_build_*`（tier2/skill/mission/pref/pers/conclusion）都调 `stores.retriever.retrieve()`——统一传入角色 ID 集合即可全链路过滤
- `memory_skill/tools.py`：`DISPATCH` 表 + `handle_weave`/`handle_search` 是 MCP 工具入口；`MemorySkill` config 有 `agent_name`（来自 MEMORY_SKILL_AGENT）
- `memory_skill/_compose.py`：`MemorySystem` 组装所有 store；新 store（CharacterStore）需挂到 `_build_system` 并在 WeaverStores 中传递
- `memory_skill/contracts.py`：`MemorySkillConfig` dataclass，加字段需兼容默认值
- 测试模式：`tests/test_fast.py`（语义+BM25 检索）、`tests/test_integration.py`、`tests/fakes.py`（fake stores 供组合测试）

### Institutional Learnings

- ChromaDB metadata 读写不一致的历史坑（update 后 search 读不到 ext_metadata_json）已有修复（未提交 WIP）——角色信息若存 metadata 会踩同一坑，故独立表更安全
- sqlite 跨线程已知缺陷（记忆模块 KNOWN-ISSUES）——CharacterStore 用 SQLite 时需注意线程模型，参考现有 DialogueStore 的连接处理

## Key Technical Decisions

- **独立 SQLite 表存角色引用**：`roles`（角色元数据）+ `role_memories`（role_id→memory_id 联合主键）+ `agent_bindings`（agent_name→role_id）。关系清晰、级联删除天然支持（FOREIGN KEY ON DELETE CASCADE 或显式清理）；避开 ChromaDB metadata 读写不一致坑
- **白名单过滤在 retriever 内**：`retrieve()` 增加 `role_memory_ids: set[str] | None` 参数——传了则语义 leg 与 BM25 leg 候选集按 ID 白名单过滤（全局不可见语义）；未传则现状不变（未绑定退化）
- **双写钩子在 ingestor**：`ingest_dialogue()` 写入 learned_store 后，若 config 有当前绑定角色 → 向 role_memories 插入引用；`source_role` 经 extra_metadata 写入
- **角色传递经 agent_name**：`handle_weave`/`handle_search` 用 config.agent_name → 查 agent_bindings → 得 role_id → 取引用集合 → 传入 retriever
- **内存缓存引用集合**：每次 weave 全量查 SQLite 引用集合（规模 1000+ 条可控，一次索引查询）；不做复杂缓存，避免失效问题

## Implementation Units

- [ ] **Unit 1: CharacterStore（角色引用存储）**

**Goal:** 新增 `memory_skill/character_store.py`，实现角色 CRUD、引用增删、agent 绑定、按 memory_id 查询引用角色的完整存储层

**Requirements:** R1-R6, R8, R9

**Dependencies:** None（新增独立模块）

**Files:**
- Create: `memory_skill/character_store.py`
- Modify: `memory_skill/contracts.py`（若需 CharacterRole dataclass）
- Test: `tests/test_character_store.py`（新建）

**Approach:**
- `CharacterStore` 类，构造接收 db_path（复用记忆库 SQLite 文件或独立文件，规划时定）
- 三张表：`roles`（id, name, description, created_at, updated_at）、`role_memories`（role_id, memory_id, added_at, 联合主键）、`agent_bindings`（agent_name PK, role_id, updated_at）
- 方法：`create_role / list_roles / get_role / update_role / delete_role`、`add_memory / remove_memory / list_memories(role_id) / list_role_ids(memory_id)`、`bind_agent / get_agent_role / unbind_agent`
- `delete_role` 级联删 role_memories 与 agent_bindings；`remove_all_memory(memory_id)` 供全局记忆删除级联用
- 线程安全：参考 DialogueStore 的连接管理模式（每操作独立连接或线程本地连接）

**Patterns to follow:**
- `memory_skill/dialogue_store.py` 的 SQLite 连接与 FTS5 用法
- `tests/fakes.py` 提供 FakeCharacterStore 供组合测试

**Test scenarios:**
- Happy path: 创建角色 → 列表可见 → 详情返回引用记忆列表
- Happy path: add_memory 后 list_memories 返回该 memory_id；remove_memory 后消失
- Happy path: bind_agent → get_agent_role 返回正确角色；unbind_agent 后返回 None
- Edge case: 重复 add_memory（幂等，联合主键防重）；删除不存在的 memory 引用不报错
- Edge case: delete_role 后该角色的全部引用与绑定消失，其他角色不受影响
- Error path: 空角色名/超长描述拒绝；删除不存在角色返回明确错误

**Verification:**
- pytest tests/test_character_store.py 全绿
- 手动 SQL 检查：create→add→delete 后三表状态一致，无孤儿行

- [ ] **Unit 2: 删除级联（learned_store 挂点）**

**Goal:** 全局记忆删除时，级联清理所有角色中的引用

**Requirements:** R8

**Dependencies:** Unit 1（CharacterStore 的 remove_all_memory）

**Files:**
- Modify: `memory_skill/learned_store.py`
- Modify: `memory_skill/_compose.py`（将 CharacterStore 挂到 MemorySystem）
- Test: `tests/test_character_store.py` 或 `tests/test_integration.py`

**Approach:**
- `LearnedStore.delete()` 不直接依赖 CharacterStore（保持存储层解耦）；由上层 `MemorySystem.delete_entry()`（若存在）或 tools 层在删除后调用 `character_store.remove_all_memory(entry_id)`
- 若无现成 MemorySystem.delete_entry，新增薄封装：learned_store.delete + character_store.remove_all_memory
- CharacterStore 挂到 `_build_system`（_compose.py），MemorySystem 暴露 `delete_entry` 与 `character` 属性

**Patterns to follow:**
- `MemorySystem` 现有薄封装模式（ingest/retrieve/boost_weight 均为 delegate）
- `_compose.py` `_build_system` 的 store 组装

**Test scenarios:**
- Integration: 创建角色 + add 两条记忆 → 删除其中一条全局记忆 → 角色引用中该条消失、另一条保留
- Integration: 删除被 2 个角色引用的记忆 → 两个角色引用均移除
- Edge case: 删除未被任何角色引用的记忆 → 角色数据无变化

**Verification:**
- pytest 集成测试通过；手动验证删除后 role_memories 无孤儿 memory_id

- [ ] **Unit 3: 检索白名单过滤（retriever）**

**Goal:** `Retriever.retrieve()` 支持角色引用 ID 白名单过滤，实现"绑定角色后全局不可见"

**Requirements:** R10, R11

**Dependencies:** Unit 1

**Files:**
- Modify: `memory_skill/retriever.py`
- Test: `tests/test_fast.py` 或 `tests/test_character_store.py`

**Approach:**
- `retrieve()` 增加 `role_memory_ids: set[str] | None = None` 参数
- 语义 leg：`learned_store.search(filters=...)` 后按白名单过滤结果（语义 leg 的 ChromaDB filters 无法直接表达 ID 集合，故查后过滤）
- BM25 leg：现有 category 过滤逻辑旁增加白名单过滤
- 白名单过滤在候选集合并处统一应用（语义 + BM25 + 时间衰减前），保证 RRF 排序只对可见记忆生效
- 未传 role_memory_ids（None）→ 完全现状，零行为变化（R11）

**Patterns to follow:**
- `retrieve()` 现有 BM25 leg 的 category 手动过滤模式（同位置加白名单过滤）
- `Retriever.retrieve` 的参数扩展需向后兼容（默认 None）

**Test scenarios:**
- Happy path: 传 role_memory_ids={id1,id2}，检索结果只含这两条（即使查询相关性与全局其他记忆更高）
- Happy path: 未传 role_memory_ids → 返回全局全部相关记忆（现状行为）
- Edge case: role_memory_ids 为空集合 → 返回空 envelope（角色无记忆时 weave 无注入）
- Edge case: 白名单中的记忆在语义 leg 与 BM25 leg 均出现 → 正常去重合并
- Integration: 角色 A 引用 id1，角色 B 引用 id2；分别传不同白名单 → 结果互不泄漏

**Verification:**
- pytest 通过；手工用现有 memory search CLI 验证白名单过滤生效

- [ ] **Unit 4: 双写钩子 + weave 角色传递（ingestor/weaver/tools）**

**Goal:** 角色会话记忆自动双写（含来源标记），weave 时经 agent 绑定查询角色引用并过滤注入

**Requirements:** R7, R9, R10, R11

**Dependencies:** Unit 1, Unit 2, Unit 3

**Files:**
- Modify: `memory_skill/ingestor.py`
- Modify: `memory_skill/weaver.py`
- Modify: `memory_skill/tools.py`
- Modify: `memory_skill/_compose.py`（WeaverStores 传 CharacterStore）
- Modify: `memory_skill/contracts.py`（config 增加角色相关字段，如 `character_role`）
- Test: `tests/test_integration.py`

**Approach:**
- Ingestor：`ingest_dialogue()` 在 learned_store 写入后，若 config.character_role 有值 → `character_store.add_memory(role, entry_id)`；`extra_metadata` 注入 `source_role`
- Weaver：`WeaverStores` 增加 `character_store` 字段；`weave()` 解析当前绑定角色 → 取引用集合 → 传给 `_apply_standard_blocks` 中各 `_build_*` 的 retriever 调用
- Tools：`handle_weave`/`handle_search` 用 config.agent_name 查 agent_bindings 得 role_id → 取引用集合 → 传 retriever
- Config：`MemorySkillConfig` 增加 `character_role: str | None = None`（默认 None → 未绑定退化现状）与可选 `character_db_path`
- MemorySystem.weave 组装角色上下文传入

**Patterns to follow:**
- `WeaverStores` dataclass 现有字段传递模式
- `_build_*` 函数统一经 `stores.retriever` 调用的模式（改一处签名即可全链路生效）

**Test scenarios:**
- Happy path: 绑定角色 agent ingest 一条记忆 → 全局库可见 + 角色引用 +1 + metadata 含 source_role
- Happy path: 绑定角色 agent weave → 返回的记忆全部在角色引用集合内
- Happy path: 未绑定 agent ingest/weave → 与现状完全一致（无角色钩子触发）
- Edge case: 角色被删除后，绑定该角色的 agent weave 退化为无角色过滤（或报错并提示重新绑定，规划时定）
- Integration: 完整链路——创建角色→add 记忆→绑定 agent→weave 只返回引用记忆

**Verification:**
- pytest tests/test_integration.py 通过
- 用 memory CLI/MCP 手动验证：绑定角色后 weave 输出不含角色外记忆

## System-Wide Impact

- **Interaction graph:** `MemorySystem` 新增 `character` store 与 `delete_entry`；`WeaverStores` 新增字段；所有 `_build_*` 函数签名变化（统一传角色白名单）
- **Error propagation:** 角色引用查询失败（SQLite 错误）→ weave 降级为无角色过滤并记录日志，不阻断会话
- **State lifecycle risks:** 删除全局记忆必须与角色级联清理同事务或严格顺序执行，避免孤儿引用；角色删除后绑定需清理
- **API surface parity:** MCP 工具增加角色相关工具（memory_character_*，规划阶段定）；现有 21 个工具行为不变
- **Unchanged invariants:** 未绑定角色的所有现有行为（weave/ingest/search）完全不变；现有数据（1056 条）不迁移

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| SQLite 跨线程访问（KNOWN-ISSUES） | CharacterStore 复用 DialogueStore 的连接管理模式；每次操作独立连接 |
| 白名单过滤影响 RRF 排序质量 | 过滤在候选集合并前统一应用；角色引用量大时排序仍有效 |
| 角色删除/绑定失效导致 weave 异常 | 角色查询失败降级为无过滤 + 日志；tools 层捕获异常 |
| 与 main 上未提交 WIP 冲突 | 角色改动只新增模块与增量修改，不重构 WIP 涉及逻辑 |

## Documentation / Operational Notes

- `docs/INTEGRATION.md` 需补充角色绑定配置说明（MEMORY_SKILL_AGENT → 角色映射）——与前端阶段一起做
- 角色 SQLite 表建表语句与升级路径（已有数据库无角色表 → 首次运行自动建表）

## Sources & References

- **Origin document:** [character-memory-requirements.md](/home/pc/projects/memory/memory-ui/docs/brainstorms/2026-09-01-character-memory-requirements.md)
- Related code: `memory_skill/retriever.py`、`memory_skill/ingestor.py`、`memory_skill/weaver.py`、`memory_skill/_compose.py`、`memory_skill/tools.py`
- 用户已确认：角色引用存独立 SQLite 表
