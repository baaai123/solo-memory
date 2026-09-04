# 酒馆模式（Tavern Mode）功能规格说明

---

## 1. 背景与目标

### 1.1 背景

当前 `memory-skill` 面向程序开发场景设计，采用**累积式语义记忆**架构，包含 `pref` / `pers` / `skill` / `mission` / `conclusion` 五类记忆。该架构假定记忆随交互历史持续累积，检索基于相关性，记忆条目只增不减。

酒馆玩法（AI 角色扮演 / 故事创作，支持多 AI 角色互相对话）引入了一种语义根本不同的记忆需求：**故事状态记忆**。故事状态是**覆盖式**的——每轮对话后角色所处的场景、心情、穿着、持有物等可能发生变化，系统只需保留最新状态，历史状态无独立检索价值。

### 1.2 目标

为 `memory-skill` 新增**酒馆模式（Tavern Mode）**，在现有开发模式之外提供一套独立的记忆存储与注入机制，满足以下需求：

1. **状态记忆**（StateStore）：每角色 8 维覆盖式快照，仅保留最新值
2. **人设记忆**（Character Store）：技能、外貌、性格等**累积式**特征（复用现有机制）
3. **关系记忆**（Relation Graph）：角色间有向社交关系，可独立演化

所有新增能力须**增量扩展**，不污染现有开发模式记忆，且不改动 `core` 层（ADR-0002：weave/ingest 为纯上下文组装，不调用内部 LLM）。

---

## 2. 模式分离设计

### 2.1 分界原则

按**角色（role）**区分记忆模式：

| 模式 | 适用对象 | 记忆机制 |
|------|----------|----------|
| **开发模式（现有）** | 普通开发 Agent（未绑定为故事角色） | 累积式语义记忆（pref/pers/skill/mission/conclusion） |
| **酒馆模式（新增）** | 绑定为「故事角色」的 Role | 状态记忆（覆盖式）+ 人设（累积式）+ 关系（有向图） |

### 2.2 隔离保证

- 两种模式**存储隔离**——酒馆模式使用独立表空间（`state_store`、`role_relations`），不触碰现有 `memories` 表
- 人设维度（skills/appearance/personality）复用现有 `character_store` 的「角色绑定记忆引用」机制，但**按 role 隔离命名空间**，不混入开发 agent 的 pers 类记忆
- `agent_bindings` 复用——多个 AI 角色 = 多个 Agent 各绑定一个 Role，状态按 Role 隔离

### 2.3 模式识别时机

- 在 `weave` 上下文组装阶段，根据当前 `agent_id` 关联的 `role` 判断是否启用酒馆模式
- 若 role 标记为 `is_tavern = true`，则注入状态快照 + 人设 + 关系摘要
- 否则走现有开发模式注入逻辑

---

## 3. 维度定义表

### 3.1 状态组（StateStore）——覆盖式，8 维

| 维度 | 字段名 | 类型 | 说明 | 示例值 |
|------|--------|------|------|--------|
| 心情 | `mood` | TEXT | 当前情绪状态 | `"焦虑"`, `"愉悦"`, `"平静"` |
| 身体需求 | `need` | TEXT | 生理/本能需求 | `"饥饿"`, `"疲惫"`, `"舒适"` |
| 健康/受伤 | `health` | TEXT | 身体状况 | `"健康"`, `"轻伤"`, `"昏迷"` |
| 穿着 | `clothing` | TEXT | 当前衣物/装备 | `"破旧斗篷"`, `"皮甲"` |
| 持有物 | `item` | TEXT | 当前携带的关键物品 | `"生锈的剑"`, `"空钱包"` |
| 动作 | `action` | TEXT | 当前正在做的动作 | `"奔跑"`, `"低声细语"`, `"蹲下"` |
| 场景/位置 | `scene` | TEXT | 所处地点/场景 | `"酒馆角落"`, `"黑暗森林小径"` |
| 天气/时间 | `weather` | TEXT | 环境时间/气候 | `"午夜暴雨"`, `"黄昏薄雾"` |

> 注：所有维度均为自由文本，由 AI 提取时生成，不设枚举约束。

### 3.2 人设组（Character Store）——累积式，4 维

| 维度 | 字段名 | 复用机制 | 说明 |
|------|--------|----------|------|
| 技能 | `skills` | character_store（记忆引用） | 角色具备的能力/技艺，累积追加 |
| 外貌 | `appearance` | character_store（记忆引用） | 角色外观特征，可修正更新 |
| 性格 | `personality` | character_store（记忆引用） | 角色性格特质，可修正更新 |
| 社交关系 | `relation` | **新增 role_relations 表** | 有向关系图，独立存储 |

> 人设组中 skills/appearance/personality 保留「累积式」语义——因为角色设定通常是长期稳定的，修正时应视为「更新描述」而非「覆盖」。

---

## 4. 存储 Schema（SQLite）

### 4.1 `state_store` 表——8 维覆盖式状态快照

```sql
CREATE TABLE state_store (
    role_id       TEXT NOT NULL,          -- 角色唯一标识
    mood          TEXT,                   -- 心情
    need          TEXT,                   -- 身体需求
    health        TEXT,                   -- 健康/受伤
    clothing      TEXT,                   -- 穿着
    item          TEXT,                   -- 持有物
    action        TEXT,                   -- 动作
    scene         TEXT,                   -- 场景/位置
    weather       TEXT,                   -- 天气/时间
    updated_at    INTEGER NOT NULL,       -- Unix 时间戳（毫秒）
    PRIMARY KEY (role_id)
);
```

**语义**：
- 每个 `role_id` 仅存一行，代表该角色最新状态快照
- 更新时执行 `INSERT OR REPLACE`，每维独立覆盖
- `updated_at` 记录最后一次状态变更时间（任意维度变化即更新）

### 4.2 人设维度（复用现有 character_store）

现有 `character_store` 已实现「角色 = 全局记忆的引用集合」，三张表：`roles`（角色元数据）/ `role_memories`（role↔memory 引用，复合主键防重）/ `agent_bindings`（agent→role 绑定）。

人设维度挂载方式：`skills` / `appearance` / `personality` **无需新表**——通过 `role_memories` 把角色绑定到一批长期记忆条目（本质是 pref 类记忆的按角色分组）。仅 `relation` 需新增 `role_relations` 表（见 4.3）。

> 人设条目与开发模式的 pref/pers 记忆同源，但按 role 过滤，逻辑隔离。

### 4.3 `role_relations` 表——有向关系图

```sql
CREATE TABLE role_relations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    from_role_id  TEXT NOT NULL,          -- 关系主体（A）
    to_role_id    TEXT NOT NULL,          -- 关系客体（B）
    relation_type TEXT NOT NULL,          -- 关系类型，如 "盟友", "敌对", "爱慕", "畏惧"
    strength      INTEGER DEFAULT 50,     -- 强度 0-100
    updated_at    INTEGER NOT NULL,       -- Unix 时间戳（毫秒）
    UNIQUE(from_role_id, to_role_id)
);
```

**语义**：
- 有向边：`from_role_id → to_role_id`，表示 A 对 B 的关系
- A 对 B 的看法 ≠ B 对 A，方向独立存储
- `strength` 为 0-100 整数，允许 AI 在更新时调整
- 更新时 `INSERT OR REPLACE` 覆盖同向边

---

## 5. 更新链路（AI 写入）

### 5.1 整体流程

```
每轮对话结束
    ↓
[触发条件] 当前 agent 为故事角色（is_tavern = true）
    ↓
[Step 1] 对话上下文 + 当前状态快照 → 封装为提取请求
    ↓
[Step 2] 网页端（ds-web 桥，子代理模式）调用 LLM 提取状态变化
    ↓
[Step 3] 提取结果（8 维新值 + relation 更新建议）返回
    ↓
[Step 4] agent 审查纠错（校验维度合理性、关系方向合法性）
    ↓
[Step 5] 写入存储：
         - state_store: INSERT OR REPLACE（覆盖）
         - role_relations: INSERT OR REPLACE（覆盖有向边）
         - character_store: 追加/修正人设（累积式）
    ↓
完成
```

### 5.2 提取提示词模板（概要）

系统向子代理提供以下上下文用于提取：

```
基于以下对话内容，提取角色 [角色名] 的状态变化：

【当前状态】
- mood: {当前值}
- need: {当前值}
- ...（8 维全量）

【对话内容】
{对话日志}

【任务】
1. 判断每一维是否发生变化，若变化则输出新值，否则输出 null
2. 若涉及与其他角色的关系变化，输出关系更新建议（方向 + 类型 + 强度）
3. 仅输出结构化 JSON，无额外解释
```

### 5.3 审查纠错（Agent 侧）

- 校验维度值长度/格式（防止注入或异常值）
- 校验 `relation_type` 白名单（可配置扩展）
- 校验 `strength` 落在 0-100 范围
- 若提取失败或不完整，保留原状态不变，记录审计日志

---

## 6. 注入链路（weave 时组装）

### 6.1 注入位置

复用现有 `scene_summary` 注入位（`weave` 上下文组装阶段），来源从「外部临时参数」切换为「持久化状态」。

### 6.2 注入内容

对于启用酒馆模式的 Role，`weave` 时向上下文注入以下三块：

1. **当前状态摘要**（来自 `state_store`）：
   ```
   [当前状态]
   心情：焦虑 | 身体需求：饥饿 | 健康：轻伤 | 穿着：破旧斗篷
   持有物：生锈的剑 | 动作：倚靠墙角 | 场景：酒馆角落 | 天气：午夜暴雨
   ```

2. **人设摘要**（来自 `character_store` 累积条目）：
   ```
   [角色设定]
   外貌：... | 性格：... | 技能：...
   ```

3. **关系摘要**（来自 `role_relations`，仅注入与当前对话相关的方向）：
   ```
   [社交关系]
   对 艾琳：盟友（70） | 对 黑剑：畏惧（85）
   ```

### 6.3 注入条件

- 若 `state_store` 中无该 `role_id` 记录，则注入空状态提示（首次对话）
- 关系摘要仅返回与当前 Role 相关的出边（`from_role_id = current_role_id`），入边不注入（避免单向认知不对称）

---

## 7. 多角色架构

### 7.1 架构模型

```
┌─────────────────────────────────────────────────────────────┐
│                        weaver                               │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐      │
│  │  Agent A    │   │  Agent B    │   │  Agent C    │      │
│  │ (role: 艾琳)│   │ (role: 黑剑)│   │ (role: 旅人)│      │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘      │
│         │                 │                 │              │
│         ▼                 ▼                 ▼              │
│  ┌─────────────────────────────────────────────────┐       │
│  │              agent_bindings                    │       │
│  │      (agent_id ↔ role_id 映射)                │       │
│  └─────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   酒馆模式存储层                            │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  │
│  │ state_store   │  │character_store│  │role_relations │  │
│  │ (按 role 隔离) │  │(按 role 隔离) │  │(有向图)       │  │
│  └───────────────┘  └───────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 关键原则

- 每个 AI 角色 = 一个 Agent，绑定一个 Role（复用 `agent_bindings`）
- 多个 Agent 可共存在同一 weaver 实例中，各自独立调用
- **状态按 Role 隔离**，不同角色之间的状态互不干扰
- **关系是角色间的有向边**，通过 `role_relations` 全局维护，不归属于任何单一 Agent

---

## 8. 实现单元拆分

| Unit | 名称 | 职责 | 依赖 | 产出 |
|------|------|------|------|------|
| **Unit 1** | StateStore 存储层 | 实现 `state_store` 表的 CRUD，提供 `get_state(role_id)` / `update_state(role_id, dims)` 接口，覆盖式更新 | SQLite 现有连接 | 存储接口 + 表迁移脚本 |
| **Unit 2** | Relation 有向关系图 | 实现 `role_relations` 表 CRUD，提供 `get_relations(role_id)`（出边）/ `upsert_relation(from, to, type, strength)` / `delete_relation(from, to)` | SQLite 现有连接 | 关系图接口 + 表迁移脚本 |
| **Unit 3** | weave 注入 `[当前状态]` | 修改 weave 上下文组装逻辑，当目标 role 为酒馆模式时，从 Unit 1 + Unit 2 读取数据，填充 `scene_summary` 注入位 | Unit 1, Unit 2 | 上下文注入增强 |
| **Unit 4** | AI 提取写入链路 | 实现「对话后触发 → 子代理提取 → Agent 审查 → 写入存储」的完整链路，包含提取提示词模板、审查纠错逻辑、审计日志 | Unit 1, Unit 2, ds-web 桥 | 自动化状态更新流水线 |
| **Unit 5**（可选） | 模式识别与路由 | 在 Agent 初始化时识别 `role.is_tavern` 标志，路由至酒馆模式或开发模式 | 现有 Agent 配置 | 模式切换逻辑 |

---

## 9. 边界与约束

### 9.1 不修改 core 层

- **ADR-0002** 约束：`weave` / `ingest` 是纯上下文组装，不调用内部 LLM
- 状态提取的 LLM 调用发生在 `ds-web 桥（子代理模式）`，非 core 内部
- core 只负责「组装已就绪的数据」，不主动触发存储写入

### 9.2 不污染开发模式记忆

- 酒馆模式不使用现有 `memories` 表（pref/pers/skill/mission/conclusion）
- `character_store` 中人设条目与开发模式 `pers` 条目**逻辑隔离**（按 role 过滤，查询时互不交叉）
- 若同一 role 同时存在开发模式绑定和酒馆模式绑定，**酒馆模式优先**，开发模式记忆不被加载

### 9.3 数据一致性

- `state_store` 每维独立覆盖，不存在部分更新导致的不一致（原子行级替换）
- `role_relations` 有向边唯一约束保证不产生重复关系

### 9.4 性能

- `state_store` 和 `role_relations` 按 `role_id` 索引，查询 O(1)
- 每轮对话至多触发 1 次状态提取 LLM 调用（子代理模式）

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| AI 提取状态变化不稳定（漏提/误提） | 状态漂移，故事连贯性下降 | Agent 审查纠错层兜底；保留 `updated_at` 审计；支持手动修正接口 |
| 关系更新与角色状态不同步 | 关系图与对话事实矛盾 | 提取时同时提供对话上下文和当前关系快照；审查层校验方向合法性 |
| 多角色并发写入同一状态 | 状态覆盖竞争，丢失中间变化 | 目前单 weaver 单线程调度，无并发冲突；未来分布式场景加乐观锁 |
| `scene_summary` 注入位长度膨胀 | 超出上下文窗口 | 状态摘要压缩为单行文本，人设/关系按需截断（如关系只取前 10 条） |
| 模式识别逻辑遗漏导致开发模式被污染 | 开发 Agent 上下文包含无关状态 | 在 weaver 入口显式判断 `role.is_tavern`，默认 false（保守）；增加集成测试覆盖 |
| 子代理提取调用失败 | 状态停留在旧值，故事中断 | 失败时静默保留原状态，记录 error 日志；不阻断对话主流程 |

---

## 附录 A：接口定义（概要）

### A.1 StateStore 接口

```python
def get_state(role_id: str) -> dict | None:
    """返回 {mood, need, health, clothing, item, action, scene, weather, updated_at} 或 None"""

def update_state(role_id: str, dims: dict, timestamp: int) -> None:
    """dims 中包含任意维度的新值，缺失维度保持不变（不覆盖为 NULL）"""
```

### A.2 Relation 接口

```python
def get_outgoing_relations(role_id: str) -> list[dict]:
    """返回该角色的所有出边 [(to_role_id, relation_type, strength)]"""

def upsert_relation(from_role: str, to_role: str, rel_type: str, strength: int, timestamp: int) -> None:
    """插入或覆盖有向关系"""

def delete_relation(from_role: str, to_role: str) -> None:
    """删除指定有向边"""
```