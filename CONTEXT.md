# CONTEXT.md — 域名词汇表

共享术语表，供代理、架构审查和未来会话使用。按主题分组，条目按需增补。

## 架构边界

| 术语 | 定义 |
|---|---|
| **记忆模块（Memory Module）** | 纯存储+检索引擎。所有 LLM 决策（分类、标题生成、结论提取、任务拆解）由主 agent 完成，模块只负责持久化和检索。 |
| **主 agent / agent** | 调用记忆模块的 AI agent。负责：分类对话块、判断技能缺口、搜索学习、教学写入、拆解任务、向用户确认。记忆模块不越俎代庖。 |

## 写链（Ingest Pipeline）

| 术语 | 定义 |
|---|---|
| **ingest** | 纯存储操作。`ingest(turn)` → `ingest_dialogue(turn)`，将对话轮写入 dialogue_store + learned_store。无 LLM 阶段，无 enrich 参数。 |
| **ingest_dialogue** | 底层持久化：逐字存储原文 + 嵌入 learned_store。返回 `IngestReceipt`（entry_id / deduped / weight / timestamp）。 |
| **deduped** | 入库时是否命中语义去重（embedding 相似度 ≥0.85 则 weight+0.05 合并，否则新建 weight=0.5）。 |
| **结构化写入** | `ingest_skill_ex` / `ingest_mission_ex` / `ingest_pref` / `ingest_pers` —— agent 显式调用的分类存储函数。不接受 LLM 参数，只接受内容。 |
| **teach_skill** | 教学写入。强制要求 `source_urls` 非空（防止训练数据捏造），weight=0.85 高置信入库。写入后自动从 `_pending_gaps` 中移除对应技能。 |
| **update_skill** | 直接重写技能条目（非语义合并）。用户或 agent 修正技能内容时使用。 |

## 读链（Read Pipeline）

| 术语 | 定义 |
|---|---|
| **weave** | 组装分层记忆上下文的读操作，输出 WeaveContext（tier1/tier2/nudge/gap_context 等区块）。 |
| **协议门控（Protocol Gate）** | weave() 入口的三道强制检查（状态由 ProtocolState 单一拥有）：① ClassificationRequired——上一轮未分类则拒绝；② SkillCheckRequired——mission 已分类但未查已有技能则拒绝；③ GapRequired——mission 的技能缺口未补齐则拒绝。迫使 agent 按闭环执行。 |
| **ProtocolState** | 协议状态（classify_pending / pending_gaps / mission_pending_check）的单一拥有者。MemorySystem 持引用，tools/skill_writer 经其 API 读写，不再用魔法字段跨模块裸访问。 |
| **classify** | agent 每轮必须调用的工具。`memory_classify(category, gaps=[])`——分类为 chat/skill/mission/pref/pers；mission 类可附带 gaps 列表触发补齐封锁。 |
| **tier2 / 对话单元** | 检索命中后展开为"±2 轮对话上下文"的单元，而非孤立片段。 |
| **RRF 融合** | 检索排序：BM25（权重 2.5）× semantic（0.5）× time（0.5）加权。 |
| **双证逻辑 / corroborated** | "语义 ≥0.72 且 query↔content 显著 token 重叠"的判定，用于 can_answer（校准见 KNOWN-ISSUES #1）。 |

## 主动学习闭环

| 术语 | 定义 |
|---|---|
| **classification gate** | weave() 第一道锁：`_classify_pending` 非空且 user_message 不同 → `ClassificationRequired`。强制 agent 每轮分类。 |
| **gap gate** | weave() 第二道锁：`_pending_gaps` 非空 → `GapRequired`。强制 agent 补齐 mission 所需技能后才可继续。 |
| **learning_queue** | SQLite 持久化队列。item 生命周期：open → done/skipped。agent 可手动入队长期待办，teach_skill 后自动标记 done。 |
| **check_skill** | 按 `category=skill` 检索 + `can_answer` 语义判定。返回 known（sem≥0.85 / ≥0.72+collaboration）/ partial / unknown。agent 据此决定是否学习。 |
| **mission 拆解** | 由 agent 自行分析，不再有内部 LLM 自动拆解。agent 手动调用 `check_skill` 逐技能判断缺口。 |

## 检索辅助

| 术语 | 定义 |
|---|---|
| **can_answer** | CapabilityRegistry 判定给定 query 是否可回答（sem≥0.85 直接可信；≥0.72 需 token overlap 佐证）。 |
| **semantic_score** | 嵌入向量的余弦相似度。低于 0.5 的命中被 damp 削弱。 |
| **nudge** | 高权重（≥0.85）无时间衰减的强制注入，用于关键提醒。critical（≥0.95）不受 `user_message` 相关性门控。 |

## 模块名

| 术语 | 定义 |
|---|---|
| **tree** | TreeManager，记忆树（任务/技能分类导航）。同时持有 LLM 凭据供树的 LLM 操作使用（待候选 3 移除）。 |
| **_compose** | MemorySystem 工厂 + weave() 协议门控。不再包含 LLM 阶段。 |
| **weaver** | 读链上下文组装。纯数据拼接，不含 LLM 调用。 |
| **retriever** | RRF 混合检索引擎。BM25 + semantic + time 三信号融合。 |

## 已废弃（历史参考）

| 术语 | 定义 |
|---|---|
| **enrich** | 已删除。原是 ingest() 的布尔开关，控制是否跑 LLM 阶段（tag_title + extract_structured）。2026-08-11 随"去掉所有内部 LLM"重构移除。 |
| **extract_structured** | 已删除。原是 ingester 内的分类器，调用 LLM 的 `classify_and_extract`。现由 agent 手动分类替代。 |
| **GapDetector / LearningDecider** | 已删除。原是自动知识缺口检测+LLM 决策 skip/ask/learn。现由 Protocol Gate 强制 + agent 自主决策替代。 |
| **LearningTaskManager** | 已删除。原是闭环自动学习（爬取→合成→验证）。现由 agent 显式执行（websearch → teach_skill → update_skill）。 |
| **extract_conclusion_entry** | 已删除。原是 LLM 从 assistant 回复中提取结论。现由 agent 自行记录。 |
