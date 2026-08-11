# ADR-0002 — 移除所有内部 LLM：记忆模块退化为纯存储+检索引擎

**Status**: Accepted · **Date**: 2026-08-11
**Supersedes**: [ADR-0001](./0001-ingest-pipeline.md)
**Decision driver**: 主动学习联动重构

## 背景

ADR-0001 设计的 ingest 管道包含多达 4 个内部 LLM 阶段：

- `tag_title` — 为 user 消息生成标题
- `extract_structured` — 分类器 (`classify_and_extract`)，判断 skill/mission/pref/pers
- `extract_conclusion_entry` — 从 assistant 回复中提取结论
- `gap_detector` + `LearningDecider` — 知识缺口检测 + LLM 决策 skip/ask/learn

此外 `MissionDecomposer` 内部 LLM 自动拆解任务并对比 skill。

这些"副 agent"替主 agent 做决策，导致三个断点：
1. **触发不可靠** — MCP 路径 `enrich=False` 跳过所有 LLM 阶段，gap 检测从未真正运行
2. **自动执行无调度器** — gap 检测到后无自动学习回路
3. **agent 不自觉** — 依赖 agent 手动调用 `memory_learn`，实际从不被调用

更深层的问题：记忆模块越界了。它在做决策，而不是提供服务。

## 决策

**移除记忆模块内所有 LLM 调用。** 模块退化为纯存储（ChromaDB + SQLite）+ 检索（RRF 融合）+ weave 上下文组装。

| 删除 | 替代 |
|---|---|
| `structured_extractor.py` (classify_and_extract) | agent 手动调用 `memory_classify` |
| `mission_decomposer.py` (LLM 拆解) | agent 自行推理拆解 |
| `memory_extract.extract_conclusion_entry` | agent 显式记录结论 |
| `memory_extract.tag_title` / `generate_title` | agent 提供标题 |
| `learning_task.py` (LearningTaskManager) | agent 执行 websearch → teach_skill |
| `learning_decider.py` (LLM 决策 skip/ask/learn) | 删除，不替代 |
| `gap_detector.py` / `gap_store.py` | `learning_queue.py`（agent 驱动队列） |

**新增协议层（Protocol Gate）强制 agent 行为：**

1. **classification gate** — 每轮 weave 后 `_classify_pending` 追踪；下一轮未分类 → `ClassificationRequired` 拒绝服务
2. **gap gate** — `memory_classify(mission, gaps=[...])` 设锁；所有 gap 被 `teach_skill` 移除前 → `GapRequired` 拒绝服务
3. **source gate** — `teach_skill` 强制 `source_urls` 非空

**新增 agent 工具：**

- `memory_check_skill` — 按 category=skill 检索 + can_answer 判定
- `memory_teach_skill` — 教学写入（weight 0.85，强制 source_urls）
- `memory_update_skill` — 直接重写（用户/agent 修正）
- `memory_classify` — 分类 + 设 gap 锁
- `memory_learning_queue` — 学习队列查看

## 理由

- **一个脑子** — 所有 LLM 决策由主 agent 完成，记忆模块只服务不决策
- **强制闭环** — Protocol Gate 消除 agent 不自觉的问题（旧系统靠自觉，实际从不执行）
- **agent 有更好的搜索能力** — 主 agent 有 websearch 工具，比内部爬虫资源多
- **用户确认环节** — agent 呈现学习结果给用户确认后再写入，避免"假学习"

## 不做的事（明确排除）

- 不删除 `tree` 模块（TreeManager）——树的管理操作需要 LLM，但树的 LLM 调用是树自己的事，不是记忆模块的
- 不修改检索链（retriever/weaver）的排序逻辑
- 不删除 `capability_registry` —— `can_answer` 是纯计算，不调 LLM

## 涉及文件

- `_compose.py` — ingest() 退化为 1 行 `ingest_dialogue`；weave() 加协议门控；工厂删除 GapDetector 接线
- `mcp_tools.py` — TOOLS 7→11→10，删 memory_learn/memory_mission_decompose，加 5 个新工具
- `weaver.py` — gap_context 改为 agent 指令区 + 分类指令
- `memory_extract.py` — 删除 6 个 LLM 函数，保留 4 个纯存储函数
- `learning_queue.py` — 新增（替代 gap_store）
- `skill_registry.py` / `skill_writer.py` — 新增（agent 学习闭环）

**删除：** `structured_extractor.py`, `mission_decomposer.py`, `learning_task.py`, `learning_decider.py`, `gap_detector.py`, `gap_store.py`

## 后果

- 正面：1 个决策中心（agent）；强制闭环消除漏分类/漏补齐；记忆模块职责清晰
- 负面：agent 每轮必须调 classify（多一次 MCP 往返）；无自动分类兜底，agent 忘了就触发封锁
- 未来审查不应重新建议"在记忆模块内加 LLM 阶段"——已决策移除
