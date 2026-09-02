---
title: feat: 待办约束体系（Default 归档 + 学习队列硬门）
type: feat
status: active
date: 2026-09-02
---

# 待办约束体系（Default 归档 + 学习队列硬门）

**Target repo:** memory for solo（memory-skill 后端 + opencode-auto-memory 插件层）

## Overview

解决两个同源问题——①default 分类积累 988 条（agent 不归档）②learning_queue 积累 169 条 open（agent 不处理待办）。根因一致：**记忆模块把认知决策交给主 agent（08-11 架构），但"何时必须决策"没有强制约束**，weave 里的 [待学习] 区只是文字提醒（非硬门）。本方案将两者统一为**待办约束体系**：复用现有 ProtocolGate 硬门机制（classify_pending 模式），为归档与队列待办加真硬门 + 提供 agent 操作工具。

## Problem Frame

- **default 988 条**：通用 ingest 不做智能分类（架构决策），依赖 agent 用专用工具；agent 从不主动分类 → 内容堆 default。
- **learning_queue 169 open**：每轮对话产生的新 mission/gap 自动入队（每日 10-40 条），但 weave 的 [待学习] 区只显示前 5 条且无强制 → 处理速度远小于入队速度。
- **约束缺口**：ProtocolGate 已对 classify/mission/gaps 三类 raise 阻断，但 **learning_queue 的 [待学习] 区只是文字声明"不执行则拒绝"——实际不阻断**；default 归档则完全没有约束。
- 用户要求："定期硬性让 agent 使用，不然又会忽略，和 conclusion 一样"。

## Requirements Trace

（需求来自对话确认，无独立需求文档）
- R1. 新增 `memory_review_default` 工具：列出 default 分类候选（含内容/时间/ID，分页）
- R2. 新增 `memory_reclassify(entry_id, category)` 工具：改条目 category + 补全缺失 title（不重嵌）
- R3. weave 注入 default 积累状态（"default 已积累 N 条"）
- R4. default 归档硬门：按轮数激活（每 N 轮 weave 强制 agent 用 review_default 响应）
- R5. 队列待办硬门：weave 的 [待学习]/[待拆解] 区从文字提醒改为真硬门（显示 open 计数 + 顶部 N 条，agent 必须 mark done/响应才继续）
- R6. mission 自动闭环：完成的 mission 自动标 done（减少 open 堆积）
- R7. 入队闸门（可选）：降低误入队灵敏度
- R8. 硬门为"必须响应"而非"必须全做完"（agent 可 mark done/跳过后继续，防误判）

## Scope Boundaries

- 只改 memory for solo（memory_skill/ + opencode-auto-memory/index.js）与测试
- 不改 memory-ui（前端归档页另行规划）；不改 dsh-memory-protocol（Python 侧改动自动生效）
- 不动 default 988 条存量的一次性清理（本方案只建立机制，存量靠后续硬门驱动 agent 逐步消化）
- 不做 LLM 自动分类（违反 08-11 零 LLM 架构）
- 不改现有三类硬门（classify/mission/gaps）行为

### Deferred to Separate Tasks

- memory-ui 归档页面（可视化 review/reclassify）——后续独立规划
- default 988 条存量的一次性 reclassify 脚本运行——机制落地后由 agent/用户驱动

## Context & Research

### Relevant Code and Patterns

- `memory_skill/protocol_gate.py`：`ProtocolGate.check()` 在 weave 前检查三类状态（`classify_pending`/`mission_pending_check`/`pending_gaps`）→ raise 阻断。**新硬门模式照此扩展**
- `memory_skill/protocol_state.py`：`ProtocolState` dataclass（classify_pending/pending_gaps/mission_pending_check 字段 + mark_weave 方法）——**加新状态位**
- `memory_skill/_compose.py`：`MemorySystem.weave` 调 `ProtocolGate(self).check()`；`ingest` 是纯存储（分类是 agent 责任）
- `memory_skill/weaver.py`：`_build_gap_context`（643 行，[待学习]/[待拆解] 区，只显示前 5 条）、`_build_pending_context`（663 行，distill 候选）
- `memory_skill/tools.py`：`_enqueue_if_learning`（204 行，classify 后自动入队）、`handle_learning_queue`（229）、`handle_learning_mark`（380）——**新工具照此模式**
- `memory_skill/mission.py`：`MissionStore`，148 行自动 enqueue mission——**自动闭环挂点**
- `memory_skill/learning_queue.py`：`LearningQueue.enqueue/open_items/mark/count_open`
- `opencode-auto-memory/index.js`：`tool.execute.before/after` hook（nudge 层，非硬门）
- **硬门核心在 Python 侧**（ProtocolGate），opencode/dsh 都走 weave → 加 Python 侧一处即可全 agent 生效

### Institutional Learnings

- 08-11 重构：零 LLM 核心路径——分类/学习决定权在主 agent，系统不做自动判断（本方案遵守：工具只提供数据可见性，决策归 agent）
- weave 硬门（b47f962）已验证：`classify_pending` raise 阻断有效且不误伤（agent 分类后清除）
- `mission_pending_check`（08-11 后加）：classify(mission) 后强制 check_skill——**同模式复用到归档/队列**

## Key Technical Decisions

- **硬门加 ProtocolGate + ProtocolState（Python 侧）**：一处实现，opencode/dsh/所有 weave 调用方统一生效；index.js 只做 nudge 辅助（不承担强制）
- **新状态位复用 mark_weave 机制**：weave 时根据计数器/队列状态设置 `archive_pending`/`queue_pending`，下次 check() raise
- **硬门严格度 = "必须响应"**：agent 需调用对应工具（mark done / review default 后 reclassify 至少 1 条）清除状态，不要求清空全部——防误判阻塞对话（R8）
- **按轮数触发（归档）**：config 加 `archive_interval: int = 10`（每 N 轮），由 ProtocolState 的 weave 计数驱动
- **队列硬门显示优化**：weaver 的待学习区加 open 计数 + 只列前 3-5 条（现状）+ 明确"必须响应"
- **归位 = 改 category + 补 title**：learned_store.update 已有 content 重嵌能力；category 变更轻量（metadata 更新），无需全量重嵌
- **mission 自动闭环**：MissionStore 状态变更（done）时同步 mark 对应 queue item

## Implementation Units

- [ ] **Unit 1: 归档工具（review_default + reclassify）**

**Goal:** 新增 `memory_review_default` 与 `memory_reclassify` 两个 MCP 工具，让 agent 能查看 default 候选并归位

**Requirements:** R1, R2

**Dependencies:** None

**Files:**
- Modify: `memory_skill/tools.py`（handler + DISPATCH + TOOL_SCHEMAS）
- Modify: `memory_skill/learned_store.py`（若需 reclassify 辅助方法）
- Test: `tests/test_archive_tools.py`（新建）

**Approach:**
- `handle_review_default(skill, args)`：调 `learned_store.list_by_category('default', limit=args.limit 默认 10, offset)` 返回条目（content 摘要/title/category/updated_at/id）
- `handle_reclassify(skill, args)`：参数 entry_id + category（校验 ∈ CATEGORIES）；调 learned_store 更新 category（+ 若 title 空则用 content 截断补 title）
- reclassify 需改 category 字段——查 learned_store.update 是否支持改 category（metadata 更新 vs 重嵌），不支持则加辅助方法
- 复用现有 handler 模式（try/except + logger + error dict）

**Patterns to follow:**
- `handle_character_get`/`handle_learning_queue` 的 handler 写法（tools.py）
- DISPATCH/TOOL_SCHEMAS 注册（character 工具模式）

**Test scenarios:**
- Happy path: review_default 返回 default 条目列表（limit 生效）
- Happy path: reclassify 将 default 条目改为 pref，get 后 category 正确
- Edge case: reclassify 不存在 entry_id → 明确 error
- Edge case: reclassify 非法 category → 400 风格 error
- Edge case: review_default 空 default → 空列表（不报错）
- Integration: reclassify 后该条目在 weave 检索中按新分类返回

**Verification:**
- pytest 通过；MCP 实测 review/reclassify 往返

- [ ] **Unit 2: 硬门状态与 weave 注入（ProtocolState + weaver）**

**Goal:** ProtocolState 加状态位、weaver 注入 default 计数与队列硬门提示、ProtocolGate 加检查

**Requirements:** R3, R4, R5, R8

**Dependencies:** Unit 1（工具存在才能硬门引用）

**Files:**
- Modify: `memory_skill/protocol_state.py`（状态位 + weave 计数）
- Modify: `memory_skill/protocol_gate.py`（check 新增两分支）
- Modify: `memory_skill/weaver.py`（gap_context 增强 + default 计数注入）
- Modify: `memory_skill/contracts.py`（archive_interval 配置）
- Modify: `memory_skill/_compose.py`（状态初始化/传递）
- Test: `tests/test_protocol_archive.py`（新建）

**Approach:**
- ProtocolState 加字段：`archive_pending: bool`、`queue_pending: bool`、`weave_count: int`；mark_weave 时 weave_count++，达 archive_interval 且 default 非空 → archive_pending=True；queue 有 open → queue_pending=True
- ProtocolGate.check()：若 archive_pending → raise ArchiveRequired（提示先 review_default）；若 queue_pending → raise QueueRequired（提示先响应待办）——**但需"可跳过"机制**：agent 调用 review_default/reclassify 或 learning_mark 后状态清除（after hook 检测）
- 权衡：queue_pending 每轮都开太烦——设计为 queue open > 阈值（如 20）才激活
- weaver：gap_context 头部加 "队列待办 N 条 open（M mission）"；加 default 计数注入（archive_context）
- 状态传递：MemorySystem 持有 ProtocolState，weaver 读 stores

**Patterns to follow:**
- 现有 classify_pending/mission_pending_check 的完整链路（state 标记 → gate 检查 → 工具调用后清除）

**Test scenarios:**
- Happy path: 每 N 轮后 weave 触发 archive_pending（check raise）
- Happy path: agent 调 review_default + reclassify ≥1 → 状态清除，下轮 weave 正常
- Edge case: 队列 open < 阈值 → queue_pending 不激活（不烦扰）
- Edge case: default 空 → archive_pending 不激活
- Edge case: agent 明确跳过（如 reclassify 标记"无值得归档"）→ 状态清除（防死锁）
- Integration: 触发硬门 → agent 响应 → 恢复的完整循环

**Verification:**
- pytest 通过；模拟 10 轮 weave 验证硬门触发 + 响应后恢复

- [ ] **Unit 3: mission 自动闭环**

**Goal:** mission 完成后自动标 queue item done，减少 open 堆积

**Requirements:** R6

**Dependencies:** None（独立）

**Files:**
- Modify: `memory_skill/mission.py`（状态变更时同步 queue）
- Test: `tests/test_mission.py` 追加

**Approach:**
- MissionStore 状态 set done 时：查 queue 中对应 mission item（detail 含 mission_id）→ mark done
- 现状：mission enqueue 时 detail 带 `mission_id=xxx`（148 行）——用此关联反查
- 失败降级：找不到 queue item 或 mark 失败 → log，不阻断

**Patterns to follow:**
- mission.py 现有 enqueue 关联逻辑

**Test scenarios:**
- Happy path: mission 完成 → 对应 queue item 自动 done
- Edge case: mission 无对应 queue item → 静默（不报错）
- Integration: mission 生命周期（open→decompose→done）后 queue open 数减少

**Verification:**
- pytest 通过

- [ ] **Unit 4: 入队闸门（可选评估）**

**Goal:** 降低 learning_queue 误入队，减少无效 open 堆积

**Requirements:** R7

**Dependencies:** None

**Files:**
- Modify: `memory_skill/tools.py`（_enqueue_if_learning 灵敏度）
- Modify: `memory_skill/learning_queue.py`（如加去重增强）
- Test: `tests/test_learning_queue.py` 追加

**Approach:**
- 评估 _enqueue_if_learning 的触发条件：目前 classify 后若 skill/mission 有 gaps 即入队
- 加启发：仅当用户明确表达学习/任务意图（非每轮闲聊推断）才入队；或提高去重（同 query 近期已 done 不重入）
- 若评估后发现误入队少，可跳过此单元（规划时保留决策点）

**Test scenarios:**
- Happy path: 明确学习请求入队；闲聊不误入队
- Edge case: 重复请求去重

**Verification:**
- pytest 通过

## System-Wide Impact

- **Interaction graph:** ProtocolGate.check 新增 raise 分支 → 所有 weave 调用方受影响（opencode/dsh）；weaver gap_context 增强 → weave 输出变化
- **Error propagation:** ArchiveRequired/QueueRequired 新异常——tools 层与 agent 提示需识别并响应
- **State lifecycle risks:** archive_pending/queue_pending 必须可清除（agent 响应或明确跳过），否则死锁（对话永久阻断）
- **API surface parity:** 新增 2 工具（review_default/reclassify）；现有 30 工具不变
- **Unchanged invariants:** 未触发硬门时 weave 行为不变；classify/mission/gaps 三类现有硬门不变；default 存量不自动迁移

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| 硬门误触发阻塞正常对话 | "必须响应"非"必须清空"（R8）；可跳过机制；阈值可配（archive_interval/queue 阈值） |
| 队列硬门每轮烦扰 | queue open 超阈值（20+）才激活；激活后 agent mark 一批即清除 |
| mission 自动闭环误标 done | 只标明确 mission_id 关联的；失败降级 log 不阻断 |
| 与 embedding 校准改动（未提交 WIP）冲突 | tools/learned_store 增量修改，不动 embedder/retriever 已改部分 |

## Documentation / Operational Notes

- INTEGRATION.md 补：归档工具 + 硬门行为说明（agent 遇到 ArchiveRequired/QueueRequired 如何响应）
- opencode-auto-memory SKILL.md 补：待办响应指引（可选，若发现 agent 不响应再加）

## Sources & References

- Related code: `memory_skill/protocol_gate.py`、`protocol_state.py`、`weaver.py`、`tools.py`、`mission.py`、`learning_queue.py`
- 历史：b47f962 weave 硬门（classify_pending 模式）、08-11 零 LLM 架构决策（记忆 conclusion_20260813_045240）
