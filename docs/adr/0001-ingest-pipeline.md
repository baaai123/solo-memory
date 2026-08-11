# ADR-0001 — 深化写链:单一 IngestPipeline 接口

> **⚠️ SUPERSEDED by [ADR-0002](./0002-no-internal-llm.md) — 2026-08-11**
> This ADR's `enrich` pipeline, `extract_structured`, `tag_title`,
> `gap_detector`, and `extract_conclusion` stages were removed in the
> "zero internal LLM" rewrite. The ingest pipeline is now a single
> `ingest_dialogue(turn)` call. This document is retained as a
> historical record of the original design.

**Status**: Accepted · **Date**: 2026-08-09
**Decision driver**: improve-codebase-architecture 审查候选 1(写链入口泛滥)+ grilling 决策树
**Relates to**: KNOWN-ISSUES #9(过程 vs 知识)、#10(推理链)、候选 7(conclusion 条目,预留位置)

## 背景

写链(ingest)有 5 个入口,各自手工拼 `DialogueTurn`、id 前缀、截断规则、LLM 阶段语义,导致:

- `ingest`(MCP/CLI/room_adapter)→ 跑 tag_title + extract_structured 两轮 LLM
- `auto_ingest`(transparent_proxy 自动写)→ 完全不跑 LLM,连 title 都不生成
- `ingest_staged` → 与 `ingest` 几乎相同的链,错误处理不同,**无生产调用方**(只有测试用)
- `_ingest_structured` / `_weave` 内联 → 各自再拼一份

KNOWN-ISSUES #9 自认"同一轮被存两份"直接源于入口泛滥;截断规则三套(`[:2000]` 截尾丢结论 vs `_clip_auto_ingest` 头尾保留 vs 不截断)。

## 决策

将写链收敛为单一深接口 `ingest(turn, *, enrich=True, report=False)`:

1. **范围**:`ingest` + `ingest_staged` 合并;`auto_ingest` 保留但内部走同一管道;`_ingest_structured` 和 `_weave` 内联收敛;删除 `ingest_staged`。
2. **返回**:写链返回 `IngestReceipt`(`entry_id` / `deduped` / `weight` / `staged` / `timestamp`),替代空壳 `MemoryEnvelope`。`deduped` 让去重路径可测。
3. **id**:保留来源前缀(`auto_` / `mcp_` / `{category}_`),时间格式和唯一性由管道统一。
4. **截断**:所有入口统一 `_clip_auto_ingest`(头 490 + 尾 300,总 ≤800,保结论)。`auto_ingest` 不再 `[:2000]`。
5. **enrich 开关**:`enrich=False` 对应 proxy 高频自动写(不跑 LLM);`enrich=True`(默认)跑 tag + extract,未来挂 conclusion 提炼。
6. **结构化显式报告**:tag/extract 在跳过时返回原因(`"reason": "tree disabled"` / `"not user role"`),不再静默 return。
7. **gap 检测**:从 `ingestor.ingest_dialogue` 移出,挂进 enrich 管道(与 tag/extract 同级);`_compose.gaps` 不再穿私有属性 `_gap_detector`。
8. **测试**:新增 `TestIngestPipelineContract` 契约测试(三店写入 / enrich 开关 / dedup 反映);保留现有集成测试。
9. **conclusion 阶段**(候选 7,2026-08-09 实施):enrich 管道新增第四阶段 `staged["conclusion"]`,只对 assistant 轮运行。`extract_conclusion`(LLM)判定回复是否含可复用结论;是则存为 `conclusion:` 类别条目(标题 + 结论 + 依据),原文照存、结论多存一份。理由:结论来自 assistant 的分析/修复,不是 user 提问;每轮最多 +1 次 LLM 且只在 assistant 轮付。
10. **nudge 相关性门控**(候选 8,2026-08-09 实施):`_build_nudge` / `_has_high_weight` 增加 `user_message` 参数,注入前用 `_token_overlap`(capability_registry 双证逻辑之一)过滤——与当前消息无 token 重叠的高权重条目不再每轮注入(消除"pip install failed"式历史噪声)。`user_message` 为空时跳过门控保持旧行为。

## 理由

- 消除同义入口语义分裂(proxy vs MCP 行为差异从"两套代码"变成"一个参数")
- `enrich` 开关为候选 7(conclusion)预留注入点
- 头尾保留截断已在 #8 校准(超时修复),统一后 proxy/MCP 行为一致,修复 `auto_ingest` 丢结论
- 显式报告让"为什么没提炼"可观察,测试可断言

## 不做的事(明确排除)

- 不碰检索链(retriever/weaver)——候选 2/6 的范围
- 不动 tree 模块结构——凭据穿透是候选 3 的范围
- ~~不实现 conclusion 提炼~~ → 已于 2026-08-09 随本 ADR 第 9 条实施(候选 7)

## 涉及文件

- `_compose.py`(ingest/ingest_staged 合并、auto_ingest 收敛、gaps 改道、conclusion 阶段)
- `ingestor.py`(gap 检测移出、返回 IngestReceipt、截断统一)
- `mcp_tools.py`(_weave 内联收敛、_ingest 改调管道)
- `memory_extract.py`(显式报告跳过原因、extract_conclusion_entry)
- `structured_extractor.py`(extract_conclusion 判定函数)
- `contracts.py`(新增 `IngestReceipt`、`clip_auto_ingest` 下沉)

## 后果

- 正面:写链单一测试面;dedup 可观察;为 conclusion 铺路;`_compose.gaps` 不再穿私有属性
- 负面/代价:一次性改动面较大(5 文件);`enrich=False` 的 proxy 行为不变但实现路径变化需回归
- 未来审查不应重新建议"合并 ingest 入口"——已决策
