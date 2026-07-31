# Memory 模块 — 长期 RAG 记忆 + 自动演化

## 核心理念

Memory 是 Room 的独立插件——Agent 感知不到它的存在，但它持续优化对话质量。

**每轮对话自动做三件事：注入上下文、存储记忆、反馈演化。**
**每隔 N 轮后台做四件事：归纳观察、清理冲突、调整权重、发现矛盾。**

---

## 架构

```
MemorySkill                           ← 对外接口
├── Embedder (ONNX bge-small-zh)       ← 文本→1024维向量
├── DialogueStore (SQLite FTS5)        ← 原始对话 + BM25 全文搜索
├── LearnedStore (ChromaDB)            ← 向量存储 + 语义搜索
├── Retriever                          ← 3路信号 RRF 融合检索
├── Weaver                             ← 自动上下文注入 (3层自适应)
├── Ingestor                           ← 写入管道
├── ObservationConsolidator            ← TextRank 关键词归纳
├── EvolutionLoop                      ← 反馈驱动权重演化
├── Cleaner                            ← 去重 + 过期清理
├── Reflect                            ← 空闲后台复盘
│   ├── consolidate → TextRank 归纳
│   ├── evolve → 权重演化
│   ├── nudge → 高权重记忆标记
│   └── contradiction → 矛盾降权
└── Feedback                           ← 自动结果检测
```

## 数据流

```
每轮 weave():
  用户消息 → Embedder → Retriever(RRF融合) → Weaver → 3层上下文 → Agent prompt

每轮 ingest():
  Agent说话 → DialogueTurn → Ingestor → SQLite + ChromaDB
  每10轮 → consolidate() → TextRank 提取关键词 → 生成观察
  每50轮 → clean() → 合并重复 + 标记过期

每轮 feedback():
  Agent回复 → auto_detect_outcome() → 正/负/中性
  → record_outcome() → EvolutionLoop 权重调整

检索 (回忆一下工具):
  用户查询 → 3路信号 RRF 融合:
    语义向量 ×1.5 + BM25全文 ×1.5 + 时间衰减 ×0.5
  → 按融合分数排序 → 返回 top-N

后台 reflect():
  对话沉默时触发（仅一次，不注入消息）
  → consolidate → evolve → nudge → contradiction
  → 结果用于后续 weave()，不影响对话流
```

## 三路检索融合

```
语义搜索 (ChromaDB)
  ↓ 向量嵌入 ×1.5
BM25 全文 (SQLite FTS5)
  ↓ 关键词匹配 ×1.5  
时间衰减
  ↓ exp(-age_hours / 24h) ×0.5
  ↓
RRF (k=60) 加权融合
  ↓
排序结果 + 权重提升 + 实体共现提升
```

## TextRank 关键词归纳

```
ObservationConsolidator:
  收集最近 200 条对话
  → TextRank4ZH 提取 top-30 关键词
  → 筛选出现 ≥5 次的词
  → 按词分组证据 → 生成结构化观察
  → 存入 learned_store (category="observation")

关键词维度: 自动发现（非硬编码）
当前关键词: 主人、声音、身体、姐姐、饼干...
```

## 矛盾检测

```
仅检测真正语义对立的词对:
  喜欢 ↔ 讨厌
  爱 ↔ 恨
  用过 ↔ 没用过

发现矛盾 → 双方权重各降 0.15 → 不再优先检索
```

## Weaver 三层注入

```
<3 轮:   tier1 (~80 字)   最近对话 + 场景
3-10轮:  tier1 + tier2 (~150字)  按 partner 分组的语义检索
                          + nudge (~30字)   权重≥0.8 的高优先级记忆
>10 轮:  + observations + heartbeat
```

## Evolution 演化

```
反馈驱动权重调整:
  正面反馈 → 权重 +0.1
  负面反馈 → 权重 -0.2
  中性 → 不变
  每轮变化上限 0.05（防震荡）
  结果持久化到 memory_outcomes.json
```

## 文件

```
memory.db               ← SQLite (对话存储 + FTS5全文索引)
memory.db_chroma/       ← ChromaDB (向量存储)
memory_outcomes.json    ← 演化结果持久化
```

## 状态指标

```
mode=onnx               ← 嵌入模型已加载 (CPU 模式)
learned=N               ← ChromaDB 中的记忆条目数
dialogue=N              ← SQLite 中的原始对话数
evo=N                   ← 已执行的演化 tick 数 (会话内计数)
```

## 导入外部对话

导入外部对话可通过 MemorySkillAdapter 的 `ingest_batch()` 方法完成，详见 `memory_skill/room_adapter.py`。
