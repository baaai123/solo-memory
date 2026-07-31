# Memory Skill — 完整技术文档

> 版本: v0.5.0 | 最后更新: 2026-07-13

---

## 目录

1. [项目理念](#1-项目理念)
2. [架构总览](#2-架构总览)
3. [写链 — Agent 发言 → 持久化](#3-写链--agent-发言-持久化)
4. [读链 — 用户消息 → 记忆注入](#4-读链--用户消息-记忆注入)
5. [检索算法 — 三信号 RRF 融合](#5-检索算法--三信号-rrf-融合)
6. [反馈与演化](#6-反馈与演化)
7. [配置参数](#7-配置参数)
8. [竞品调研](#8-竞品调研)
9. [嵌入模型选择](#9-嵌入模型选择)
10. [使用协议](#10-使用协议)
11. [版本历史](#11-版本历史)

---

## 1. 项目理念

**一句话**: 一个对 Agent 完全透明的长期记忆插件——Agent 不知道自己有记忆，但行为被记忆驱动。

### 核心设计原则

| 原则 | 含义 | 体现 |
|------|------|------|
| **无知觉** | Agent 不感知记忆系统存在，不调用 `memory_search`，不知道 weave 在运行。Agent 只收到组织得更好的 system prompt | 记忆以 user 消息注入，不暴露 XML 标签或工具接口 |
| **逐字原文** | Never summarize. Store verbatim. 摘要丢失语境、态度、细节 | SQLite FTS5 存储原文，ChromaDB 存储原文向量 |
| **BM25 主信号** | 中文对话 exact word match > vector similarity | BM25=2.5, semantic=0.5 |
| **行为驱动** | 记忆注入不只是信息——是行为指令 | "你很清楚...""务必...""好感度偏高..." |
| **独立插件** | Memory 不知道 Persona、Emotion、TTS 的存在 | 桥接在 RoomAssembly 层 |

### 为什么不是别的方式

| 替代方案 | 为什么没选 |
|----------|-----------|
| Letta/MemGPT | Agent 不应该感知到记忆系统 |
| CrewAI task memory | 角色对话不需要 task planning |
| MemVerse GPT 摘要 | 摘要丢失语境。verbatim+BM25 更可靠 |
| 纯向量检索 | 嵌入质量不可靠。BM25 作为主信号 |

### 已知权衡

- Evolution 反馈链路长——delta=0.15，需多轮
- GPU 依赖——CPU fallback (SHA-256) 仅测试用
- jieba 必需——FTS5 不切中文

---

## 2. 架构总览

```
                    ┌─────────────────────────────────┐
                    │         Room Agent              │
                    │  (Agent 不感知 Memory 存在)      │
                    └──────┬──────────────┬───────────┘
                           │              │
                    ② 注入 │              │ ① 存储（发言后）
                    (读链) │              │ (写链)
                           ▼              ▼
              ┌─────────────────────────────────────┐
              │      MemorySkillAdapter             │
              │  实现 MemoryProtocol, 桥接 Room      │
              └──────────────┬──────────────────────┘
                             │
              ┌──────────────▼──────────────────────┐
              │          MemorySkill                │
              │        (组合层, 薄委托)              │
              └───┬───┬───┬───┬───┬───┬───┬────────┘
                  │   │   │   │   │   │   │
          ┌───────┼───┼───┼───┼───┼───┼───┼───────┐
          │ Weaver│Ret│Emb│DS │LS │Evo│Feed│       │
          │ (编织)│   │   │   │   │   │   │       │
          └───────┴───┴───┴───┴───┴───┴───┴───────┘
```

### 模块职责

| 模块 | 职责 | 文件 |
|------|------|------|
| `MemorySkill` | 薄组合层，对外接口 | `skill.py` |
| `Embedder` | ONNX bge-large-en-v1.5，1024维向量 | `embedder.py` |
| `DialogueStore` | SQLite FTS5 + jieba 分词，BM25全文搜索 | `dialogue_store.py` |
| `LearnedStore` | ChromaDB 向量存储，HNSW cosine检索 | `learned_store.py` |
| `Retriever` | 3信号RRF融合检索 | `retriever.py` |
| `Weaver` | 自动4层上下文编织（tier1/tier2/emotion/nudge） | `weaver.py` |
| `Ingestor` | 写入管道（重要性门控 + 双存储） | `ingestor.py` |
| `ObservationConsolidator` | TextRank关键词归纳 → 结构化观察 | `observation.py` |
| `EvolutionLoop` | 反馈驱动权重演化 | `evolution.py` |
| `Feedback` | 自动结果检测（LLM + 规则双路径） | `feedback.py` |
| `Cleaner` | 去重 + 过期清理 | `cleaner.py` |
| `Reflect` | 空闲后台复盘 | `reflect.py` |
| `SawRingBuffer` | 环状缓冲区（容量1000） | `saw_buffer.py` |
| `ImportanceScorer` | 重要性门控（阈值0.3） | `importance.py` |
| `NoiseFilter` | 屏幕流噪声过滤（n-gram余弦） | `noise_filter.py` |
| `MiniCPM5Rewriter` | 查询扩展（可选） | `rewriter.py` |
| `MemorySkillAdapter` | Room框架桥接 | `room_adapter.py` |
| `mcp_server` | MCP stdio JSON-RPC服务 | `mcp_server.py` |
| `mcp_tools` | 5个MCP工具定义 | `mcp_tools.py` |
| `cli` | Click命令行入口 | `cli.py` |

### 数据流全景

```
每轮 weave():
  用户消息 → Embedder → Retriever(RRF融合) → Weaver → 4层上下文 → Agent prompt

每轮 ingest():
  Agent发言 → DialogueTurn → Ingestor → SQLite + ChromaDB
  每10轮 → consolidate() → TextRank提取关键词 → 生成观察
  每50轮 → clean() → 合并重复 + 标记过期

每轮 feedback():
  Agent回复 → auto_detect_outcome() → 正/负/中性
  → record_outcome() → EvolutionLoop 权重调整

后台 reflect():
  对话沉默时触发（仅一次，不注入消息）
  → consolidate → evolve → nudge → contradiction
  → 结果用于后续 weave()，不影响对话流
```

---

## 3. 写链 — Agent 发言 → 持久化

### 3.1 入口

```
Room.ConversationLoop
  └─ await agent.memory.add(msg, partner=partner)
```

每次 Agent 或 User 发言后，`ConversationLoop` 调用 `MemorySkillAdapter.add()`。

### 3.2 Adapter.add()

- 使用 `Message.role` 枚举判定角色（`agent`/`assistant` → `"assistant"`，其余 → `"user"`）
- `display_name` 控制 Weaver 显示标签，`agent_name` 控制 namespace
- 自动维护：每10轮触发 `consolidate()`，每50轮触发 `clean()`

### 3.3 Ingestor.ingest_dialogue()

```
DialogueTurn
  │
  ├─ ① SawRingBuffer.put()          ← 始终存储（环形缓冲区, 容量 1000）
  │
  ├─ ② ImportanceScorer.evaluate()   ← 重要性门控
  │     规则: 关键词("记住"/"重要") +0.3, 偏好词("我喜欢") +0.2
  │     阈值: 0.3 → 低于则跳过（仅 SawRingBuffer）
  │
  ├─ ③ DialogueStore.insert(turn)    ← SQLite 持久化
  │     ├─ dialogue_turns 表: INSERT OR IGNORE
  │     └─ dialogue_fts 表(FTS5): _cn_tokenize(content)
  │         jieba 分词 → FTS5 unicode61 索引
  │
  ├─ ④ _extract_entities(content)    ← 正则提取专名 (英文大写词)
  │
  └─ ⑤ LearnedStore.insert(entry)    ← ChromaDB 向量存储
       ├─ Embedder.embed(content) → ONNX bge-large → [1024] float
       ├─ MemoryEntry(id, content, weight=0.5, category=ns, tags, metadata)
       └─ ChromaDB collection.add() (HNSW + cosine)
```

### 3.4 批量摄入

`Ingestor.ingest_batch()` 提供高性能路径：

- 文本分块（512 token 窗口, 256 token 步长, 50% 重叠）
- 单次 ONNX `embed_batch` 调用
- 单次 ChromaDB `collection.add()`
- 单次 SQLite `executemany` 事务
- 目标 ≥500 msg/s

### 3.5 DialectStore 数据模型

```sql
CREATE TABLE dialogue_turns (
    id       TEXT PRIMARY KEY,
    role     TEXT NOT NULL,       -- "user" | "assistant"
    content  TEXT NOT NULL,
    timestamp REAL NOT NULL,
    saw_index INTEGER,
    UNIQUE(role, content, timestamp)
);

CREATE VIRTUAL TABLE dialogue_fts USING fts5(
    turn_id, content,
    tokenize = 'unicode61'
);
```

### 3.6 重要性评分算法

```python
score = 0.5                              # 基础分
score += min(high_hits, 2) * 0.1        # 关键词boost ("记住"/"重要"/"别忘了")
score += pref_hits * 0.15               # 偏好boost ("我喜欢"/"我讨厌")
if len(text) <= 4: score -= 0.3         # 超短惩罚
if unique_ratio < 0.4: score -= 0.3    # 低多样性惩罚
if trivial_pattern.match: score = 0.05  # 已知无意义形式
return clamp(score, 0.0, 1.0)
```

---

## 4. 读链 — 用户消息 → 记忆注入

### 4.1 入口

```
Room.agent._step()
  ├─ memory_ctx = await self.memory.weave(user_msg)
  └─ msgs = self._template.render(agent, history, memory_ctx)
```

### 4.2 Adapter.weave()

```python
ctx = self._skill.weave(user_message=message, partner=p, scene_summary=scene_summary)
block = ctx.to_prompt_block()

# display_name 替换
if self.display_name != self.agent_name:
    block = block.replace(f"{self.agent_name}: ", f"{self.display_name}: ")

# 截断: 默认 3000 chars (~750 tokens)
if len(block) > self._max_context_chars:
    block = block[:self._max_context_chars].rsplit("\n", 1)[0] + "\n..."
```

### 4.3 Weaver.weave() — 四层编织

```
weave(skill, user_message, scene_summary, partner)
  │
  ├─ 深度门控:
  │   compact:  recent(5min) < 3 AND total_stored == 0 → 仅 tier1
  │   standard: recent(5min) ≤10 OR has stored data → tier1+tier2+emotion+nudge
  │   deep:     recent(5min) > 10 → 以上 + heartbeat
  │
  ├─ tier1 = _build_tier1(scene_summary)
  │   ├─ [当前场景] {scene_summary} | [当前感知] SawBuffer[-1]
  │   └─ [最近对话] get_recent(3) — 各条 ≤80 字符, 总 ≤420 字符
  │
  ├─ tier2 = _build_tier2(user_message, ns)      ← 标准/深度
  │   └─ Retriever.retrieve() → 3信号 RRF 融合
  │
  ├─ emotion = _build_emotion_context(partner)
  │   └─ Evolution._outcomes[-50:] → 正/负比例 → 好感度提示 (≥3 样本)
  │
  └─ nudge = _build_nudge()
      └─ learned_store.search(weight≥0.85) → ⚠/💡 行为提示 (top 3)
```

#### Tier2 输出格式 (V8)

```
以下是你知道的事实。你只能使用这些信息，不得编造:
  - [Agent说的] (fact text with source attribution)
  - [User说的] (user's own words)
（如果你不确定某条信息是否在上面列出——就不要提及它。）
```

**反幻觉约束**:
1. 每条事实标记 `[source说的]` — source 由 `role` 推断
2. 自引用跳过：跳过与当前用户消息前25字符相同的事实
3. 去重：同一事实只出现一次（前60字符key）
4. 显式约束："你只能使用这些信息，不得编造"

#### 四层注入的Token预算

| 层级 | 上限 | 内容 |
|------|------|------|
| tier1 | ~320 chars (~80 tokens) | 当前场景 + 最近3条对话 |
| tier2 | ~1200 chars (~300 tokens) | RRF检索的事实片段 |
| emotion | ~100 chars (~25 tokens) | 好感度一句提示 |
| nudge | ~120 chars (~30 tokens) | ⚠/💡 行为提醒 (top 3) |

#### 为什么注入 user 而非 system 消息

DeepSeek 角色扮演模型无视 system 指令。记忆以 `[系统提醒]` 前缀 + `role="user"` 注入，DeepSeek 服从 user 消息。

### 4.4 最终消息列表

```
[0] system   ── ### 你是谁 / ### 你在哪里 / ### 行为规则
[1] user     ── [系统提醒] 以下是你的记忆记录...
                 [当前场景] ...
                 [最近对话] ...
                 以下是你知道的事实...
                 [你当前对user的感受] ...
[2] user     ── User: (历史用户消息)
[3] asst     ── Agent: (历史Agent回复)
[4] user     ── User: (当前用户消息)
```

---

## 5. 检索算法 — 三信号 RRF 融合

### 5.1 算法

```
retrieve(query, limit=5, filters={category: "agent_a"})

  ├─ ① SEMANTIC: learned_store.search(query, limit=50, filters)
  │     ChromaDB HNSW cosine 相似度
  │     权重: 0.5

  ├─ ② BM25: dialogue_store.search(query[:80], limit=50)
  │     jieba分词 → FTS5 MATCH → BM25排序
  │     权重: 2.5  ← 主信号

  ├─ ③ TEMPORAL: _time_decay(entry, now)
  │     exp(-age_hours / 24.0)
  │     权重: 0.5

  └─ RRF融合 (k=60):
      rrf_score = 0.5/(60+sem_rank) + 2.5/(60+bm25_rank) + 0.5/(60+temp_rank)
      final_score = rrf_score × (0.5 + entry.weight)          ← Evolution权重
      final_score ×= (1.0 + 0.1 × entity_overlap)             ← 实体共现 (max 1.5×)
      → Top-N → MemoryEnvelope
```

### 5.2 为什么 BM25=2.5 是主信号

中文嵌入质量依赖训练数据。在特定内容上向量相似度不稳定，exact word match (BM25) 更可靠。语义作为兜底。

Weight分配经过消融实验验证（`docs/reports/ablation_report.md`）。

### 5.3 查询可选扩展

`MiniCPM5Rewriter` 可执行查询扩展：

```
输入: "auth"
输出: "authentication JWT OAuth login"
```

使用 MiniCPM-5 1B GGUF 模型 + llama-cpp-python。Lazy-load，默认关闭。

---

## 6. 反馈与演化

### 6.1 反馈自动检测

两条路径，LLM优先：

| 路径 | 机制 | 激活条件 |
|------|------|----------|
| **LLM** | MiniCPM-5 1B GGUF 分类 → positive/negative/neutral | 默认启用，失败降级 |
| **Rule** | keyword overlap: 响应词 ∩ 检索结果词 > 30% → positive, 0% → negative | LLM路径失败时 |

### 6.2 Evolution 权重演化

```
每轮演化 (tick):
  │
  ├─ 收集最后 100 条 outcomes
  ├─ 计算 per-memory deltas:
  │   positive → +0.1, negative → -0.2, neutral → 0
  ├─ 单轮 cap: max_delta = 0.15
  ├─ 边界约束: weight ∈ [0.05, 0.95]
  ├─ learned_store.set_weight(mem_id, new_weight)
  └─ 清除已处理 outcomes → 保存到 {db}_outcomes.json
```

**演化周期**: 每 5 条 outcomes 触发一次，最小 5 条样本，跨 session 持久化。

### 6.3 空闲反思 (Reflect)

对话沉默时自动触发：

```
reflect(skill):
  ├─ consolidate() → TextRank 关键词归纳 → 生成观察
  ├─ evolve()     → 权重演化 tick
  ├─ nudge        → 高权重记忆标记 (weight ≥ 0.85)
  └─ contradiction → 矛盾降权 (双方各 -0.15)
```

### 6.4 记忆清理 (Cleaner)

安全清理，永不硬删除：

```
clean(skill):
  ├─ 分类: protected (weight≥0.7 | category=preference/observation) vs candidates
  ├─ 去重: 同 category 内 cosine similarity > 0.95 → 合并为一条
  └─ 过期: weight加权TTL (14-90天) → 标记 stale
```

### 6.5 Observation Consolidation

TextRank 关键词提取 → 结构化观察：

```
ObservationConsolidator:
  收集最近 200 条对话
  → TextRank4ZH 提取 top-30 关键词
  → 筛选出现 ≥5 次的词
  → 按词分组证据 → 生成结构化观察
  → 存入 learned_store (category="observation", weight=0.6, is_system=True)
```

---

## 7. 配置参数

### 7.1 MemorySkillConfig

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `rrf_bm25_weight` | 2.5 | BM25 信号权重 (主信号) |
| `rrf_semantic_weight` | 0.5 | 语义向量权重 |
| `rrf_temporal_weight` | 0.5 | 时间衰减权重 |
| `rrf_k` | 60 | RRF 融合常数 |
| `temporal_boost_hours` | 24.0 | 时间衰减半衰期 (小时) |
| `evolution_max_delta` | 0.15 | 单轮最大权重调整 |
| `evolution_min_samples` | 5 | 最小演化样本数 |
| `evolution_interval_seconds` | 300 | 演化间隔 |
| `embedding_dim` | 1024 | 嵌入向量维度 |
| `max_learned_entries` | 100,000 | ChromaDB 最大条目数 |
| `saw_buffer_capacity` | 1000 | 环形缓冲区容量 |
| `import_chunk_tokens` | 256 | 批量摄入重叠步长 |
| `import_overlap_tokens` | 128 | 批量摄入重叠窗口 |

### 7.2 MemorySkillAdapter

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `agent_name` | `""` | namespace key (ChromaDB category) |
| `display_name` | `""` | Weaver 显示标签 |
| `partner` | `"user"` | 默认对话对象 |
| `max_context_chars` | 3000 | 记忆上下文截断阈值 (~750 tokens) |

### 7.3 Weaver 常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `_TIER1_MAX_CHARS` | 320 | tier1 场景+对话上限 |
| `_TIER2_MAX_CHARS` | 1200 | tier2 单条事实上限 |
| `_NUDGE_MAX_CHARS` | 120 | nudge 单条上限 |
| `_MAX_RECENT_TURNS` | 3 | 最近对话条数 |
| `_NUDGE_WEIGHT_THRESHOLD` | 0.85 | nudge 最低权重 |
| `_NUDGE_CRITICAL_THRESHOLD` | 0.95 | ⚠ 强制提醒权重 |

---

## 8. 竞品调研

调研了 7 个现有 AI Agent 记忆系统方案（详见 RESEARCH.md），核心发现：

### 8.1 横向对比

| 维度 | Supermemory | Hindsight | Mem0 | OpenViking | MemPalace | MemVerse |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 检索 | RAG+Memory混合 | TEMPR 4路 | 3信号融合 | 目录递归 | BM25+向量 | 图谱RAG |
| 存储 | 云端Postgres | PostgreSQL | Qdrant | AGFS+向量 | ChromaDB | JSONL |
| 画像 | ~50ms | ❌ | user/session | 8类实体 | ❌ | ❌ |
| 遗忘 | ✅ 自动 | ❌ | ❌ | ❌ | ❌ | ❌ |
| 自托管 | ❌ SaaS | ✅ | ✅ | ✅ | ✅ | ✅ |
| 本地嵌入 | ❌ | ❌ | ❌ | ❌ | ✅ ONNX | ❌ |
| LongMemEval | #1 (81.6%) | 91.4% | 94.8% | — | **96.6%** | — |

### 8.2 关键启发

| 来源 | 启发 | 本项目的实现 |
|------|------|-------------|
| **MemPalace** | verbatim 存储 + 本地嵌入 > GPT 摘要 | SQLite原文 + ChromaDB原文向量 + ONNX本地推理 |
| **Hindsight** | Observation Consolidation（统计合成高阶知识） | TextRank关键词提取 → 结构化观察 |
| **Mnemis** | System-1/System-2 分层路由（不应所有query都深度检索） | compact/standard/deep 三层weave深度 |
| **OpenViking** | 单次LLM提取（Schema编译同prompt）+ 文件系统即记忆 | —（本项目零LLM核心路径） |
| **Supermemory** | 自动遗忘，User Profile > 原始检索 | weight加权TTL过期 |
| **Letta** | Heartbeat自省机制 | 高权重记忆自动追加一轮注入 |

### 8.3 本项目的差异化

1. **无知觉架构** — Agent从不感知记忆系统的存在（唯一实现）
2. **BM25主信号** — 中文embedding不可靠，exact match更准
3. **反馈驱动权重进化** — 不依赖LLM反思，靠用户反馈自动调权
4. **零LLM核心路径** — 检索、嵌入、存储全部本地完成

---

## 9. 嵌入模型选择

### 9.1 背景

V1/V2 使用 `all-MiniLM-L6-v2`（384-dim, 256-token token限制）。消融实验显示向量检索 Hits@10 = **0%**。根因是 256-token 导致长消息被截断。

### 9.2 对比

| 维度 | all-MiniLM-L6-v2 | bge-large-en-v1.5 |
|------|------------------|-------------------|
| 模型家族 | Sentence-BERT (MiniLM) | BERT (large) |
| 参数量 | 22M | **335M** (15×) |
| 输出维度 | 384 | **1024** (2.67×) |
| Token 上限 | 256 | **512** (2×) |
| ONNX 模型大小 | ~80 MB | **~1.3 GB** |
| MTEB 评分 | 61.0 | **64.0** |
| 推理速度 (CPU) | ~2ms | ~20-50ms |
| 硬件需求 | CPU 即可 | **GPU 推荐 (CUDA)** |

### 9.3 选择理由

1. 512-token 覆盖绝大多数消息，减少分块依赖
2. 1024-dim 提供更丰富的语义表达空间
3. bge-large 官方提供 ONNX 导出
4. 部署环境有 CUDA 支持
5. MTEB 榜单验证，风险低

### 9.4 部署

```python
self._session = ort.InferenceSession(
    onnx_file,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

- GPU (CUDA): 预期 batch embedding 吞吐 ≥100 msg/s
- CPU 回退: 推理速度 ~20-50ms/次
- Fallback: ONNX 模型不存在时，SHA-256 deterministic fallback

---

## 10. 使用协议

### 10.1 MCP 工具

| 工具 | 时机 | 输入 | 输出 |
|------|------|------|------|
| `memory_status` | 会话启动 | 无 | 条目数、模型状态、演化tick数 |
| `memory_weave` | 每次回复前 | `user_message`, 可选 `scene_summary` | 4层上下文块 |
| `memory_search` | 需要深度检索 | `query`, 可选 `limit` | 排序后的记忆条目 + 相关性分数 |
| `memory_ingest` | 重要对话后 | `role` (user/assistant), `content` | 确认 |
| `memory_feedback` | 检索使用后 | `memory_ids` (列表), `outcome` (positive/negative/neutral) | 演化tick结果 |

### 10.2 何时摄入 vs 跳过

**摄入** 当:
- 用户陈述项目/个人事实
- 设计中做出决策
- 发现并修复 bug
- 用户表达强烈偏好

**跳过** 当:
- 问候、闲聊
- 无新知识的纯命令执行
- 即将修复的错误消息（摄入修复，不摄入错误）

### 10.3 反馈指南

在使用了 `memory_weave` 或 `memory_search` 的任何轮次之后:
- `positive` — 检索到的记忆准确且有用
- `negative` — 检索到的记忆错误或具误导性
- `neutral` — 存在记忆但与当前上下文无关

这会训练演化系统：准确的记忆获得更高权重，错误的记忆被抑制。

### 10.4 记忆系统记忆的内容

- **项目事实**: "this project uses SQLite, not PostgreSQL"
- **用户偏好**: "prefers TypeScript over JavaScript", "uses 2-space indentation"
- **历史决策**: "switched from ChromaDB to LanceDB for performance"
- **Bug与修复**: "the ONNX position encoding bug was fixed by truncating to 512"
- **模式**: "user always writes tests before implementation"

---

## 11. 版本历史

### v0.5.0 (2026-06-04)

**新增**:
- 透明记忆上下文：weave 输出使用自然语言格式
- 基于角色的初始权重：user消息 weight=0.6，assistant消息 weight=0.4
- 评估框架：`evaluation/duel.py` 自动化agent对决评分
- 跨会话演化：EvolutionLoop 持久化到 `{db}_outcomes.json`
- Room Agent 集成：`MemorySkillAdapter` 直接 Python 导入（<50ms/调用）
- 结构化日志：Python `logging` 模块，`MEMORY_SKILL_LOG_LEVEL` 环境变量

**变更**:
- Weave上下文按partner标记记忆
- Namespace模型简化：跨partner召回为有意设计
- Bridge weave顺序修复：先检索后摄入，修复自动摄入污染

**修复**:
- Nudge阈值：`$gt` 0.85 → `$gte` 0.85
- Bridge搜索：传递 `partner` 参数
- Tier2门控测试：修正期望值

### v0.4.0 (2026-06-01)

- MCP 服务器：4个工具（search, ingest, status, feedback）
- 反馈自动检测：LLM + 规则双路径
- ONNX位置编码修复：`min(max_len, 512)` 截断
- OpenCode 插件：5个原生工具

### v0.3.0 (2026-06-01)

- bge-large-en-v1.5 嵌入（1024-dim, 512-token）替代 all-MiniLM-L6-v2
- MiniCPM5 查询重写
- CUDAExecutionProvider 支持 + CPU 回退
- 向量 recall@10: 0% → 90%

### v0.2.0 (2026-06-01)

- 批量摄入：`Ingestor.ingest_batch()` + `IngestProfile`
- 文本分块：sentence-boundary对齐
- 消融实验：发现 256-token 根因
- RRF校准：网格搜索 (240组合)

### v0.1.0 (2026-06-01)

- 初始实现：15+ 模块，174 测试，TDD
- 核心：MemorySkill, Embedder, Retriever, Ingestor, EvolutionLoop
- 存储：SQLite FTS5, ChromaDB, SawRingBuffer
- Weaver：tier1/tier2/nudge 分层上下文注入
