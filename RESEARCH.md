# Memory 项目调研记录

> **V3 采用**: bge-large-en-v1.5 (ONNX, 1024-dim) 替代 all-MiniLM-L6-v2 (ONNX, 384-dim)

---

## 调研目标

为构建 AI Agent 的长时记忆系统，调研现有开源方案和学术前沿。

---

## 1. MemVerse — 多模态终身学习记忆框架

- **论文**: [arXiv:2512.03627](https://arxiv.org/abs/2512.03627)
- **代码**: [github.com/KnowledgeXLab/MemVerse](https://github.com/KnowledgeXLab/MemVerse) (⭐144)
- **机构**: 上海人工智能实验室
- **状态**: 已克隆到 `./MemVerse/`，已修复三层检索缺陷

### 1.1 架构

```
输入（text/image/video/audio）
  │
  ├─ 多模态处理 → 统一文本
  │   ├─ Image → GPT-4o 描述
  │   ├─ Video → FFmpeg 1fps 抽帧 → GPT-4o 逐帧 → GPT-4o-mini 摘要
  │   └─ Audio → Whisper-1 转写
  │
  ├─ 短期记忆（deque, K=10）
  │
  └─ 长期记忆（三层知识图谱）
      ├─ Core Memory     → 用户身份/偏好/人格
      ├─ Episodic Memory → 时间线事件日记
      └─ Semantic Memory → 概念/定义/知识
        │
        └─ LightRAG 知识图谱（6种检索mode：hybrid/local/global/mix/naive/bypass）
                │
                └─ 可选：参数化记忆（Qwen2.5-7B 周期蒸馏，快 89%）
```

### 1.2 关键指标

| Benchmark | 成绩 |
|-----------|------|
| ScienceQA | 85.48%（GPT-4o-mini + MemVerse） |
| MSR-VTT (t2v R@1) | 90.40% |
| LoCoMo F1 | 60.00%（GPT-3.5-Turbo-16k） |

### 1.3 API 调用成本

| 操作 | API 调用次数 | 主要消耗 |
|------|:---:|------|
| 纯文本 Insert | 13 次 | 3× GPT-4o-mini Agent + 3× LightRAG 实体/关系提取 + 4× embedding |
| 纯文本 Query | ~4 次 | 3× LightRAG keyword extraction + 1× final answer |
| 30s 视频 Insert | ~44 次 | 30× GPT-4o frame caption + 13× 基础 insert |

### 1.4 已修复问题

代码中 `rag_retrieve()` 原本只查询 `mem_core`，episodic/semantic 图谱写入后从未被检索。
已改为 `asyncio.gather` 并行查询三层，结果按 `[Core Memory]` / `[Episodic Memory]` / `[Semantic Memory]` 分段标注。

### 1.5 局限性

- 全链路 OpenAI 依赖，成本高
- 无持久数据库（JSONL 文件存储）
- 无测试覆盖
- 论文 vs 代码存在 gap（三层检索未连通、STM 未接入主流程）
- 参数化记忆需 GPU（Qwen2.5-7B ≈ 16GB VRAM）

---

## 2. Mnemis — 双路由层级图谱检索

- **论文**: [arXiv:2602.15313](https://arxiv.org/abs/2602.15313)
- **会议**: ACL 2026（已接收）
- **作者**: Zihao Tang et al.
- **代码**: 未找到开源仓库

### 2.1 核心创新

**双路由检索 = System-1（快）+ System-2（深）**，受 Kahneman《思考，快与慢》启发。

```
Query
  │
  ├─ System-1（快速路由，总是先走）
  │   └─ 基础图相似度检索
  │       → 简单查询直接返回（"我昨天做了什么"）
  │
  └─ System-2（深度路由，按需激活）
      └─ 层级语义图自顶向下遍历
          → 复杂推理才启动（"我和张三关系怎么演变的"）
```

**关键**：不是所有 query 都走全量检索，而是根据问题复杂度自动路由。

### 2.2 对比 MemVerse

| | Mnemis | MemVerse |
|---|---|---|
| 检索策略 | 分层路由（简单/复杂自动判断） | 统一图谱检索（6 mode 但本质同层） |
| 记忆组织 | 基础图 + 层级语义图（跨层遍历） | 三个独立图（并行查询） |
| 效率 | 按需深度，轻量 query 开销小 | 每次至少走 core 全量 |
| LoCoMo | **93.9**（GPT-4.1-mini） | 60.0（GPT-3.5-Turbo） |
| 开源 | ❌ | ✅ |
| 发表级别 | ACL 2026 | arXiv preprint |

### 2.3 关键启示

1. **不应该所有 query 都深度检索** — 简单事实查询走相似度就够了
2. **层级图 > 独立图** — 一次遍历跨层比并行查多个独立图更高效
3. **System-1/System-2 路由** — 轻量分类器判断 query 类型，按需选择检索深度

---

## 3. MemPalace — 本地优先、verbatim 存储的生产级记忆工具

- **代码**: [github.com/MemPalace/mempalace](https://github.com/MemPalace/mempalace) (⭐53k, 🍴7k)
- **版本**: v3.3.6 | License: MIT | Python 3.9+
- **定位**: 本地优先 CLI 工具，非科研框架

### 3.1 核心理念

与 MemVerse 的根本不同：

| | MemPalace | MemVerse |
|---|---|---|
| 存储方式 | **逐字原文（verbatim）**，永不摘要 | GPT 提取 + 结构化摘要 |
| 检索 | 混合 BM25 + 向量，零 API 调用 | 图谱 RAG，依赖 GPT API |
| 部署 | 本地 CLI + MCP Server (stdio) | FastAPI REST 服务 + Docker |
| LLM 依赖 | **核心检索零依赖** | 全程依赖 OpenAI API |
| LongMemEval R@5 | **96.6%**（纯本地，零 API） | 未测试 |

**一句话**：MemVerse 是"让 GPT 帮你记住"，MemPalace 是"你自己有搜索引擎就能找到"。

### 3.2 记忆宫殿架构（Method of Loci）

```
PALACE (根)
├── WING (人/项目)           → wing_myapp, wing_alice
│   ├── ROOM (天/话题)        → pricing-discussion, 2026-05-15
│   │   ├── DRAWER (原文)     → 对话原文，逐字不动
│   │   └── CLOSET (索引)     → AAAK 压缩指针，指向 drawer
├── KNOWLEDGE GRAPH (SQLite) → 时序实体-关系图
├── HALLWAYS (翼内连接)      → 共现实体路径
└── TUNNELS (跨翼连接)       → 显式跨项目桥接
```

**关键原则**：Drawer 永远 verbatim（不摘要、不改写），Closet 是索引层（排名信号，不挡检索）。即使 closet 没命中，drawer 查询仍然跑。

### 3.3 检索机制

**三级混合检索**（`searcher.py`, 1077 行）：

| 层级 | 机制 | 说明 |
|------|------|------|
| Tier 1 | 纯向量（ChromaDB cosine） | 默认，零 LLM |
| Tier 2 | BM25 + 向量混合 | 0.6×向量 + 0.4×BM25，closet 命中加排名分 |
| Tier 3 | 候选联合 | BM25（SQLite FTS5）+ 向量 pool 合并 |

**降级策略**：HNSW 索引损坏时自动回退到纯 SQLite BM25，不崩溃。

### 3.4 技术栈

| 层 | 技术 |
|---|---|
| 向量存储 | ChromaDB 1.5（可插拔 ABC 接口） |
| 嵌入 | ONNX 本地推理（`embeddinggemma-300m` 多语言 / `all-MiniLM-L6-v2` 英文） |
| 知识图谱 | SQLite（时序三元组，含有效窗口） |
| 硬件加速 | CUDA / DirectML / CoreML / CPU |
| LLM（可选） | Ollama / OpenAI / Anthropic（仅 init 阶段实体检测用） |
| MCP | 29 个工具，stdio JSON-RPC |
| 测试 | 80 个测试文件，85% 覆盖率 |

### 3.5 MCP 工具（29 个）

| 类别 | 工具数 | 示例 |
|------|:---:|------|
| 读取 | 8 | `mempalace_search`, `mempalace_get_drawer`, `mempalace_list_wings` |
| 写入 | 6 | `mempalace_add_drawer`, `mempalace_diary_write` |
| 知识图谱 | 5 | `mempalace_kg_query`, `mempalace_kg_timeline` |
| 图遍历 | 7 | `mempalace_traverse`, `mempalace_find_tunnels` |
| 系统 | 3 | `mempalace_hook_settings`, `mempalace_reconnect` |

### 3.6 Benchmark 成绩

| Benchmark | 指标 | 成绩 | LLM |
|-----------|------|------|:---:|
| LongMemEval (raw) | R@5 | **96.6%** | ❌ |
| LongMemEval (hybrid v4) | R@5 | 98.4% | ❌ |
| LongMemEval (hybrid + rerank) | R@5 | ≥99% | ✅ 任意模型 |
| LoCoMo (hybrid v5) | R@10 | 88.9% | ❌ |
| ConvoMem | Avg recall | 92.9% | ❌ |
| MemBench (ACL 2025) | R@5 | 80.3% | ❌ |

**关键**：96.6% 成绩不需要 API key、不需要云、不需要 LLM。纯本地完成。

### 3.7 关键启示

1. **Verbatim > 摘要** — MemPalace 证明逐字存储 + 混合检索在 benchmark 上远超 GPT 提取方案
2. **本地嵌入就够了** — embeddinggemma-300m 多语言模型，300MB，完全替代 text-embedding-3-small
3. **空间组织 > 语义分类** — wing/room/drawer 的空间隐喻比 core/episodic/semantic 更直观且可 scope
4. **SQLite 时序 KG** — 简单但足够，比 LightRAG 的复杂度低一个数量级
5. **29 MCP 工具** — 说明"让 Agent 自己管理记忆"比"把记忆藏在 API 后面"更灵活

---

## 4. OpenViking (VikingMem) — 字节跳动生产级记忆库

- **论文**: [arXiv:2605.29640](https://arxiv.org/abs/2605.29640) — VLDB 2026 已接收
- **代码**: [volcengine/OpenViking](https://github.com/volcengine/OpenViking) (AGPL-3.0)
- **注意**: `BytedanceFu/VikingMem` 只是 LoCoMo 测评脚本壳，真身在 `volcengine/OpenViking`
- **定位**: 工业级、多租户、事件-实体代数驱动的"上下文数据库"

### 4.1 核心创新

**① 文件系统即记忆** — `viking://user/memories/preferences/coding`，目录层级 + 语义搜索双路。

**② 事件-实体代数** — 可编程记忆模型：
```
entity := SELECT OP(event.content) FROM Events
          WHERE filters(event) GROUP BY keys(event)
```
6 个算子：SUM / COUNT / AVG / MAX / LLM_MERGE / TIME_COMPRESS。记忆设计从 prompt 工程变成声明式。

**③ 单次 LLM 提取** — k 种记忆类型只调 1 次 LLM（Schema 编译到同 prompt），比 MemVerse 3 次提取省 (k-1)× token。

**④ EUA Patch 更新** — Search/Replace 风格的实体增量更新，无需第二次 LLM 调用。

### 4.2 架构

```
Client Layer (Embedded / HTTP / CLI)
       │
Service Layer
  ├─ SessionService  → 两阶段提交 + 记忆提取
  ├─ SearchService   → 意图分析 + 层级递归检索 + Rerank
  ├─ ResourceService → 知识库管理
  └─ ...
       │
Storage Layer
  ├─ AGFS (内容存储, POSIX-like)
  └─ Vector Index (语义搜索, C++ 引擎)
```

### 4.3 8 类记忆 + 三层加载

| 类别 | 作用域 | 可合并 |
|------|--------|:---:|
| profile | 用户身份属性 | ✅ |
| preferences | 用户偏好 | ✅ |
| entities | 人物/项目 | ✅ |
| events | 事件决策 | ❌ |
| cases | 问题+方案 | ❌ |
| patterns | 可复用模式 | ✅ |
| tools | 工具知识 | ✅ |
| skills | 工作流策略 | ✅ |

L0 (~100 tokens) → L1 (~2k tokens) → L2 (unlimited)，按需加载。

### 4.4 关键指标

| 指标 | 数值 |
|------|------|
| 检索准确率提升 | 最高 38% vs baselines |
| 存储成本降低 | 83.2% vs naive RAG |
| P95 延迟优化 | 900ms 降低 |
| 生产规模 | 单租户 10 亿+ tokens/天 |

---

## 5. Hindsight — 仿生记忆（非 RAG、非 KG）

- **代码**: [vectorize-io/hindsight](https://github.com/vectorize-io/hindsight) (⭐15.3k, 🍴865)
- **论文**: [arXiv:2512.12818](https://arxiv.org/abs/2512.12818)（与 Virginia Tech + Washington Post 合著）
- **公司**: Vectorize.io（商业产品 + 开源 MIT）
- **版本**: v0.7.1 | Python + TypeScript + Rust

### 5.1 仿生记忆层级

```
Mental Models → 用户精选摘要（pinned，优先查询）
     ↑
Observations → 后台自动合成的高阶信念（proof_count + freshness_trend）
     ↑
Experiences → Agent 自身经历（"我碰了炉子，很疼"）
     +
World Facts  → 客观事实（"炉子会变热"）
```

**关键**：Observation Consolidation 是后台自动流程——多个 raw fact → 统计合成高阶信念。这是"学习"机制，不靠 LLM 反思，靠统计聚合。

### 5.2 TEMPR 四路并行检索

```
Query → ┬ Semantic (pgvector)    ┐
        ├ BM25 (pg_bm25)        ├─ RRF融合 → Cross-Encoder重排 → 结果
        ├ Graph (实体/因果链接)   │
        └ Temporal (时间范围过滤) ┘
```

四路同时跑，RRF 融合 + Cross-Encoder 重排。比 MemPalace 两路（BM25+向量）多两路，比 MemVerse 单路多三路。

### 5.3 CARA Reflection Agent

不是简单的 top-k 检索 → 注入 prompt。CARA 是一个**工具调用循环**：

```
Query → 查 Mental Models（精选摘要）
      → 查 Observations（合成信念，带 freshness）
      → 查 Raw Facts（原始事实，ground truth）
      → 按 Mission + Directives + Disposition 生成最终回答
```

**Disposition** 可配置（skepticism/literalism/empathy），控制 Agent 的回答风格。

### 5.4 Benchmark 表现（LongMemEval S 组）

| 系统 | 成绩 |
|------|------|
| Hindsight (Gemini-3 Pro) | **91.4%** |
| Hindsight (OSS-120B) | 89.0% |
| Supermemory (Gemini-3) | 85.2% |
| Hindsight (OSS-20B) | 83.6% |
| Zep (GPT-4o) | 71.2% |
| Full-context GPT-4o | 60.2% |

独立验证：Virginia Tech + Washington Post 复现。

### 5.5 关键启示

1. **Observation Consolidation** — 其他方案都没有的"学习"层，从 raw facts 自动合成高阶知识
2. **四路并行检索 > 两路/单路** — TEMPR 的 graph + temporal 路是差异化优势
3. **Bank 隔离模型** — 类似 OpenViking account，per-bank 独立 LLM 配置
4. **Per-operation LLM** — retain/recall/reflect/consolidation 各用不同模型，优化成本

### 5.6 TEMPR 四路检索详解

```
Query
  ├─ Semantic (pgvector HNSW, 5x 过采样, ef_search=200)
  ├─ BM25    (pg_bm25 / PostgreSQL tsvector / pg_search)
  ├─ Graph   (实体共现 + 语义kNN + 因果链, 可加性评分 ∈ [0,3])
  │            entity_score   = tanh(count × 0.5)
  │            semantic_score = max_knn_weight
  │            causal_score   = max_link_weight
  └─ Temporal (时间窗口 + 传播激活, BFS max 5 层)
       │
       ▼
  RRF融合: score(d) = Σ 1/(60 + rank(d))   (k=60)
       │
       ▼
  Cross-Encoder重排 × 三个boost:
  final = CE × (1+0.2(R-0.5)) × (1+0.2(T-0.5)) × (1+0.1(P-0.5))
  其中 R=recency(365天线性衰减), T=temporal proximity, P=proof_count(log归一化)
```

### 5.7 CARA 反思 Agent

强制层级工具调用循环（max 10 轮）：

```
Iter 0: 强制 search_mental_models → 精选摘要（最高质量）
Iter 1: 强制 search_observations → 合成信念（带 freshness_trend）
Iter 2: 强制 recall            → 原始事实（ground truth）
Iter 3+: auto 工具选择          → expand(获取上下文)
Iter 9:  强制最终合成            → 纯 LLM, 无工具
```

**反幻觉 guardrail**: 所有引用 ID 必须来自工具返回的 available_ids，不在集合内则静默删除。

### 5.8 流式 Retention 管线

```
Producer (32并发 LLM):
  chunk → LLM提取5维事实 → 生成嵌入 → enqueue

Queue (asyncio.Queue, maxsize=200)

Consumer (串行, 3阶段):
  Phase 1: 实体解析 (独立连接, 不在事务中, Trigram GIN)
  Phase 2: DB写入 (SELECT...FOR UPDATE 文档所有权门)
           - INSERT memory_units, unit_entities
           - 创建 temporal/semantic/causal links
  Phase 3: 最终 ANN 遍历 (post-commit, best-effort)
```

### 5.9 API 表面

| 类别 | 数量 | 关键 |
|------|:---:|------|
| REST 端点 | **55+** | retain/recall/reflect + banks/entities/mental-models/documents/webhooks |
| MCP 工具 | **30** | retain, sync_retain, recall, reflect + CRUD |
| 可配置字段 | ~30/bank | disposition(skepticism/literalism/empathy), missions, chunk策略 |

### 5.10 数据库架构

PostgreSQL (pgvector HNSW) + 可配置全文搜索:

| 表 | 用途 |
|---|---|
| `memory_units` | 主记忆存储 (384d向量 + tsvector/BM25) |
| `memory_links` | 图谱边 (temporal/semantic/entity/causal) |
| `entities` | 规范实体 (Trigram模糊匹配) |
| `unit_entities` | 记忆↔实体 M2M |
| `entity_cooccurrences` | 共现统计 |
| `observation_sources` | 合成信念溯源 |
| `mental_models` | 精选摘要 (可刷新的活文档) |
| `async_operations` | 异步任务追踪 |

---

## 6. Supermemory — 全托管 TypeScript 记忆引擎

- **代码**: [supermemoryai/supermemory](https://github.com/supermemoryai/supermemory) (⭐23.1k, 🍴2.1k)
- **语言**: TypeScript 63% / Python 6.3%
- **部署**: Cloudflare Workers + Drizzle ORM（SaaS 优先）

### 6.1 核心卖点

- **User Profiles 一等公民**：`profile.static`(长期事实) + `profile.dynamic`(近期动态)，~50ms
- **自动遗忘**：临时事实过期消失，矛盾自动解决
- **Hybrid Search**：RAG(知识库) + Memory(个性化) 合并单次查询
- **Connectors**：Google Drive / Gmail / Notion / OneDrive / GitHub 实时同步
- **MemoryBench**：自建标准化对比框架
- **Benchmark**: LongMemEval #1, LoCoMo #1, ConvoMem #1

### 6.2 关键启示

1. **User Profile > 原始检索** — 预计算画像比每次检索更高效（~50ms vs 秒级）
2. **自动遗忘是刚需** — 所有方案中只有 Supermemory 原生支持记忆过期
3. **全托管降低门槛** — npm install 直接用，但牺牲了本地/自托管灵活性
4. **TypeScript 生态** — 如果目标用户是 JS/TS 开发者，这是最友好的方案
5. **不是开源框架** — 核心引擎是闭源 SaaS，开源的是 SDK + 插件

---

## 7. 六方案横向对比

| 维度 | Supermemory | Hindsight | Mem0 | OpenViking | MemPalace | MemVerse |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 成熟度 | ⭐⭐⭐⭐⭐ 商业 | ⭐⭐⭐⭐⭐ 商业 | ⭐⭐⭐⭐⭐ 产品 | ⭐⭐⭐⭐⭐ 工业 | ⭐⭐⭐⭐⭐ 社区 | ⭐⭐ 原型 |
| 语言 | **TypeScript** | Python | Python | Python+Rust | Python | Python |
| 部署 | Cloudflare SaaS | Docker/嵌入PG | Lib/Server/Cloud | Docker/K8s | CLI 本地 | Docker |
| 存储 | 云端 (Postgres?) | PostgreSQL | Qdrant(30种) | AGFS+向量 | ChromaDB | JSONL+图谱 |
| 检索 | RAG+Memory混合 | TEMPR 4路 | 3信号融合 | 目录递归 | BM25+向量 | 图谱 RAG |
| 画像 | **~50ms Profile** | ❌ | user/session/agent | 8类实体 | ❌ | ❌ |
| 遗忘 | ✅ **自动过期** | ❌ | ❌ | ❌ | ❌ | ❌ |
| 自托管 | ❌ SaaS优先 | ✅ | ✅ | ✅ | ✅ | ✅ |
| 开源 | MIT (SDK) | MIT | Apache 2.0 | AGPL-3.0 | MIT | MIT |
| LongMemEval | **#1** (81.6%) | 91.4%* | 94.8% | — | 96.6%† | — |
| Stars | **23.1k** | 15.3k | 57.2k | — | 53.1k | 144 |

---

## 8. 设计方向总结

基于五方分析，一个更好的长期记忆系统应该：

| 维度 | 采用方案 | 来源 |
|------|---------|------|
| 记忆存储 | **Verbatim 原文 + 结构化提取双轨** | MemPalace + Hindsight |
| 记忆载体 | PostgreSQL (pgvector) + 虚拟文件系统层级 | Hindsight + OpenViking |
| 记忆模型 | **仿生层级**（World→Experience→Observation→Mental Model） | Hindsight |
| 学习机制 | **Observation Consolidation** 自动合成高阶信念 | Hindsight |
| 提取策略 | **单次 LLM 批量提取**（Schema 编译到同 prompt） | OpenViking |
| 检索策略 | **TEMPR 多路并行**（Semantic+BM25+Graph+Temporal）→ RRF融合 | Hindsight |
| 检索实现 | 混合 BM25 + 向量 + 图谱 + 时序 + 目录递归 | Hindsight + OpenViking + MemPalace |
| 反思 | **CARA Agent** 工具调用循环（Mission+Directives+Disposition） | Hindsight |
| 更新机制 | **Patch 算法增量更新**（EUA），避免全量重处理 | OpenViking |
| 嵌入模型 | ONNX 本地推理（embeddinggemma-300m）+ pgvector | MemPalace + Hindsight |
| 空间组织 | 虚拟文件系统层级（viking:// URI） | OpenViking |
| 摘要/提取 | **可选后处理**，不阻塞检索路径 | MemPalace + Mnemis |
| 多模态 | 统一文本化 pipeline（含文档解析: PDF/DOCX/代码） | OpenViking + MemVerse |
| 接口 | MCP 工具集（30 工具，Agent 自主管理） | MemPalace + Hindsight |
| 多租户 | account/user/agent + Bank 隔离 | OpenViking + Hindsight |
| 隐私 | 写入时占位符替换 + 读取时恢复 | OpenViking |
| 部署 | Docker + 嵌入式模式（pg0） | Hindsight |

---

## 9. 本地 MemVerse 代码修改记录

### 修改文件：`MemVerse/orchestrator.py`

**`rag_retrieve()`** — 单通道 → 三通道并行：

```python
async def rag_retrieve(query, mode="hybrid"):
    core_result, epi_result, sem_result = await asyncio.gather(
        mem_core.aquery(query, param=QueryParam(mode=mode)),
        mem_epi.aquery(query, param=QueryParam(mode=mode)),
        mem_sem.aquery(query, param=QueryParam(mode=mode)),
        return_exceptions=True,
    )
    return {"core": ..., "episodic": ..., "semantic": ...}
```

**`handle_query()`** — 适配新的 dict 格式，按 `[Core Memory]` / `[Episodic Memory]` / `[Semantic Memory]` 分段注入 prompt。

---

## 10. 环境信息

```bash
# 项目路径
/memory/MemVerse/

# Python 环境
python3 -m venv venv
source venv/bin/activate  # Python 3.14.4

# 关键依赖
fastapi==0.136.3
openai==2.38.0
networkx==3.6.1
numpy==2.4.6

# 启动
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# 需要环境变量
OPENAI_API_KEY=...
OPENAI_API_BASE=...
```

---

## 11. 嵌入模型选择: all-MiniLM-L6-v2 vs bge-large-en-v1.5

### 背景

V1/V2 使用 `all-MiniLM-L6-v2`（384-dim, 256-token），在 Ablation 测试中向量检索 Hits@10 = **0%**。
根因是 256-token 上下文限制导致长消息被截断，短 query 无法在 HNSW top-10 中命中。

V3 升级为 `bge-large-en-v1.5`（1024-dim, 512-token），目标是恢复向量检索信号。

### 技术对比

| 维度 | all-MiniLM-L6-v2 | bge-large-en-v1.5 | 差值 |
|------|------------------|-------------------|------|
| 模型家族 | Sentence-BERT (MiniLM) | BERT (large) | — |
| 参数量 | 22M | **335M** | +313M (15×) |
| 输出维度 | 384 | **1024** | +640 (2.67×) |
| Token 上限 | 256 | **512** | +256 (2×) |
| ONNX 模型大小 | ~80 MB | **~1.3 GB** | +1.22 GB |
| MTEB 评分 | 61.0 | **64.0** | +3.0 |
| 推理速度 (CPU) | ~2ms/文本 | ~20-50ms/文本 | 更慢 |
| 硬件需求 | CPU 即可 | **GPU 推荐 (CUDA)** | 更高 |
| 运行时精度 | FP32 | FP32 | 相同 |
| 池化策略 | mean pooling | mean pooling + L2 norm | bge 额外标准化 |

### 为什么选 bge-large-en-v1.5

| 因素 | 考量 | 决策 |
|------|------|------|
| 256-token 瓶颈 | V1 ablation 表明向量 Hits@10 = 0% 的根本原因 | 512-token 覆盖绝大多数消息，减少分块依赖 |
| 384-dim 稀疏 | 低维向量在 HNSW 中区分度不足，短 query 难命中 | 1024-dim 提供更丰富的语义表达空间 |
| 本地部署 | 必须纯本地 ONNX 推理，不能依赖 API | bge-large 官方提供 ONNX 导出 |
| GPU 可用 | 部署环境有 CUDA 支持 | CUDAExecutionProvider 加速推理 |
| 社区成熟度 | MTEB 榜单验证、HuggingFace 高频下载、ONNX 导出方案成熟 | 低风险选择 |
| 备选淘汰 | bge-m3 (1024-dim, 多语言) 对纯英文场景过重；gte-large 无官方 ONNX | bge-large-en-v1.5 最平衡 |

### 为什么不继续用 all-MiniLM

1. **256-token 硬限制不可绕过** — 文本分块 (V2) 是缓解措施，但分块后的短语义单元仍然对短 query 不友好
2. **384-dim 的表达容量不足** — 在长消息 (500-3000 chars) 场景下，大量信息被压缩到 384 个浮点数中，相似度信号被稀释
3. **22M 参数的学习容量** — 无法捕捉复杂的技术术语和上下文关联

### 为什么没选更大的模型

| 模型 | 维度 | Token | 大小 | 理由 |
|------|------|-------|------|------|
| bge-large-en-v1.5 | 1024 | 512 | ~1.3 GB | ✅ 选定 |
| bge-m3 | 1024 | 8192 | ~2.2 GB | 多语言、超长上下文，对纯英文场景过重 |
| instructor-xl | 768 | 512 | ~1.5 GB | 需 instruction prefix，增加查询复杂度 |
| gte-large | 1024 | 512 | ~1.4 GB | ONNX 导出方案不成熟 |
| nomic-embed-text-v1 | 768 | 8192 | ~700 MB | MTEB 评分略低 (62.4) |

### 部署决策

```python
# ONNX 推理 providers 列表 — GPU 优先，CPU 回退
self._session = ort.InferenceSession(
    onnx_file,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
```

- **GPU (CUDA)**: 预期 batch embedding 吞吐 ≥100 msg/s
- **CPU 回退**: 无 GPU 时自动降级，推理速度 ~20-50ms/次（单文本）
- **Fallback**: ONNX 模型文件不存在时，SHA-256 deterministic fallback 确保 test-safe

---

## 12. Letta (formerly MemGPT) — 结构化记忆块 + Agent 自主记忆管理

- **代码**: [letta-ai/letta](https://github.com/letta-ai/letta) (⭐23.1k, Apache-2.0)
- **原项目**: cpacker/MemGPT (arXiv 2310.08560)
- **定位**: 完整 Agent 框架，Agent 通过工具调用**主动**管理自己的记忆

### 12.1 核心理念对比

| | Letta | memory-skill |
|------|------|------|
| 记忆管理 | **主动**：Agent 调 `core_memory_append` / `core_memory_replace` | **被动**：系统自动 weave 注入 |
| 记忆结构 | 结构化 Block（value/label/limit/description/read_only） | 扁平 MemoryEntry + category |
| LLM 依赖 | 全程需要（提取、总结、反思都用 LLM） | 零 LLM 核心路径 |
| 上下文管理 | 按字符限制 + 超限自动总结 | 按 tiers 分层注入 + token budget |
| Agent 类型 | 9 种不同工具集的 Agent 类型 | 无区分，统一接口 |

### 12.2 可借鉴但需保持"无知觉"的模式

以下是 Letta 的优良设计，但采用时必须确保 Agent 不知道记忆系统的存在：

**① Heartbeat 自省机制**（强烈推荐）
```
Letta: Agent 在工具调用时设 request_heartbeat=true → 系统自动触发新一轮思考
我们: weave 上下文注入后，Agent 自然产生回复 → 系统在回复后再注入一次上下文
→ Agent 感知：就是自己在"多想了一下"，不是系统在推记忆
```
- 不需要 Agent 主动设置 heartbeat
- 在 weave 注入后，如果检索到高权重相关记忆（weight > 0.7），系统自动追加一轮注入

**② 结构化记忆块**（可选采纳）
```
Letta: <memory_blocks><human>...</human><persona>...</persona></memory_blocks>
我们: "关于 partner，你记得: ..."（已经是自然语言，不需要 XML 标签）
→ 保持当前自然语言格式，不引入 XML 标签
→ 但可以借鉴 description + chars_current 的元数据思路，作为内部字段不暴露给 Agent
```

**③ 分层 Agent 类型的工具集**（推荐）
```
Letta: 不同 AgentType 有不同的工具集（core 工具 / sleep 工具 / voice 工具）
我们: 可以根据场景提供不同深度的注入
  - compact 模式：只注 tier1（最近对话），适合高频简短对话
  - standard 模式：tier1 + tier2 + nudge（当前默认）
  - deep 模式：tier1 + tier2 + nudge + observations + archival search
```
- Agent 不知道模式存在，只是收到的上下文长度不同

**④ 读保护标记**（推荐）
```
Letta: Block.read_only = True → Agent 不能编辑 system 管理的记忆
我们: MemoryEntry 加 is_system 标记 → system 注入的记忆不会被 Agent 的自我修正覆盖
```
- 纯内部机制，Agent 无感知

### 12.3 不可采纳的模式（会破坏无知觉）

| Letta 模式 | 为什么不能要 |
|------|------|
| Agent 调 `core_memory_append("人设", "记住：用户叫 Alice")` | Agent 知道自己有记忆系统 |
| 系统 prompt 里出现 `<memory_blocks>` 标签 | 暴露了记忆机制 |
| Agent 看到 `156 total memories stored in archival memory` | Agent 知道自己在用数据库 |
| 系统 prompt 解释 heartbeat 机制 | Agent 知道系统在自动触发思考 |

### 12.4 适合我们且不破坏无知觉的补充

| 灵感 | 实现方式 | Agent 感知 |
|------|------|:--:|
| Heartbeat 自省 | weave 后自动追加一轮注入（当高权重记忆存在时） | "我多想了下" |
| 分层深度 | compact/standard/deep 三种 weave 深度，按场景自动选择 | 上下文长短不同 |
| 读保护 | `MemoryEntry.is_system = True` — Agent 无法覆盖系统注入 | 无感知 |
| 上下文压力警告 | 当 token 超过阈值，在 weave 中优先注入最重要记忆 | 自然行为 |
| 记忆向量化总结 | consolidate 输出更结构化（目标/事件/细节/状态/下一步） | 总结质量更高 |
