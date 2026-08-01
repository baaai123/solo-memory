# Memory Skill

**面向中文 AI Agent 的长期记忆插件** — 本地优先、双模型记忆、可自我进化。

为 AI Agent（Claude / OpenAI / 自研 LLM）提供持久化的长期记忆：每次对话自动存取，检索时注入相关记忆上下文，检测知识缺口后主动学习。**零 API 检索**（本地向量检索），LLM 仅用于增强（分类/合成）。

> 面向中文优化：jieba 中文分词 BM25 + 中文提示词 + 中英双语嵌入（bge-large-en-v1.5）。

---

## 特性

| 特性 | 说明 |
|---|---|
| **两半记忆模型** | 非结构化对话（user_mem）+ 结构化知识（pref/pers/skill/mission） |
| **自动存取** | weave 自动注入上下文；透明代理下 Agent 零改动 |
| **主动检索** | Agent 引用记忆标题 → 自动展开为完整上下文 |
| **主动学习** | 知识缺口检测 → 爬取 → 合成 → 验证闭环 |
| **反馈演化** | 记忆权重随使用自动演化（去重+0.05 / 引用+0.02 / 反馈+0.05） |
| **三层注入** | tier1 场景感知 + tier2 对话片段 + nudge 高优记忆 |
| **透明接入** | MCP 工具 / OpenAI 兼容代理 / Python API 三通道 |

---

## 架构

```
┌────────────────────────────── Agent 层 ──────────────────────────────┐
│  MCP 工具 (7个)    透明代理 (auto_context)     Python API            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌─────────────────────────── MemorySystem ────────────────────────────┐
│  IngestPipeline           Weaver (8区块)        RetrievalCoordinator │
│  │  ingest_dialogue       │  tier1/tier2       │  RRF 三信号融合     │
│  │  tag_title (LLM)       │  nudge/gap/emotion │  BM25 ×2.5          │
│  └  extract_structured    └  树导航/skill/mission ── semantic ×0.5  │
│                                                      temporal ×0.5   │
├──────────────────────────────────────────────────────────────────────┤
│  SQLite FTS5 (jieba 中文分词)      ChromaDB (1024-dim 向量)          │
│  SawRingBuffer (短期观察)          TreeManager (记忆树导航)           │
│  GapDetector → LearningDecider → WebCrawler → KnowledgeSynth         │
└──────────────────────────────────────────────────────────────────────┘
```

### 数据流

```
对话 → Ingestor → [SQLite 对话库] + [ChromaDB 向量库] + [记忆树]
                      ↓                             ↓
                    FTS5 BM25                  向量检索
                      └────────┬──────────────────┘
                               ↓
                      RRF 融合 (k=60)
                               ↓
                     Weaver 组装 8 区块上下文
                               ↓
                      注入 Agent 提示词
```

### 检索信号（RRF 融合）

| 信号 | 权重 | 来源 |
|---|---|---|
| BM25 全文 | 2.5 | SQLite FTS5，jieba 中文分词 |
| 语义向量 | 0.5 | ChromaDB，bge-large-en-v1.5 (1024-dim) |
| 时间衰减 | 0.5 | `weight × exp(-0.01 × hours)` |

### 7 个 MCP 工具

| 工具 | 用途 |
|---|---|
| `memory_search` | 检索记忆（RRF 融合） |
| `memory_weave` | 注入三层记忆上下文（含自动存取） |
| `memory_ingest` | 存储对话 |
| `memory_status` | 健康检查 |
| `memory_feedback` | 反馈权重演化 |
| `memory_gaps` | 查看知识缺口 |
| `memory_learn` | 爬取学习闭环 |

---

## 安装

### 依赖

| 依赖 | 用途 | 必需 |
|---|---|---|
| `chromadb` | 向量存储 | ✅ |
| `numpy` | 向量运算 | ✅ |
| `jieba` | 中文分词（BM25） | ✅ |
| `mcp` | MCP 服务器 | ✅（工具模式） |
| `click` | CLI | ✅ |
| `pydantic` / `tenacity` / `openai` / `requests` | LLM 调用 | ✅ |
| `python-dotenv` | 环境变量 | ✅ |
| `onnxruntime` + `tokenizers` | ONNX 嵌入 | ⚠️ 可选（缺则 SHA-256 fallback，检索精度大幅下降） |
| `llama-cpp-python` | 本地 LLM（查询改写/自动反馈） | ⚠️ 可选 |

```bash
# 基础安装
pip install -e .                # 核心（含 mcp/jieba）
pip install -e ".[onnx]"        # 加 ONNX 嵌入（推荐，检索精度关键）
pip install -e ".[full]"        # 全部（ONNX + 本地 LLM）
# 或直接
pip install -r requirements.txt
```

### 下载嵌入模型

```bash
./download_model.sh             # 下载 bge-large-en-v1.5 → models/
```

### 配置环境变量

复制 `.env.example` 为 `.env` 并填入：

```
IMPORTANCE_API_KEY=sk-xxx       # LLM 分类/合成用（DeepSeek 等）
MEMORY_SKILL_DB_PATH=memory.db  # 数据库路径
MEMORY_MODEL_PATH=models/bge-large-en-v1.5
```

---

## 快速开始

### 方式 1：透明代理（Agent 零改动）

```bash
DEEPSEEK_API_KEY=sk-xxx ./start.sh --port 8888

# Agent 设置
export OPENAI_API_BASE=http://127.0.0.1:8888/v1
```

每次 chat 请求自动注入记忆、响应自动存回——Agent 完全不感知记忆系统。

### 方式 2：MCP 工具（OpenCode / Claude Code 等）

```json
{
  "mcp": {
    "opencode-memory": {
      "type": "local",
      "command": ["/abs/path/venv/bin/python", "-m", "memory_skill.mcp_server"],
      "environment": {
        "MEMORY_SKILL_DB_PATH": "/abs/path/opencode_memory.db",
        "IMPORTANCE_API_KEY": "sk-xxx"
      }
    }
  }
}
```

详见 [docs/INTEGRATION.md](docs/INTEGRATION.md)。

### 方式 3：Python API

```python
from memory_skill import MemorySkill, MemorySkillConfig, DialogueTurn

skill = MemorySkill(MemorySkillConfig(db_path="memory.db"))

# 存储对话
skill.ingest(DialogueTurn(role="user", content="我推荐使用 FastAPI", ...))

# 注入记忆上下文
ctx = skill.weave("FastAPI 是什么？")
print(ctx.to_prompt_block())

# 主动检索
skill.expand("FastAPI")

# 主动学习
skill.learn("Docker", ["https://docs.docker.com/..."])
```

---

## 文档

| 文档 | 内容 |
|---|---|
| [SKILL.md](SKILL.md) | Agent 使用协议（8 区块 weave） |
| [COMPREHENSIVE.md](COMPREHENSIVE.md) | 完整架构设计 |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | OpenCode / Cursor / 代理接入指南 |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | 记忆协议与工具规范 |
| [CHANGELOG.md](CHANGELOG.md) | 版本历史 |

---

## 性能

| 指标 | 数值 |
|---|---|
| 中文检索精度 | 93%（300 条记忆） |
| 检索延迟 | 35-100ms |
| 测试 | 46 快速（0.07s 纯内存）+ 65 集成 |

---

## License

[Apache License 2.0](LICENSE)

