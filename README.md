# Memory Skill

**为 AI Agent 打造的长期记忆插件**（中文为主，中英双语可用）— 本地优先、双模型记忆、可自我进化。

为 AI Agent（Claude / OpenAI / 自研 LLM）提供持久化的长期记忆：每次对话自动存取，检索时注入相关记忆上下文，检测知识缺口后主动学习。**零 API 检索**（本地向量检索），LLM 仅用于增强（分类/合成）。

> **语言支持**：中英双语均可存取，检索信号各有侧重——中文由 BM25（jieba 分词）主导，英文由语义向量（bge-large-en-v1.5）主导。插件本身语言无关，中文/英文对话都能自动记忆。

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
| BM25 全文 | 2.5 | SQLite FTS5，jieba 中文分词（中文主导） |
| 语义向量 | 0.5 | ChromaDB，bge-large-en-v1.5 (1024-dim)（英文主导） |
| 时间衰减 | 0.5 | `weight × exp(-0.01 × hours)` |

> **语言说明**：检索是 RRF 融合——中文内容主要靠 BM25（jieba 对中文分词准确），英文内容主要靠语义向量（bge-large-en-v1.5 是英文专用模型）。两路互补：中文记忆靠 BM25 召回，英文记忆靠语义召回，均可在同库中检索。若需单模型统一中英语义检索，可替换为多语言嵌入模型（如 bge-m3，需重新嵌入历史记忆）。

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

## 使用教程（从零到会用）

> 下面以 **OpenCode + MCP 方式** 为例，展示完整接入流程。其他 Agent（Claude Code / Cursor）流程相同，只是配置文件名不同。

### 第 1 步：下载并安装

```bash
git clone <your-repo-url> memory-skill
cd memory-skill

# 创建虚拟环境并安装依赖（含 ONNX 嵌入）
python3 -m venv venv
source venv/bin/activate
pip install -e ".[onnx]"

# 下载嵌入模型（bge-large-en-v1.5 → models/）
./download_model.sh
```

### 第 2 步：配置密钥

```bash
cp .env.example .env
# 编辑 .env，填入 LLM API Key（用于记忆分类/合成/学习）
# IMPORTANCE_API_KEY=sk-xxx
```

### 第 3 步：把 SKILL.md 交给 Agent

`SKILL.md` 是 Agent 的记忆使用协议——把它放进你的 Agent 知识库，或在配置中引用：

- **OpenCode**: 放到项目根（Agent 自动读取 `AGENTS.md`/技能目录），或通过 `prompt_append` 注入协议
- **Claude Code**: 放入 `CLAUDE.md` 引用，或作为 skill 文件
- **Cursor**: 放入 `.cursor/rules/` 或项目 rules

协议核心（SKILL.md 全文见仓库）：

```
BEFORE responding:   memory_weave(user_message)   → 注入记忆上下文
AFTER 重要交互:      memory_ingest(role, content) → 存入记忆
需要更多时:          memory_search(query)         → 深度检索
会话开始:            memory_status                 → 健康检查
```

### 第 4 步：配置 MCP 服务器

在 Agent 的 MCP 配置中注册（OpenCode 示例）：

```json
{
  "mcp": {
    "opencode-memory": {
      "type": "local",
      "command": ["/abs/path/memory-skill/venv/bin/python", "-m", "memory_skill.mcp_server"],
      "environment": {
        "MEMORY_SKILL_DB_PATH": "/abs/path/memory-skill/memory.db",
        "IMPORTANCE_API_KEY": "sk-xxx"
      }
    }
  }
}
```

> ⚠️ 注意：`command` 必须用 **venv 的绝对路径 python**，并先 `pip install -e .`（否则 `memory_skill` 包无法 import）。详见 [docs/INTEGRATION.md](docs/INTEGRATION.md)。

### 第 5 步：重启 Agent 并验证

重启 Agent 会话，让 Agent 调用记忆工具：

```
# Agent 应能看到并调用这些工具：
memory_search / memory_weave / memory_ingest / memory_status
memory_feedback / memory_gaps / memory_learn
```

**快速验证**：让 Agent 说一句重要信息（如"我偏好用 Python 写后端"），重启会话后再问它——如果它还记得，说明记忆已生效。

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

### 方式 4：自动记忆插件（推荐，agent 零感知）

```json
{
  "plugin": ["/abs/path/to/solo-memory/opencode-auto-memory"]
}
```

`chat.message` 自动注入记忆、`event` 自动存储——agent 不需要记得调任何工具。详见 `opencode-auto-memory/README.md`。

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

