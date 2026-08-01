# Memory Skill

[English](README.en.md) | [中文](README.md)

**Long-term memory plugin for AI Agents** — local-first, dual-signal retrieval, self-evolving. Bilingual (English & Chinese), with a first-class experience for Chinese conversations.

Gives AI Agents (Claude / OpenAI / your own LLM) persistent long-term memory: every conversation is stored automatically, relevant memories are injected into context on retrieval, and knowledge gaps trigger self-directed learning. **Zero API retrieval** — all search happens locally; the LLM is used only for enhancement (classification / synthesis).

> **Language support**: Both English and Chinese can be stored and retrieved, with retrieval signals weighted differently per language — Chinese is dominated by BM25 (jieba tokenization), English by semantic vectors (bge-large-en-v1.5). The plugin itself is language-agnostic; conversations in either language are remembered automatically.

---

## Features

| Feature | Description |
|---|---|
| **Two-Half Memory Model** | Unstructured dialogue (user_mem) + structured knowledge (pref/pers/skill/mission) |
| **Automatic Read/Write** | `weave` injects context automatically; zero agent changes under the transparent proxy |
| **Proactive Retrieval** | Agent references a memory title → auto-expands into full context |
| **Self-Directed Learning** | Knowledge-gap detection → crawl → synthesize → verify closed loop |
| **Feedback-Driven Evolution** | Memory weights evolve with usage (dedup +0.05 / reference +0.02 / feedback +0.05) |
| **Three-Tier Injection** | tier1 scenario awareness + tier2 dialogue snippets + nudge high-priority memories |
| **Transparent Integration** | MCP tools / OpenAI-compatible proxy / Python API — three channels |

---

## Architecture

```
┌────────────────────────────── Agent Layer ─────────────────────────────┐
│  MCP Tools (7)   Transparent Proxy (auto_context)   Python API         │
└──────────────────────────────┬────────────────────────────────────────┘
                               │
┌─────────────────────────── MemorySystem ───────────────────────────────┐
│  IngestPipeline           Weaver (8 blocks)        RetrievalCoordinator│
│  │  ingest_dialogue       │  tier1/tier2          │  RRF 3-signal fusion│
│  │  tag_title (LLM)       │  nudge/gap/emotion    │  BM25 ×2.5          │
│  └  extract_structured    └  tree-nav/skill/mission ── semantic ×0.5   │
│                                                      temporal ×0.5      │
├────────────────────────────────────────────────────────────────────────┤
│  SQLite FTS5 (jieba Chinese tokenization)      ChromaDB (1024-dim vec) │
│  SawRingBuffer (short-term observation)        TreeManager (memory nav)│
│  GapDetector → LearningDecider → WebCrawler → KnowledgeSynth           │
└────────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
conversation → Ingestor → [SQLite dialogue store] + [ChromaDB vector store] + [memory tree]
                      ↓                             ↓
                    FTS5 BM25                  vector search
                      └────────┬──────────────────┘
                               ↓
                      RRF fusion (k=60)
                               ↓
                     Weaver assembles 8-block context
                               ↓
                     Injected into agent prompt
```

### Retrieval Signals (RRF Fusion)

| Signal | Weight | Source |
|---|---|---|
| BM25 full-text | 2.5 | SQLite FTS5, jieba Chinese tokenization (dominates Chinese) |
| Semantic vector | 0.5 | ChromaDB, bge-large-en-v1.5 (1024-dim) (dominates English) |
| Temporal decay | 0.5 | `weight × exp(-0.01 × hours)` |

> **A note on languages**: Retrieval is RRF fusion — Chinese content is recalled primarily by BM25 (jieba tokenizes Chinese accurately), English content primarily by semantic vectors (bge-large-en-v1.5 is an English-specific model). The two paths are complementary: Chinese memories are recalled via BM25, English memories via semantic recall, both searchable in the same store. If you want a single model for unified bilingual semantic retrieval, swap in a multilingual embedding model (e.g. bge-m3) — this requires re-embedding existing memories.

### The 7 MCP Tools

| Tool | Purpose |
|---|---|
| `memory_search` | Search memories (RRF fusion) |
| `memory_weave` | Inject three-tier memory context (includes automatic read/write) |
| `memory_ingest` | Store a conversation |
| `memory_status` | Health check |
| `memory_feedback` | Feedback-driven weight evolution |
| `memory_gaps` | Inspect knowledge gaps |
| `memory_learn` | Crawl-learn closed loop |

---

## Installation

### Dependencies

| Dependency | Purpose | Required |
|---|---|---|
| `chromadb` | Vector store | ✅ |
| `numpy` | Vector math | ✅ |
| `jieba` | Chinese tokenization (BM25) | ✅ |
| `mcp` | MCP server | ✅ (tool mode) |
| `click` | CLI | ✅ |
| `pydantic` / `tenacity` / `openai` / `requests` | LLM calls | ✅ |
| `python-dotenv` | Environment variables | ✅ |
| `onnxruntime` + `tokenizers` | ONNX embeddings | ⚠️ optional (falls back to SHA-256 hashing — retrieval quality drops significantly) |
| `llama-cpp-python` | Local LLM (query rewriting / auto-feedback) | ⚠️ optional |

```bash
# Basic install
pip install -e .                # core (incl. mcp/jieba)
pip install -e ".[onnx]"        # + ONNX embeddings (recommended — critical for retrieval quality)
pip install -e ".[full]"        # everything (ONNX + local LLM)
# or directly
pip install -r requirements.txt
```

### Download the Embedding Model

```bash
./download_model.sh             # downloads bge-large-en-v1.5 → models/
```

### Configure Environment Variables

Copy `.env.example` to `.env` and fill in:

```
IMPORTANCE_API_KEY=sk-xxx       # for LLM classification/synthesis
MEMORY_SKILL_DB_PATH=memory.db  # database path
MEMORY_MODEL_PATH=models/bge-large-en-v1.5
```

#### LLM Model Configuration (defaults to DeepSeek V4 Flash; any OpenAI-compatible model works)

The system calls an LLM through an OpenAI-compatible interface (for memory classification / synthesis / learning). **It defaults to DeepSeek V4 Flash, but you can use any OpenAI-compatible model or service** — just change 3 environment variables:

```bash
IMPORTANCE_API_BASE=https://api.deepseek.com/v1   # API base (OpenAI-compatible)
IMPORTANCE_API_KEY=sk-xxx                          # your key
IMPORTANCE_MODEL=deepseek-v4-flash                 # model name

# Example: switch to OpenAI
# IMPORTANCE_API_BASE=https://api.openai.com/v1
# IMPORTANCE_MODEL=gpt-4o-mini

# Example: switch to a local vLLM / Ollama
# IMPORTANCE_API_BASE=http://127.0.0.1:8000/v1
# IMPORTANCE_MODEL=qwen2.5-7b-instruct
```

> Compatible with any service exposing `/v1/chat/completions` (OpenAI, Qwen, GLM, Moonshot, local vLLM, etc.). The defaults are tuned for DeepSeek V4 Flash (e.g. reserved `max_tokens`); if you switch models and classification/synthesis results look off, adjust the `IMPORTANCE_*` parameters.

---

## Usage Guide (from zero to production)

### Option A: Let the AI install it (fastest, recommended)

Hand the repo URL to your AI Agent and tell it:

```
Install https://github.com/baaai123/solo-memory and connect it to my OpenCode.

Steps:
1. git clone https://github.com/baaai123/solo-memory
2. Run ./setup.sh (creates venv + installs dependencies + configures embedding model)
3. Register the opencode-auto-memory plugin in opencode.json
4. Fill in my own IMPORTANCE_API_KEY in .env (use my own LLM API key)

Note: ./setup.sh does the whole environment setup in one shot; the opencode-auto-memory
plugin auto-injects memory context and auto-stores conversations — the Agent never has
to call memory tools manually.
```

The AI will autonomously complete clone → environment setup → plugin registration. The only thing you do is put **your own LLM API key** in `.env` (used for memory classification/synthesis/learning, billed to your own API account).

> **Why this works**: `setup.sh` encapsulates environment setup; the `opencode-auto-memory` plugin includes first-run bootstrapping (auto-creates venv if missing). The only manual step is providing an API key — no memory system can hold your private key for you.

### Option B: Manual Installation (step by step)

> The walkthrough below uses **OpenCode + the auto-memory plugin** as an example. Other Agents (Claude Code / Cursor) follow the same flow, just with a different config filename.

#### Step 1: Download and install

```bash
git clone https://github.com/baaai123/solo-memory
cd solo-memory

# One-shot environment setup (venv + dependencies + embedding model)
./setup.sh

# or manually:
# python3 -m venv venv && source venv/bin/activate && pip install -e ".[onnx]"
# ./download_model.sh   # bge-large-en-v1.5 → models/
```

#### Step 2: Configure your key

```bash
cp .env.example .env
# edit .env, fill in the LLM API Key (used for memory classification/synthesis/learning)
# IMPORTANCE_API_KEY=sk-xxx
```

#### Step 3: Hand SKILL.md to the Agent

`SKILL.md` is the Agent's memory protocol — put it in your Agent's knowledge base, or reference it in your configuration:

- **OpenCode**: place it at the project root (the Agent reads `AGENTS.md`/skills automatically), or inject the protocol via `prompt_append`
- **Claude Code**: reference it from `CLAUDE.md`, or use it as a skill file
- **Cursor**: put it in `.cursor/rules/` or the project rules

The core protocol (full SKILL.md lives in the repo):

```
BEFORE responding:   memory_weave(user_message)   → inject memory context
AFTER  key exchange: memory_ingest(role, content) → store to memory
Need more:           memory_search(query)         → deep retrieval
Session start:       memory_status                 → health check
```

#### Step 4: Register the auto-memory plugin

Add the plugin path to the `plugin` array in `~/.config/opencode/opencode.json`:

```json
{
  "plugin": [
    "/abs/path/to/solo-memory/opencode-auto-memory"
  ]
}
```

The plugin auto-injects memory context (via the `chat.message` hook) and auto-stores conversations (via the `event` hook) — the Agent never has to call tools manually.

> For the MCP tool mode (manually calling `memory_search` etc.), see [Quick Start → Option 2](#option-2mcp-tools-opencode--claude-code-etc) below.

#### Step 5: Restart the Agent and verify

Restart the Agent session so it picks up the memory tools:

```
# The Agent should now see and be able to call these tools:
memory_search / memory_weave / memory_ingest / memory_status
memory_feedback / memory_gaps / memory_learn
```

**Quick check**: tell the Agent something important (e.g. "I prefer Python for backend work"), restart the session, then ask it again — if it remembers, the memory is working.

---

## Quick Start

### Option 1: Transparent proxy (zero agent changes)

```bash
DEEPSEEK_API_KEY=sk-xxx ./start.sh --port 8888

# Agent settings
export OPENAI_API_BASE=http://127.0.0.1:8888/v1
```

Memory is injected into every chat request and responses are stored back automatically — the Agent is completely unaware of the memory system.

### Option 2: MCP tools (OpenCode / Claude Code, etc.)

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

> **Hermes Agent**: also supports MCP — connect this server under `mcp_servers` as augmented memory (RRF dual-signal retrieval + learning loop). See [docs/INTEGRATION.md](docs/INTEGRATION.md#2b-hermes-agent-接入mcp-增强记忆).

### Option 4: Auto-memory plugin (recommended — zero agent awareness)

```json
{
  "plugin": ["/abs/path/to/solo-memory/opencode-auto-memory"]
}
```

`chat.message` auto-injects memory, `event` auto-stores — the agent doesn't need to remember to call any tools. See `opencode-auto-memory/README.md` for details.

See [docs/INTEGRATION.md](docs/INTEGRATION.md).

### Option 3: Python API

```python
from memory_skill import MemorySkill, MemorySkillConfig, DialogueTurn

skill = MemorySkill(MemorySkillConfig(db_path="memory.db"))

# Store a conversation
skill.ingest(DialogueTurn(role="user", content="I recommend FastAPI", ...))

# Inject memory context
ctx = skill.weave("What is FastAPI?")
print(ctx.to_prompt_block())

# Proactive retrieval
skill.expand("FastAPI")

# Self-directed learning
skill.learn("Docker", ["https://docs.docker.com/..."])
```

---

## Documentation

| Doc | Contents |
|---|---|
| [SKILL.md](SKILL.md) | Agent usage protocol (8-block weave) |
| [COMPREHENSIVE.md](COMPREHENSIVE.md) | Full architecture design |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | OpenCode / Cursor / proxy integration guide |
| [docs/PROTOCOL.md](docs/PROTOCOL.md) | Memory protocol & tool specs |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

> Note: linked docs are currently Chinese-only. English versions coming soon.

---

## Performance

| Metric | Value |
|---|---|
| Chinese retrieval accuracy | 93% (300 memories) |
| Retrieval latency | 35-100ms |
| Tests | 46 fast (0.07s pure in-memory) + 65 integration |

---

## License

[Apache License 2.0](LICENSE)
