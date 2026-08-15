# Memory Skill

[English](README.en.md) | [中文](README.md)

**为 AI Agent 打造的长期记忆插件**（中文为主，中英双语可用）— 本地优先、双模型记忆、可自我进化。

> **哼，杂鱼又忘事了吧？** 过去聊过什么、你爱用什么技术栈、哪个 bug 踩过几遍，我全替你记着呢。下次开口前先给你递小抄，**省得你像个金鱼一样三秒重置，把 token 浪费在重复自我介绍和重复 websearch 上**。已经学过的东西我会拦着不让你再学一遍，没学过的才放你去搜——**帮你省 token、省时间，别不识好歹**。当然啦，才、才不是特地为你准备的，只是看不得你每次都从零开始犯蠢而已。

为 AI Agent（Claude / OpenAI / 自研 LLM）提供持久化的长期记忆：每次对话自动存取，检索时注入相关记忆上下文，对话碎片经提炼后沉淀为结构化知识。**零 API 检索**（本地向量检索），**所有 LLM 决策由主 agent 完成**（模块为纯存储+检索，不越俎代庖——见 ADR-0002）。

> **语言支持**：中英双语均可存取，检索信号各有侧重——中文由 BM25（jieba 分词）主导，英文由语义向量（bge-large-en-v1.5）主导。插件本身语言无关，中文/英文对话都能自动记忆。

---

## 特性

| 特性 | 说明 |
|---|---|
| **两半记忆模型** | 非结构化对话 + 结构化知识（pref/pers/skill/mission/conclusion） |
| **自动存取** | weave 自动注入上下文；透明代理下 Agent 零改动 |
| **主动检索** | Agent 引用记忆标题 → 自动展开为完整上下文 |
| **碎片隔离** | 未分类对话碎片不污染 weave 注入（tier2/nudge/[近期记忆] 只显示结构化记忆），碎片仍可显式搜索 |
| **候选提炼** | distill 将对话碎片压缩为带证据的候选卡 → 主 agent 审核 → 自动转正结构化记忆 |
| **反馈演化** | 记忆权重随使用自动演化（去重+0.05 / 引用+0.02 / 反馈+0.05） |
| **三层注入** | tier1 场景感知 + tier2 结构化记忆 + nudge 高优记忆 |
| **透明接入** | MCP 工具 / OpenAI 兼容代理 / Python API 三通道 |

---

## 架构

```
┌────────────────────────────── Agent 层 ──────────────────────────────┐
│  MCP 工具 (15个)   透明代理 (auto_context)     Python API             │
│  决策权全部在主 agent：分类/拆解/教学/审核 —— 模块不越俎代庖          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌─────────────────────────── MemorySystem ────────────────────────────┐
│  写链 (IngestPipeline)    读链 (Weaver 10区块)     检索 (RRF)        │
│  │  ingest_dialogue       │  tier1/tier2          │  BM25 ×2.5       │
│  │  dedup (语义合并)      │  nudge/[历史结论]     │  semantic ×0.5   │
│  │  碎片 → default 分类   │  skill/mission/pref/pers │  temporal ×0.5 │
│  └  teach_skill (结构化)  └  树导航/[待审核提炼]   └                  │
├──────────────────────────────────────────────────────────────────────┤
│  提炼层 (distill)     审核层 (pending_store)      存储层              │
│  碎片→候选卡(带证据)  accepted→自动转正           SQLite FTS5         │
│  offset 窗口遍历历史   rejected→丢弃              ChromaDB (1024-dim) │
│  只压缩不断言(防捏造)  skill 保留人工 teach        SawRingBuffer       │
│  evidence 必须真实存在  (source_urls 铁律)         TreeManager         │
└──────────────────────────────────────────────────────────────────────┘
```

### 数据流（闭环）

```
对话 → Ingestor → [SQLite 对话库] + [ChromaDB 向量库] + [记忆树]
                      │
                      ├── 碎片 (default 分类) ──→ distill ──→ pending 候选
                      │                              │          │
                      │                    [待审核提炼]提醒       ├─ accepted → 自动转正
                      │                              │          │             ↓
                      │                              │          └─ rejected → 丢弃
                      │                              │                    结构化记忆
                      │                              │                    (skill/pref/pers/
                      │                              │                     mission/conclusion)
                      └── 检索 (RRF k=60) ←──────────┘                     ↓
                                          ↓                          Weaver 组装 10 区块
                                    注入 Agent 提示词 ←────────────────────┘
```

### 检索信号（RRF 融合）

| 信号 | 权重 | 来源 |
|---|---|---|
| BM25 全文 | 2.5 | SQLite FTS5，jieba 中文分词（中文主导） |
| 语义向量 | 0.5 | ChromaDB，bge-large-en-v1.5 (1024-dim)（英文主导） |
| 时间衰减 | 0.5 | `weight × exp(-0.01 × hours)` |

> **语言说明**：检索是 RRF 融合——中文内容主要靠 BM25（jieba 对中文分词准确），英文内容主要靠语义向量（bge-large-en-v1.5 是英文专用模型）。两路互补：中文记忆靠 BM25 召回，英文记忆靠语义召回，均可在同库中检索。若需单模型统一中英语义检索，可替换为多语言嵌入模型（如 bge-m3，需重新嵌入历史记忆）。

### 15 个 MCP 工具

| 工具 | 用途 |
|---|---|
| `memory_weave` | 注入分层记忆上下文（含自动存取） |
| `memory_search` | 检索记忆（RRF 融合，碎片也可显式查） |
| `memory_ingest` | 存储对话 |
| `memory_status` | 健康检查 |
| `memory_feedback` | 反馈权重演化 |
| `memory_classify` | 分类对话（chat/skill/mission/pref/pers）——协议门控要求每轮调用 |
| `memory_check_skill` | 检查技能是否已掌握（known/partial/unknown） |
| `memory_teach_skill` | 教学写入（强制 source_urls 防捏造） |
| `memory_update_skill` | 更新技能 |
| `memory_learning_queue` | 查看学习队列（待学习/待拆解） |
| `memory_learning_mark` | 关闭学习队列条目 |
| `memory_distill` | 提炼对话碎片为候选卡（offset 遍历历史） |
| `memory_pending` | 查看待审核候选 |
| `memory_pending_mark` | 确认/拒绝候选（accepted 自动转正） |
| `memory_conclusions` | 查询结论条目 |

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
IMPORTANCE_API_KEY=sk-xxx       # LLM 分类/合成用
MEMORY_SKILL_DB_PATH=memory.db  # 数据库路径
MEMORY_MODEL_PATH=models/bge-large-en-v1.5
```

#### LLM 模型配置（默认 DeepSeek V4 Flash，可换任意 OpenAI 兼容模型）

系统通过 OpenAI 兼容接口调用 LLM（用于记忆分类/合成/学习）。**默认指向 DeepSeek V4 Flash，但你可以用任何 OpenAI 兼容模型/服务**——只需改 3 个环境变量：

```bash
IMPORTANCE_API_BASE=https://api.deepseek.com/v1   # API 地址（OpenAI 兼容）
IMPORTANCE_API_KEY=sk-xxx                          # 你的 key
IMPORTANCE_MODEL=deepseek-v4-flash                 # 模型名

# 示例：换 OpenAI
# IMPORTANCE_API_BASE=https://api.openai.com/v1
# IMPORTANCE_MODEL=gpt-4o-mini

# 示例：换本地 vLLM / Ollama
# IMPORTANCE_API_BASE=http://127.0.0.1:8000/v1
# IMPORTANCE_MODEL=qwen2.5-7b-instruct
```

> 兼容任何提供 `/v1/chat/completions` 的服务（OpenAI、Qwen、GLM、Moonshot、本地 vLLM 等）。默认值经过 DeepSeek V4 Flash 调优（如 `max_tokens` 预留），换模型后若分类/合成结果异常，可调整 `IMPORTANCE_*` 相关参数。

---

## 使用教程（从零到会用）

### 方式 A：让 AI 自己安装（最快，推荐）

把仓库 URL 直接交给你的 AI Agent，告诉它：

```
安装 https://github.com/baaai123/solo-memory 并接入我的 OpenCode。

步骤：
1. git clone https://github.com/baaai123/solo-memory
2. 运行 ./setup.sh（创建 venv + 安装依赖 + 配置嵌入模型）
3. 在 opencode.json 注册插件 opencode-auto-memory
4. 在 .env 里填我自己的 IMPORTANCE_API_KEY（用我自己的 LLM API key）

注：./setup.sh 一键完成环境搭建；opencode-auto-memory 插件会自动注入记忆
上下文并自动存储对话，Agent 无需手动调用记忆工具。
```

AI 会自主完成 clone → 环境搭建 → 插件注册。你只需在 `.env` 里填**你自己的 LLM API key**（用于记忆分类/合成/学习，走你自己的 API 账号计费）。

> **为什么可行**：`setup.sh` 已封装环境搭建；`opencode-auto-memory` 插件含首次运行自动引导（venv 缺失时自动创建）。唯一人肉步骤是提供 API key——任何记忆系统都无法替你保管私钥。

### 方式 B：手动安装（逐步）

> 下面以 **OpenCode + 自动记忆插件** 为例。其他 Agent（Claude Code / Cursor）流程相同，只是配置文件名不同。

#### 第 1 步：下载并安装

```bash
git clone https://github.com/baaai123/solo-memory
cd solo-memory

# 一键环境搭建（创建 venv + 安装依赖 + 配置嵌入模型）
./setup.sh

# 或手动：
# python3 -m venv venv && source venv/bin/activate && pip install -e ".[onnx]"
# ./download_model.sh   # bge-large-en-v1.5 → models/
```

#### 第 2 步：配置密钥

```bash
cp .env.example .env
# 编辑 .env，填入 LLM API Key（用于记忆分类/合成/学习）
# IMPORTANCE_API_KEY=sk-xxx
```

#### 第 3 步：把 SKILL.md 交给 Agent

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

#### 第 4 步：注册自动记忆插件

在 `~/.config/opencode/opencode.json` 的 `plugin` 数组加入插件路径：

```json
{
  "plugin": [
    "/abs/path/to/solo-memory/opencode-auto-memory"
  ]
}
```

插件会自动注入记忆上下文（`chat.message` hook）并自动存储对话（`event` hook）——Agent 无需手动调工具。

> 如需 MCP 工具方式（手动调用 `memory_search` 等），见下方 [快速开始 → 方式 2](#方式-2mcp-工具opencode--claude-code-等)。

#### 第 5 步：重启 Agent 并验证

重启 Agent 会话，让 Agent 调用记忆工具：

```
# Agent 应能看到并调用这些工具（15 个，核心 5 个）：
memory_search / memory_weave / memory_ingest / memory_status
memory_feedback / memory_classify / memory_teach_skill / memory_distill
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

> **Hermes Agent**：也支持 MCP——在 `mcp_servers` 配置段接入本 server 作为增强记忆（RRF 双信号检索 + 学习闭环）。配置见 [docs/INTEGRATION.md](docs/INTEGRATION.md#2b-hermes-agent-接入mcp-增强记忆)。

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

# 提炼候选（对话碎片 → 待审核候选）
skill.distill()   # 或 MCP: memory_distill

# 查看/审核候选
skill.pending()   # 或 MCP: memory_pending / memory_pending_mark
```

---

## 提炼与审核（主动学习 v2）

08-11 重写后，记忆模块为纯存储+检索，**所有学习决策由主 agent 完成**（ADR-0002）。主动学习闭环变为：

```
对话碎片 ── memory_distill ──→ 候选卡 (topic/summary/evidence/suggested)
                                   │  evidence 必须引用真实对话 id（防捏造）
                                   │  只压缩不断言，suggested 只是建议
                                   ↓
                              pending_store (SQLite，不进检索库)
                                   │
                    weave 注入 [待审核提炼] 提醒（每轮可见）
                                   ↓
                          主 agent 审核 (memory_pending)
                                   │
              ┌────────────────────┼────────────────────┐
              ↓                    ↓                    ↓
        accepted              rejected            skill 候选
        (conclusion/pref/pers)   → 丢弃              → 保留人工 teach
        自动转正入库                                (source_urls 铁律)
```

**关键设计（防捏造防线）：**
- `distill` 只总结已有对话，绝不新增事实；每条 `evidence` 必须是真实存在的 dialogue id，否则候选被拒收
- 候选存独立 `pending_store`，**不参与检索**——审核前不会污染 weave
- skill 候选不自动转正：`teach_skill` 强制 `source_urls` 非空（ADR-0002 防止主 agent 凭训练数据捏造）
- `memory_distill` 支持 `offset/limit` 窗口遍历历史——**旧记忆也能被提炼**，不只是最新对话

---

## 文档

| 文档 | 内容 |
|---|---|
| [SKILL.md](SKILL.md) | Agent 使用协议（分层 weave 注入 + 提炼闭环） |
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
| 测试 | 115 快速/集成（25 network/slow 需真实 API key 时运行） |

---

## License

[Apache License 2.0](LICENSE)

