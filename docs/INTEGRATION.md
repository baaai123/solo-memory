# Integration Guide

## 三种接入方式

### 1. 透明代理（推荐 — Agent 零代码改动）

```bash
# 一键启动
./start.sh
# 或
DEEPSEEK_API_KEY=sk-xxx memory-proxy --port 8888
```

Agent 设置 `OPENAI_API_BASE=http://127.0.0.1:8888/v1` 即可。记忆自动注入、自动存储、Agent 可主动检索。

### 2. MCP 协议 (OpenCode / Cursor)

**OpenCode** — 添加 `~/.config/opencode/opencode.json`（注意键是 `mcp`，不是 `mcpServers`；`command` 用数组形式；`memory_skill` 包需已安装到 venv，见下）:
```json
{
  "mcp": {
    "opencode-memory": {
      "type": "local",
      "command": ["/绝对路径/memory for solo/venv/bin/python", "-m", "memory_skill.mcp_server"],
      "environment": {
        "MEMORY_SKILL_DB_PATH": "/绝对路径/opencode_memory.db",
        "MEMORY_SKILL_AGENT": "opencode",
        "DEEPSEEK_API_KEY": "sk-xxx",
        "DEEPSEEK_API_BASE": "https://api.deepseek.com/v1",
        "DEEPSEEK_MODEL": "deepseek-v4-flash",
        "IMPORTANCE_API_KEY": "sk-xxx",
        "IMPORTANCE_API_BASE": "https://api.deepseek.com/v1",
        "IMPORTANCE_MODEL": "deepseek-v4-flash"
      },
      "timeout": 60000
    }
  }
}
```

> ⚠ 常见坑: `python -m memory_skill.mcp_server` 依赖 `memory_skill` 包在 **venv site-packages** 中（`pip install -e .`），否则只有 cwd 恰好是项目目录时才能 import。另外 `IMPORTANCE_*` 变量（树分类 LLM）缺了会导致重要性评分静默失败——建议与 `DEEPSEEK_*` 配同样的值。

**Cursor** — 配置 `.cursor/mcp.json`（Cursor 使用 `mcpServers` 键）:
```json
{
  "mcpServers": {
    "memory": {
      "command": "venv/bin/python",
      "args": ["-m", "memory_skill.mcp_server"],
      "env": {
        "DEEPSEEK_API_KEY": "sk-xxx",
        "DEEPSEEK_API_BASE": "https://api.deepseek.com/v1",
        "DEEPSEEK_MODEL": "deepseek-v4-flash",
        "MEMORY_SKILL_DB_PATH": "/绝对路径/opencode_memory.db",
        "IMPORTANCE_API_KEY": "sk-xxx"
      }
    }
  }
}
```

**Cody** — 同 MCP 协议配置，参考平台文档。

### 2b. Hermes Agent 接入（MCP 增强记忆）

[Hermes Agent](https://hermesagent.org.cn/)（开源自托管 AI Agent，Nous Research）支持标准 MCP。我们的 MCP server 可直接接入，作为 Hermes 的**增强记忆系统**（RRF 双信号检索 + 学习闭环，替代或补充其自带记忆）。

**配置** — 编辑 Hermes 配置文件（`~/.hermes/config.yaml` 或 `hermes setup`），添加 `mcp_servers` 段：
```yaml
mcp_servers:
  solo_memory:
    command: "/绝对路径/solo-memory/venv/bin/python"
    args: ["-m", "memory_skill.mcp_server"]
    env:
      MEMORY_SKILL_DB_PATH: "/绝对路径/solo-memory/opencode_memory.db"
      MEMORY_SKILL_AGENT: "hermes"
      IMPORTANCE_API_KEY: "sk-xxx"
      IMPORTANCE_API_BASE: "https://api.deepseek.com/v1"
      IMPORTANCE_MODEL: "deepseek-v4-flash"
```

**建议工具过滤**（白名单）——只需暴露记忆核心 5 个工具：
```yaml
mcp_servers:
  solo_memory:
    command: "/绝对路径/solo-memory/venv/bin/python"
    args: ["-m", "memory_skill.mcp_server"]
    env:
      MEMORY_SKILL_DB_PATH: "/绝对路径/solo-memory/opencode_memory.db"
      MEMORY_SKILL_AGENT: "hermes"
      IMPORTANCE_API_KEY: "sk-xxx"
    include:
      - memory_weave
      - memory_search
      - memory_ingest
      - memory_status
      - memory_gaps
```

**使用协议** — 让 Hermes 遵循（参考 [SKILL.md](../SKILL.md)）：
```
每次回复前:   memory_weave   → 注入记忆上下文
重要交互后:   memory_ingest  → 存入记忆
需要更多时:   memory_search  → 深度检索
会话开始:     memory_status  → 健康检查
```

> **定位说明**：Hermes 自带长期记忆，本模块作为**增强记忆**接入——提供更精确的 RRF 双信号检索（BM25×2.5 + 语义×0.5）和主动学习闭环（知识缺口检测 → 爬取 → 合成 → 验证）。适合需要更强检索精度或跨 Agent 统一记忆的场景。

### 3. Python API

```python
from memory_skill import MemorySkill, MemorySkillConfig

config = MemorySkillConfig(db_path="memory.db", agent_name="my-agent")
skill = MemorySkill(config)

# 对话自动存储
skill.ingest(turn)

# 获取记忆上下文  
ctx = skill.weave(user_message)
print(ctx.to_prompt_block())

# Agent 主动检索
print(skill.expand("FastAPI"))
```

### 环境变量

| 变量 | 默认 | 说明 |
|------|------|------|
| `DEEPSEEK_API_KEY` | — | LLM API 密钥 |
| `DEEPSEEK_API_BASE` | `https://api.deepseek.com/v1` | API 地址 |
| `DEEPSEEK_MODEL` | `deepseek-v4-flash` | 模型名 |
| `IMPORTANCE_API_KEY` | — | 树分类/重要性 LLM 密钥（建议与 `DEEPSEEK_API_KEY` 相同） |
| `IMPORTANCE_API_BASE` | `https://api.deepseek.com/v1` | 树分类 API 地址 |
| `IMPORTANCE_MODEL` | `deepseek-v4-flash` | 树分类模型名 |
| `MEMORY_SKILL_DB_PATH` | `memory.db` | 数据库路径 |
| `MEMORY_SKILL_AGENT` | `memory-skill` | Agent 名称 |
