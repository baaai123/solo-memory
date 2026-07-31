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

**OpenCode** — 添加 `~/.config/opencode/opencode.json`:
```json
{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "memory_skill.mcp_server"],
      "env": {
        "DEEPSEEK_API_KEY": "sk-xxx"
      }
    }
  }
}
```

**Cursor** — 配置 `.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "memory": {
      "command": "venv/bin/python",
      "args": ["-m", "memory_skill.mcp_server"],
      "env": { "DEEPSEEK_API_KEY": "sk-xxx" }
    }
  }
}
```

**Cody** — 同 MCP 协议配置，参考平台文档。

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
| `MEMORY_SKILL_DB_PATH` | `memory.db` | 数据库路径 |
| `MEMORY_SKILL_AGENT` | `memory-skill` | Agent 名称 |
