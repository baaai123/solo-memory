# Memory Skill — 接入测试

## 方式 1: 透明代理 (最快)

```bash
# 启动
DEEPSEEK_API_KEY=sk-xxx ./start.sh

# Agent 改一行
export OPENAI_API_BASE=http://127.0.0.1:8888/v1
```

Agent 照常调 `POST /v1/chat/completions`，记忆自动生效。

## 方式 2: Python API

```python
from memory_skill import MemorySkill, MemorySkillConfig

skill = MemorySkill(MemorySkillConfig(db_path="memory.db"))
skill.ingest(turn)              # 存储
ctx = skill.weave("用户问题")   # 获取上下文
skill.expand("FastAPI")         # 主动检索
skill.ingest_pers("简洁")       # 设置人格
```

## 方式 3: MCP 协议

```json
// ~/.config/opencode/opencode.json
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

工具: `memory_weave`, `memory_search`, `memory_ingest`, `memory_status`, `memory_feedback`

## 当前数据库

已有一个 30 条咨询案例的数据库: `/tmp/multi.db`

```python
skill = MemorySkill(MemorySkillConfig(db_path="/tmp/multi.db", agent_name="multi"))
```
