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
| `MEMORY_SKILL_AGENT` | `memory-skill` | Agent 名称（角色记忆按此查询绑定关系，见下节） |

## 角色记忆（Character Memory）

角色记忆把记忆库升级为「全局库 + 角色视图」两层结构：**角色是全局记忆的引用集合，不复制数据**。全局库仍是唯一数据源（persona / skill / mission / pref / pers / conclusion / default 分类），角色只是一层可见性过滤视图。绑定角色后，weave 只注入该角色引用的记忆（全局其余不可见）；未绑定 agent 行为完全不变。

### 1. 概念

- **全局库 = 唯一数据源**：所有记忆仍沉淀在全局库，分类体系不变。
- **角色 = 引用集合**：角色只记录「引用了哪些全局记忆 ID」，不复制内容；同一条记忆可被多个角色引用。
- **绑定即过滤**：agent 绑定角色后，weave / 检索只命中该角色的引用记忆；未绑定 agent 退化为现状（注入全部记忆），向后兼容。

### 2. 数据存储

角色数据存在记忆库 SQLite 文件的三张新表，首次运行自动建表（`CREATE TABLE IF NOT EXISTS`），已有数据库无需迁移：

| 表 | 字段 | 说明 |
|----|------|------|
| `roles` | id / name / description / created_at / updated_at | 角色元数据 |
| `role_memories` | role_id / memory_id / added_at（联合主键） | 角色→记忆引用，幂等防重 |
| `agent_bindings` | agent_name（主键）/ role_id / updated_at | agent→角色绑定 |

### 3. agent 接入方式

1. 在 MCP 配置的 `environment` 中设置 `MEMORY_SKILL_AGENT` 为当前 agent 名（如 `opencode`、`dsh`）——现有配置无需改动，角色绑定只是给该变量增加一层映射。
2. 用 MCP 工具 `memory_character_bind_agent`（或 memory-ui 接入页）把该 agent 绑定到某个角色。
3. 之后 weave 时系统自动：`agent_name` → 查 `agent_bindings` → 得 `role_id` → 取引用记忆集合 → 作为白名单传入检索器，只注入角色内记忆。
4. 未设置 `MEMORY_SKILL_AGENT` 或未绑定角色时，weave 注入全局全部记忆，与现状一致。

### 4. MCP 工具清单

以下角色管理 MCP 工具已注册（`memory_skill/tools.py` 的 `DISPATCH` / `TOOL_SCHEMAS`），agent 可在会话内直接调用：

| 工具 | 用途 |
|------|------|
| `memory_character_list` | 列出所有角色 |
| `memory_character_create` | 创建角色（名称 / 描述，可选初始记忆） |
| `memory_character_get` | 查看角色详情与引用记忆列表 |
| `memory_character_delete` | 删除角色（只删引用集合，不动全局记忆） |
| `memory_character_add_memory` | 将一条全局记忆加入角色引用 |
| `memory_character_remove_memory` | 从角色移除一条记忆引用 |
| `memory_character_bind_agent` | 将 agent 绑定到角色 |
| `memory_character_agent_role` | 查询某 agent 当前绑定的角色 |

> 💡 角色管理同样可通过 memory-ui 的「🎭 角色」页图形化操作（见第 7 节），两者操作同一份数据。

### 5. 双写机制

绑定角色的 agent 会话生成新记忆时，系统自动双写：

1. 记忆照常写入全局库（沿用现有分类与 ingest 流程）；
2. 引用自动加入该角色的引用集合（`role_memories` 幂等插入）；
3. 记忆 metadata 写入 `source_role`，标记来源角色，全局库可按来源追溯。

### 6. 级联删除

- **删除全局记忆** → 自动清理所有角色中的引用（`remove_all_memory`），无悬空引用。
- **删除角色** → 只删该角色的引用集合与 agent 绑定，全局库记忆不受影响。

### 7. Web UI

memory-ui（localhost:8000）的「🎭 角色」页可图形化管理角色、引用与 agent 绑定（角色 CRUD、全局 / 角色双视图、接入页绑定）。
