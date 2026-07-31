# Memory Protocol

## 每轮必须执行的记忆操作

每次对话，Agent 必须按以下顺序操作：

```
1. memory_weave   — 获取记忆上下文（BEFORE responding）
2. memory_ingest  — 存储重要发现（AFTER interaction）
```

## 工具说明

| 工具 | 参数 | 说明 |
|------|------|------|
| `memory_weave` | `user_message`, `scene_summary` | 返回记忆上下文块，注入到 prompt 中 |
| `memory_search` | `query`, `limit` | 深度搜索历史记忆 |
| `memory_ingest` | `content`, `role` | 存储关键信息（bug、决策、偏好） |
| `memory_status` | — | 检查记忆系统健康状态 |
| `memory_feedback` | `query_id`, `outcome`, `cited_ids` | 反馈检索质量，训练权重 |

## 接入方式

**OpenCode**: 复制 `~/.config/opencode/opencode.json` 中的 `mcp.opencode-memory` 段和 `agent.sisyphus.prompt_append` 段。

**透明代理**: `./start.sh` 启动，Agent 设置 `OPENAI_API_BASE=http://127.0.0.1:8888/v1`。

**Python**: `from memory_skill import MemorySkill, MemorySkillConfig`

## 反模式

- ❌ 回复前不调 `memory_weave` — 丢失上下文注入机会
- ❌ 只调 `memory_search` 而不调 `memory_weave` — weave 自动注入更高效
- ❌ 存储无意义的内容 — 重要性门控会自动过滤

## 人格配置

Agent 人格通过 `ingest_pers(trait)` 设置，自动累积为人物卡注入到每次对话中。

```python
skill.ingest_pers("简洁")
skill.ingest_pers("代码优先")
skill.ingest_pers("用中文回复")
```
