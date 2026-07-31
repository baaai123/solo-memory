# Memory Skill

Long-term memory for AI Agents — local-first, two-sided, self-evolving.

## 特性

- **两半记忆模型**: 非结构化对话 + 结构化知识 (pref/pers/skill/mission)
- **自动提取**: 每轮对话 LLM 分类 → 结构化分支
- **主动检索**: Agent 引用记忆标题 → 自动展开
- **主动学习**: gap → crawl → synth → verify 闭环
- **透明代理**: Agent 零改动接入

## 快速开始

```bash
# 一键启动透明代理
DEEPSEEK_API_KEY=sk-xxx ./start.sh

# Agent 设置
export OPENAI_API_BASE=http://127.0.0.1:8888/v1
```

## Python API

```python
from memory_skill import MemorySkill, MemorySkillConfig

skill = MemorySkill(MemorySkillConfig(db_path="memory.db"))

skill.ingest(turn)              # 存储对话
ctx = skill.weave("问题")       # 获取记忆上下文
skill.expand("FastAPI")         # 主动检索
skill.ingest_pers("简洁")       # 设置人格
skill.learn("Docker", urls)     # 主动学习
```

## 文档

- [SKILL.md](SKILL.md) — Agent 使用协议
- [COMPREHENSIVE.md](COMPREHENSIVE.md) — 架构文档
- [docs/INTEGRATION.md](docs/INTEGRATION.md) — 接入指南
- [docs/PROTOCOL.md](docs/PROTOCOL.md) — 记忆协议

## 架构

```
记忆模型: user_mem (非结构) + pref/pers/skill/mission (结构)
存储:     SQLite FTS5 (BM25) + ChromaDB (向量)
检索:     RRF 融合 (BM25×2.5 + semantic×0.5 + time×0.5)
组装:     weave 8 区块上下文
学习:     gap → crawl → synth → ingest → verify
```

## 指标

- 中文检索精度: 93% (300条)
- 检索延迟: 35-100ms
- 测试: 15 单元 + 65 集成
