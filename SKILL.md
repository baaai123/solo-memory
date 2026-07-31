---
name: memory-skill
description: Long-term memory for coding sessions — remember project facts, user preferences, past decisions, learned skills, and ongoing tasks across sessions. Call memory_weave before responding, memory_ingest after significant interactions, memory_search for deep retrieval. Use whenever context from past conversations could improve the current answer.
---

# Memory Skill for OpenCode

Long-term memory for coding sessions. Remembers project facts, user preferences,
past decisions, learned skills, and ongoing tasks — across sessions.

## What Memory Remembers

- **Project facts** (user_mem): "this project uses SQLite", "the auth module is in src/auth/"
- **User preferences** (pref): "prefers TypeScript", "uses 2-space indentation"
- **Past decisions** (user_mem): "switched from ChromaDB to LanceDB", "rejected gRPC"
- **Bugs & fixes** (user_mem): "the ONNX position encoding bug, fixed by truncating"
- **Skills** (skill): "can deploy FastAPI with uvicorn", "knows Docker compose"
- **Ongoing tasks** (mission): "migrate auth module by Friday [doing]"
- **Your personality** (pers): accumulates traits — "concise", "code-first"

## Usage Protocol

```
BEFORE responding:
  opencode-memory_memory_weave(user_message)  → inject 8-block context

AFTER important decisions, bugs, findings:
  opencode-memory_memory_ingest(role, content) → persist for future

WHEN you need more than weave provides:
  opencode-memory_memory_search(query)         → deep retrieval

SESSION START:
  opencode-memory_memory_status                 → health check
```

## Weave Context (what you receive)

```
[人格设定]         # Agent 人物卡 (pers)
[用户偏好]         # key-value preferences (pref)
[当前场景]         # scene_summary
[最近对话]         # last 3 turns
[检索记忆]         # RRF-semantic recall from past
[已掌握的技能]     # skill titles only
[当前任务]         # mission steps with skill status
[知识缺口]         # things to learn
```

## When to Ingest vs Skip

**Ingest:**
- User states a fact about the project or themselves
- A design decision is made ("let's use Redis")
- A bug is found and fixed
- User expresses a strong preference
- You learned a new skill or technique

**Skip:**
- Greetings, small talk
- Pure command execution without new knowledge
- Error messages (ingest the fix, not the error)

## Architecture (transparent to Agent)

```
memory_weave → Embedder(bge-large) → Retriever(RRF: BM25×2.5 + semantic×0.5 + time×0.5) → Weaver(8-block)
memory_ingest → 两边存储:
  ├── user_mem: raw dialogue → BM25 + vector index → RRF retrieval
  └── structured: classify_and_extract(LLM) → pref/pers/skill/mission routing
```

You also have **active retrieval**: reference a topic in your response, and
the system expands it with related memories from the same time window.
