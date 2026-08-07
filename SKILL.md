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

### MANDATORY per-turn loop (no exceptions)

Every exchange follows this exact order — this is the ONLY way both sides
of the conversation get persisted:

```
1. USER message arrives
2. CALL memory_weave(
     user_message=<current user message>,
     assistant_content=<YOUR PREVIOUS reply, verbatim>   ← MUST pass
   )
   → auto-ingests user + your last reply, returns 8-block context
3. Respond to the user using the woven context
4. Your reply becomes assistant_content for the NEXT turn's memory_weave
```

Skipping `assistant_content` means your replies are never stored — half
the conversation is lost. memory_weave auto-ingests both fields.

### Supplemental tools

```
WHEN you need more than weave provides:
  memory_search(query)  → deep retrieval

AFTER standalone facts/decisions (not part of a user→assistant exchange):
  memory_ingest(role, content)

SESSION START:
  memory_status  → health check
```

## Search-First Discipline (KNOWN-ISSUES #11)

`memory_weave` is passive context; it is NOT a substitute for retrieval.
When you hit an information gap, SEARCH FIRST before re-doing work:

- **Before re-crawling/re-computing anything** ("has this been researched
  before?"), call `memory_search` — a marathon session re-crawled 302
  pages after `/tmp` was wiped because the agent never searched its own
  memory first (hours wasted).
- **Before starting a new sub-topic**, search for related past work.
- **Actively `memory_ingest` key decisions** (e.g. "TuLED and elderroll
  share D:\game\SkyrimSE") — do not rely only on auto-ingest.
- A session with 100+ turns and ~2 `memory_search` calls is the failure
  mode this section exists to prevent.

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

## Preserve the Reasoning Chain (KNOWN-ISSUES #10)

Memory stores *what* happened, not *why it was right*. When ingesting a
conclusion, attach its basis so a future session can reuse the reasoning
instead of re-deriving it:

```
# GOOD — conclusion + basis:
"MO2 界面顺序 = 文件顺序镜像(由用户锚点'界面最后=卡西安娜=文件第1行'验证)"
# BAD — bare conclusion:
"MO2 界面顺序 = 文件顺序镜像"
```

If the conclusion came from a user-provided anchor, a test, or a verified
log — name that source in the ingested content. Bare conclusions without
their basis force every future session to re-validate from scratch.

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

## Architecture (transparent to Agent)

```
memory_weave → Embedder(bge-large) → Retriever(RRF: BM25×2.5 + semantic×0.5 + time×0.5) → Weaver(8-block)
memory_ingest → 两边存储:
  ├── user_mem: raw dialogue → BM25 + vector index → RRF retrieval
  └── structured: classify_and_extract(LLM) → pref/pers/skill/mission routing
```

You also have **active retrieval**: reference a topic in your response, and
the system expands it with related memories from the same time window.

## Install (if memory is not yet available)

If this memory system is not connected, you can set it up yourself:

```
1. git clone https://github.com/baaai123/solo-memory
2. cd solo-memory && ./setup.sh        # venv + deps + embedding model (one command)
3. Ask the user to fill .env with their own IMPORTANCE_API_KEY (their personal LLM API key)
4. Register the plugin in opencode.json:
   { "plugin": ["/abs/path/to/solo-memory/opencode-auto-memory"] }
```

After setup, memory is automatic: `chat.message` injects context, `event`
stores dialogue. You do not need to call the tools manually, but the protocol
above (weave/ingest/search/status) still works for explicit control.

