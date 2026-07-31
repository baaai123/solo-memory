# Memory Skill for OpenCode

Long-term memory for coding sessions. Remembers project facts, user preferences, past decisions, and learned patterns — across sessions.

## What Memory Remembers

- **Project facts**: "this project uses SQLite, not PostgreSQL", "the auth module is in src/auth/"
- **User preferences**: "prefers TypeScript over JavaScript", "uses 2-space indentation"
- **Past decisions**: "switched from ChromaDB to LanceDB for performance", "rejected gRPC in favor of REST"
- **Bugs & fixes**: "the ONNX position encoding bug was fixed by truncating to 512"
- **Patterns**: "user always writes tests before implementation"

## Usage Protocol

```
FIRST TURN of session:
  memory_status → check system health

BEFORE EVERY response to user:
  memory_weave(user_message) → get auto-injected context

AFTER significant exchanges:
  memory_ingest(role, content) → save for future recall

WHEN weave isn't enough:
  memory_search(query) → explicit deep search

AFTER seeing retrieval results:
  memory_feedback(memory_ids, outcome) → train the system
```

## Tool Reference

| Tool | When | Input | Output |
|------|------|-------|--------|
| `opencode-memory_memory_status` | Session start | none | Entry counts, model status, evolution ticks |
| `opencode-memory_memory_weave` | Before every response | `user_message`, optional `scene_summary` | 3-tier context block (recent + recall + nudges) |
| `opencode-memory_memory_search` | Deep dive needed | `query`, optional `limit` | Ranked memory entries with relevance scores |
| `opencode-memory_memory_ingest` | After important turns | `role` (user/assistant), `content` | Confirmation |
| `opencode-memory_memory_feedback` | After retrieval used | `memory_ids` (list), `outcome` (positive/negative/neutral) | Evolution tick result |

## When to Ingest vs Skip

**Ingest** when:
- User states a fact about the project or themselves
- A design decision is made
- A bug is found and fixed
- User expresses a strong preference

**Skip** when:
- Greetings, small talk
- Pure command execution without new knowledge
- Error messages you're about to fix (ingest the fix, not the error)

## Feedback

After any turn where `memory_weave` or `memory_search` was used:
- `positive` — retrieved memories were accurate and helpful
- `negative` — retrieved memories were wrong or misleading
- `neutral` — memories were present but irrelevant

This trains the evolution system: accurate memories gain weight, wrong ones are suppressed.

## Architecture (transparent to Agent)

```
memory_weave → Embedder(ONNX bge-large) → Retriever(RRF fusion) → Weaver(3-tier context)
memory_ingest → ImportanceGate → SQLite(BM25) + ChromaDB(vectors)
memory_feedback → EvolutionLoop → weight ±0.15 per feedback
```

- **RRF fusion**: BM25 keyword ×2.5 + semantic vector ×0.5 + time decay ×0.5
- **Evolution**: Feedback-driven weight adjustment. Verified facts rank higher over time.
- **Consolidation**: Every 10 turns, TextRank extracts keywords → structured observations.
- **Contradiction detection**: Opposite claims → both sides down-weighted.

## States

| State | Meaning |
|-------|---------|
| `mode=onnx` | Embedding model loaded (CPU, ~30s cold start) |
| `learned=N` | Entries in ChromaDB vector store |
| `dialogue=N` | Raw turns in SQLite |
| `evo=N` | Evolution ticks this session |
