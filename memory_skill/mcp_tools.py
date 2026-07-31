"""MCP tool definitions and handlers for Memory Skill.

Exposes 4 tools over the MCP JSON-RPC protocol:
  - ``memory_search``   — semantic search across dialogue + learned memory
  - ``memory_ingest``   — ingest a dialogue turn into the memory system
  - ``memory_status``   — health summary (entry counts, model status)
  - ``memory_feedback`` — record retrieval outcome + boost cited memories
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_skill.skill import MemorySkill


logger = logging.getLogger("memory_skill.mcp")

# ═══════════════════════════════════════════════════════════════════════════════
# Protocol Injection (delivered with every status call + tool descriptions)
# ═══════════════════════════════════════════════════════════════════════════════

MEMORY_PROTOCOL = """IMPORTANT — Memory Skill Usage Protocol (v5 auto-injection):
1. ON WAKE-UP: Call memory_status to check health and model availability.
2. BEFORE RESPONDING: Call memory_weave to get auto-injected context for the
   current conversation. This is the PRIMARY way to use memory — the system
   automatically assembles relevant past conversations and learned knowledge.
3. AFTER INTERACTIONS: Call memory_ingest to save significant dialogue turns.
4. FOR DEEP QUERIES: Use memory_search when memory_weave isn't enough and you
   need to explicitly search for specific past information.
5. FOR LEARNING: Call memory_feedback with cited memory IDs to boost their weights.

The weave() paradigm means the Agent does NOT need to call memory_search before
every turn. memory_weave handles context injection automatically.

MCP servers and middleware can also use ``skill.auto_context(messages)`` and
``skill.auto_ingest(user_msg, assistant_response)`` for fully transparent
memory injection — the LLM has zero knowledge of the memory system.
These methods handle context injection and ingestion without any tool calls
from the LLM side."""

# ═══════════════════════════════════════════════════════════════════════════════
# Tool Definitions (MCP schema format)
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS: list[dict[str, Any]] = [
    {
        "name": "memory_search",
        "description": (
            "Search agent memory for relevant past conversations and learned "
            "knowledge. Returns ranked results with content, relevance scores, "
            "and memory IDs. Use this BEFORE responding to any question about "
            "past events, user preferences, or previously discussed topics."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query. Be specific — "
                    "include names, topics, dates for best results.",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results to return (1-100).",
                    "minimum": 1,
                    "maximum": 100,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "memory_weave",
        "description": (
            "Auto-assemble layered memory context for the current conversation. "
            "Call BEFORE responding to any user message. Returns a context block "
            "ready for system-prompt injection. tier1: scene perception, "
            "tier2: deep memory (gated), nudge: high-importance reminders.\n\n"
            "V10: Both user_message AND assistant_content are auto-ingested "
            "(ImportanceScorer filters trivial content). This mirrors the Room "
            "framework's conversation-block pattern where both sides of every "
            "exchange are persisted without manual ingest calls."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "user_message": {
                    "type": "string",
                    "description": "The user's current message.",
                },
                "assistant_content": {
                    "type": "string",
                    "description": "Optional: your OWN response from the PREVIOUS "
                    "exchange. Auto-ingested into memory so both sides of every "
                    "conversation block are persisted. Pass your full last response.",
                },
                "scene_summary": {
                    "type": "string",
                    "description": "Optional: what the user is doing now.",
                },
            },
        },
    },
    {
        "name": "memory_ingest",
        "description": (
            "Save a dialogue turn into the memory system for future retrieval. "
            "Call this after significant interactions — user questions, "
            "important answers, decisions made, facts learned. Each turn is "
            "indexed for both semantic search and full-text retrieval.\n\n"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content to save — typically the user's "
                    "question or your response that should be remembered.",
                },
                "role": {
                    "type": "string",
                    "default": "user",
                    "description": "Speaker role: 'user', 'assistant', or 'system'.",
                    "enum": ["user", "assistant", "system"],
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_status",
        "description": (
            "Get a health summary of the memory system: entry counts, "
            "embedder status, and configuration. "
            "Call this on first wake-up to understand the memory system state.\n\n"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "memory_feedback",
        "description": (
            "Record feedback on a previous memory_search result. Provide the "
            "query_id from the search response, the outcome ('positive', "
            "'negative', 'neutral', or 'auto' for automatic detection), "
            "and the list of memory IDs you cited or found useful. "
            "To enable auto-detection, also pass search_results and "
            "final_response. This boosts the weights of cited memories "
            "to improve future retrieval.\n\n"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_id": {
                    "type": "string",
                    "description": "The query_id from a previous memory_search "
                    "response. Used to correlate feedback with the search.",
                },
                "outcome": {
                    "type": "string",
                    "description": "How useful the search results were: "
                    "'positive' (found what I needed), 'negative' (results "
                    "were irrelevant), 'neutral' (mixed or unsure), or "
                    "'auto' (let the system auto-detect).",
                    "enum": ["positive", "negative", "neutral", "auto"],
                },
                "cited_ids": {
                    "type": "array",
                    "description": "List of memory entry IDs from the search "
                    "results that you used or cited in your response.",
                    "items": {"type": "string"},
                },
                "search_results": {
                    "type": "array",
                    "description": "Optional: the search results from the "
                    "previous memory_search call (for auto-detection). "
                    "Each item should have 'id' and 'content' fields.",
                    "items": {"type": "object"},
                },
                "final_response": {
                    "type": "string",
                    "description": "Optional: the final response the agent "
                    "produced using the search results (for auto-detection).",
                },
            },
            "required": ["query_id"],
        },
    },
    {
        "name": "memory_learn",
        "description": (
            "Closed-loop knowledge acquisition: crawl URLs, synthesize markdown, "
            "ingest into memory, and verify comprehension. Use when the agent "
            "detects a knowledge gap and has source URLs to learn from. "
            "Returns task status (crawling/synthesizing/verifying/done/failed)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The knowledge topic to learn about.",
                },
                "urls": {
                    "type": "array",
                    "description": "List of source URLs to crawl for knowledge.",
                    "items": {"type": "string"},
                },
            },
            "required": ["topic", "urls"],
        },
    },
    {
        "name": "memory_gaps",
        "description": (
            "List knowledge gaps detected during conversation — topics the agent "
            "could not answer confidently. Use to discover what the agent does not "
            "know, then call memory_learn to fill gaps with source URLs."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# ToolHandler
# ═══════════════════════════════════════════════════════════════════════════════


class ToolHandler:
    """Handles MCP tool invocations against a MemorySkill instance.

    Parameters
    ----------
    skill:
        A fully-initialized ``MemorySkill`` instance.
    """

    def __init__(self, skill: MemorySkill) -> None:
        self._skill = skill

    # ── Dispatch ──────────────────────────────────────────────────────────

    def handle(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call to the appropriate handler.

        Returns
        -------
        dict
            A JSON-serializable result dict. On error, ``{"error": str(...)}``.
        """
        try:
            if tool_name == "memory_search":
                return self._search(
                    query=arguments.get("query", ""),
                    limit=arguments.get("limit", 10),
                )
            elif tool_name == "memory_weave":
                return self._weave(
                    user_message=arguments.get("user_message", ""),
                    scene_summary=arguments.get("scene_summary", ""),
                    assistant_content=arguments.get("assistant_content", ""),
                )
            elif tool_name == "memory_ingest":
                return self._ingest(
                    content=arguments.get("content", ""),
                    role=arguments.get("role", "user"),
                )
            elif tool_name == "memory_status":
                return self._status()
            elif tool_name == "memory_feedback":
                return self._feedback(
                    query_id=arguments.get("query_id", ""),
                    outcome=arguments.get("outcome", "auto"),
                    cited_ids=arguments.get("cited_ids", []),
                    search_results=arguments.get("search_results"),
                    final_response=arguments.get("final_response"),
                )
            elif tool_name == "memory_learn":
                return self._learn(
                    topic=arguments.get("topic", ""),
                    urls=arguments.get("urls", []),
                )
            elif tool_name == "memory_gaps":
                return self._gaps()
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as exc:
            logger.exception("Tool %s failed", tool_name)
            return {"error": f"{type(exc).__name__}: {exc}"}

    # ── Tool Handlers ──────────────────────────────────────────────────────

    def _weave(self, user_message: str, scene_summary: str,
               assistant_content: str = "") -> dict[str, Any]:
        """Auto-assemble layered memory context, with auto-ingest of both sides.

        V10: Every weave call auto-ingests the user's message AND the previous
        assistant response (if provided). This mirrors the Room framework's
        conversation-block pattern — both sides of every exchange are persisted
        without manual ingest. The ImportanceScorer filters out trivial content.
        """
        # ── Auto-ingest user message ────────────────────────────────────
        if user_message and user_message.strip():
            self._ingest(content=user_message, role="user")

        # ── Auto-ingest previous assistant response ─────────────────────
        if assistant_content and assistant_content.strip():
            self._ingest(content=assistant_content, role="assistant")

        ctx = self._skill.weave(
            user_message=user_message,
            scene_summary=scene_summary,
        )
        return {
            "tier1_context": ctx.tier1_context,
            "tier2_context": ctx.tier2_context,
            "memory_nudge": ctx.memory_nudge,
            "prompt_block": ctx.to_prompt_block(),
            "is_empty": ctx.is_empty
        }

    def _search(self, query: str, limit: int) -> dict[str, Any]:
        """Search agent memory for relevant past conversations."""
        if not query.strip():
            return {"results": [], "note": "Empty query — no search performed", "query_id": ""}

        limit = max(1, min(limit, 100))

        try:
            envelope = self._skill.retrieve(query, limit=limit)
        except Exception as exc:
            logger.warning("Retrieval failed: %s", exc)
            return {
                "results": [],
                "note": f"Retrieval error: {type(exc).__name__}: {exc}",
                "query_id": query,
            }

        results = []
        for entry in envelope.entries:
            results.append({
                "id": entry.id,
                "content": entry.content,
                "relevance": round(entry.weight, 4),
                "category": entry.category,
                "created_at": entry.created_at.isoformat(),
                "tags": entry.tags,
            })

        note = None
        if not results:
            note = "No matching memories found"
        elif envelope.truncated:
            note = (
                f"Results truncated — {envelope.total_candidates} total "
                f"candidates, showing top {len(results)}"
            )

        return {
            "results": results,
            "count": len(results),
            "total_candidates": envelope.total_candidates,
            "query_id": query,
            **({"note": note} if note else {}),
        }

    def _ingest(self, content: str, role: str) -> dict[str, Any]:
        """Save a dialogue turn into the memory system."""
        if not content.strip():
            return {"error": "Empty content — nothing to ingest"}

        from memory_skill.contracts import DialogueTurn

        turn_id = f"mcp_{role}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}"
        turn = DialogueTurn(
            id=turn_id,
            role=role,
            content=content,
            timestamp=datetime.now(UTC),
        )

        try:
            envelope = self._skill.ingest(turn)
        except Exception as exc:
            logger.exception("Ingest failed")
            return {"error": f"Ingest error: {type(exc).__name__}: {exc}"}

        return {
            "status": "ingested",
            "turn_id": turn_id,
            "role": role,
            "content_length": len(content),
            "count": envelope.total_candidates,
            "timestamp": envelope.timestamp.isoformat()
        }

    def _status(self) -> dict[str, Any]:
        """Return a health summary of the memory system."""
        try:
            health = self._skill.health()
        except Exception as exc:
            logger.exception("Health check failed")
            return {"error": f"Health check error: {type(exc).__name__}: {exc}"}

        # ── Protocol injection ──
        health["protocol"] = MEMORY_PROTOCOL

        return health

    def _feedback(
        self,
        query_id: str,
        outcome: str,
        cited_ids: list[str],
        search_results: list[dict[str, Any]] | None = None,
        final_response: str | None = None,
    ) -> dict[str, Any]:
        """Record feedback and boost weights for cited memories."""
        if not query_id.strip():
            return {"error": "query_id is required"}

        # ── Auto-detect outcome if requested ──────────────────────────────
        if outcome == "auto" or (not outcome):
            from memory_skill.feedback import auto_detect_outcome

            outcome = auto_detect_outcome(
                query=query_id,
                search_results=search_results or [],
                final_response=final_response or "",
            )
            logger.debug(
                "Auto-detected outcome: %s for query %r", outcome, query_id
            )
        elif outcome not in ("positive", "negative", "neutral"):
            return {
                "error": (
                    f"Invalid outcome '{outcome}'. "
                    "Must be one of: positive, negative, neutral, auto"
                )
            }

        cited_ids = cited_ids or []

        # ── Boost weights for cited memories ───────────────────────────────
        for mid in cited_ids[:3]:
            try:
                self._skill.boost_weight(mid)
            except Exception:
                pass

        return {
            "status": "recorded",
            "query_id": query_id,
            "outcome": outcome,
            "recorded": len(cited_ids)
        }

    # ── Learning loop tools ─────────────────────────────────────────────

    def _learn(self, topic: str, urls: list[str]) -> dict[str, Any]:
        """Closed-loop knowledge acquisition: crawl → synthesize → ingest → verify."""
        if not topic.strip():
            return {"error": "topic is required"}
        if not urls:
            return {"error": "at least one URL is required"}

        try:
            task = self._skill.learn(topic, urls)
            return {
                "task_id": task.id,
                "topic": task.topic,
                "status": task.status,
                "attempts": task.attempts,
                "status_log": [
                    {"status": s, "detail": d}
                    for s, d, _ in task.status_log
                ],
            }
        except Exception as exc:
            logger.exception("Learn task failed for %r", topic)
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _gaps(self) -> dict[str, Any]:
        """Return currently detected knowledge gaps."""
        gaps = self._skill.gaps
        return {
            "count": len(gaps),
            "gaps": [
                {
                    "query": g.query[:120],
                    "branch": g.branch,
                    "severity": g.severity,
                    "confidence": g.confidence,
                }
                for g in gaps[-20:]
            ],
        }
