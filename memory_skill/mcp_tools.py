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
        "name": "memory_check_skill",
        "description": (
            "Check whether a skill is already known to memory (scoped to the "
            "skill category). Returns 'known' | 'partial' | 'unknown' with "
            "matching skill entries. Use BEFORE learning a new topic — if "
            "known, reuse or update the existing skill instead of re-learning."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "The skill name to check, e.g. 'Docker Compose'.",
                },
            },
            "required": ["skill_name"],
        },
    },
    {
        "name": "memory_teach_skill",
        "description": (
            "Persist a taught skill at high confidence. Use AFTER learning a "
            "topic (via web search or user instruction): pass the skill title "
            "and the learned content. Returns the stored entry_id. The user "
            "may later correct it via memory_update_skill."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Skill title, e.g. 'Docker Compose 多容器部署'.",
                },
                "content": {
                    "type": "string",
                    "description": "The learned knowledge (markdown ok).",
                },
                "source_urls": {
                    "type": "array",
                    "description": "Source URLs used for learning — required, "
                    "teaching must be backed by external references, not guesswork.",
                    "items": {"type": "string"},
                },
            },
            "required": ["title", "content", "source_urls"],
        },
    },
    {
        "name": "memory_update_skill",
        "description": (
            "Rewrite a skill entry's content — a correction from the user or "
            "the agent after learning. Directly replaces the stored content "
            "(not a semantic merge), so skills stay writable for teaching."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entry_id": {
                    "type": "string",
                    "description": "The skill entry id to update.",
                },
                "content": {
                    "type": "string",
                    "description": "The corrected skill content (markdown ok).",
                },
            },
            "required": ["entry_id", "content"],
        },
    },
    {
        "name": "memory_classify",
        "description": (
            "Classify the current user message. REQUIRED after every "
            "memory_weave call — the next weave() is rejected until "
            "classification is done. Pass gaps when category='mission': "
            "system blocks weave until all gaps are taught."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Classification: 'chat' | 'skill' | 'mission' | 'pref' | 'pers'",
                    "enum": ["chat", "skill", "mission", "pref", "pers"],
                },
                "note": {
                    "type": "string",
                    "description": "Optional note: skill name, mission summary, etc.",
                },
                "gaps": {
                    "type": "array",
                    "description": "Required skill names for a mission (blocks weave until taught)",
                    "items": {"type": "string"},
                },
            },
            "required": ["category"],
        },
    },
    {
        "name": "memory_learning_queue",
        "description": (
            "List pending learning items: skills the classifier flagged as "
            "not-yet-mastered and missions awaiting decomposition. Open items "
            "are rendered by weave as [待学习]/[待拆解] directives — act on "
            "them and confirm results with the user."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "memory_learning_mark",
        "description": (
            "Mark a learning-queue item as done or skipped (e.g. a completed "
            "mission). Missions are never auto-closed — the agent closes them "
            "explicitly after decomposition/execution. Pass the item id from "
            "memory_learning_queue."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "item_id": {
                    "type": "string",
                    "description": "The learning-queue item id (lq_...).",
                },
                "status": {
                    "type": "string",
                    "description": "'done' (default) or 'skipped'.",
                    "enum": ["done", "skipped"],
                },
            },
            "required": ["item_id"],
        },
    },
    {
        "name": "memory_distill",
        "description": (
            "Turn dialogue fragments into reviewable candidate cards "
            "(topic/summary/evidence/suggested). Pure decision-support — "
            "nothing is promoted; review via memory_pending. Walks history "
            "window by window: offset=0 is newest 60 turns, pass offset=60 "
            "to reach older memories."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "since_days": {
                    "type": "integer",
                    "description": "Look back this many days (default 7).",
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip this many newest turns (default 0).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Window size per call (default 60).",
                },
            },
        },
    },
    {
        "name": "memory_pending",
        "description": (
            "List open distill candidates awaiting agent review. After "
            "reviewing, accept or reject via memory_pending_mark; promote "
            "valuable content with memory_teach_skill (skills)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max candidates to return (default 20).",
                },
            },
        },
    },
    {
        "name": "memory_pending_mark",
        "description": (
            "Accept or reject a distill candidate. Accepted candidates must "
            "still be promoted by the agent (teach_skill / structured ingest); "
            "this only records the review decision."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "candidate_id": {
                    "type": "string",
                    "description": "Candidate id from memory_pending (dc_...).",
                },
                "status": {
                    "type": "string",
                    "enum": ["accepted", "rejected"],
                },
            },
            "required": ["candidate_id", "status"],
        },
    },
    {
        "name": "memory_conclusions",
        "description": (
            "List reusable conclusion entries extracted from assistant replies "
            "(category=conclusion, title + summary + evidence format). Queries "
            "the store directly by category — NOT semantic search — so the "
            "result is complete and accurate. Use to verify what conclusions "
            "the memory system has distilled."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max conclusions to return (newest first).",
                },
            },
        },
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# ToolHandler
# ═══════════════════════════════════════════════════════════════════════════════

# KNOWN-ISSUES #8: full-length assistant_content (e.g. a whole summary block)
# makes the auto-ingest embedding slow enough to trip MCP -32001 timeouts.
# Cap what gets embedded at ingest time; retrieval/weave still sees the
# full query where it matters.
#
# Head+tail retention: Chinese replies often put the conclusion at the END,
# so naive head-only truncation would drop it (KNOWN-ISSUES #9 "结论散落").
# Keep the first `_HEAD` chars + last `_TAIL` chars, ellipsis in between.

class ToolHandler:
    """Dispatches MCP tool calls to standalone handler functions in tools.py."""

    def __init__(self, skill: MemorySkill) -> None:
        self._skill = skill

    def handle(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from memory_skill.tools import DISPATCH
        handler = DISPATCH.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(self._skill, arguments)
        except Exception as exc:
            logger.exception("Tool %s failed", tool_name)
            return {"error": f"{type(exc).__name__}: {exc}"}
