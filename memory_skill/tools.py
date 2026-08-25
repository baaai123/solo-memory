"""Standalone tool handler functions — extracted from mcp_tools.ToolHandler.

Each function is a pure handler: it takes a MemorySkill instance and
arguments, returns a result dict. No class state, no dispatch logic.
ToolHandler.handle() delegates here via a name→function mapping.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("memory_skill.tools")


# ── Lazy helpers (module-level singletons) ────────────────────────────

# Per-instance caches (WeakKeyDictionary so skills can be GC'd).
from weakref import WeakKeyDictionary

_registries: WeakKeyDictionary = WeakKeyDictionary()
_writers: WeakKeyDictionary = WeakKeyDictionary()


def _get_registry(skill):
    # Per-instance cache so multiple MemorySystem instances (test dbs, room
    # agents) never share one registry bound to the first skill's tree.
    if skill not in _registries:
        from memory_skill.capability_registry import CapabilityRegistry
        from memory_skill.skill_registry import SkillRegistry
        capability = CapabilityRegistry(skill.tree, skill.retriever)
        _registries[skill] = SkillRegistry(skill.retriever, capability)
    return _registries[skill]


def _get_writer(skill):
    if skill not in _writers:
        from memory_skill.skill_writer import SkillWriter
        _writers[skill] = SkillWriter(skill)
    return _writers[skill]


# ── Handler functions ─────────────────────────────────────────────────

def handle_weave(skill, args: dict[str, Any]) -> dict[str, Any]:
    from memory_skill.contracts import clip_auto_ingest

    user_message = args.get("user_message", "")
    assistant_content = args.get("assistant_content", "")

    if user_message and user_message.strip():
        handle_ingest(skill, {"content": clip_auto_ingest(user_message), "role": "user"})
    if assistant_content and assistant_content.strip():
        handle_ingest(skill, {"content": clip_auto_ingest(assistant_content),
                              "role": "assistant"})

    try:
        ctx = skill.weave(
            user_message=user_message,
            scene_summary=args.get("scene_summary", ""),
            partner=args.get("partner"),
        )
        return {
            "tier1_context": ctx.tier1_context,
            "tier2_context": ctx.tier2_context,
            "memory_nudge": ctx.memory_nudge,
            "prompt_block": ctx.to_prompt_block(),
            "is_empty": ctx.is_empty,
        }
    except Exception as exc:
        logger.exception("weave failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_search(skill, args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query", "")
    if not query.strip():
        return {"error": "query is required"}
    # A mission that searches memory has checked for existing knowledge.
    skill.protocol.mark_skill_checked()
    try:
        result = skill.retrieve(query, limit=args.get("limit", 10))
        from datetime import UTC, datetime
        query_id = f"q_{datetime.now(UTC):%Y%m%d_%H%M%S}_{hash(query) & 0xFFFF:04x}"
        return {
            "query_id": query_id,
            "total_candidates": result.total_candidates,
            "results": [
                {
                    "entry_id": e.id,
                    "content": e.content,
                    "semantic_score": getattr(e, "semantic_score", None),
                    "weight": e.weight,
                    "metadata": getattr(e, "metadata", None),
                }
                for e in result.entries
            ],
        }
    except Exception as exc:
        logger.exception("search failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_ingest(skill, args: dict[str, Any]) -> dict[str, Any]:
    from memory_skill.contracts import DialogueTurn, clip_auto_ingest
    from datetime import UTC, datetime

    content = args.get("content", "")
    role = args.get("role", "user")
    if not content.strip():
        return {"error": "content is required"}
    try:
        turn = DialogueTurn(
            id=f"mcp_{datetime.now(UTC):%Y%m%d_%H%M%S}_{hash(content) & 0xFFFF:04x}",
            role=role,
            content=clip_auto_ingest(content),
            timestamp=datetime.now(UTC),
        )
        receipt = skill.ingest(turn, enrich=False)
        return {
            "entry_id": receipt.entry_id,
            "deduped": receipt.deduped,
            "weight": receipt.weight,
        }
    except Exception as exc:
        logger.exception("ingest failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_status(skill, args: dict[str, Any]) -> dict[str, Any]:
    h = skill.health()
    if h.get("embedder", {}).get("degraded"):
        h["warning"] = (
            "⚠ EMBEDDER DEGRADED — 语义检索/去重/学习判定已降级为 SHA-256 hash"
            "（无 ONNX 模型）。修复: 运行 ./download_model.sh 下载 bge-large-en-v1.5，"
            "或 export MEMORY_MODEL_PATH=<模型路径> 后重启。"
        )
    return h


def handle_feedback(skill, args: dict[str, Any]) -> dict[str, Any]:
    query_id = args.get("query_id", "")
    outcome = args.get("outcome", "auto")
    cited_ids = args.get("cited_ids", [])
    if not query_id:
        return {"error": "query_id is required"}
    try:
        boosted: list[str] = []
        for entry_id in cited_ids:
            skill.learned_store.boost_weight(entry_id)
            boosted.append(entry_id)
        return {
            "status": "feedback_recorded",
            "query_id": query_id,
            "outcome": outcome,
            "boosted": boosted,
            "cited": cited_ids,
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_check_skill(skill, args: dict[str, Any]) -> dict[str, Any]:
    skill_name = args.get("skill_name", "")
    if not skill_name.strip():
        return {"error": "skill_name is required"}
    # Checking existing skills satisfies the mission gate.
    skill.protocol.mark_skill_checked()
    try:
        return _get_registry(skill).check_skill(skill_name)
    except Exception as exc:
        logger.exception("check_skill failed for %r", skill_name)
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_teach_skill(skill, args: dict[str, Any]) -> dict[str, Any]:
    return _get_writer(skill).teach_skill(
        title=args.get("title", ""),
        content=args.get("content", ""),
        source_urls=args.get("source_urls"),
    )


def handle_update_skill(skill, args: dict[str, Any]) -> dict[str, Any]:
    entry_id = args.get("entry_id", "")
    if not entry_id.strip():
        return {"error": "entry_id is required"}
    try:
        return _get_writer(skill).update_skill(entry_id, args.get("content", ""))
    except Exception as exc:
        logger.exception("update_skill failed for %s", entry_id)
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_classify(skill, args: dict[str, Any]) -> dict[str, Any]:
    category = args.get("category", "chat")
    note = args.get("note", "")
    gaps = args.get("gaps")
    skill.protocol.mark_classified(category, gaps)
    _enqueue_if_learning(skill, category, note, gaps)
    return {"status": "classified", "category": category, "note": note}


def _enqueue_if_learning(skill, category: str, note: str, gaps) -> None:
    """Enqueue skill/mission items into the learning queue.

    The classifier routing a turn to ``skill`` or ``mission`` is the only
    write-side entry of the active-learning loop (see learning_queue docs).
    Without this, [待学习]/[待拆解] directives never appear for new turns.
    Dedup happens inside ``enqueue`` (identical open items are skipped).
    """
    if category not in ("skill", "mission"):
        return
    query = (note or "").strip()
    if not query:
        return
    queue = getattr(skill, "learning_queue", None)
    if queue is None:
        return
    detail = ""
    if category == "mission" and gaps:
        detail = "所需技能: " + ", ".join(str(g) for g in gaps if g)
    try:
        queue.enqueue(category, query, detail=detail)
    except Exception as exc:
        logger.warning("learning_queue enqueue failed: %s", exc)


def handle_learning_queue(skill, args: dict[str, Any]) -> dict[str, Any]:
    gaps = skill.gaps
    return {
        "count": len(gaps),
        "items": [
            {
                "kind": g.kind,
                "query": getattr(g, "query", "")[:120],
                "detail": getattr(g, "detail", "")[:200],
                "status": getattr(g, "status", "open"),
                "id": getattr(g, "id", ""),
            }
            for g in gaps[-20:]
        ],
    }


def handle_conclusions(skill, args: dict[str, Any]) -> dict[str, Any]:
    limit = args.get("limit", 10)
    try:
        entries = skill.learned_store.list_by_category("conclusion", limit=limit)
        return {
            "conclusions": [
                {
                    "id": e.id,
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", e.content[:120]),
                    "content": e.content,
                    "weight": e.weight,
                }
                for e in entries
            ],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_learning_mark(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Mark a learning-queue item done/skipped (e.g. a completed mission).

    Missions are never auto-closed by teach_skill (that only matches
    ``kind=skill``); the agent closes them explicitly once the mission has
    been decomposed or executed. Requires the item id from
    ``memory_learning_queue``.
    """
    item_id = args.get("item_id", "")
    status = args.get("status", "done")
    if not item_id.strip():
        return {"error": "item_id is required"}
    if status not in ("done", "skipped"):
        return {"error": f"status must be 'done' or 'skipped', got {status!r}"}
    queue = getattr(skill, "learning_queue", None)
    if queue is None:
        return {"error": "learning_queue not available (tree disabled?)"}
    try:
        ok = queue.mark(item_id.strip(), status)
        return {"status": "marked" if ok else "not_found_or_already_closed",
                "item_id": item_id.strip(), "new_status": status}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_distill(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Turn dialogue fragments into reviewable candidate cards.

    Pure decision-support: compresses fragments into topic/summary/evidence
    candidates and writes them to the pending store.  Nothing is asserted
    or promoted — the agent reviews via memory_pending and decides.

    ``offset``/``limit`` walk the history window by window (default: newest
    60 turns).  Pass ``offset=60`` next to reach older memories.
    """
    pending = getattr(skill, "pending_store", None)
    if pending is None:
        return {"error": "pending_store not available (tree disabled?)"}
    from memory_skill.distill import distill as _distill
    try:
        return _distill(
            skill.dialogue_store, pending,
            since_days=args.get("since_days", 7),
            offset=args.get("offset", 0),
            limit=args.get("limit", 60),
        )
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_pending(skill, args: dict[str, Any]) -> dict[str, Any]:
    """List open distill candidates awaiting agent review."""
    pending = getattr(skill, "pending_store", None)
    if pending is None:
        return {"error": "pending_store not available (tree disabled?)"}
    try:
        items = pending.list_open(limit=args.get("limit", 20))
        return {
            "count": len(items),
            "items": [
                {
                    "candidate_id": c.candidate_id,
                    "topic": c.topic,
                    "summary": c.summary,
                    "evidence": c.evidence,
                    "suggested": c.suggested,
                    "confidence": c.confidence,
                    "created_at": c.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                }
                for c in items
            ],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_pending_mark(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Accept or reject a distill candidate.

    Accepting a ``conclusion``/``pref``/``pers`` candidate auto-promotes it
    to the structured store (the agent's decision is the review gate).
    Accepting a ``skill`` candidate only records the decision — the agent
    must still teach it via ``memory_teach_skill`` with source_urls (the
    ADR-0002 anti-hallucination gate; distill evidence ids are dialogue
    ids, not URLs).
    """
    pending = getattr(skill, "pending_store", None)
    if pending is None:
        return {"error": "pending_store not available (tree disabled?)"}
    candidate_id = args.get("candidate_id", "")
    status = args.get("status", "")
    if not candidate_id.strip():
        return {"error": "candidate_id is required"}
    if status not in ("accepted", "rejected"):
        return {"error": f"status must be 'accepted' or 'rejected', got {status!r}"}

    cand = None
    for c in pending.list_open(limit=100):
        if c.candidate_id == candidate_id.strip():
            cand = c
            break
    if cand is None:
        return {"error": "candidate not found or already closed"}
    if not pending.mark(candidate_id.strip(), status):
        return {"status": "not_found_or_already_closed",
                "candidate_id": candidate_id.strip()}

    promoted: str | None = None
    if status == "accepted" and cand.suggested in ("conclusion", "pref", "pers"):
        try:
            from memory_skill.memory_extract import _ingest_structured
            result = _ingest_structured(
                skill, cand.topic, cand.summary, cand.suggested,
            )
            promoted = getattr(result, "entry_id", None) or cand.suggested
        except Exception as exc:
            return {"status": "marked",
                    "candidate_id": candidate_id.strip(),
                    "promotion_error": f"{type(exc).__name__}: {exc}"}

    extra: dict[str, Any] = {}
    if promoted:
        extra["promoted_to"] = promoted
    if status == "accepted" and cand.suggested == "skill":
        extra["note"] = "已接受但需 memory_teach_skill(带 source_urls) 转正"
    return {"status": "marked",
            "candidate_id": candidate_id.strip(),
            **extra}


# ── Dispatch map ───────────────────────────────────────────────────────

DISPATCH = {
    "memory_weave": handle_weave,
    "memory_search": handle_search,
    "memory_ingest": handle_ingest,
    "memory_status": handle_status,
    "memory_feedback": handle_feedback,
    "memory_check_skill": handle_check_skill,
    "memory_teach_skill": handle_teach_skill,
    "memory_update_skill": handle_update_skill,
    "memory_classify": handle_classify,
    "memory_learning_queue": handle_learning_queue,
    "memory_learning_mark": handle_learning_mark,
    "memory_distill": handle_distill,
    "memory_pending": handle_pending,
    "memory_pending_mark": handle_pending_mark,
    "memory_conclusions": handle_conclusions,
}

# ── Tool schemas (single source of truth) ────────────────────────
# Each tool's MCP schema lives next to its handler. mcp_tools.TOOLS
# is derived from this map so adding a tool means editing ONE place.
TOOL_SCHEMAS: dict[str, dict] = {
    "memory_check_skill": {
        "description": "Check whether a skill is already known to memory (scoped to the skill category). Returns 'known' | 'partial' | 'unknown' with matching skill entries. Use BEFORE learning a new topic \u2014 if known, reuse or update the existing skill instead of re-learning.",
        "inputSchema": {"type": "object", "properties": {"skill_name": {"type": "string", "description": "The skill name to check, e.g. 'Docker Compose'."}}, "required": ["skill_name"]},
    },
    "memory_classify": {
        "description": "Classify the current user message. REQUIRED after every memory_weave call \u2014 the next weave() is rejected until classification is done. Pass gaps when category='mission': system blocks weave until all gaps are taught.",
        "inputSchema": {"type": "object", "properties": {"category": {"type": "string", "description": "Classification: 'chat' | 'skill' | 'mission' | 'pref' | 'pers'", "enum": ["chat", "skill", "mission", "pref", "pers"]}, "note": {"type": "string", "description": "Optional note: skill name, mission summary, etc."}, "gaps": {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}], "description": "Required skill names for a mission (blocks weave until taught)"}}, "required": ["category"]},
    },
    "memory_conclusions": {
        "description": "List reusable conclusion entries extracted from assistant replies (category=conclusion, title + summary + evidence format). Queries the store directly by category \u2014 NOT semantic search \u2014 so the result is complete and accurate. Use to verify what conclusions the memory system has distilled.",
        "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10, "description": "Max conclusions to return (newest first)."}}},
    },
    "memory_distill": {
        "description": "Turn dialogue fragments into reviewable candidate cards (topic/summary/evidence/suggested). Pure decision-support \u2014 nothing is promoted; review via memory_pending. Walks history window by window: offset=0 is newest 60 turns, pass offset=60 to reach older memories.",
        "inputSchema": {"type": "object", "properties": {"since_days": {"type": "integer", "description": "Look back this many days (default 7)."}, "offset": {"type": "integer", "description": "Skip this many newest turns (default 0)."}, "limit": {"type": "integer", "description": "Window size per call (default 60)."}}},
    },
    "memory_feedback": {
        "description": "Record feedback on a previous memory_search result. Provide the query_id from the search response, the outcome ('positive', 'negative', 'neutral', or 'auto' for automatic detection), and the list of memory IDs you cited or found useful. To enable auto-detection, also pass search_results and final_response. This boosts the weights of cited memories to improve future retrieval.\n\n",
        "inputSchema": {"type": "object", "properties": {"query_id": {"type": "string", "description": "The query_id from a previous memory_search response. Used to correlate feedback with the search."}, "outcome": {"type": "string", "description": "How useful the search results were: 'positive' (found what I needed), 'negative' (results were irrelevant), 'neutral' (mixed or unsure), or 'auto' (let the system auto-detect).", "enum": ["positive", "negative", "neutral", "auto"]}, "cited_ids": {"type": "array", "description": "List of memory entry IDs from the search results that you used or cited in your response.", "items": {"type": "string"}}, "search_results": {"type": "array", "description": "Optional: the search results from the previous memory_search call (for auto-detection). Each item should have 'id' and 'content' fields.", "items": {"type": "object"}}, "final_response": {"type": "string", "description": "Optional: the final response the agent produced using the search results (for auto-detection)."}}, "required": ["query_id"]},
    },
    "memory_ingest": {
        "description": "Save a dialogue turn into the memory system for future retrieval. Call this after significant interactions \u2014 user questions, important answers, decisions made, facts learned. Each turn is indexed for both semantic search and full-text retrieval.\n\n",
        "inputSchema": {"type": "object", "properties": {"content": {"type": "string", "description": "The content to save \u2014 typically the user's question or your response that should be remembered."}, "role": {"type": "string", "default": "user", "description": "Speaker role: 'user', 'assistant', or 'system'.", "enum": ["user", "assistant", "system"]}}, "required": ["content"]},
    },
    "memory_learning_mark": {
        "description": "Mark a learning-queue item as done or skipped (e.g. a completed mission). Missions are never auto-closed \u2014 the agent closes them explicitly after decomposition/execution. Pass the item id from memory_learning_queue.",
        "inputSchema": {"type": "object", "properties": {"item_id": {"type": "string", "description": "The learning-queue item id (lq_...)."}, "status": {"type": "string", "description": "'done' (default) or 'skipped'.", "enum": ["done", "skipped"]}}, "required": ["item_id"]},
    },
    "memory_learning_queue": {
        "description": "List pending learning items: skills the classifier flagged as not-yet-mastered and missions awaiting decomposition. Open items are rendered by weave as [\u5f85\u5b66\u4e60]/[\u5f85\u62c6\u89e3] directives \u2014 act on them and confirm results with the user.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "memory_pending": {
        "description": "List open distill candidates awaiting agent review. After reviewing, accept or reject via memory_pending_mark; promote valuable content with memory_teach_skill (skills).",
        "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "description": "Max candidates to return (default 20)."}}},
    },
    "memory_pending_mark": {
        "description": "Accept or reject a distill candidate. Accepted candidates must still be promoted by the agent (teach_skill / structured ingest); this only records the review decision.",
        "inputSchema": {"type": "object", "properties": {"candidate_id": {"type": "string", "description": "Candidate id from memory_pending (dc_...)."}, "status": {"type": "string", "enum": ["accepted", "rejected"]}}, "required": ["candidate_id", "status"]},
    },
    "memory_search": {
        "description": "Search agent memory for relevant past conversations and learned knowledge. Returns ranked results with content, relevance scores, and memory IDs. Use this BEFORE responding to any question about past events, user preferences, or previously discussed topics.",
        "inputSchema": {"type": "object", "properties": {"query": {"type": "string", "description": "Natural language search query. Be specific \u2014 include names, topics, dates for best results."}, "limit": {"type": "integer", "default": 10, "description": "Maximum number of results to return (1-100).", "minimum": 1, "maximum": 100}}, "required": ["query"]},
    },
    "memory_status": {
        "description": "Get a health summary of the memory system: entry counts, embedder status, and configuration. Call this on first wake-up to understand the memory system state.\n\n",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "memory_teach_skill": {
        "description": "Persist a taught skill at high confidence. Use AFTER learning a topic (via web search or user instruction): pass the skill title and the learned content. Returns the stored entry_id. The user may later correct it via memory_update_skill.",
        "inputSchema": {"type": "object", "properties": {"title": {"type": "string", "description": "Skill title, e.g. 'Docker Compose \u591a\u5bb9\u5668\u90e8\u7f72'."}, "content": {"type": "string", "description": "The learned knowledge (markdown ok)."}, "source_urls": {"type": "array", "description": "Source URLs used for learning \u2014 required, teaching must be backed by external references, not guesswork.", "items": {"type": "string"}}}, "required": ["title", "content", "source_urls"]},
    },
    "memory_update_skill": {
        "description": "Rewrite a skill entry's content \u2014 a correction from the user or the agent after learning. Directly replaces the stored content (not a semantic merge), so skills stay writable for teaching.",
        "inputSchema": {"type": "object", "properties": {"entry_id": {"type": "string", "description": "The skill entry id to update."}, "content": {"type": "string", "description": "The corrected skill content (markdown ok)."}}, "required": ["entry_id", "content"]},
    },
    "memory_weave": {
        "description": "Auto-assemble layered memory context for the current conversation. Call BEFORE responding to any user message. Returns a context block ready for system-prompt injection. tier1: scene perception, tier2: deep memory (gated), nudge: high-importance reminders.\n\nV10: Both user_message AND assistant_content are auto-ingested (ImportanceScorer filters trivial content). This mirrors the Room framework's conversation-block pattern where both sides of every exchange are persisted without manual ingest calls.",
        "inputSchema": {"type": "object", "properties": {"user_message": {"type": "string", "description": "The user's current message."}, "assistant_content": {"type": "string", "description": "Optional: your OWN response from the PREVIOUS exchange. Auto-ingested into memory so both sides of every conversation block are persisted. Pass your full last response."}, "scene_summary": {"type": "string", "description": "Optional: what the user is doing now."}}},
    },
}
