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

_registry = None
_writer = None


def _get_registry(skill):
    global _registry
    if _registry is None:
        from memory_skill.capability_registry import CapabilityRegistry
        from memory_skill.skill_registry import SkillRegistry
        capability = CapabilityRegistry(skill.tree, skill.retriever)
        _registry = SkillRegistry(skill.retriever, capability)
    return _registry


def _get_writer(skill):
    global _writer
    if _writer is None:
        from memory_skill.skill_writer import SkillWriter
        _writer = SkillWriter(skill)
    return _writer


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
    try:
        result = skill.retrieve(query, limit=args.get("limit", 10))
        return {
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
    return skill.health()


def handle_feedback(skill, args: dict[str, Any]) -> dict[str, Any]:
    query_id = args.get("query_id", "")
    outcome = args.get("outcome", "auto")
    cited_ids = args.get("cited_ids", [])
    if not query_id:
        return {"error": "query_id is required"}
    try:
        skill.learned_store.boost_weight(query_id)
        return {"status": "feedback_recorded", "outcome": outcome, "cited": cited_ids}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_check_skill(skill, args: dict[str, Any]) -> dict[str, Any]:
    skill_name = args.get("skill_name", "")
    if not skill_name.strip():
        return {"error": "skill_name is required"}
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
    skill._classify_pending = None
    if category == "mission" and gaps:
        skill._pending_gaps = set(g.strip() for g in gaps if g.strip())
    else:
        skill._pending_gaps.clear()
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
