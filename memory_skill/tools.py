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
    return {"status": "classified", "category": category, "note": note}


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
        entries = [
            e for e in skill.learned_store._entries.values()
            if getattr(e, "category", None) == "conclusion"
        ]
        entries.sort(key=lambda e: getattr(e, "created_at", 0), reverse=True)
        return {
            "conclusions": [
                {
                    "id": e.id,
                    "title": getattr(e, "title", ""),
                    "summary": getattr(e, "summary", e.content[:120]),
                    "content": e.content,
                    "weight": e.weight,
                }
                for e in entries[:limit]
            ],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


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
    "memory_conclusions": handle_conclusions,
}
