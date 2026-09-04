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


def handle_ingest_pers(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Write a persona rule/trait into the pers category.

    Delegates to ``skill.ingest_pers`` (memory_extract): appends the trait
    to the longest existing ``# ``-prefixed persona card, or creates a new
    pers entry when no card exists.  The weaver's [人格特征] block reads
    the pers category, so traits written here surface in every weave.
    A ``None`` result means the store deduped the write (duplicate).
    """
    raw = args.get("trait")
    if not isinstance(raw, str) or not raw.strip():
        return {"error": "trait is required"}
    trait = raw.strip()
    ingest_pers = getattr(skill, "ingest_pers", None)
    if ingest_pers is None:
        return {"error": "ingest_pers not available on this skill"}
    try:
        result = ingest_pers(trait)
    except Exception as exc:
        logger.exception("ingest_pers failed for %r", trait)
        return {"error": f"{type(exc).__name__}: {exc}"}
    if result is None:
        return {"status": "duplicate_skipped", "trait": trait}
    return {"status": "ingested", "trait": trait, "persisted": True}


def handle_ingest_conclusion(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Persist a root-cause / knowledge conclusion into the conclusion category.

    Delegates to ``skill.ingest_conclusion`` (memory_extract).  The weaver's
    [历史结论] block reads the conclusion category (newest first), so
    conclusions written here surface in every weave.  Call this whenever a
    debugging session or investigation reaches a root-cause verdict.
    """
    raw_title = args.get("title")
    if not isinstance(raw_title, str) or not raw_title.strip():
        return {"error": "title is required"}
    title = raw_title.strip()
    content = args.get("content", "")
    if not isinstance(content, str):
        content = str(content)
    ingest_conclusion = getattr(skill, "ingest_conclusion", None)
    if ingest_conclusion is None:
        return {"error": "ingest_conclusion not available on this skill"}
    try:
        result = ingest_conclusion(title, content.strip())
    except Exception as exc:
        logger.exception("ingest_conclusion failed for %r", title)
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"status": "ingested", "title": title, "persisted": True}


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


# ── Mission structured records ────────────────────────────────────────

def _get_mission_store(skill):
    store = getattr(skill, "mission_store", None)
    if store is None:
        from memory_skill.mission import MissionStore
        store = MissionStore(
            learned_store=skill.learned_store,
            learning_queue=getattr(skill, "learning_queue", None),
        )
        try:
            object.__setattr__(skill, "mission_store", store)
        except Exception:
            pass
    return store


def handle_mission_create(skill, args: dict[str, Any]) -> dict[str, Any]:
    try:
        mission = _get_mission_store(skill).create(
            content=args.get("content", ""),
            title=args.get("title", ""),
        )
        return {"status": "created", "mission": mission.to_dict()}
    except Exception as exc:
        logger.exception("mission_create failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_mission_get(skill, args: dict[str, Any]) -> dict[str, Any]:
    mission_id = args.get("mission_id", "")
    if not mission_id.strip():
        return {"error": "mission_id is required"}
    try:
        mission = _get_mission_store(skill).get(mission_id)
        if mission is None:
            return {"error": f"mission {mission_id!r} not found"}
        return {"mission": mission.to_dict()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_mission_list(skill, args: dict[str, Any]) -> dict[str, Any]:
    try:
        missions = _get_mission_store(skill).list_missions(
            status=args.get("status"),
            limit=args.get("limit", 100),
        )
        return {"count": len(missions),
                "missions": [m.to_dict() for m in missions]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_mission_add_step(skill, args: dict[str, Any]) -> dict[str, Any]:
    mission_id = args.get("mission_id", "")
    if not mission_id.strip():
        return {"error": "mission_id is required"}
    try:
        mission = _get_mission_store(skill).add_step(
            mission_id=mission_id,
            text=args.get("text", ""),
            skill_id=args.get("skill_id", ""),
        )
        return {"mission": mission.to_dict()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_mission_update_step(skill, args: dict[str, Any]) -> dict[str, Any]:
    mission_id = args.get("mission_id", "")
    if not mission_id.strip():
        return {"error": "mission_id is required"}
    try:
        mission = _get_mission_store(skill).update_step(
            mission_id=mission_id,
            index=args.get("index", -1),
            text=args.get("text"),
            skill_id=args.get("skill_id"),
            done=args.get("done"),
        )
        return {"mission": mission.to_dict()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_mission_remove_step(skill, args: dict[str, Any]) -> dict[str, Any]:
    mission_id = args.get("mission_id", "")
    if not mission_id.strip():
        return {"error": "mission_id is required"}
    try:
        mission = _get_mission_store(skill).remove_step(
            mission_id=mission_id,
            index=args.get("index", -1),
        )
        return {"mission": mission.to_dict()}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_mission_set_status(skill, args: dict[str, Any]) -> dict[str, Any]:
    mission_id = args.get("mission_id", "")
    if not mission_id.strip():
        return {"error": "mission_id is required"}
    try:
        mission = _get_mission_store(skill).set_status(
            mission_id=mission_id,
            status=args.get("status", "open"),
        )
        return {"mission": mission.to_dict()}
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
        _clear_queue_pending(skill)
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


# ── Archive: review + reclassify default-category entries ─────────────

# Legal learned-store categories (verified against the live chroma store
# 2026-09-02: default 993 / skill 53 / mission 64 / pref 7 / pers 3 /
# conclusion 25).  Kept in tools.py — contracts.py is server-layer
# territory (Unit 2 hard gate) and must not be touched here.
VALID_CATEGORIES = frozenset(
    {"pref", "pers", "skill", "mission", "conclusion", "default"}
)


def _clear_archive_pending(skill) -> None:
    """Disarm the archive hard-gate after any archive-tool response.

    Unit 2 anti-deadlock rule (R8): a successful review_default /
    reclassify call proves the agent responded, so the gate clears even
    when zero entries moved.  Defensive: shells without a protocol
    reference (e.g. minimal test compositions) skip silently.
    """
    protocol = getattr(skill, "protocol", None)
    if protocol is not None and hasattr(protocol, "clear_archive_pending"):
        protocol.clear_archive_pending()


def _clear_queue_pending(skill) -> None:
    """Disarm the queue hard-gate after any learning_mark response."""
    protocol = getattr(skill, "protocol", None)
    if protocol is not None and hasattr(protocol, "clear_queue_pending"):
        protocol.clear_queue_pending()


def handle_review_default(skill, args: dict[str, Any]) -> dict[str, Any]:
    """List default-category entries awaiting reclassification.

    The classifier (08-11 architecture) leaves every turn the agent never
    explicitly categorized as ``default``; this tool is how the agent
    reviews that backlog and reclassifies it via ``memory_reclassify``.
    """
    try:
        limit = int(args.get("limit", 10))
        offset = int(args.get("offset", 0))
    except (TypeError, ValueError):
        return {"error": "limit/offset must be integers"}
    limit = max(0, min(limit, 50))
    offset = max(0, offset)
    try:
        all_entries = skill.learned_store.list_by_category(
            "default", limit=0, sort_by="updated_at", descending=True,
        )
        total = len(all_entries)
        page = all_entries[offset:] if limit == 0 else all_entries[offset:offset + limit]
        _clear_archive_pending(skill)
        return {
            "count": len(page),
            "total": total,
            "items": [
                {
                    "id": e.id,
                    "content": e.content[:200],
                    "title": e.metadata.get("title", ""),
                    "category": e.category,
                    "updated_at": e.updated_at.isoformat() if e.updated_at else None,
                }
                for e in page
            ],
        }
    except Exception as exc:
        logger.exception("review_default failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_reclassify(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Move an entry out of ``default`` into a real category.

    Lightweight: only the ``category`` metadata field changes — content is
    not re-embedded.  When the entry has no title, one is derived from the
    first 30 chars of the content.
    """
    entry_id = str(args.get("entry_id") or "").strip()
    category = str(args.get("category") or "").strip()
    if not entry_id:
        return {"error": "entry_id is required"}
    if not category:
        return {"error": "category is required"}
    if category not in VALID_CATEGORIES:
        return {"error": f"invalid category: {category}"}
    try:
        entry = skill.learned_store.get_entry(entry_id)
        if entry is None:
            return {"error": f"entry not found: {entry_id}"}
        old_category = entry.category
        old_title = str(entry.metadata.get("title") or "").strip()
        skill.learned_store.set_category(entry_id, category)
        title = old_title
        if not title:
            title = entry.content[:30]
            skill.learned_store.set_title(entry_id, title)
        _clear_archive_pending(skill)
        return {
            "ok": True,
            "entry_id": entry_id,
            "category": category,
            "old_category": old_category,
            "title": title,
        }
    except Exception as exc:
        logger.exception("reclassify failed for %s", entry_id)
        return {"error": f"{type(exc).__name__}: {exc}"}


# ── Character (role) management ───────────────────────────────────────

def _get_character_store(skill):
    """Return the skill's CharacterStore, or None when not composed."""
    return getattr(skill, "character", None)


def handle_character_list(skill, args: dict[str, Any]) -> dict[str, Any]:
    """List every role (character) with its memory ref_count."""
    character = _get_character_store(skill)
    if character is None:
        return {"error": "character store not available"}
    try:
        return {"roles": character.list_roles()}
    except Exception as exc:
        logger.exception("character_list failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_character_create(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Create a role — a reference set over global memories, not a copy."""
    character = _get_character_store(skill)
    if character is None:
        return {"error": "character store not available"}
    name = str(args.get("name") or "").strip()
    if not name:
        return {"error": "name is required"}
    try:
        role_id = character.create_role(
            name,
            description=str(args.get("description") or ""),
        )
        return {"status": "created", "role_id": role_id}
    except Exception as exc:
        logger.exception("character_create failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_character_get(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Return a role's details plus its referenced memory ids."""
    character = _get_character_store(skill)
    if character is None:
        return {"error": "character store not available"}
    role_id = str(args.get("role_id") or "").strip()
    if not role_id:
        return {"error": "role_id is required"}
    try:
        role = character.get_role(role_id)
        if role is None:
            return {"error": f"role not found: {role_id}"}
        return {
            "role": role,
            "memory_ids": character.list_memories(role_id),
        }
    except Exception as exc:
        logger.exception("character_get failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_character_delete(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Delete a role; cascades its memory references and agent bindings."""
    character = _get_character_store(skill)
    if character is None:
        return {"error": "character store not available"}
    role_id = str(args.get("role_id") or "").strip()
    if not role_id:
        return {"error": "role_id is required"}
    try:
        if not character.delete_role(role_id):
            return {"error": f"role not found: {role_id}"}
        return {"status": "deleted", "role_id": role_id}
    except Exception as exc:
        logger.exception("character_delete failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_character_add_memory(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Add a global memory reference to a role (idempotent)."""
    character = _get_character_store(skill)
    if character is None:
        return {"error": "character store not available"}
    role_id = str(args.get("role_id") or "").strip()
    memory_id = str(args.get("memory_id") or "").strip()
    dimension = str(args.get("dimension") or "general").strip() or "general"
    if not role_id:
        return {"error": "role_id is required"}
    if not memory_id:
        return {"error": "memory_id is required"}
    try:
        if not character.add_memory(role_id, memory_id, dimension=dimension):
            return {"error": f"role not found: {role_id}"}
        return {"status": "added", "dimension": dimension}
    except Exception as exc:
        logger.exception("character_add_memory failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_character_remove_memory(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Remove a memory reference from a role (idempotent)."""
    character = _get_character_store(skill)
    if character is None:
        return {"error": "character store not available"}
    role_id = str(args.get("role_id") or "").strip()
    memory_id = str(args.get("memory_id") or "").strip()
    if not role_id:
        return {"error": "role_id is required"}
    if not memory_id:
        return {"error": "memory_id is required"}
    try:
        if not character.remove_memory(role_id, memory_id):
            return {"error": f"role not found: {role_id}"}
        return {"status": "removed"}
    except Exception as exc:
        logger.exception("character_remove_memory failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_character_bind_agent(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Bind an agent to a role; an empty role_id unbinds instead.

    Rebinding an already-bound agent switches it to the new role.
    A bound agent's weave only injects memories inside the role's
    reference set.
    """
    character = _get_character_store(skill)
    if character is None:
        return {"error": "character store not available"}
    agent_name = str(args.get("agent_name") or "")
    role_id = str(args.get("role_id") or "").strip()
    try:
        if not role_id:
            character.unbind_agent(agent_name)
            return {"status": "unbound", "agent_name": agent_name,
                    "role_id": None}
        if not character.bind_agent(agent_name, role_id):
            return {"error": f"role not found: {role_id}"}
        return {"status": "bound", "agent_name": agent_name,
                "role_id": role_id}
    except Exception as exc:
        logger.exception("character_bind_agent failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_character_agent_role(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Return the role id an agent is bound to (None when unbound)."""
    character = _get_character_store(skill)
    if character is None:
        return {"error": "character store not available"}
    agent_name = str(args.get("agent_name") or "")
    try:
        return {"agent_name": agent_name,
                "role_id": character.get_agent_role(agent_name)}
    except Exception as exc:
        logger.exception("character_agent_role failed")
        return {"error": f"{type(exc).__name__}: {exc}"}


def handle_state_extract(skill, args: dict[str, Any]) -> dict[str, Any]:
    """Extract (and optionally apply) a tavern role's current state.

    Sub-agent pattern: ``apply=false`` returns a JSON preview for agent
    review; ``apply=true`` overwrites the role's state snapshot.  State
    extraction never runs inside ingest/weave (ADR-0002).
    """
    store = getattr(skill, "state_store", None)
    if store is None:
        return {"error": "state_store not available (not composed)"}
    role_id = str(args.get("role_id") or "")
    conversation = str(args.get("conversation") or "")
    if not role_id or not conversation:
        return {"error": "role_id and conversation required"}
    from memory_skill.tavern_extract import extract_state

    try:
        current = store.get_state(role_id)
        extracted = extract_state(conversation, current)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if not extracted:
        return {"error": "state extraction failed (bridge down or bad response)"}
    if args.get("apply"):
        store.update_state(role_id, extracted)
        return {"applied": True, "state": store.get_state(role_id)}
    return {"applied": False, "preview": extracted}


# ── Dispatch map ───────────────────────────────────────────────────────

DISPATCH = {
    "memory_weave": handle_weave,
    "memory_search": handle_search,
    "memory_ingest": handle_ingest,
    "memory_ingest_pers": handle_ingest_pers,
    "memory_ingest_conclusion": handle_ingest_conclusion,
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
    "memory_review_default": handle_review_default,
    "memory_reclassify": handle_reclassify,
    "memory_mission_create": handle_mission_create,
    "memory_mission_get": handle_mission_get,
    "memory_mission_list": handle_mission_list,
    "memory_mission_add_step": handle_mission_add_step,
    "memory_mission_update_step": handle_mission_update_step,
    "memory_mission_remove_step": handle_mission_remove_step,
    "memory_mission_set_status": handle_mission_set_status,
    "memory_character_list": handle_character_list,
    "memory_character_create": handle_character_create,
    "memory_character_get": handle_character_get,
    "memory_character_delete": handle_character_delete,
    "memory_character_add_memory": handle_character_add_memory,
    "memory_character_remove_memory": handle_character_remove_memory,
    "memory_character_bind_agent": handle_character_bind_agent,
    "memory_character_agent_role": handle_character_agent_role,
    "memory_state_extract": handle_state_extract,
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
    "memory_review_default": {
        "description": "列出 default 分类下待归位的记忆条目（agent 未主动分类而积压的候选）。返回 count/total/items（id、content 截断 200 字符、title、category、updated_at），按 updated_at 倒序。归档硬门触发时用此工具查看候选，再用 memory_reclassify 归位。",
        "inputSchema": {"type": "object", "properties": {"limit": {"type": "integer", "default": 10, "maximum": 50, "description": "返回条数（默认 10，最大 50）。"}, "offset": {"type": "integer", "default": 0, "description": "跳过条数，用于分页（默认 0）。"}}},
    },
    "memory_reclassify": {
        "description": "将记忆条目归位到 pref/pers/skill/mission/conclusion（或改回 default）。只更新 category 元数据、不重嵌 content（轻量）；title 为空时自动用 content 前 30 字符补全。归档硬门触发时归档候选后调用。",
        "inputSchema": {"type": "object", "properties": {"entry_id": {"type": "string", "description": "条目 id（memory_review_default 返回的 id）。"}, "category": {"type": "string", "enum": ["pref", "pers", "skill", "mission", "conclusion", "default"], "description": "目标分类。"}}, "required": ["entry_id", "category"]},
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
    "memory_ingest_pers": {
        "description": "将人格规则/特质写入人格记忆（pers 分类）。weave 的 [人格特征] 区会显示。用于持久化用户明确要求的规则（如\"禁止使用内置 todo\"）。规则会自动追加到人格卡片。",
        "inputSchema": {"type": "object", "properties": {"trait": {"type": "string", "description": "人格规则/特质描述，必填。自动追加到 pers 分类中最长的 # 开头人格卡片；无卡片时新建。"}}, "required": ["trait"]},
    },
    "memory_ingest_conclusion": {
        "description": "将排查/调查得出的根因结论写入结论记忆（conclusion 分类）。weave 的 [历史结论] 区会显示（最新优先）。每当调试/排查/架构审查得出根因判断时调用，title 简短总结结论，content 补充细节。",
        "inputSchema": {"type": "object", "properties": {"title": {"type": "string", "description": "结论标题（简短，如：find_duplicate bug 由余弦距离反转导致），必填。"}, "content": {"type": "string", "description": "结论细节（可选）：根因、影响、修复方式。"}}, "required": ["title"]},
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
    "memory_mission_create": {
        "description": "Create a structured mission record (category=mission ChromaDB entry + open learning-queue item). The mission holds a steps list and an open/done status in its metadata; add steps with memory_mission_add_step.",
        "inputSchema": {"type": "object", "properties": {"content": {"type": "string", "description": "The mission description / task content."}, "title": {"type": "string", "description": "Optional short title."}}, "required": ["content"]},
    },
    "memory_mission_get": {
        "description": "Return a full mission record: content, status (open/done), steps list with per-step skill linkage and done flags.",
        "inputSchema": {"type": "object", "properties": {"mission_id": {"type": "string", "description": "The mission entry id (dialogue:mission_...)."}}, "required": ["mission_id"]},
    },
    "memory_mission_list": {
        "description": "List structured missions, optionally filtered by status (open/done), newest first.",
        "inputSchema": {"type": "object", "properties": {"status": {"type": "string", "description": "Optional filter: 'open' or 'done'.", "enum": ["open", "done"]}, "limit": {"type": "integer", "default": 100, "description": "Max missions to return (default 100)."}}},
    },
    "memory_mission_add_step": {
        "description": "Append a step to a mission. Optionally link a skill by its entry id; the skill title is resolved and stored on the step.",
        "inputSchema": {"type": "object", "properties": {"mission_id": {"type": "string", "description": "The mission entry id."}, "text": {"type": "string", "description": "Step description."}, "skill_id": {"type": "string", "description": "Optional linked skill entry id."}}, "required": ["mission_id", "text"]},
    },
    "memory_mission_update_step": {
        "description": "Modify one step by index: text, skill_id, or done flag. Unspecified fields are left unchanged.",
        "inputSchema": {"type": "object", "properties": {"mission_id": {"type": "string", "description": "The mission entry id."}, "index": {"type": "integer", "description": "Zero-based step index."}, "text": {"type": "string", "description": "New step text."}, "skill_id": {"type": "string", "description": "New linked skill id."}, "done": {"type": "boolean", "description": "Step completion flag."}}, "required": ["mission_id", "index"]},
    },
    "memory_mission_remove_step": {
        "description": "Delete a step by index from a mission.",
        "inputSchema": {"type": "object", "properties": {"mission_id": {"type": "string", "description": "The mission entry id."}, "index": {"type": "integer", "description": "Zero-based step index to remove."}}, "required": ["mission_id", "index"]},
    },
    "memory_mission_set_status": {
        "description": "Set a mission's status to 'open' or 'done'. Mirroring closes the corresponding learning-queue item when marking done.",
        "inputSchema": {"type": "object", "properties": {"mission_id": {"type": "string", "description": "The mission entry id."}, "status": {"type": "string", "enum": ["open", "done"]}}, "required": ["mission_id", "status"]},
    },
    "memory_character_list": {
        "description": "列出所有角色（全局记忆的引用集合，不复制记忆内容）及其引用计数。绑定角色后 weave 只注入该角色的记忆。",
        "inputSchema": {"type": "object", "properties": {}},
    },
    "memory_character_create": {
        "description": "创建新角色。角色=全局记忆的引用集合（不复制内容），绑定 agent 后 weave 只注入角色记忆。name 必填，description 可选，返回 role_id。",
        "inputSchema": {"type": "object", "properties": {"name": {"type": "string", "description": "角色名，必填。"}, "description": {"type": "string", "description": "角色描述，可选。"}}, "required": ["name"]},
    },
    "memory_character_get": {
        "description": "获取角色详情及其引用的全部 memory_id。角色不存在时返回 error。",
        "inputSchema": {"type": "object", "properties": {"role_id": {"type": "string", "description": "角色 id。"}}, "required": ["role_id"]},
    },
    "memory_character_delete": {
        "description": "删除角色，级联删除其记忆引用和 agent 绑定。角色不存在时返回 error。",
        "inputSchema": {"type": "object", "properties": {"role_id": {"type": "string", "description": "角色 id。"}}, "required": ["role_id"]},
    },
    "memory_character_add_memory": {
        "description": "向角色添加一条全局记忆引用（幂等，不复制内容）。dimension 为人设维度（skills/appearance/personality，酒馆模式用；默认 general）。角色绑定后 weave 只注入其引用集合内的记忆。",
        "inputSchema": {"type": "object", "properties": {"role_id": {"type": "string", "description": "角色 id。"}, "memory_id": {"type": "string", "description": "全局记忆 entry id（如 dialogue:xxx）。"}, "dimension": {"type": "string", "description": "人设维度：general/skills/appearance/personality（默认 general）。"}}, "required": ["role_id", "memory_id"]},
    },
    "memory_character_remove_memory": {
        "description": "从角色移除一条记忆引用（幂等）。角色不存在时返回 error。",
        "inputSchema": {"type": "object", "properties": {"role_id": {"type": "string", "description": "角色 id。"}, "memory_id": {"type": "string", "description": "记忆 entry id。"}}, "required": ["role_id", "memory_id"]},
    },
    "memory_character_bind_agent": {
        "description": "将 agent 绑定到角色（重复绑定会切换到新角色）。role_id 为空字符串或省略时解绑。绑定后该 agent 的 weave 只注入角色记忆。",
        "inputSchema": {"type": "object", "properties": {"agent_name": {"type": "string", "description": "Agent 名。"}, "role_id": {"type": "string", "description": "角色 id；留空或省略 = 解绑。"}}, "required": ["agent_name"]},
    },
    "memory_character_agent_role": {
        "description": "查询 agent 当前绑定的角色 id（未绑定返回 null）。",
        "inputSchema": {"type": "object", "properties": {"agent_name": {"type": "string", "description": "Agent 名。"}}, "required": ["agent_name"]},
    },
    "memory_weave": {
        "description": "Auto-assemble layered memory context for the current conversation. Call BEFORE responding to any user message. Returns a context block ready for system-prompt injection. tier1: scene perception, tier2: deep memory (gated), nudge: high-importance reminders.\n\nV10: Both user_message AND assistant_content are auto-ingested (ImportanceScorer filters trivial content). This mirrors the Room framework's conversation-block pattern where both sides of every exchange are persisted without manual ingest calls.",
        "inputSchema": {"type": "object", "properties": {"user_message": {"type": "string", "description": "The user's current message."}, "assistant_content": {"type": "string", "description": "Optional: your OWN response from the PREVIOUS exchange. Auto-ingested into memory so both sides of every conversation block are persisted. Pass your full last response."}, "scene_summary": {"type": "string", "description": "Optional: what the user is doing now."}}},
    },
    "memory_state_extract": {
        "description": "Extract (and optionally apply) a tavern role's current 8-dim state (mood/need/health/clothing/item/action/scene/weather) from a dialogue snippet via the LLM bridge. apply=false returns a preview for agent review; apply=true overwrites the role's state snapshot. Tavern Mode Unit 4.",
        "inputSchema": {"type": "object", "properties": {"role_id": {"type": "string", "description": "The role whose state to update."}, "conversation": {"type": "string", "description": "The recent dialogue to extract state from."}, "apply": {"type": "boolean", "description": "When true, overwrite the state store; when false, return a preview only (default false)."}}, "required": ["role_id", "conversation"]},
    },
}


# ── Consistency guard ─────────────────────────────────────────────
# DISPATCH (handlers) and TOOL_SCHEMAS (MCP schemas) must stay in sync:
# adding a tool means touching BOTH. This invariant is cheap to check
# at import time and fails fast instead of shipping an Unknown tool.
_MISSING_SCHEMA = sorted(set(DISPATCH) - set(TOOL_SCHEMAS))
_MISSING_HANDLER = sorted(set(TOOL_SCHEMAS) - set(DISPATCH))
if _MISSING_SCHEMA or _MISSING_HANDLER:
    raise RuntimeError(
        "DISPATCH/TOOL_SCHEMAS out of sync: "
        f"no schema for {_MISSING_SCHEMA}; "
        f"no handler for {_MISSING_HANDLER}"
    )
del _MISSING_SCHEMA, _MISSING_HANDLER
