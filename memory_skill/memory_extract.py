"""Structured extraction and ingest — separated from _compose.py.

Handles classify_and_extract → routing to skill/pref/pers/mission ingests,
and LLM-based title generation for user_mem entries.
"""

from __future__ import annotations

from datetime import UTC, datetime as dt
import logging

_logger = logging.getLogger(__name__)


def extract_structured(ms, turn) -> None:
    """Run classify_and_extract on a user turn and route to structured ingests."""
    if turn.role != "user":
        return
    if not ms.tree:
        return
    api = ms.tree._api_base if hasattr(ms.tree, "_api_base") else ""
    key = ms.tree._api_key if hasattr(ms.tree, "_api_key") else ""
    model = ms.tree._model if hasattr(ms.tree, "_model") else ""

    from memory_skill.structured_extractor import classify_and_extract

    result = classify_and_extract(api, key, model, turn.content)
    t = result.get("type", "none")
    if t == "pref":
        ingest_pref(ms, result.get("key", ""), result.get("value", ""))
    elif t == "pers":
        ingest_pers(ms, result.get("trait", ""))
    elif t == "skill":
        ingest_skill_ex(ms, result.get("title", ""),
                        f"# {result.get('title', '')}\n\n学习目标: {result.get('goal', '')}")
    elif t == "mission":
        ingest_mission_ex(ms, result.get("title", ""),
                          result.get("summary") or turn.content)


def generate_title(ms, content: str) -> str:
    """LLM-generated 2-5 word title for user_mem entry."""
    if not ms.tree:
        return content[:40]
    from memory_skill._llm_utils import call_llm
    api = ms.tree._api_base if hasattr(ms.tree, "_api_base") else ""
    key = ms.tree._api_key if hasattr(ms.tree, "_api_key") else ""
    model = ms.tree._model if hasattr(ms.tree, "_model") else ""
    raw = call_llm(api, key, model,
                   f"Summarize in 5 words: {content[:200]}",
                   max_tokens=100, temperature=0.0)
    return raw.strip()[:50] if raw else content[:40]


def tag_title(ms, turn) -> None:
    """Generate and store a title for a dialogue turn."""
    if not ms.tree:
        return
    title = generate_title(ms, turn.content)
    if not title:
        return
    try:
        ms.learned_store.set_title(f"dialogue:{turn.id}", title)
    except Exception:
        pass


# ── Structured ingest helpers ─────────────────────────────────────────────


def _ingest_structured(ms, title: str, content: str, category: str,
                       source_urls: list[str] | None = None) -> object:
    from memory_skill.contracts import DialogueTurn
    now = dt.now(UTC)
    turn = DialogueTurn(
        id=f"{category}_{now:%Y%m%d_%H%M%S}_{hash(content) & 0xFFFF:04x}",
        role="system",
        content=content,
        timestamp=now,
        saw_index=0,
    )
    extra = {}
    if source_urls:
        extra["source_urls"] = source_urls
    result = ms.ingestor.ingest_dialogue(turn, category=category, extra_metadata=extra)
    if ms.tree:
        try:
            if category == "skill":
                from memory_skill.tree_classifier import classify_skill_path
                path = classify_skill_path(title)
                if path:
                    ms.tree.add_skill_node(path)
            else:
                branch_short = {"pref": "pref", "pers": "pers", "mission": "task"}.get(category)
                if branch_short:
                    root = "user" if category == "pref" else "assistant"
                    ms.tree.add_node(content=content, memory_id=turn.id,
                                     root=root, branch=branch_short, timestamp=now)
        except Exception:
            pass
    return result


def ingest_skill_ex(ms, title: str, content: str,
                    source_urls: list[str] | None = None) -> object:
    return _ingest_structured(ms, title, content, "skill", source_urls=source_urls)


def ingest_mission_ex(ms, title: str, content: str) -> object:
    return _ingest_structured(ms, title, content, "mission")


def ingest_pref(ms, key: str, value: str) -> object:
    return _ingest_structured(ms, key, f"{key}: {value}", "pref")


def ingest_pers(ms, trait: str) -> object:
    existing = ms.retriever.retrieve("all", limit=10, filters={"category": "pers"})
    cards = [e for e in existing.entries if e.content.startswith("# ")]
    if cards:
        current = max(cards, key=lambda e: len(e.content)).content
        if f"- {trait}" not in current:
            idx = current.find("## 规则")
            if idx > 0:
                updated = current[:idx].rstrip() + f"\n- {trait}\n\n" + current[idx:]
            else:
                updated = current.rstrip() + f"\n- {trait}"
            return _ingest_structured(ms, trait, updated, "pers")
        return None
    card = f"""# Agent 人物卡

## 设定
- 角色: 技术助手

## 风格
- {trait}

## 规则
- 不要编造不知道的信息"""
    return _ingest_structured(ms, trait, card, "pers")
