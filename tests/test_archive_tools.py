"""MCP tool handlers for archive review + reclassification.

Exercises ``handle_review_default`` and ``handle_reclassify`` against a
real ``LearnedStore`` (ChromaDB in a tmp dir, FakeEmbedder) mounted on a
minimal ``MemorySystem`` shell — mirroring the composition style of
``test_character_tools.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from memory_skill._compose import MemorySystem
from memory_skill.contracts import MemoryEntry, MemorySkillConfig
from memory_skill.learned_store import LearnedStore
from memory_skill.tools import (
    VALID_CATEGORIES,
    handle_reclassify,
    handle_review_default,
)
from tests.fakes import FakeEmbedder


def _entry(
    entry_id: str,
    content: str,
    *,
    category: str = "default",
    title: str = "",
    hours_ago: int = 0,
) -> MemoryEntry:
    now = datetime.now(UTC)
    return MemoryEntry(
        id=entry_id,
        content=content,
        created_at=now - timedelta(hours=hours_ago),
        updated_at=now - timedelta(hours=hours_ago),
        weight=0.5,
        category=category,
        tags=[],
        metadata={"title": title} if title else {},
    )


def _skill_with_store(tmp_path) -> MemorySystem:
    """A bare MemorySystem shell carrying a real LearnedStore."""
    config = MemorySkillConfig(db_path=str(tmp_path / "chroma"))
    store = LearnedStore(config, FakeEmbedder(dim=config.embedding_dim))
    ms = object.__new__(MemorySystem)
    ms.__dict__["learned_store"] = store
    return ms


# ── review_default ────────────────────────────────────────────────────────


def test_review_default_returns_default_entries_only(tmp_path):
    """Given default + non-default entries, when reviewing, then only the
    default ones come back with id/content/title/category/updated_at."""
    skill = _skill_with_store(tmp_path)
    skill.learned_store.insert(_entry("d1", "x" * 250 + " tail", title="t1", hours_ago=2))
    skill.learned_store.insert(_entry("d2", "second default", hours_ago=1))
    skill.learned_store.insert(_entry("p1", "a preference", category="pref"))

    result = handle_review_default(skill, {})

    assert result["count"] == 2
    assert result["total"] == 2
    assert [item["id"] for item in result["items"]] == ["d2", "d1"]  # newest first
    assert result["items"][1]["content"] == "x" * 200  # truncated to 200
    assert result["items"][1]["title"] == "t1"
    assert all(item["category"] == "default" for item in result["items"])
    assert all(item["updated_at"] for item in result["items"])


def test_review_default_empty_store(tmp_path):
    """Given no default entries, when reviewing, then an empty result
    (not an error)."""
    skill = _skill_with_store(tmp_path)

    result = handle_review_default(skill, {})

    assert result == {"count": 0, "total": 0, "items": []}


def test_review_default_pagination(tmp_path):
    """Given five default entries, when paging with limit/offset, then
    the requested slice comes back with the full total."""
    skill = _skill_with_store(tmp_path)
    for i in range(5):
        skill.learned_store.insert(_entry(f"d{i}", f"entry {i}", hours_ago=i))

    result = handle_review_default(skill, {"limit": 2, "offset": 1})

    assert result["count"] == 2
    assert result["total"] == 5
    assert [item["id"] for item in result["items"]] == ["d1", "d2"]


def test_review_default_clamps_limit(tmp_path):
    """Given a limit above the cap, when reviewing, then limit is clamped
    to 50."""
    skill = _skill_with_store(tmp_path)
    for i in range(3):
        skill.learned_store.insert(_entry(f"d{i}", f"entry {i}"))

    result = handle_review_default(skill, {"limit": 999})

    assert result["count"] == 3


def test_review_default_rejects_bad_offset(tmp_path):
    """Given a non-integer offset, when reviewing, then error."""
    skill = _skill_with_store(tmp_path)

    result = handle_review_default(skill, {"offset": "abc"})

    assert result == {"error": "limit/offset must be integers"}


# ── reclassify ────────────────────────────────────────────────────────────


def test_reclassify_moves_entry_and_fills_title(tmp_path):
    """Given a default entry without a title, when reclassifying to pref,
    then category changes, title is derived from content[:30], and the
    entry is still retrievable via get_entry."""
    skill = _skill_with_store(tmp_path)
    content = "用户喜欢冰美式，每天下午都要来一杯。" * 3
    skill.learned_store.insert(_entry("d1", content))

    result = handle_reclassify(skill, {"entry_id": "d1", "category": "pref"})

    assert result["ok"] is True
    assert result["entry_id"] == "d1"
    assert result["category"] == "pref"
    assert result["old_category"] == "default"
    assert result["title"] == content[:30]
    entry = skill.learned_store.get_entry("d1")
    assert entry.category == "pref"
    assert entry.metadata.get("title") == content[:30]


def test_reclassify_keeps_existing_title(tmp_path):
    """Given an entry that already has a title, when reclassifying, then
    the title is preserved verbatim."""
    skill = _skill_with_store(tmp_path)
    skill.learned_store.insert(_entry("d1", "long content here", title="既有标题"))

    result = handle_reclassify(skill, {"entry_id": "d1", "category": "skill"})

    assert result["title"] == "既有标题"
    assert skill.learned_store.get_entry("d1").metadata.get("title") == "既有标题"


def test_reclassify_missing_entry(tmp_path):
    """Given an unknown entry_id, when reclassifying, then
    'entry not found' error and no exception leaks."""
    skill = _skill_with_store(tmp_path)

    result = handle_reclassify(skill, {"entry_id": "ghost", "category": "pref"})

    assert result == {"error": "entry not found: ghost"}


@pytest.mark.parametrize("category", ["bogus", "PREF"])
def test_reclassify_invalid_category(tmp_path, category):
    """Given a category outside the legal set, when reclassifying, then
    'invalid category' error."""
    skill = _skill_with_store(tmp_path)
    skill.learned_store.insert(_entry("d1", "content"))

    result = handle_reclassify(skill, {"entry_id": "d1", "category": category})

    assert result == {"error": f"invalid category: {category}"}


@pytest.mark.parametrize("category", ["", None])
def test_reclassify_blank_category(tmp_path, category):
    """Given a blank category, when reclassifying, then 'category is
    required' error."""
    skill = _skill_with_store(tmp_path)
    skill.learned_store.insert(_entry("d1", "content"))

    result = handle_reclassify(skill, {"entry_id": "d1", "category": category})

    assert result == {"error": "category is required"}


def test_reclassify_missing_entry_id(tmp_path):
    """Given no entry_id, when reclassifying, then error."""
    skill = _skill_with_store(tmp_path)

    result = handle_reclassify(skill, {"category": "pref"})

    assert result == {"error": "entry_id is required"}


def test_reclassify_all_legal_categories(tmp_path):
    """Given the full legal category set, when reclassifying to each, then
    every one is accepted."""
    skill = _skill_with_store(tmp_path)

    for category in sorted(VALID_CATEGORIES):
        skill.learned_store.insert(_entry(f"e_{category}", "content"))
        result = handle_reclassify(
            skill, {"entry_id": f"e_{category}", "category": category}
        )
        assert result["ok"] is True


# ── integration ───────────────────────────────────────────────────────────


def test_reclassify_retrievable_under_new_category(tmp_path):
    """Given a reclassified entry, when listing the new category and
    reviewing default again, then the entry shows up in the new category
    only."""
    skill = _skill_with_store(tmp_path)
    skill.learned_store.insert(_entry("d1", "把这条归到 mission"))

    handle_reclassify(skill, {"entry_id": "d1", "category": "mission"})

    mission_entries = skill.learned_store.list_by_category("mission", limit=0)
    assert [e.id for e in mission_entries] == ["d1"]
    review = handle_review_default(skill, {})
    assert review == {"count": 0, "total": 0, "items": []}
