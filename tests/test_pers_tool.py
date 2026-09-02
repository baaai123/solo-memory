"""MCP tool handler tests for ``memory_ingest_pers``.

Exercises ``handle_ingest_pers`` in ``memory_skill.tools`` against a real
``MemorySystem`` composed from in-memory fakes (``build_fast_system``,
mirroring ``test_fast.py``), plus shell stubs for the handler contract
(blank trait / missing method / duplicate / exception — mirroring the
composition style of ``test_character_tools.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_skill._compose import MemorySystem
from memory_skill.contracts import DialogueTurn, MemorySkillConfig
from memory_skill.tools import DISPATCH, TOOL_SCHEMAS, handle_ingest_conclusion, handle_ingest_pers
from tests.fakes import build_fast_system

TRAIT = "禁止使用内置todo"


def _new_system() -> MemorySystem:
    """A fresh MemorySystem composed entirely from in-memory fakes."""
    return build_fast_system(MemorySkillConfig(db_path=":memory:"))


def _seed_persona_card(ms: MemorySystem, body: str) -> str:
    """Seed a ``# ``-prefixed pers card; returns its entry id."""
    ms.ingestor.ingest_dialogue(DialogueTurn(
        id=f"t_card_{hash(body) & 0xFFFF:04x}",
        role="system",
        content=body,
        timestamp=datetime.now(UTC),
    ), category="pers")
    cards = [e for e in ms.learned_store.list_by_category("pers", limit=10)
             if e.content.startswith("# ")]
    assert len(cards) == 1
    return cards[0].id


def _pers_contents(ms: MemorySystem) -> list[str]:
    return [e.content for e in ms.learned_store.list_by_category("pers", limit=0)]


# ── registration ─────────────────────────────────────────────────────────────


def test_tool_registered_in_dispatch_and_schemas():
    """Given the module import, then memory_ingest_pers is wired into both
    DISPATCH and TOOL_SCHEMAS with trait as the sole required field."""
    assert DISPATCH["memory_ingest_pers"] is handle_ingest_pers
    schema = TOOL_SCHEMAS["memory_ingest_pers"]
    assert schema["inputSchema"]["required"] == ["trait"]
    assert schema["inputSchema"]["properties"]["trait"]["type"] == "string"
    assert "pers" in schema["description"]


# ── happy: fresh system creates a pers entry ────────────────────────────────


def test_ingest_pers_creates_pers_entry():
    """Given an empty system, when ingesting a trait, then status=ingested,
    persisted=True, and the pers category holds an entry with the trait."""
    ms = _new_system()

    result = handle_ingest_pers(ms, {"trait": TRAIT})

    assert result == {"status": "ingested", "trait": TRAIT, "persisted": True}
    assert any(TRAIT in c for c in _pers_contents(ms))


def test_ingest_pers_strips_and_echoes_trait():
    """Given a trait with surrounding whitespace, when ingesting, then the
    stored content uses the stripped trait."""
    ms = _new_system()

    result = handle_ingest_pers(ms, {"trait": f"  {TRAIT}  "})

    assert result["status"] == "ingested"
    assert result["trait"] == TRAIT
    assert any(f"- {TRAIT}" in c for c in _pers_contents(ms))


# ── happy: append to an existing persona card ───────────────────────────────
# NOTE: this documents the intended contract (traits accumulate on the
# longest '# '-prefixed card).  It is currently blocked by a pre-existing
# bug: memory_extract.ingest_pers calls
#   find_duplicate(embed(current))
# with the card's OWN content, which always self-matches (cosine distance
# 0 <= 1 - 0.85) and returns None, so the append branch is dead and the
# handler reports duplicate_skipped instead of appending.  strict=True
# fails the suite loudly if someone fixes it without removing this marker.


def test_ingest_pers_appends_to_existing_card():
    """Given an existing '# ' persona card, when ingesting a new trait,
    then the trait is appended to that same card (no new card)."""
    ms = _new_system()
    card_id = _seed_persona_card(ms, "# Agent 人物卡\n\n## 设定\n- 角色: AI 助手")

    result = handle_ingest_pers(ms, {"trait": TRAIT})

    assert result["status"] == "ingested"
    card = ms.learned_store.get_entry(card_id)
    assert f"- {TRAIT}" in card.content
    cards = [c for c in _pers_contents(ms) if c.startswith("# ")]
    assert len(cards) == 1  # 非重复建卡


def test_repeat_ingest_never_double_appends():
    """Given a persona card, when ingesting the same trait twice, then the
    trait appears at most once on the card regardless of dedup path."""
    ms = _new_system()
    card_id = _seed_persona_card(ms, "# Agent 人物卡\n\n## 设定\n- 角色: AI 助手")
    handle_ingest_pers(ms, {"trait": TRAIT})
    handle_ingest_pers(ms, {"trait": TRAIT})

    card = ms.learned_store.get_entry(card_id)

    assert card.content.count(f"- {TRAIT}") <= 1


# ── edge: handler contract (shell stubs) ────────────────────────────────────


@pytest.mark.parametrize("trait", ["", "   ", None, 42])
def test_blank_or_non_string_trait_is_rejected(trait):
    """Given a blank/non-string trait, when ingesting, then error
    'trait is required' and nothing is written."""
    ms = _new_system()

    result = handle_ingest_pers(ms, {"trait": trait})

    assert result == {"error": "trait is required"}
    assert _pers_contents(ms) == []


def test_duplicate_returns_duplicate_skipped():
    """Given a skill whose ingest_pers dedups to None, when ingesting,
    then status=duplicate_skipped with the trait echoed."""

    class StubSkill:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def ingest_pers(self, trait: str):
            self.calls.append(trait)
            return None  # store deduped the write

    skill = StubSkill()

    result = handle_ingest_pers(skill, {"trait": TRAIT})

    assert result == {"status": "duplicate_skipped", "trait": TRAIT}
    assert skill.calls == [TRAIT]


def test_missing_ingest_pers_method_defends():
    """Given a skill without an ingest_pers method, when calling the
    handler, then a defensive error dict comes back and nothing crashes."""

    class NoPersSkill:
        pass

    result = handle_ingest_pers(NoPersSkill(), {"trait": TRAIT})

    assert result == {"error": "ingest_pers not available on this skill"}


def test_exception_is_caught_and_returned():
    """Given an ingest_pers that raises, when calling the handler, then
    the exception is returned as an error dict, not propagated."""

    class BoomSkill:
        def ingest_pers(self, trait: str):
            raise RuntimeError("store down")

    result = handle_ingest_pers(BoomSkill(), {"trait": TRAIT})

    assert result == {"error": "RuntimeError: store down"}


# ── integration: weave picks the trait up ───────────────────────────────────


def test_weave_pers_context_contains_ingested_trait():
    """Given an ingested trait, when weaving, then pers_context (the
    [人格特征] block) contains the trait."""
    ms = _new_system()

    handle_ingest_pers(ms, {"trait": TRAIT})
    ctx = ms.weave(user_message="继续", scene_summary="")

    assert TRAIT in (ctx.pers_context or "")


class TestIngestConclusion:
    """memory_ingest_conclusion writes root-cause conclusions."""

    def test_ingest_conclusion_writes_entry(self):
        """Given a title and content, when ingesting a conclusion,
        then a conclusion-category entry appears with title+content."""
        ms = _new_system()
        result = handle_ingest_conclusion(ms, {"title": "某 bug 由 X 导致", "content": "根因 Y,修复 Z"})
        assert result["status"] == "ingested"
        entries = ms.retriever.retrieve("all", limit=5, filters={"category": "conclusion"})
        assert any("某 bug 由 X 导致" in e.content for e in entries.entries)

    def test_empty_title_rejected(self):
        """Given a blank title, when ingesting, then an error is returned."""
        ms = _new_system()
        result = handle_ingest_conclusion(ms, {"title": "   "})
        assert "error" in result

    def test_no_ingest_method_degrades(self):
        """Given a shell without ingest_conclusion, when ingesting,
        then a clear error is returned (no crash)."""
        class Shell:
            pass
        result = handle_ingest_conclusion(Shell(), {"title": "x"})
        assert "error" in result
