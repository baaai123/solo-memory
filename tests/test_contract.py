"""Contract tests — one suite, both adapters.

Every store contract (insert/search/weight/duplicate) is asserted against
BOTH the real implementation (chroma/SQLite) and the in-memory fake.  A
behavior change in either side fails here, so the 0.07s fast suite stays
honest: the fakes are a verified test double, not a hand-copied twin.
"""

from __future__ import annotations

import pytest
from datetime import UTC, datetime

from memory_skill.contracts import DialogueTurn, MemoryEntry, MemorySkillConfig
from tests.fakes import FakeEmbedder, FakeLearnedStore, InMemoryDialogueStore

# ── Adapter fixtures ─────────────────────────────────────────────────────

def _naive_now() -> datetime:
    return datetime.now()


def _make_turn(turn_id: str, content: str, role: str = "user") -> DialogueTurn:
    return DialogueTurn(id=turn_id, role=role, content=content, timestamp=_naive_now())


def _make_entry(entry_id: str, content: str, category: str = "default") -> MemoryEntry:
    now = _naive_now()
    return MemoryEntry(
        id=entry_id,
        content=content,
        created_at=now,
        updated_at=now,
        weight=0.5,
        category=category,
        tags=[],
        metadata={},
    )


# ── DialogueStore contracts ───────────────────────────────────────────────

class TestDialogueStoreContract:
    @pytest.fixture(params=["real", "fake"])
    def store(self, request, tmp_path):
        if request.param == "fake":
            return InMemoryDialogueStore()
        from memory_skill.dialogue_store import DialogueStore
        cfg = MemorySkillConfig(db_path=str(tmp_path / "d.db"))
        return DialogueStore(cfg)

    def test_insert_then_get_by_id(self, store):
        store.insert(_make_turn("t1", "hello world"))
        turn = store.get_by_id("t1")
        assert turn is not None
        assert turn.content == "hello world"

    def test_insert_idempotent_by_id(self, store):
        store.insert(_make_turn("t1", "first"))
        store.insert(_make_turn("t1", "first"))
        assert store.count() == 1

    def test_count(self, store):
        store.insert(_make_turn("t1", "a"))
        store.insert(_make_turn("t2", "b"))
        assert store.count() == 2

    def test_get_recent_returns_newest_last(self, store):
        store.insert(_make_turn("t1", "old"))
        store.insert(_make_turn("t2", "new"))
        recent = store.get_recent(5)
        assert recent[-1].id == "t2"  # newest last (real store contract)
        assert recent[0].id == "t1"

    def test_search_finds_token(self, store):
        store.insert(_make_turn("t1", "python backend framework"))
        results = store.search("python", limit=5)
        assert any(t.id == "t1" for t in results)

    def test_search_multiword_partial_match(self, store):
        """OR semantics: a multi-word query surfaces documents matching any word."""
        store.insert(_make_turn("t1", "python backend framework"))
        store.insert(_make_turn("t2", "frontend javascript"))
        results = store.search("python backend ORM", limit=5)
        assert any(t.id == "t1" for t in results)


# ── LearnedStore contracts ────────────────────────────────────────────────

class TestLearnedStoreContract:
    @pytest.fixture(params=["real", "fake"])
    def store(self, request, tmp_path):
        if request.param == "fake":
            s = FakeLearnedStore()
            s.attach_embedder(FakeEmbedder(dim=16))
            return s
        from memory_skill.embedder import Embedder
        from memory_skill.learned_store import LearnedStore
        cfg = MemorySkillConfig(db_path=str(tmp_path / "l.db"))
        embedder = Embedder(cfg)
        return LearnedStore(cfg, embedder, chroma_path=str(tmp_path / "chroma"))

    def test_insert_then_search_finds_related(self, store):
        store.insert(_make_entry("e1", "I love python backend development"))
        results = store.search("python backend", limit=5)
        assert any(e.id == "e1" for e in results)

    def test_get_weight_returns_set_value(self, store):
        store.insert(_make_entry("e1", "content"))
        assert store.get_weight("e1") == 0.5
        store.set_weight("e1", 0.7)
        assert store.get_weight("e1") == 0.7

    def test_boost_weight_increments_and_caps(self, store):
        store.insert(_make_entry("e1", "content"))
        new = store.boost_weight("e1", delta=0.1)
        assert new == 0.6
        new = store.boost_weight("e1", delta=0.5, cap=0.95)
        assert new == 0.95  # capped

    def test_boost_missing_entry_returns_default(self, store):
        assert store.boost_weight("nonexistent") == 0.5

    def test_search_respects_category_filter(self, store):
        store.insert(_make_entry("e1", "python", category="work"))
        store.insert(_make_entry("e2", "python", category="personal"))
        results = store.search("python", limit=5, filters={"category": "work"})
        ids = {e.id for e in results}
        assert "e1" in ids
        assert "e2" not in ids

    def test_search_respects_limit(self, store):
        for i in range(5):
            store.insert(_make_entry(f"e{i}", f"topic {i}"))
        results = store.search("topic", limit=3)
        assert len(results) <= 3
