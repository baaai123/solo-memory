"""Unit 3 — Retriever role-whitelist filtering (character-bound retrieval).

Verifies that ``Retriever.retrieve(role_memory_ids=...)`` filters both the
semantic and BM25 legs before candidate merging, so a bound character role
sees only its referenced memories (R10) while an unbound call behaves
exactly as before (R11).  Uses fake stores — no chroma/SQLite/LLM.
"""

from __future__ import annotations

from datetime import UTC, datetime

from memory_skill.contracts import DialogueTurn, MemoryEntry, MemorySkillConfig
from memory_skill.retriever import Retriever
from tests.fakes import FakeLearnedStore, InMemoryDialogueStore


def _entry(eid: str, content: str) -> MemoryEntry:
    return MemoryEntry(
        id=eid,
        content=content,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        weight=0.5,
        category="pref",
        tags=[],
        metadata={},
    )


def _make_retriever(
    learned_entries: list[MemoryEntry] | None = None,
    dialogue_turns: list[DialogueTurn] | None = None,
) -> Retriever:
    cfg = MemorySkillConfig(db_path=":memory:", agent_name="test")
    dialogue_store = InMemoryDialogueStore()
    learned_store = FakeLearnedStore()
    for e in learned_entries or []:
        learned_store.insert(e)
    for t in dialogue_turns or []:
        dialogue_store.insert(t)
    return Retriever(
        config=cfg,
        dialogue_store=dialogue_store,
        learned_store=learned_store,
    )


class TestRoleWhitelist:
    def test_whitelist_returns_only_referenced(self):
        """Given id m3 matches the query best but is unreferenced,
        when retrieving with role_memory_ids={m1, m2},
        then only m1 and m2 are returned."""
        ret = _make_retriever(learned_entries=[
            _entry("m1", "用户喜欢 蓝色 主题"),
            _entry("m2", "用户喜欢 蓝色 主题 和 深色 模式"),
            _entry("m3", "用户喜欢 蓝色 主题 且 讨厌 红色"),
        ])
        envelope = ret.retrieve("蓝色 主题", limit=10, role_memory_ids={"m1", "m2"})
        assert {e.id for e in envelope.entries} == {"m1", "m2"}

    def test_no_whitelist_returns_global(self):
        """Given the same store, when retrieving without role_memory_ids,
        then all query-matching global memories are returned (status quo)."""
        ret = _make_retriever(learned_entries=[
            _entry("m1", "用户喜欢 蓝色 主题"),
            _entry("m3", "用户喜欢 蓝色 主题 且 讨厌 红色"),
        ])
        envelope = ret.retrieve("蓝色 主题", limit=10)
        assert {e.id for e in envelope.entries} == {"m1", "m3"}

    def test_empty_whitelist_returns_empty_envelope(self):
        """Given a role with no referenced memories,
        when retrieving with role_memory_ids=set(),
        then the envelope is empty (no injection for a memory-less role)."""
        ret = _make_retriever(learned_entries=[_entry("m1", "用户喜欢 蓝色 主题")])
        envelope = ret.retrieve("蓝色 主题", limit=10, role_memory_ids=set())
        assert envelope.entries == []
        assert envelope.total_candidates == 0
        assert envelope.truncated is False
        assert envelope.type == "recall"

    def test_whitelist_entry_in_both_legs_is_deduped(self):
        """Given a referenced memory present in both the semantic leg
        (learned entry ``dialogue:t1``) and the BM25 leg (turn t1),
        when retrieving with that id whitelisted,
        then it appears exactly once in the candidate set."""
        turn = DialogueTurn(
            id="t1", role="user", content="用户喜欢 蓝色 主题 风格",
            timestamp=datetime.now(UTC),
        )
        ret = _make_retriever(
            learned_entries=[
                _entry("dialogue:t1", "用户喜欢 蓝色 主题 风格"),
                _entry("m2", "无关 内容"),
            ],
            dialogue_turns=[turn],
        )
        envelope = ret.retrieve(
            "蓝色 主题", limit=10, role_memory_ids={"dialogue:t1"},
        )
        assert envelope.total_candidates == 1
        assert [e.id for e in envelope.entries] == ["dialogue:t1"]

    def test_roles_do_not_leak(self):
        """Given role A referencing m1 and role B referencing m2,
        when retrieving with each role's whitelist,
        then each role sees only its own referenced memories."""
        ret = _make_retriever(learned_entries=[
            _entry("m1", "用户喜欢 蓝色 主题"),
            _entry("m2", "用户喜欢 红色 主题"),
        ])
        env_a = ret.retrieve("主题", limit=10, role_memory_ids={"m1"})
        env_b = ret.retrieve("主题", limit=10, role_memory_ids={"m2"})
        assert {e.id for e in env_a.entries} == {"m1"}
        assert {e.id for e in env_b.entries} == {"m2"}

    def test_unreferenced_dialogue_turns_are_filtered(self):
        """Given dialogue turns matching the query whose ids are not in the
        whitelist, when retrieving with a bound role,
        then those BM25 entries are excluded (role references point at
        learned ids only)."""
        turn_kept = DialogueTurn(
            id="t1", role="user", content="用户喜欢 蓝色 主题 风格",
            timestamp=datetime.now(UTC),
        )
        turn_dropped = DialogueTurn(
            id="t2", role="user", content="用户喜欢 蓝色 主题 的 壁纸",
            timestamp=datetime.now(UTC),
        )
        ret = _make_retriever(
            learned_entries=[_entry("dialogue:t1", "用户喜欢 蓝色 主题 风格")],
            dialogue_turns=[turn_kept, turn_dropped],
        )
        envelope = ret.retrieve(
            "蓝色 主题", limit=10, role_memory_ids={"dialogue:t1"},
        )
        assert {e.id for e in envelope.entries} == {"dialogue:t1"}
