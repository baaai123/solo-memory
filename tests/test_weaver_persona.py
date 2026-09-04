"""Integration tests for weaver._build_persona_context (tavern persona dims).

Real CharacterStore (temp sqlite) + fake learned store that maps
memory_id -> content. Style mirrors test_character_store.py: plain pytest,
no mocks beyond a tiny stand-in for the learned store.
"""
import pytest

from memory_skill.character_store import CharacterStore
from memory_skill.weaver import _build_persona_context
from memory_skill.weaver import WeaverStores


class _FakeEntry:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLearned:
    """Maps memory_id -> content; missing entries behave like the real
    store (get_entry returns None)."""

    def __init__(self, entries: dict[str, str]) -> None:
        self._entries = {k: _FakeEntry(v) for k, v in entries.items()}

    def get_entry(self, entry_id: str):
        return self._entries.get(entry_id)


_ENTRIES = {
    "mem_skill_1": "精通 Python 编程，擅长异步与性能优化",
    "mem_skill_2": "熟悉 Rust，能写高性能系统级代码",
    "mem_appearance_1": "身高 175cm，黑色短发，戴眼镜，常穿休闲西装",
    "mem_personality_1": "性格沉稳，善于分析问题，做事细致认真",
    "mem_general_1": "这是通用记忆，不应出现在人设注入中",
    "mem_long_skill": "这是一个非常非常长的技能描述，" * 30,
}


@pytest.fixture
def stores(tmp_path):
    db = str(tmp_path / "persona.db")
    character = CharacterStore(db)
    learned = _FakeLearned(_ENTRIES)
    return WeaverStores(
        saw_buffer=None,
        dialogue_store=None,
        learned_store=learned,
        retriever=None,
        agent_name="t",
        namespace="test",
        character=character,
    ), character


def _mk_role(character: CharacterStore, dims: dict[str, str]) -> str:
    rid = character.create_role("测试角色", is_tavern=True)
    for mid, dim in dims.items():
        character.add_memory(rid, mid, dimension=dim)
    return rid


def test_persona_groups_three_dims_and_skips_general(stores):
    """Given a tavern role with skills/appearance/personality/general refs,
    when persona context is built, then the three persona blocks appear and
    general content is absent."""
    _, character = stores
    rid = _mk_role(character, {
        "mem_skill_1": "skills",
        "mem_appearance_1": "appearance",
        "mem_personality_1": "personality",
        "mem_general_1": "general",
    })
    out = _build_persona_context(rid, stores[0])
    assert "[角色设定·技能]" in out
    assert "[角色设定·外貌]" in out
    assert "[角色设定·性格]" in out
    assert "通用记忆" not in out
    assert "精通 Python" in out
    assert "身高 175cm" in out
    assert "性格沉稳" in out


def test_only_general_returns_empty(stores):
    """Given only a general ref, when built, then result is empty."""
    _, character = stores
    rid = _mk_role(character, {"mem_general_1": "general"})
    assert _build_persona_context(rid, stores[0]) == ""


def test_none_role_returns_empty(stores):
    """Given no bound role, then result is empty."""
    assert _build_persona_context(None, stores[0]) == ""


def test_missing_stores_returns_empty(tmp_path):
    """Given character or learned_store is None, then result is empty."""
    db = str(tmp_path / "x.db")
    character = CharacterStore(db)
    rid = character.create_role("测试角色")
    character.add_memory(rid, "mem_skill_1", dimension="skills")

    s1 = WeaverStores(saw_buffer=None, dialogue_store=None, learned_store=None,
                      retriever=None, agent_name="t", namespace="n",
                      character=character)
    assert _build_persona_context(rid, s1) == ""

    s2 = WeaverStores(saw_buffer=None, dialogue_store=None,
                      learned_store=_FakeLearned(_ENTRIES), retriever=None,
                      agent_name="t", namespace="n", character=None)
    assert _build_persona_context(rid, s2) == ""


def test_same_dimension_joined_and_truncated(stores):
    """Given two skills refs plus an over-long one, then they are joined
    with '；' and each item is capped at 120 chars."""
    _, character = stores
    rid = _mk_role(character, {
        "mem_skill_1": "skills",
        "mem_skill_2": "skills",
    })
    out = _build_persona_context(rid, stores[0])
    assert "[角色设定·技能]" in out
    assert "精通 Python" in out
    assert "熟悉 Rust" in out
    assert "；" in out

    rid2 = _mk_role(character, {"mem_long_skill": "skills"})
    out2 = _build_persona_context(rid2, stores[0])
    body = out2.split("[角色设定·技能] ", 1)[1]
    # single entry, no join char, hard cap at 120 chars (no ellipsis added)
    assert "；" not in body
    assert len(body) <= 120


def test_get_entry_exception_skips_dim(stores):
    """Given a learned store whose get_entry raises for one dimension, then
    that dimension is skipped and the others still render (weave never
    blocks on a store hiccup)."""
    _, character = stores

    class _FlakyLearned(_FakeLearned):
        def get_entry(self, entry_id):
            if entry_id == "mem_personality_1":
                raise RuntimeError("boom")
            return super().get_entry(entry_id)

    flaky = _FlakyLearned(_ENTRIES)
    s = WeaverStores(saw_buffer=None, dialogue_store=None, learned_store=flaky,
                     retriever=None, agent_name="t", namespace="n",
                     character=character)
    rid = _mk_role(character, {
        "mem_skill_1": "skills",
        "mem_appearance_1": "appearance",
        "mem_personality_1": "personality",
    })
    out = _build_persona_context(rid, s)
    assert "[角色设定·技能]" in out
    assert "[角色设定·外貌]" in out
    assert "[角色设定·性格]" not in out


def test_missing_content_skipped(stores):
    """Given a persona ref whose content is absent (None), then it is
    skipped without error."""
    _, character = stores
    rid = character.create_role("测试角色")
    character.add_memory(rid, "mem_ghost", dimension="skills")
    character.add_memory(rid, "mem_skill_1", dimension="skills")
    out = _build_persona_context(rid, stores[0])
    assert "[角色设定·技能]" in out
    assert "精通 Python" in out
