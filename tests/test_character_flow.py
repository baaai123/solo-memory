"""Unit 4 — character role ingest double-write + weave role filtering.

Covers the full character flow with a real ``CharacterStore`` (SQLite in a
tmp dir) combined with fake stores, mirroring the Unit 2/3 test patterns:

- ``MemorySystem.ingest`` double-writes the stored entry into the bound
  role's reference set and stamps ``source_role`` on the entry metadata.
- ``MemorySystem.weave`` resolves the agent's bound role and filters every
  learned-memory retrieval to the role's reference set.
- Unbound systems (and degraded lookups) keep the original behaviour.
- The Ingestor double-write hook itself, including the dedup-merge path.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_skill._compose import MemorySystem
from memory_skill.character_store import CharacterStore
from memory_skill.contracts import DialogueTurn, MemorySkillConfig
from memory_skill.ingestor import Ingestor
from memory_skill.protocol_state import ProtocolState
from memory_skill.retriever import Retriever
from tests.fakes import (
    FakeEmbedder,
    FakeLearnedStore,
    FakeSawBuffer,
    InMemoryDialogueStore,
)


def _now() -> datetime:
    return datetime.now(UTC)


def build_character_system(tmp_path, *, agent_name: str = "test_agent"):
    """Compose a MemorySystem from fakes + a real CharacterStore."""
    config = MemorySkillConfig(db_path=":memory:", agent_name=agent_name)
    character = CharacterStore(str(tmp_path / "character_flow.db"))
    embedder = FakeEmbedder(dim=config.embedding_dim)
    saw_buffer = FakeSawBuffer()
    dialogue_store = InMemoryDialogueStore()
    learned_store = FakeLearnedStore()
    learned_store.attach_embedder(embedder)
    retriever = Retriever(
        config=config,
        dialogue_store=dialogue_store,
        learned_store=learned_store,
    )
    ingestor = Ingestor(
        config=config,
        saw_buffer=saw_buffer,
        dialogue_store=dialogue_store,
        learned_store=learned_store,
        embedder=embedder,
        tree=None,
        learning_queue=None,
        character_store=character,
    )
    ms = object.__new__(MemorySystem)
    ms.__dict__.update(
        config=config,
        embedder=embedder,
        saw_buffer=saw_buffer,
        dialogue_store=dialogue_store,
        learned_store=learned_store,
        tree=None,
        ingestor=ingestor,
        retriever=retriever,
        learning_queue=None,
        pending_store=None,
        character=character,
        protocol=ProtocolState(),
        _composed_at=_now(),
    )
    return ms, character


# ── Ingest double-write ──────────────────────────────────────────────────────


def test_ingest_double_writes_bound_role(tmp_path):
    """Given an agent bound to a role, when MemorySystem.ingest stores a turn,
    then the role references the new entry and its metadata carries
    source_role."""
    ms, character = build_character_system(tmp_path, agent_name="test_agent")
    role_id = character.create_role("雪奈", "温柔冷静")
    assert character.bind_agent("test_agent", role_id)

    receipt = ms.ingest(DialogueTurn(
        id="t1", role="user", content="你好呀", timestamp=_now(),
    ))

    assert receipt.entry_id == "dialogue:t1"
    assert character.list_memories(role_id) == ["dialogue:t1"]
    entry = ms.learned_store.get_entry("dialogue:t1")
    assert entry.metadata["source_role"] == role_id


def test_ingest_unbound_leaves_no_reference(tmp_path):
    """Given a role that exists but no agent binding, when ingesting,
    then no reference and no source_role are added (R11)."""
    ms, character = build_character_system(tmp_path, agent_name="test_agent")
    role_id = character.create_role("雪奈")

    receipt = ms.ingest(DialogueTurn(
        id="t1", role="user", content="未绑定记忆", timestamp=_now(),
    ))

    assert receipt.entry_id == "dialogue:t1"
    assert character.list_memories(role_id) == []
    entry = ms.learned_store.get_entry("dialogue:t1")
    assert "source_role" not in entry.metadata


# ── Weave role filtering ─────────────────────────────────────────────────────


def _seed_structured(ms, *, role_id: str | None) -> None:
    """Seed one in-role skill entry (via the double-write hook when bound)
    and two out-of-role structured entries (skill + title-less pref)."""
    now = _now()
    ms.ingestor.ingest_dialogue(DialogueTurn(
        id="t_in", role="system", content="FastAPI alphamarker 性能好",
        timestamp=now,
    ), category="skill", extra_metadata={"title": "FastAPI"},
        character_role=role_id)
    ms.ingestor.ingest_dialogue(DialogueTurn(
        id="t_out", role="user", content="Docker betamarker 部署指南",
        timestamp=now,
    ), category="skill", extra_metadata={"title": "Docker"})
    ms.ingestor.ingest_dialogue(DialogueTurn(
        id="t_pref", role="user", content="betamarker 用户偏好 晚上工作",
        timestamp=now,
    ), category="pref")


def test_weave_filters_out_of_role_memories(tmp_path):
    """Given a bound role plus structured memories inside and outside the
    role's reference set, when weaving, then only the in-role entry becomes
    a tier2 unit and out-of-role entries stay out of every learned-memory
    slot (title_preview/pref/skill)."""
    ms, character = build_character_system(tmp_path, agent_name="")
    role_id = character.create_role("雪奈")
    assert character.bind_agent("", role_id)
    _seed_structured(ms, role_id=role_id)
    assert character.list_memories(role_id) == ["dialogue:t_in"]

    ctx = ms.weave(user_message="FastAPI", scene_summary="")

    tier2 = ctx.tier2_context or ""
    assert tier2.count("[记忆片段]") == 1
    assert "alphamarker" in tier2
    assert "betamarker" not in (ctx.title_preview or "")
    assert "betamarker" not in (ctx.pref_context or "")
    assert "betamarker" not in (ctx.skill_context or "")


def test_weave_unbound_shows_all_memories(tmp_path):
    """Given the same seed data with no agent binding, when weaving,
    then every structured entry becomes a unit and out-of-role memories
    surface in the structured slots (zero behaviour change)."""
    ms, character = build_character_system(tmp_path, agent_name="")
    character.create_role("雪奈")  # role exists, agent never bound
    _seed_structured(ms, role_id=None)

    ctx = ms.weave(user_message="FastAPI", scene_summary="")

    assert (ctx.tier2_context or "").count("[记忆片段]") == 3
    assert "betamarker" in (ctx.title_preview or "")
    assert "betamarker" in (ctx.pref_context or "")


def test_weave_deleted_role_degrades_to_unbound(tmp_path):
    """Given a role deleted after ingest, when weaving, then the binding is
    gone (get_agent_role → None) and the weave is unfiltered."""
    ms, character = build_character_system(tmp_path, agent_name="test_agent")
    role_id = character.create_role("雪奈")
    character.bind_agent("test_agent", role_id)
    ms.ingest(DialogueTurn(
        id="t1", role="user", content="绑定时期记忆", timestamp=_now(),
    ))

    assert character.delete_role(role_id)

    receipt = ms.ingest(DialogueTurn(
        id="t2", role="user", content="删除后记忆", timestamp=_now(),
    ))
    assert character.get_agent_role("test_agent") is None
    assert character.list_memories(role_id) == []
    entry = ms.learned_store.get_entry("dialogue:t2")
    assert receipt.entry_id == "dialogue:t2"
    assert "source_role" not in entry.metadata


# ── Full-chain integration ───────────────────────────────────────────────────


def test_full_chain_create_add_bind_ingest_weave(tmp_path):
    """Given the complete flow (create role → reference a pre-existing
    memory → bind agent → ingest → weave), then both referenced memories
    are visible and out-of-role memories stay hidden."""
    ms, character = build_character_system(tmp_path, agent_name="")
    role_id = character.create_role("雪奈")

    now = _now()
    ms.ingestor.ingest_dialogue(DialogueTurn(
        id="t_pre", role="system", content="premarker 预置人格设定",
        timestamp=now,
    ), category="pers", extra_metadata={"title": "人格卡"})
    assert character.add_memory(role_id, "dialogue:t_pre")
    assert character.bind_agent("", role_id)

    ms.ingest(DialogueTurn(
        id="t_new", role="user", content="新会话记忆", timestamp=now,
    ))
    assert set(character.list_memories(role_id)) == {
        "dialogue:t_pre", "dialogue:t_new",
    }

    ms.ingestor.ingest_dialogue(DialogueTurn(
        id="t_out", role="system", content="betamarker 角色外技能",
        timestamp=now,
    ), category="skill", extra_metadata={"title": "角色外"})

    ctx = ms.weave(user_message="premarker", scene_summary="")

    assert "premarker" in ctx.to_prompt_block()
    # 检索增强后可召回多个角色记忆, 断言至少含目标片段即可
    assert (ctx.tier2_context or "").count("[记忆片段]") >= 1
    assert "betamarker" not in (ctx.tier2_context or "")
    assert "betamarker" not in (ctx.pers_context or "")
    assert "betamarker" not in (ctx.skill_context or "")
    assert "betamarker" not in (ctx.title_preview or "")


# ── Ingestor hook unit tests ─────────────────────────────────────────────────


def _bare_ingestor(tmp_path):
    """An Ingestor over fakes with a real CharacterStore wired in."""
    config = MemorySkillConfig(db_path=":memory:", agent_name="test_agent")
    character = CharacterStore(str(tmp_path / "ingestor_flow.db"))
    embedder = FakeEmbedder(dim=config.embedding_dim)
    saw_buffer = FakeSawBuffer()
    dialogue_store = InMemoryDialogueStore()
    learned_store = FakeLearnedStore()
    learned_store.attach_embedder(embedder)
    ingestor = Ingestor(
        config=config,
        saw_buffer=saw_buffer,
        dialogue_store=dialogue_store,
        learned_store=learned_store,
        embedder=embedder,
        tree=None,
        learning_queue=None,
        character_store=character,
    )
    return ingestor, character, learned_store


def test_ingestor_hook_double_writes_merge_target(tmp_path):
    """Given two identical turns with a character_role, when ingesting,
    then the dedup merge target is referenced exactly once."""
    ingestor, character, learned_store = _bare_ingestor(tmp_path)
    role_id = character.create_role("A")

    ingestor.ingest_dialogue(DialogueTurn(
        id="t1", role="user", content="重复内容", timestamp=_now(),
    ), character_role=role_id)
    receipt = ingestor.ingest_dialogue(DialogueTurn(
        id="t2", role="user", content="重复内容", timestamp=_now(),
    ), character_role=role_id)

    assert receipt.deduped is True
    assert receipt.entry_id == "dialogue:t1"
    assert character.list_memories(role_id) == ["dialogue:t1"]
    assert learned_store.get_entry("dialogue:t1") is not None


def test_ingestor_hook_inert_without_character_role(tmp_path):
    """Given a wired character store but no character_role, when ingesting,
    then no reference is added (hook inert)."""
    ingestor, character, _ = _bare_ingestor(tmp_path)
    role_id = character.create_role("A")

    ingestor.ingest_dialogue(DialogueTurn(
        id="t1", role="user", content="普通记忆", timestamp=_now(),
    ))

    assert character.list_memories(role_id) == []


def test_ingestor_hook_survives_missing_character_store(tmp_path):
    """Given a character_role but no character_store (minimal composition),
    when ingesting, then the write succeeds and the hook is a no-op."""
    config = MemorySkillConfig(db_path=":memory:", agent_name="test_agent")
    embedder = FakeEmbedder(dim=config.embedding_dim)
    ingestor = Ingestor(
        config=config,
        saw_buffer=FakeSawBuffer(),
        dialogue_store=InMemoryDialogueStore(),
        learned_store=FakeLearnedStore(),
        embedder=embedder,
    )
    receipt = ingestor.ingest_dialogue(DialogueTurn(
        id="t1", role="user", content="普通记忆", timestamp=_now(),
    ), character_role="character:ghost")
    assert receipt.entry_id == "dialogue:t1"
