"""Tests for Unit 2 — CharacterStore hooked into MemorySystem.

Covers:
- ``_build_system`` exposing ``ms.character`` (default db_path and the
  ``character_db_path`` override).
- ``MemorySystem.delete_entry`` cascading a global memory delete into
  ``remove_all_memory`` so no orphan character references remain.
- The defensive guard when ``character`` is absent (minimal compositions).

Real ``_build_system`` builds are used for wiring tests; cascade tests run
against a composed MemorySystem with a real CharacterStore and a minimal
fake learned store (no chroma/ONNX needed).
"""

from __future__ import annotations

import sqlite3

import pytest

from memory_skill._compose import MemorySystem, _build_system
from memory_skill.character_store import CharacterStore
from memory_skill.contracts import MemorySkillConfig


class _DeletingFakeLearnedStore:
    """Minimal fake exposing only what ``delete_entry`` needs."""

    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete(self, entry_id: str) -> None:
        self.deleted.append(entry_id)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def built_system(tmp_path):
    """A real ``_build_system`` output backed by a temporary DB (no tree)."""
    config = MemorySkillConfig(
        db_path=str(tmp_path / "real.db"),
        tree_enabled=False,
    )
    ms = _build_system(config)
    yield ms
    # Best-effort cleanup of file-backed connections.
    for store_name in ("character", "dialogue_store"):
        store = getattr(ms, store_name, None)
        conn = getattr(store, "_conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
    try:
        ms.learned_store._client.clear_system_cache()
    except Exception:
        pass


@pytest.fixture
def cascade_system(tmp_path):
    """MemorySystem composed from a real CharacterStore + deleting fake."""
    character = CharacterStore(str(tmp_path / "cascade.db"))
    learned = _DeletingFakeLearnedStore()
    ms = MemorySystem(character=character, learned_store=learned)
    yield ms
    character._conn.close()


# ── _build_system wiring ────────────────────────────────────────────────────


def test_build_system_exposes_character_store(built_system):
    """Given _build_system, When accessed, Then ms.character is a working
    CharacterStore (create_role → list_roles)."""
    ms = built_system
    assert isinstance(ms.character, CharacterStore)

    role_id = ms.character.create_role("雪奈", "温柔冷静")
    roles = ms.character.list_roles()
    assert len(roles) == 1
    assert roles[0]["id"] == role_id
    assert roles[0]["name"] == "雪奈"


def test_build_system_character_db_path_override(tmp_path):
    """Given character_db_path, When _build_system runs, Then the character
    tables live in that file, not in the main memory DB."""
    main_db = tmp_path / "main.db"
    char_db = tmp_path / "char.db"
    config = MemorySkillConfig(
        db_path=str(main_db),
        character_db_path=str(char_db),
        tree_enabled=False,
    )
    ms = _build_system(config)
    try:
        ms.character.create_role("A")
        char_conn = sqlite3.connect(str(char_db))
        try:
            count = char_conn.execute("SELECT COUNT(*) FROM roles").fetchone()[0]
        finally:
            char_conn.close()
        assert count == 1

        main_conn = sqlite3.connect(str(main_db))
        try:
            tables = {
                row[0]
                for row in main_conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            main_conn.close()
        assert "roles" not in tables
    finally:
        ms.character._conn.close()
        try:
            ms.dialogue_store._conn.close()
        except Exception:
            pass
        try:
            ms.learned_store._client.clear_system_cache()
        except Exception:
            pass


# ── delete_entry cascade ────────────────────────────────────────────────────


def test_delete_entry_cascades_to_character_references(cascade_system):
    """Given a role referencing two memories, When delete_entry removes one,
    Then that reference disappears from the role and the other is kept."""
    ms = cascade_system
    role_id = ms.character.create_role("A")
    ms.character.add_memory(role_id, "memory:gone")
    ms.character.add_memory(role_id, "memory:kept")

    ms.delete_entry("memory:gone")

    assert ms.learned_store.deleted == ["memory:gone"]
    assert ms.character.list_role_ids("memory:gone") == []
    assert ms.character.list_memories(role_id) == ["memory:kept"]


def test_delete_entry_removes_from_all_roles(cascade_system):
    """Given a memory referenced by two roles, When delete_entry, Then both
    references are removed while each role's other memories remain."""
    ms = cascade_system
    role_a = ms.character.create_role("A")
    role_b = ms.character.create_role("B")
    ms.character.add_memory(role_a, "memory:shared")
    ms.character.add_memory(role_b, "memory:shared")
    ms.character.add_memory(role_a, "memory:kept")

    ms.delete_entry("memory:shared")

    assert ms.character.list_role_ids("memory:shared") == []
    assert ms.character.list_memories(role_a) == ["memory:kept"]
    assert ms.character.list_memories(role_b) == []


def test_delete_entry_unreferenced_memory_is_noop(cascade_system):
    """Given a memory no character references, When delete_entry, Then no
    error and existing character data is untouched."""
    ms = cascade_system
    role_id = ms.character.create_role("A")
    ms.character.add_memory(role_id, "memory:kept")

    ms.delete_entry("memory:never-referenced")

    assert ms.learned_store.deleted == ["memory:never-referenced"]
    assert ms.character.list_memories(role_id) == ["memory:kept"]


def test_delete_entry_without_character_store_is_defensive():
    """Given a composition with no character store, When delete_entry, Then
    the learned store delete still runs and nothing raises."""
    learned = _DeletingFakeLearnedStore()
    ms = MemorySystem(learned_store=learned)

    ms.delete_entry("memory:x")

    assert learned.deleted == ["memory:x"]
