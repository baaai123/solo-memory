"""Tests for StateStore — Tavern Mode Unit 1 (per-role covering state).

Covers first write, covering update, None-ignored semantics, invalid-dim
filtering, empty dims, explicit timestamps, multi-role isolation, delete, and
health counting.  Uses only a temporary SQLite file — no other stores, no LLM.
"""

from __future__ import annotations

import pytest

from memory_skill.state_store import StateStore

DIMS = ("mood", "need", "health", "clothing", "item", "action", "scene", "weather")


@pytest.fixture
def store(tmp_path):
    """A StateStore backed by a fresh temporary SQLite file."""
    db_path = tmp_path / "state_test.db"
    s = StateStore(str(db_path))
    yield s
    s._conn.close()


def _full_dims(**overrides) -> dict:
    base = {d: f"val-{d}" for d in DIMS}
    base.update(overrides)
    return base


# ── First write ─────────────────────────────────────────────────────────────


def test_first_write_all_dims(store):
    """Given no record, When writing all 8 dims, Then get_state returns them."""
    store.update_state("r1", _full_dims())
    st = store.get_state("r1")
    assert st is not None
    for d in DIMS:
        assert st[d] == f"val-{d}"
    assert isinstance(st["updated_at"], float)


def test_first_write_partial_fills_null(store):
    """Given a first write of only mood, Then the other 7 dims are NULL."""
    store.update_state("r1", {"mood": "焦虑"})
    st = store.get_state("r1")
    assert st["mood"] == "焦虑"
    for d in DIMS:
        if d != "mood":
            assert st[d] is None


# ── Covering update ─────────────────────────────────────────────────────────


def test_overwrite_same_dim(store):
    """Given a dim already written, When updating it, Then only latest kept."""
    store.update_state("r1", {"mood": "焦虑"})
    store.update_state("r1", {"mood": "警觉"})
    assert store.get_state("r1")["mood"] == "警觉"


def test_update_partial_keeps_others(store):
    """Given two dims written, When updating one, Then the other is kept."""
    store.update_state("r1", {"mood": "焦虑", "health": "轻伤"})
    store.update_state("r1", {"clothing": "皮甲"})
    st = store.get_state("r1")
    assert st["mood"] == "焦虑"
    assert st["health"] == "轻伤"
    assert st["clothing"] == "皮甲"


# ── None ignored ────────────────────────────────────────────────────────────


def test_none_keeps_previous(store):
    """Given a dim written, When updating it with None, Then value unchanged."""
    store.update_state("r1", {"mood": "焦虑"})
    store.update_state("r1", {"mood": None, "health": "健康"})
    st = store.get_state("r1")
    assert st["mood"] == "焦虑"
    assert st["health"] == "健康"


def test_first_write_all_none_creates_nothing(store):
    """Given all-None dims, When writing, Then no record is created."""
    store.update_state("r1", {"mood": None, "health": None})
    assert store.get_state("r1") is None


# ── Invalid dimensions ──────────────────────────────────────────────────────


def test_invalid_dim_only_creates_nothing(store):
    """Given only an invalid key, When writing, Then no record is created."""
    store.update_state("r1", {"hacker": "x"})
    assert store.get_state("r1") is None


def test_mixed_valid_invalid(store):
    """Given valid + invalid keys, Then valid is written, invalid ignored."""
    store.update_state("r1", {"mood": "平静", "hacker": "x"})
    st = store.get_state("r1")
    assert st["mood"] == "平静"
    assert "hacker" not in st


# ── Empty dims ──────────────────────────────────────────────────────────────


def test_empty_dims_keeps_existing(store):
    """Given an existing record, When writing {}, Then nothing changes."""
    store.update_state("r1", {"mood": "焦虑"})
    before = store.get_state("r1")
    store.update_state("r1", {})
    assert store.get_state("r1") == before


# ── Explicit timestamp ──────────────────────────────────────────────────────


def test_explicit_timestamp(store):
    """Given an explicit timestamp, Then updated_at equals it exactly."""
    store.update_state("r1", {"mood": "平静"}, timestamp=123.0)
    assert store.get_state("r1")["updated_at"] == 123.0


# ── Multi-role isolation ────────────────────────────────────────────────────


def test_roles_isolated(store):
    """Given two roles, Then writing one never affects the other."""
    store.update_state("rA", {"mood": "愉悦"})
    store.update_state("rB", {"mood": "沮丧"})
    assert store.get_state("rA")["mood"] == "愉悦"
    assert store.get_state("rB")["mood"] == "沮丧"


# ── Delete ──────────────────────────────────────────────────────────────────


def test_delete_existing_and_nonexistent(store):
    """Given a record, Then delete returns True once, then False."""
    store.update_state("r1", {"mood": "焦虑"})
    assert store.delete_state("r1") is True
    assert store.get_state("r1") is None
    assert store.delete_state("r1") is False


# ── Health ──────────────────────────────────────────────────────────────────


def test_health_counts_distinct_roles(store):
    """Given writes to N roles, Then health counts N (updates don't inflate)."""
    assert store.health()["state_count"] == 0
    store.update_state("r1", {"mood": "a"})
    store.update_state("r2", {"mood": "b"})
    store.update_state("r1", {"mood": "c"})  # update, not a new role
    assert store.health()["state_count"] == 2
    store.delete_state("r1")
    assert store.health()["state_count"] == 1
