"""Tests for RelationStore — Tavern Mode Unit 2 (directed relation graph)."""

from __future__ import annotations

import pytest

from memory_skill.relation_store import RelationStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "relation_test.db"
    s = RelationStore(str(db_path))
    yield s
    s._conn.close()


def test_upsert_and_get(store):
    """Given an upserted edge, Then get_relation returns it with all fields."""
    store.upsert_relation("A", "B", "盟友", 70)
    r = store.get_relation("A", "B")
    assert r is not None
    assert r["from_role_id"] == "A"
    assert r["to_role_id"] == "B"
    assert r["relation_type"] == "盟友"
    assert r["strength"] == 70
    assert isinstance(r["updated_at"], float)


def test_direction_is_independent(store):
    """Given A→B and B→A, Then each edge keeps its own type/strength."""
    store.upsert_relation("A", "B", "爱慕", 80)
    store.upsert_relation("B", "A", "畏惧", 20)
    assert store.get_relation("A", "B")["relation_type"] == "爱慕"
    assert store.get_relation("B", "A")["relation_type"] == "畏惧"


def test_upsert_replaces_same_edge(store):
    """Given an existing edge, When upserting again, Then it is replaced."""
    store.upsert_relation("A", "B", "盟友", 50)
    store.upsert_relation("A", "B", "敌对", 10)
    r = store.get_relation("A", "B")
    assert r["relation_type"] == "敌对"
    assert r["strength"] == 10


def test_outgoing_and_incoming(store):
    """Given A→B and C→B, Then B has 1 incoming, A/C have 1 outgoing each."""
    store.upsert_relation("A", "B", "盟友", 70)
    store.upsert_relation("C", "B", "敌对", 30)
    out_a = store.get_outgoing("A")
    assert [e["to_role_id"] for e in out_a] == ["B"]
    inc_b = store.get_incoming("B")
    assert sorted(e["from_role_id"] for e in inc_b) == ["A", "C"]


def test_empty_relation_type_raises(store):
    """Given an empty relation_type, Then ValueError is raised."""
    with pytest.raises(ValueError):
        store.upsert_relation("A", "B", "  ", 50)


def test_strength_out_of_range_raises(store):
    """Given strength outside 0-100, Then ValueError is raised."""
    with pytest.raises(ValueError):
        store.upsert_relation("A", "B", "盟友", 101)
    with pytest.raises(ValueError):
        store.upsert_relation("A", "B", "盟友", -1)


def test_delete_relation(store):
    """Given an edge, When deleting, Then True once, then False."""
    store.upsert_relation("A", "B", "盟友", 50)
    assert store.delete_relation("A", "B") is True
    assert store.get_relation("A", "B") is None
    assert store.delete_relation("A", "B") is False


def test_health_counts_edges(store):
    """Given N distinct edges, Then health counts N."""
    assert store.health()["relation_count"] == 0
    store.upsert_relation("A", "B", "盟友", 50)
    store.upsert_relation("B", "C", "畏惧", 30)
    store.upsert_relation("A", "B", "爱慕", 80)  # replace, not a new edge
    assert store.health()["relation_count"] == 2
