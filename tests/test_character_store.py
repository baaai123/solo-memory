"""Tests for CharacterStore — Unit 1 of the character-store backend plan.

Covers role CRUD, memory reference add/remove/list, agent bindings, cascade
deletes, and health.  Uses only a temporary SQLite file — no other stores,
no LLM, no Chroma.
"""

from __future__ import annotations

import pytest

from memory_skill.character_store import CharacterStore


@pytest.fixture
def store(tmp_path):
    """A CharacterStore backed by a fresh temporary SQLite file."""
    db_path = tmp_path / "character_test.db"
    s = CharacterStore(str(db_path))
    yield s
    s._conn.close()


# ── Roles: create / list / get ──────────────────────────────────────────────


def test_create_role_then_list_and_get(store: CharacterStore):
    """Given a fresh store, When creating a role, Then list_roles and get_role
    both expose it with the right fields and ref_count 0."""
    role_id = store.create_role("雪奈", "温柔冷静")

    roles = store.list_roles()
    assert len(roles) == 1
    assert roles[0]["id"] == role_id
    assert roles[0]["name"] == "雪奈"
    assert roles[0]["description"] == "温柔冷静"
    assert roles[0]["ref_count"] == 0
    assert isinstance(roles[0]["created_at"], float)
    assert roles[0]["created_at"] == roles[0]["updated_at"]

    detail = store.get_role(role_id)
    assert detail is not None
    assert detail["id"] == role_id
    assert detail["name"] == "雪奈"
    assert detail["ref_count"] == 0


def test_get_role_missing_returns_none(store: CharacterStore):
    """Given an unknown id, When get_role is called, Then None is returned."""
    assert store.get_role("character:missing") is None


def test_list_roles_is_ordered_and_ref_count_is_correct(store: CharacterStore):
    """Given two roles where one references memories, When listing, Then the
    oldest comes first and ref_count matches each role's references."""
    role_a = store.create_role("A")
    role_b = store.create_role("B")
    store.add_memory(role_b, "memory:1")
    store.add_memory(role_b, "memory:2")

    roles = store.list_roles()
    assert [r["id"] for r in roles] == [role_a, role_b]
    assert roles[0]["ref_count"] == 0
    assert roles[1]["ref_count"] == 2
    role_b_detail = store.get_role(role_b)
    assert role_b_detail is not None
    assert role_b_detail["ref_count"] == 2


def test_create_role_empty_name_raises(store: CharacterStore):
    """Given an empty/whitespace name, When create_role is called, Then a
    ValueError is raised and nothing is stored."""
    with pytest.raises(ValueError):
        store.create_role("")
    with pytest.raises(ValueError):
        store.create_role("   ")
    assert store.list_roles() == []


def test_update_role_changes_only_provided_fields(store: CharacterStore):
    """Given a role, When update_role is called with one field, Then only that
    field changes and updated_at moves forward."""
    role_id = store.create_role("静流", "旧描述")
    old_updated = store.get_role(role_id)
    assert old_updated is not None
    old_updated = old_updated["updated_at"]

    assert store.update_role(role_id, name="静流改") is True
    detail = store.get_role(role_id)
    assert detail is not None
    assert detail["name"] == "静流改"
    assert detail["description"] == "旧描述"

    assert store.update_role(role_id, description="新描述") is True
    detail = store.get_role(role_id)
    assert detail is not None
    assert detail["description"] == "新描述"
    assert detail["updated_at"] >= old_updated


def test_update_role_missing_returns_false(store: CharacterStore):
    """Given an unknown id, When update_role is called, Then False."""
    assert store.update_role("character:nope", name="x") is False


def test_update_role_empty_name_raises(store: CharacterStore):
    """Given an empty new name, When update_role is called, Then ValueError."""
    role_id = store.create_role("A")
    with pytest.raises(ValueError):
        store.update_role(role_id, name="  ")


# ── Memory references ───────────────────────────────────────────────────────


def test_add_and_remove_memory_roundtrip(store: CharacterStore):
    """Given a role, When add_memory then remove_memory, Then the reference
    appears in list_memories and disappears after removal."""
    role_id = store.create_role("A")

    assert store.add_memory(role_id, "memory:1") is True
    assert store.list_memories(role_id) == ["memory:1"]
    assert store.list_role_ids("memory:1") == [role_id]

    assert store.remove_memory(role_id, "memory:1") is True
    assert store.list_memories(role_id) == []
    assert store.list_role_ids("memory:1") == []


def test_add_memory_is_idempotent(store: CharacterStore):
    """Given duplicate add_memory calls, When listing, Then the reference
    appears exactly once (composite PK prevents duplicates)."""
    role_id = store.create_role("A")
    store.add_memory(role_id, "memory:1")
    assert store.add_memory(role_id, "memory:1") is True

    assert store.list_memories(role_id) == ["memory:1"]
    detail = store.get_role(role_id)
    assert detail is not None
    assert detail["ref_count"] == 1


def test_list_memories_ordered_by_added_at(store: CharacterStore):
    """Given two references added in order, When listing, Then they come back
    in add order (added_at ascending)."""
    role_id = store.create_role("A")
    store.add_memory(role_id, "memory:first")
    store.add_memory(role_id, "memory:second")
    assert store.list_memories(role_id) == ["memory:first", "memory:second"]


def test_list_role_ids_returns_all_referencing_roles(store: CharacterStore):
    """Given one memory referenced by two roles, When list_role_ids is called,
    Then both role ids are returned."""
    role_a = store.create_role("A")
    role_b = store.create_role("B")
    store.add_memory(role_a, "memory:shared")
    store.add_memory(role_b, "memory:shared")

    assert sorted(store.list_role_ids("memory:shared")) == sorted([role_a, role_b])


def test_remove_missing_reference_is_noop(store: CharacterStore):
    """Given a role with no such reference, When remove_memory is called,
    Then no error and True (idempotent delete)."""
    role_id = store.create_role("A")
    assert store.remove_memory(role_id, "memory:never-added") is True


def test_add_memory_missing_role_returns_false(store: CharacterStore):
    """Given an unknown role, When add_memory is called, Then False and no
    reference row is created."""
    assert store.add_memory("character:nope", "memory:1") is False
    assert store.list_role_ids("memory:1") == []


def test_remove_memory_missing_role_returns_false(store: CharacterStore):
    """Given an unknown role, When remove_memory is called, Then False."""
    assert store.remove_memory("character:nope", "memory:1") is False


def test_list_memories_unknown_role_returns_empty(store: CharacterStore):
    """Given an unknown role, When list_memories is called, Then empty list."""
    assert store.list_memories("character:nope") == []


# ── Cascade deletes ─────────────────────────────────────────────────────────


def test_delete_role_cascades_references_and_bindings(store: CharacterStore):
    """Given a role with references and an agent binding, When delete_role,
    Then its references and binding disappear while another role is intact."""
    role_a = store.create_role("A")
    role_b = store.create_role("B")
    store.add_memory(role_a, "memory:1")
    store.add_memory(role_a, "memory:2")
    store.add_memory(role_b, "memory:3")
    store.bind_agent("agent_x", role_a)

    assert store.delete_role(role_a) is True

    # Role gone; its references and binding gone with it.
    assert store.get_role(role_a) is None
    assert store.list_memories(role_a) == []
    assert store.list_role_ids("memory:1") == []
    assert store.list_role_ids("memory:2") == []
    assert store.get_agent_role("agent_x") is None

    # Other role untouched.
    assert store.get_role(role_b) is not None
    assert store.list_memories(role_b) == ["memory:3"]


def test_delete_role_missing_returns_false(store: CharacterStore):
    """Given an unknown role, When delete_role, Then False."""
    assert store.delete_role("character:nope") is False


def test_remove_all_memory_removes_from_all_roles(store: CharacterStore):
    """Given a memory referenced by two roles, When remove_all_memory, Then
    both references are deleted and the row count is returned."""
    role_a = store.create_role("A")
    role_b = store.create_role("B")
    store.add_memory(role_a, "memory:shared")
    store.add_memory(role_b, "memory:shared")
    store.add_memory(role_a, "memory:kept")

    removed = store.remove_all_memory("memory:shared")
    assert removed == 2
    assert store.list_role_ids("memory:shared") == []
    assert store.list_memories(role_a) == ["memory:kept"]
    assert store.list_memories(role_b) == []


def test_remove_all_memory_unknown_returns_zero(store: CharacterStore):
    """Given an unreferenced memory id, When remove_all_memory, Then 0."""
    assert store.remove_all_memory("memory:unreferenced") == 0


# ── Agent bindings ──────────────────────────────────────────────────────────


def test_bind_get_and_rebind_agent(store: CharacterStore):
    """Given two roles, When binding and rebinding an agent, Then
    get_agent_role follows the latest binding."""
    role_a = store.create_role("A")
    role_b = store.create_role("B")

    assert store.bind_agent("agent_x", role_a) is True
    assert store.get_agent_role("agent_x") == role_a

    # INSERT OR REPLACE: rebinding switches to the new role.
    assert store.bind_agent("agent_x", role_b) is True
    assert store.get_agent_role("agent_x") == role_b


def test_unbind_agent_clears_binding(store: CharacterStore):
    """Given a bound agent, When unbind_agent, Then get_agent_role is None
    and unbind returns True."""
    role_id = store.create_role("A")
    store.bind_agent("agent_x", role_id)

    assert store.unbind_agent("agent_x") is True
    assert store.get_agent_role("agent_x") is None


def test_unbind_agent_unknown_returns_false(store: CharacterStore):
    """Given an unbound agent, When unbind_agent, Then False."""
    assert store.unbind_agent("agent_nobody") is False


def test_get_agent_role_unbound_returns_none(store: CharacterStore):
    """Given an unbound agent, When get_agent_role, Then None."""
    assert store.get_agent_role("agent_nobody") is None


def test_bind_agent_missing_role_returns_false(store: CharacterStore):
    """Given an unknown role, When bind_agent, Then False and no binding."""
    assert store.bind_agent("agent_x", "character:nope") is False
    assert store.get_agent_role("agent_x") is None


# ── Health ──────────────────────────────────────────────────────────────────


def test_health_reports_tables_and_counts(store: CharacterStore):
    """Given a fresh store, When health is called, Then status healthy with
    zero counts; counts follow data afterwards."""
    h0 = store.health()
    assert h0["status"] == "healthy"
    assert h0["backend"] == "sqlite3"
    assert h0["roles"] == 0
    assert h0["role_memories"] == 0
    assert h0["agent_bindings"] == 0

    role_id = store.create_role("A")
    store.add_memory(role_id, "memory:1")
    store.bind_agent("agent_x", role_id)

    h1 = store.health()
    assert h1["status"] == "healthy"
    assert h1["roles"] == 1
    assert h1["role_memories"] == 1
    assert h1["agent_bindings"] == 1


# ── Persona dimensions (tavern mode) ──────────────────────────────────────


def test_add_memory_defaults_to_general(store: CharacterStore):
    """Given no dimension, When add_memory, Then stored as general."""
    rid = store.create_role("A")
    assert store.add_memory(rid, "memory:1") is True
    assert store.list_memories(rid) == ["memory:1"]
    assert store.list_memories(rid, dimension="general") == ["memory:1"]
    assert store.list_memories(rid, dimension="skills") == []
    pairs = store.list_memory_dims(rid)
    assert pairs == [{"memory_id": "memory:1", "dimension": "general"}]


def test_add_memory_with_persona_dimension(store: CharacterStore):
    """Given a persona dimension, When add_memory, Then tagged and filterable."""
    rid = store.create_role("A")
    store.add_memory(rid, "memory:skill", dimension="skills")
    store.add_memory(rid, "memory:app", dimension="appearance")
    store.add_memory(rid, "memory:per", dimension="personality")
    store.add_memory(rid, "memory:gen")

    assert store.list_memories(rid) == [
        "memory:skill", "memory:app", "memory:per", "memory:gen",
    ]
    assert store.list_memories(rid, dimension="skills") == ["memory:skill"]
    assert store.list_memories(rid, dimension="appearance") == ["memory:app"]
    assert store.list_memories(rid, dimension="personality") == ["memory:per"]
    pairs = store.list_memory_dims(rid)
    assert {p["dimension"] for p in pairs} == {
        "general", "skills", "appearance", "personality",
    }


def test_add_memory_idempotent_keeps_first_dimension(store: CharacterStore):
    """Given a duplicate add with a different dimension, Then the original
    dimension is kept (INSERT OR IGNORE, composite PK wins)."""
    rid = store.create_role("A")
    store.add_memory(rid, "memory:1", dimension="skills")
    store.add_memory(rid, "memory:1", dimension="personality")
    assert store.list_memories(rid, dimension="skills") == ["memory:1"]
    assert store.list_memories(rid, dimension="personality") == []


def test_set_memory_dimension_retags(store: CharacterStore):
    """Given an existing ref, When set_memory_dimension, Then the tag
    changes; unknown refs return False."""
    rid = store.create_role("A")
    store.add_memory(rid, "memory:1")
    assert store.set_memory_dimension(rid, "memory:1", "skills") is True
    assert store.list_memories(rid, dimension="skills") == ["memory:1"]
    assert store.list_memories(rid, dimension="general") == []
    assert store.set_memory_dimension(rid, "memory:ghost", "skills") is False
    assert store.set_memory_dimension("character:ghost", "memory:1", "skills") is False
