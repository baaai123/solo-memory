"""MCP tool handlers for character (role) management.

Exercises the eight ``memory_character_*`` handlers in
``memory_skill.tools`` against a real ``CharacterStore`` (SQLite in a tmp
dir) mounted on a minimal ``MemorySystem`` shell, mirroring the
composition style of ``test_character_flow.py``.
"""

from __future__ import annotations

import pytest

from memory_skill._compose import MemorySystem
from memory_skill.character_store import CharacterStore
from memory_skill.tools import (
    handle_character_add_memory,
    handle_character_agent_role,
    handle_character_bind_agent,
    handle_character_create,
    handle_character_delete,
    handle_character_get,
    handle_character_list,
    handle_character_remove_memory,
)

ALL_HANDLERS = (
    handle_character_list,
    handle_character_create,
    handle_character_get,
    handle_character_delete,
    handle_character_add_memory,
    handle_character_remove_memory,
    handle_character_bind_agent,
    handle_character_agent_role,
)


def _skill_with_character(tmp_path, *, with_store: bool = True) -> MemorySystem:
    """A bare MemorySystem shell carrying only the CharacterStore field."""
    ms = object.__new__(MemorySystem)
    if with_store:
        ms.__dict__["character"] = CharacterStore(
            str(tmp_path / "character_tools.db")
        )
    return ms


def _create_role(skill, name: str = "雪奈", description: str = "温柔冷静") -> str:
    result = handle_character_create(
        skill, {"name": name, "description": description}
    )
    assert result["status"] == "created"
    return result["role_id"]


# ── list ─────────────────────────────────────────────────────────────────────


def test_list_empty_store(tmp_path):
    """Given an empty store, when listing, then roles is []."""
    skill = _skill_with_character(tmp_path)

    result = handle_character_list(skill, {})

    assert result == {"roles": []}


def test_list_returns_created_roles(tmp_path):
    """Given two created roles, when listing, then both come back with
    name and ref_count."""
    skill = _skill_with_character(tmp_path)
    _create_role(skill, "雪奈")
    _create_role(skill, "静流", "活泼")

    result = handle_character_list(skill, {})

    assert len(result["roles"]) == 2
    by_name = {role["name"]: role for role in result["roles"]}
    assert set(by_name) == {"雪奈", "静流"}
    assert by_name["静流"]["description"] == "活泼"
    assert all(role["ref_count"] == 0 for role in result["roles"])


# ── create ───────────────────────────────────────────────────────────────────


def test_create_returns_role_id(tmp_path):
    """Given a name, when creating, then status=created with a non-empty
    role_id and the role is retrievable."""
    skill = _skill_with_character(tmp_path)

    result = handle_character_create(skill, {"name": "雪奈", "description": "温柔"})

    assert result["status"] == "created"
    assert result["role_id"]
    assert skill.character.get_role(result["role_id"])["name"] == "雪奈"


@pytest.mark.parametrize("name", ["", "   ", None])
def test_create_rejects_blank_name(tmp_path, name):
    """Given a blank name, when creating, then error 'name is required'."""
    skill = _skill_with_character(tmp_path)

    result = handle_character_create(skill, {"name": name})

    assert result == {"error": "name is required"}


# ── store-unavailable defence ────────────────────────────────────────────────


@pytest.mark.parametrize("handler", ALL_HANDLERS)
def test_handlers_defend_missing_character_store(handler):
    """Given a skill without a character attribute, when calling any
    handler, then 'character store not available'."""
    skill = _skill_with_character(None, with_store=False)

    result = handler(skill, {"role_id": "x", "memory_id": "y", "name": "z",
                             "agent_name": "a"})

    assert result == {"error": "character store not available"}


# ── get ──────────────────────────────────────────────────────────────────────


def test_get_returns_role_and_memory_ids(tmp_path):
    """Given a role with two referenced memories, when getting, then both
    the role dict and the memory_ids list come back."""
    skill = _skill_with_character(tmp_path)
    role_id = _create_role(skill)
    skill.character.add_memory(role_id, "dialogue:m1")
    skill.character.add_memory(role_id, "dialogue:m2")

    result = handle_character_get(skill, {"role_id": role_id})

    assert result["role"]["id"] == role_id
    assert result["role"]["name"] == "雪奈"
    assert result["memory_ids"] == ["dialogue:m1", "dialogue:m2"]


def test_get_missing_role(tmp_path):
    """Given an unknown role_id, when getting, then 'role not found'."""
    skill = _skill_with_character(tmp_path)

    result = handle_character_get(skill, {"role_id": "character:ghost"})

    assert result == {"error": "role not found: character:ghost"}


# ── add / remove references ──────────────────────────────────────────────────


def test_add_remove_roundtrip(tmp_path):
    """Given a role, when adding two references then removing one, then
    the remaining reference survives and re-removal is a no-op."""
    skill = _skill_with_character(tmp_path)
    role_id = _create_role(skill)

    assert handle_character_add_memory(
        skill, {"role_id": role_id, "memory_id": "dialogue:m1"}
    ) == {"status": "added", "dimension": "general"}
    assert handle_character_add_memory(
        skill, {"role_id": role_id, "memory_id": "dialogue:m2"}
    ) == {"status": "added", "dimension": "general"}
    assert skill.character.list_memories(role_id) == ["dialogue:m1", "dialogue:m2"]

    assert handle_character_remove_memory(
        skill, {"role_id": role_id, "memory_id": "dialogue:m1"}
    ) == {"status": "removed"}
    assert skill.character.list_memories(role_id) == ["dialogue:m2"]
    assert handle_character_remove_memory(
        skill, {"role_id": role_id, "memory_id": "dialogue:m1"}
    ) == {"status": "removed"}
    assert skill.character.list_memories(role_id) == ["dialogue:m2"]


def test_add_memory_unknown_role(tmp_path):
    """Given an unknown role_id, when adding a memory, then error."""
    skill = _skill_with_character(tmp_path)

    result = handle_character_add_memory(
        skill, {"role_id": "character:ghost", "memory_id": "dialogue:m1"}
    )

    assert result == {"error": "role not found: character:ghost"}


# ── delete ───────────────────────────────────────────────────────────────────


def test_delete_removes_from_list(tmp_path):
    """Given two roles, when deleting one, then list shrinks to one and a
    second delete errors."""
    skill = _skill_with_character(tmp_path)
    keep = _create_role(skill, "静流")
    gone = _create_role(skill, "雪奈")

    assert handle_character_delete(skill, {"role_id": gone}) == {
        "status": "deleted", "role_id": gone,
    }

    names = {role["name"] for role in handle_character_list(skill, {})["roles"]}
    assert names == {"静流"}
    assert skill.character.get_role(keep) is not None
    assert handle_character_delete(skill, {"role_id": gone}) == {
        "error": f"role not found: {gone}",
    }


def test_delete_cascades_binding_and_refs(tmp_path):
    """Given a role with a bound agent and references, when deleting it,
    then the binding and references are gone too."""
    skill = _skill_with_character(tmp_path)
    role_id = _create_role(skill)
    skill.character.add_memory(role_id, "dialogue:m1")
    skill.character.bind_agent("test_agent", role_id)

    result = handle_character_delete(skill, {"role_id": role_id})

    assert result == {"status": "deleted", "role_id": role_id}
    assert skill.character.get_agent_role("test_agent") is None
    assert skill.character.list_memories(role_id) == []
    assert skill.character.get_role(role_id) is None


# ── bind / unbind / query ────────────────────────────────────────────────────


def test_bind_agent_then_query_role(tmp_path):
    """Given a role, when binding an agent, then status=bound and
    agent_role reports the role_id."""
    skill = _skill_with_character(tmp_path)
    role_id = _create_role(skill)

    result = handle_character_bind_agent(
        skill, {"agent_name": "test_agent", "role_id": role_id}
    )

    assert result == {"status": "bound", "agent_name": "test_agent",
                      "role_id": role_id}
    assert handle_character_agent_role(skill, {"agent_name": "test_agent"}) == {
        "agent_name": "test_agent", "role_id": role_id,
    }


def test_bind_unknown_role(tmp_path):
    """Given an unknown role_id, when binding, then error."""
    skill = _skill_with_character(tmp_path)

    result = handle_character_bind_agent(
        skill, {"agent_name": "test_agent", "role_id": "character:ghost"}
    )

    assert result == {"error": "role not found: character:ghost"}


def test_unbind_with_empty_role_id(tmp_path):
    """Given a bound agent, when binding with an empty role_id, then the
    binding is removed and agent_role returns None."""
    skill = _skill_with_character(tmp_path)
    role_id = _create_role(skill)
    skill.character.bind_agent("test_agent", role_id)

    result = handle_character_bind_agent(
        skill, {"agent_name": "test_agent", "role_id": ""}
    )

    assert result == {"status": "unbound", "agent_name": "test_agent",
                      "role_id": None}
    assert handle_character_agent_role(skill, {"agent_name": "test_agent"}) == {
        "agent_name": "test_agent", "role_id": None,
    }


def test_rebind_switches_role(tmp_path):
    """Given a bound agent, when rebinding to another role, then
    agent_role reports the new role_id."""
    skill = _skill_with_character(tmp_path)
    first = _create_role(skill, "雪奈")
    second = _create_role(skill, "静流")
    handle_character_bind_agent(
        skill, {"agent_name": "test_agent", "role_id": first}
    )

    result = handle_character_bind_agent(
        skill, {"agent_name": "test_agent", "role_id": second}
    )

    assert result == {"status": "bound", "agent_name": "test_agent",
                      "role_id": second}
    assert skill.character.get_agent_role("test_agent") == second


def test_agent_role_unbound(tmp_path):
    """Given an agent never bound, when querying, then role_id is None."""
    skill = _skill_with_character(tmp_path)

    result = handle_character_agent_role(skill, {"agent_name": "nobody"})

    assert result == {"agent_name": "nobody", "role_id": None}
