"""
CharacterStore — SQLite-backed storage for character (role) memory references.

A character (角色) is a *reference set* of global memories — it never copies
memory content.  This module owns three tables in the memory SQLite database:

- ``roles``          — character metadata (id, name, description, timestamps).
- ``role_memories``  — (role_id, memory_id) join rows; the composite primary
                       key makes duplicate references impossible.
- ``agent_bindings`` — agent_name -> role_id mapping used by weave/search
                       to resolve which character an agent is playing.

Uses only Python stdlib ``sqlite3`` — no external dependencies, no ORM.
Connection model follows ``DialogueStore``: a single connection created with
``check_same_thread=False`` + ``busy_timeout``, schema created idempotently
with ``CREATE TABLE IF NOT EXISTS``, one commit per operation.

This is Unit 1 of the character-store backend plan.  Retrieval whitelist
filtering, ingest double-write, and weave role passing are later units.
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from memory_skill.contracts import utcnow

# Persona dimensions for tavern roles (spec §3.2); general = pre-tavern plain ref.
PERSONA_DIMENSIONS = ("general", "skills", "appearance", "personality")

# ── Schema ────────────────────────────────────────────────────────────────────
# Module-level constant on purpose: schema edits rewrite this whole constant,
# never via substring replacement on the class body.
_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS roles (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    personality  TEXT NOT NULL DEFAULT '',
    scenario     TEXT NOT NULL DEFAULT '',
    mes_example  TEXT NOT NULL DEFAULT '',
    creator_notes TEXT NOT NULL DEFAULT '',
    first_mes    TEXT NOT NULL DEFAULT '',
    system_prompt TEXT NOT NULL DEFAULT '',
    avatar       TEXT NOT NULL DEFAULT '',
    is_tavern    INTEGER NOT NULL DEFAULT 0,
    st_name      TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS role_memories (
    role_id    TEXT NOT NULL,
    memory_id  TEXT NOT NULL,
    dimension  TEXT NOT NULL DEFAULT 'general',
    target     TEXT NOT NULL DEFAULT '',
    added_at   REAL NOT NULL,
    PRIMARY KEY (role_id, memory_id)
);

CREATE TABLE IF NOT EXISTS agent_bindings (
    agent_name TEXT PRIMARY KEY,
    role_id    TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class CharacterStore:
    """Storage layer for character memory reference sets.

    Parameters:
        db_path: Path to the SQLite file (shared memory DB or standalone).

    All timestamps are stored as Unix epoch floats (seconds); returned dicts
    carry the raw float values.
    """

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_CREATE_TABLES_SQL)
        self._migrate_is_tavern()
        self._migrate_st_name()
        self._migrate_dimension()
        self._migrate_card_fields()

    def _migrate_card_fields(self) -> None:
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(roles)").fetchall()]
        for col in ("personality", "scenario", "mes_example", "creator_notes",
                    "first_mes", "system_prompt", "avatar"):
            if col not in cols:
                self._conn.execute(
                    f"ALTER TABLE roles ADD COLUMN {col} TEXT NOT NULL DEFAULT ''"
                )
        self._conn.commit()

    def _migrate_is_tavern(self) -> None:
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(roles)").fetchall()]
        if "is_tavern" not in cols:
            self._conn.execute(
                "ALTER TABLE roles ADD COLUMN is_tavern INTEGER NOT NULL DEFAULT 0"
            )
            self._conn.commit()

    def _migrate_st_name(self) -> None:
        cols = [r[1] for r in self._conn.execute("PRAGMA table_info(roles)").fetchall()]
        if "st_name" not in cols:
            self._conn.execute(
                "ALTER TABLE roles ADD COLUMN st_name TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()

    def _migrate_dimension(self) -> None:
        cols = [r[1] for r in
                self._conn.execute("PRAGMA table_info(role_memories)").fetchall()]
        if "dimension" not in cols:
            self._conn.execute(
                "ALTER TABLE role_memories "
                "ADD COLUMN dimension TEXT NOT NULL DEFAULT 'general'"
            )
            self._conn.commit()
        if "target" not in cols:
            self._conn.execute(
                "ALTER TABLE role_memories "
                "ADD COLUMN target TEXT NOT NULL DEFAULT ''"
            )
            self._conn.commit()

    # ── Roles ─────────────────────────────────────────────────────────────

    def create_role(self, name: str, description: str = "",
                    is_tavern: bool = False,
                    st_name: str = "",
                    personality: str = "", scenario: str = "",
                    mes_example: str = "", creator_notes: str = "",
                    first_mes: str = "",
                    system_prompt: str = "", avatar: str = "") -> str:
        """Create a new character and return its id.

        *st_name* optionally binds the role to a SillyTavern character name
        (used by the tavern bridge's role lookup); ignored outside tavern use.
        The *personality/scenario/mes_example/creator_notes/first_mes* fields
        mirror a Tavern card so imported characters keep their definitions.
        *system_prompt* holds the card's global play-rule block (injected at
        the top of the role's system prompt); *avatar* an optional avatar
        data URL/base64 kept for display.
        Raises:
            ValueError: If *name* is empty or whitespace-only.
        """
        if not name.strip():
            raise ValueError("role name must not be empty")
        now = utcnow().timestamp()
        role_id = f"character:{int(now)}_{uuid.uuid4().hex[:6]}"
        self._conn.execute(
            "INSERT INTO roles (id, name, description, personality, scenario, "
            "mes_example, creator_notes, first_mes, system_prompt, avatar, "
            "is_tavern, st_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (role_id, name, description, personality, scenario, mes_example,
             creator_notes, first_mes, system_prompt, avatar,
             1 if is_tavern else 0, st_name, now, now),
        )
        self._conn.commit()
        return role_id

    def list_roles(self) -> list[dict[str, Any]]:
        """Return all characters (oldest first), each with a ``ref_count``.

        ``ref_count`` is the number of memories this character references
        (LEFT JOIN COUNT on ``role_memories``).
        """
        rows = self._conn.execute(
            "SELECT r.id, r.name, r.description, r.personality, r.scenario, "
            "       r.mes_example, r.creator_notes, r.first_mes, r.system_prompt, r.avatar, r.is_tavern, r.st_name, "
            "       r.created_at, "
             "       r.updated_at, COUNT(rm.memory_id) AS ref_count "
            "FROM roles r "
            "LEFT JOIN role_memories rm ON rm.role_id = r.id "
            "GROUP BY r.id "
            "ORDER BY r.created_at ASC"
        ).fetchall()
        return [_row_to_role(row) for row in rows]

    def get_role(self, role_id: str) -> dict[str, Any] | None:
        """Return a character's details (with ``ref_count``), or ``None``."""
        row = self._conn.execute(
            "SELECT r.id, r.name, r.description, r.personality, r.scenario, "
            "       r.mes_example, r.creator_notes, r.first_mes, r.system_prompt, r.avatar, r.is_tavern, r.st_name, "
            "       r.created_at, "
             "       r.updated_at, COUNT(rm.memory_id) AS ref_count "
            "FROM roles r "
            "LEFT JOIN role_memories rm ON rm.role_id = r.id "
            "WHERE r.id = ? "
            "GROUP BY r.id",
            (role_id,),
        ).fetchone()
        return _row_to_role(row) if row is not None else None

    def get_role_by_st_name(self, st_name: str) -> dict[str, Any] | None:
        """Return the role bound to a SillyTavern character name.

        Matching is case-insensitive so a SillyTavern rename keeps the
        binding.  Returns ``None`` when no role is bound to *st_name*.
        """
        row = self._conn.execute(
            "SELECT r.id, r.name, r.description, r.personality, r.scenario, "
            "       r.mes_example, r.creator_notes, r.first_mes, r.system_prompt, r.avatar, r.is_tavern, r.st_name, "
            "       r.created_at, r.updated_at, COUNT(rm.memory_id) AS ref_count "
            "FROM roles r "
            "LEFT JOIN role_memories rm ON rm.role_id = r.id "
            "WHERE LOWER(r.st_name) = LOWER(?) "
            "GROUP BY r.id "
            "ORDER BY r.updated_at DESC "
            "LIMIT 1",
            (st_name,),
        ).fetchone()
        return _row_to_role(row) if row is not None else None

    def update_role(
        self,
        role_id: str,
        name: str | None = None,
        description: str | None = None,
        is_tavern: bool | None = None,
        st_name: str | None = None,
        personality: str | None = None,
        scenario: str | None = None,
        system_prompt: str | None = None,
        avatar: str | None = None,
        mes_example: str | None = None,
        creator_notes: str | None = None,
        first_mes: str | None = None,
    ) -> bool:
        """Update a character's card fields.

        ``st_name=None`` leaves the binding untouched; pass ``""`` to clear.
        ``None`` on any card field also leaves it untouched.
        Returns ``False`` when the character does not exist.

        Raises:
            ValueError: If *name* is provided but empty or whitespace-only.
        """
        if not self._role_exists(role_id):
            return False
        if name is not None and not name.strip():
            raise ValueError("role name must not be empty")

        assignments: list[str] = []
        params: list[Any] = []
        if name is not None:
            assignments.append("name = ?")
            params.append(name)
        if description is not None:
            assignments.append("description = ?")
            params.append(description)
        if personality is not None:
            assignments.append("personality = ?")
            params.append(personality)
        if scenario is not None:
            assignments.append("scenario = ?")
            params.append(scenario)
        if mes_example is not None:
            assignments.append("mes_example = ?")
            params.append(mes_example)
        if creator_notes is not None:
            assignments.append("creator_notes = ?")
            params.append(creator_notes)
        if first_mes is not None:
            assignments.append("first_mes = ?")
            params.append(first_mes)
        if system_prompt is not None:
            assignments.append("system_prompt = ?")
            params.append(system_prompt)
        if avatar is not None:
            assignments.append("avatar = ?")
            params.append(avatar)
        if is_tavern is not None:
            assignments.append("is_tavern = ?")
            params.append(1 if is_tavern else 0)
        if st_name is not None:
            assignments.append("st_name = ?")
            params.append(st_name)
        if assignments:
            assignments.append("updated_at = ?")
            params.append(utcnow().timestamp())
            params.append(role_id)
            self._conn.execute(
                f"UPDATE roles SET {', '.join(assignments)} WHERE id = ?",
                params,
            )
            self._conn.commit()
        return True

    def delete_role(self, role_id: str) -> bool:
        """Delete a character and cascade its references and bindings.

        Removes the character's ``role_memories`` rows and ``agent_bindings``
        rows in the same transaction.  Returns ``False`` when the character
        does not exist.
        """
        if not self._role_exists(role_id):
            return False
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "DELETE FROM role_memories WHERE role_id = ?", (role_id,)
            )
            self._conn.execute(
                "DELETE FROM agent_bindings WHERE role_id = ?", (role_id,)
            )
            self._conn.execute("DELETE FROM roles WHERE id = ?", (role_id,))
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return True

    # ── Memory references ─────────────────────────────────────────────────

    def add_memory(self, role_id: str, memory_id: str,
                   dimension: str = "general", target: str = "") -> bool:
        """Add a memory reference to a character (idempotent).

        ``INSERT OR IGNORE`` on the composite primary key makes repeated
        calls no-ops — the first dimension wins on a duplicate add.  Tavern
        persona dimensions are ``skills`` / ``appearance`` / ``personality``
        (any other value is stored verbatim for forward compatibility).
        *target* marks an entry as "about X" (a role id or the reserved
        ``user``); an empty target means the entry is about the role itself.
        Returns ``False`` when the character does not exist.
        """
        if not self._role_exists(role_id):
            return False
        self._conn.execute(
            "INSERT OR IGNORE INTO role_memories "
            "(role_id, memory_id, dimension, target, added_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (role_id, memory_id, dimension, target or "", utcnow().timestamp()),
        )
        self._conn.commit()
        return True

    def set_memory_target(self, role_id: str, memory_id: str,
                          target: str) -> bool:
        """Mark a reference as being about *target* (role id or ``user``).
        Returns ``False`` when the reference does not exist."""
        cursor = self._conn.execute(
            "UPDATE role_memories SET target = ? "
            "WHERE role_id = ? AND memory_id = ?",
            (target or "", role_id, memory_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def remove_memory(self, role_id: str, memory_id: str) -> bool:
        """Remove a memory reference from a character.

        Idempotent — removing a reference that does not exist is a no-op.
        Returns ``False`` when the character does not exist.
        """
        if not self._role_exists(role_id):
            return False
        self._conn.execute(
            "DELETE FROM role_memories WHERE role_id = ? AND memory_id = ?",
            (role_id, memory_id),
        )
        self._conn.commit()
        return True

    def set_memory_dimension(self, role_id: str, memory_id: str,
                             dimension: str) -> bool:
        """Retag a reference's persona dimension.

        Returns ``False`` when the character or the reference does not
        exist (so the UI can surface "not found" without guessing).
        """
        cursor = self._conn.execute(
            "UPDATE role_memories SET dimension = ? "
            "WHERE role_id = ? AND memory_id = ?",
            (dimension, role_id, memory_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def list_memories(self, role_id: str,
                      dimension: str | None = None) -> list[str]:
        """Return a character's referenced memory ids (oldest first).

        With *dimension* (e.g. ``"skills"``) only ids tagged with that
        persona dimension are returned.  Returns an empty list for an unknown
        character.
        """
        if dimension is None:
            rows = self._conn.execute(
                "SELECT memory_id FROM role_memories "
                "WHERE role_id = ? ORDER BY added_at ASC",
                (role_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT memory_id FROM role_memories "
                "WHERE role_id = ? AND dimension = ? ORDER BY added_at ASC",
                (role_id, dimension),
            ).fetchall()
        return [row[0] for row in rows]

    def list_memory_dims(self, role_id: str) -> list[dict[str, str]]:
        """Return (memory_id, dimension, target) triples, oldest first."""
        rows = self._conn.execute(
            "SELECT memory_id, dimension, target FROM role_memories "
            "WHERE role_id = ? ORDER BY added_at ASC",
            (role_id,),
        ).fetchall()
        return [{"memory_id": r[0], "dimension": r[1], "target": r[2] or ""}
                for r in rows]

    def list_role_ids(self, memory_id: str) -> list[str]:
        """Return all character ids referencing *memory_id*."""
        rows = self._conn.execute(
            "SELECT role_id FROM role_memories "
            "WHERE memory_id = ? ORDER BY role_id ASC",
            (memory_id,),
        ).fetchall()
        return [row[0] for row in rows]

    def remove_all_memory(self, memory_id: str) -> int:
        """Remove *memory_id* references from every character.

        Returns the number of reference rows deleted.  Called when a global
        memory is deleted so no orphan references remain (Unit 2 hook).
        """
        cursor = self._conn.execute(
            "DELETE FROM role_memories WHERE memory_id = ?", (memory_id,)
        )
        self._conn.commit()
        return cursor.rowcount

    # ── Agent bindings ────────────────────────────────────────────────────

    def bind_agent(self, agent_name: str, role_id: str) -> bool:
        """Bind an agent to a character (``INSERT OR REPLACE``).

        Rebinding an already-bound agent switches it to the new character.
        Returns ``False`` when the character does not exist.
        """
        if not self._role_exists(role_id):
            return False
        self._conn.execute(
            "INSERT OR REPLACE INTO agent_bindings (agent_name, role_id, updated_at) "
            "VALUES (?, ?, ?)",
            (agent_name, role_id, utcnow().timestamp()),
        )
        self._conn.commit()
        return True

    def get_agent_role(self, agent_name: str) -> str | None:
        """Return the character id an agent is bound to, or ``None``."""
        row = self._conn.execute(
            "SELECT role_id FROM agent_bindings WHERE agent_name = ?",
            (agent_name,),
        ).fetchone()
        return row[0] if row is not None else None

    def unbind_agent(self, agent_name: str) -> bool:
        """Remove an agent's binding. Returns ``True`` if a binding existed."""
        cursor = self._conn.execute(
            "DELETE FROM agent_bindings WHERE agent_name = ?", (agent_name,)
        )
        self._conn.commit()
        return cursor.rowcount > 0

    # ── Diagnostics ───────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Return table presence and row counts for all three tables.

        Keys: ``status``, ``backend``, ``roles``, ``role_memories``,
        ``agent_bindings`` (counts, -1 on failure).
        """
        expected = ("roles", "role_memories", "agent_bindings")
        try:
            present = {
                row[0]
                for row in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' "
                    "AND name IN (?, ?, ?)",
                    expected,
                ).fetchall()
            }
            missing = [t for t in expected if t not in present]
            result: dict[str, Any] = {
                "status": "healthy" if not missing else f"degraded: missing {missing}",
                "backend": "sqlite3",
            }
            for table in expected:
                row = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
                result[table] = int(row[0]) if row is not None else -1
            return result
        except Exception as exc:
            return {
                "status": f"degraded: {exc}",
                "backend": "sqlite3",
                "roles": -1,
                "role_memories": -1,
                "agent_bindings": -1,
            }

    # ── helpers ───────────────────────────────────────────────────────────

    def _role_exists(self, role_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM roles WHERE id = ?", (role_id,)
        ).fetchone()
        return row is not None


def _row_to_role(row: tuple) -> dict[str, Any]:
    """Convert a roles query row to a dict."""
    (id_, name, description, personality, scenario, mes_example,
     creator_notes, first_mes, system_prompt, avatar, is_tavern, st_name,
     created_at, updated_at, ref_count) = row
    return {
        "id": id_,
        "name": name,
        "description": description,
        "personality": personality,
        "scenario": scenario,
        "mes_example": mes_example,
        "creator_notes": creator_notes,
        "first_mes": first_mes or "",
        "system_prompt": system_prompt or "",
        "avatar": avatar or "",
        "is_tavern": bool(is_tavern),
        "st_name": st_name or "",
        "created_at": created_at,
        "updated_at": updated_at,
        "ref_count": ref_count,
    }
