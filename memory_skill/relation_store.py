"""RelationStore — directed social-relation graph for Tavern Mode.

角色间有向关系：A 对 B 的看法 ≠ B 对 A 的看法，所以存成有向边
``(from_role_id → to_role_id, relation_type, strength)``，独立于状态快照。
strength 取 0-100 整数。与 StateStore 一样只用 stdlib ``sqlite3``。
"""

from __future__ import annotations

import sqlite3
from typing import Any

from memory_skill.contracts import utcnow

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS role_relations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    from_role_id  TEXT NOT NULL,
    to_role_id    TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    strength      INTEGER NOT NULL DEFAULT 50,
    updated_at    REAL NOT NULL,
    UNIQUE(from_role_id, to_role_id)
);
CREATE INDEX IF NOT EXISTS idx_relations_from ON role_relations(from_role_id);
CREATE INDEX IF NOT EXISTS idx_relations_to   ON role_relations(to_role_id);
"""

_MIN_STRENGTH = 0
_MAX_STRENGTH = 100


class RelationStore:
    """Directed relation graph between roles."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_CREATE_TABLES_SQL)

    def upsert_relation(self, from_role_id: str, to_role_id: str,
                        relation_type: str, strength: int,
                        timestamp: float | None = None) -> None:
        """Insert or replace the directed edge ``from → to``.

        Raises:
            ValueError: If *relation_type* is empty or *strength* out of 0-100.
        """
        if not relation_type.strip():
            raise ValueError("relation_type must not be empty")
        if not _MIN_STRENGTH <= strength <= _MAX_STRENGTH:
            raise ValueError(f"strength must be {_MIN_STRENGTH}-{_MAX_STRENGTH}")
        ts = timestamp if timestamp is not None else utcnow().timestamp()
        self._conn.execute(
            "INSERT OR REPLACE INTO role_relations "
            "(from_role_id, to_role_id, relation_type, strength, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (from_role_id, to_role_id, relation_type.strip(), strength, ts),
        )
        self._conn.commit()

    def get_relation(self, from_role_id: str, to_role_id: str) -> dict[str, Any] | None:
        """Return a single directed edge, or ``None`` if absent."""
        row = self._conn.execute(
            "SELECT from_role_id, to_role_id, relation_type, strength, updated_at "
            "FROM role_relations WHERE from_role_id = ? AND to_role_id = ?",
            (from_role_id, to_role_id),
        ).fetchone()
        if row is None:
            return None
        return {
            "from_role_id": row[0],
            "to_role_id": row[1],
            "relation_type": row[2],
            "strength": row[3],
            "updated_at": row[4],
        }

    def get_outgoing(self, from_role_id: str) -> list[dict[str, Any]]:
        """Return all outgoing edges of *from_role_id* (who they feel about)."""
        rows = self._conn.execute(
            "SELECT from_role_id, to_role_id, relation_type, strength, updated_at "
            "FROM role_relations WHERE from_role_id = ? ORDER BY strength DESC",
            (from_role_id,),
        ).fetchall()
        return [
            {"from_role_id": r[0], "to_role_id": r[1], "relation_type": r[2],
             "strength": r[3], "updated_at": r[4]}
            for r in rows
        ]

    def get_incoming(self, to_role_id: str) -> list[dict[str, Any]]:
        """Return all incoming edges of *to_role_id* (who feels about them)."""
        rows = self._conn.execute(
            "SELECT from_role_id, to_role_id, relation_type, strength, updated_at "
            "FROM role_relations WHERE to_role_id = ? ORDER BY strength DESC",
            (to_role_id,),
        ).fetchall()
        return [
            {"from_role_id": r[0], "to_role_id": r[1], "relation_type": r[2],
             "strength": r[3], "updated_at": r[4]}
            for r in rows
        ]

    def delete_relation(self, from_role_id: str, to_role_id: str) -> bool:
        """Delete a directed edge; returns ``False`` if it did not exist."""
        cur = self._conn.execute(
            "DELETE FROM role_relations WHERE from_role_id = ? AND to_role_id = ?",
            (from_role_id, to_role_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def health(self) -> dict[str, Any]:
        """Return a small health dict (edge count)."""
        count = self._conn.execute("SELECT COUNT(*) FROM role_relations").fetchone()[0]
        return {"relation_count": count}
