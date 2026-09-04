"""StateStore — per-role state snapshot for Tavern Mode.

覆盖式状态存储：每个角色一行快照，8 个维度（mood/need/health/clothing/
item/action/scene/weather），更新即覆盖，只保留最新值。区别于 learned_store
的累积式语义记忆——故事状态回答「现在是什么」，而非「历史有什么」。

Uses stdlib ``sqlite3`` only, mirroring ``character_store.py``.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from memory_skill.contracts import utcnow

_STATE_DIMENSIONS = (
    "mood", "need", "health", "clothing", "item", "action", "scene", "weather",
)

_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS state_store (
    role_id     TEXT PRIMARY KEY,
    mood        TEXT,
    need        TEXT,
    health      TEXT,
    clothing    TEXT,
    item        TEXT,
    action      TEXT,
    scene       TEXT,
    weather     TEXT,
    updated_at  REAL NOT NULL
);
"""


class StateStore:
    """Covering-write store for per-role story state snapshots."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_CREATE_TABLES_SQL)

    def get_state(self, role_id: str) -> dict[str, Any] | None:
        """Return the latest snapshot for *role_id*, or ``None`` if absent."""
        row = self._conn.execute(
            "SELECT mood, need, health, clothing, item, action, scene, weather, updated_at "
            "FROM state_store WHERE role_id = ?",
            (role_id,),
        ).fetchone()
        if row is None:
            return None
        keys = _STATE_DIMENSIONS + ("updated_at",)
        return dict(zip(keys, row))

    def update_state(self, role_id: str, dims: dict[str, str],
                     timestamp: float | None = None) -> None:
        """Covering-update one or more dimensions.

        ``dims`` values that are ``None`` or not in ``_STATE_DIMENSIONS`` are
        ignored, so omitted dimensions keep their previous value.
        """
        clean = {k: v for k, v in dims.items() if k in _STATE_DIMENSIONS and v is not None}
        if not clean:
            return
        ts = timestamp if timestamp is not None else utcnow().timestamp()
        existing = self.get_state(role_id) or {}
        merged = {**existing, **clean, "updated_at": ts}
        cols = _STATE_DIMENSIONS + ("updated_at",)
        placeholders = ", ".join("?" for _ in cols)
        self._conn.execute(
            f"INSERT OR REPLACE INTO state_store (role_id, {', '.join(cols)}) "
            f"VALUES (?, {placeholders})",
            (role_id, *(merged.get(c) for c in cols)),
        )
        self._conn.commit()

    def delete_state(self, role_id: str) -> bool:
        """Delete a role's snapshot; returns ``False`` if it did not exist."""
        cur = self._conn.execute("DELETE FROM state_store WHERE role_id = ?", (role_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def health(self) -> dict[str, Any]:
        """Return a small health dict (snapshot count)."""
        count = self._conn.execute("SELECT COUNT(*) FROM state_store").fetchone()[0]
        return {"state_count": count}
