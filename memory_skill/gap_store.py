"""Persistent knowledge-gap store.

Gaps detected by :class:`~memory_skill.gap_detector.GapDetector` are kept
in memory during a session; this store persists them to SQLite so gaps
survive process restarts and can be reviewed or learned later.
"""

from __future__ import annotations

import logging
import sqlite3
import uuid
from datetime import UTC, datetime

from memory_skill.gap_detector import Gap

logger = logging.getLogger("memory_skill.gap_store")


_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_gaps (
    id          TEXT PRIMARY KEY,
    query       TEXT NOT NULL,
    branch      TEXT NOT NULL,
    severity    TEXT NOT NULL,
    confidence  REAL NOT NULL,
    detected_at REAL NOT NULL,
    status      TEXT NOT NULL DEFAULT 'open',
    decision    TEXT
);
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO knowledge_gaps
    (id, query, branch, severity, confidence, detected_at, status, decision)
VALUES (?, ?, ?, ?, ?, ?, 'open', ?)
"""


def _make_decision(action: str | None):
    if not action:
        return None
    from memory_skill.learning_decider import Decision
    return Decision(action=action, reasoning="", confidence=0.0)


class GapStore:
    """SQLite-backed persistence for knowledge gaps."""

    def __init__(self, db_path: str) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.executescript(_CREATE_SQL)

    def save(self, gap: Gap) -> None:
        decision_action = gap.decision.action if gap.decision is not None else None
        try:
            self._conn.execute(
                _INSERT_SQL,
                (
                    f"gap_{gap.detected_at.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
                    gap.query,
                    gap.branch,
                    gap.severity,
                    float(gap.confidence),
                    gap.detected_at.timestamp(),
                    decision_action,
                ),
            )
            self._conn.commit()
        except Exception as exc:
            logger.warning("Failed to persist gap %r: %s", gap.query[:40], exc)

    def load(self, limit: int = 100) -> list[Gap]:
        try:
            rows = self._conn.execute(
                "SELECT query, branch, severity, confidence, detected_at, decision "
                "FROM knowledge_gaps ORDER BY detected_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except Exception as exc:
            logger.warning("Failed to load gaps: %s", exc)
            return []
        return [
            Gap(
                query=row[0],
                branch=row[1],
                severity=row[2],
                confidence=row[3],
                detected_at=datetime.fromtimestamp(row[4], tz=UTC),
                decision=_make_decision(row[5]),
            )
            for row in rows
        ]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
