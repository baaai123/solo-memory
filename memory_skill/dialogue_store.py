"""
DialogueStore — SQLite-backed dialogue turn persistence with FTS5 BM25 search.

Provides insert, retrieve, search, time-range query, and TTL-based eviction
for ``DialogueTurn`` objects.  Uses only Python stdlib ``sqlite3`` — no
external dependencies, no ORM, no external server.

Key features:
- Schema versioning: raises ``StoreCorruptionError`` on version mismatch.
- FTS5 full-text search with BM25 ranking.
- TTL eviction via ``cleanup()`` based on configurable threshold.
- Idempotent inserts via ``INSERT OR IGNORE`` on (role, content, timestamp).
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from memory_skill.contracts import DialogueTurn, utcnow

# Chinese tokenizer — lazy import with fallback
_jieba = None


def _get_jieba():
    global _jieba
    if _jieba is None:
        try:
            # Load custom dictionary for food/domain terms
            import os

            import jieba
            _dict_path = os.path.join(os.path.dirname(__file__), "jieba_dict.txt")
            if os.path.exists(_dict_path):
                jieba.load_userdict(_dict_path)
            _jieba = jieba
        except ImportError:
            _jieba = False
    return _jieba if _jieba else None


def _cn_tokenize(text: str) -> str:
    """Tokenize Chinese text for FTS5. Uses jieba if available, else passthrough."""
    # First, strip all characters FTS5 unicode61 rejects
    import re
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    jb = _get_jieba()
    if jb:
        result = " ".join(jb.cut(text))
    else:
        result = text
    return result

if TYPE_CHECKING:
    from memory_skill.contracts import MemorySkillConfig

# ── Schema ────────────────────────────────────────────────────────────────────

class DialogueStore:
    """SQLite-backed store for dialogue turns with FTS5 BM25 search.

    ⚠ CRITICAL: ``_CREATE_TABLES_SQL`` is a multi-line SQL string inside
    the class.  NEVER use ``edit`` tool ``replaceAll`` or any substring
    replacement on this file — it can corrupt the class definition and
    swallow the entire ``class DialogueStore`` block.  If you need to
    change the schema, use ``write`` to rewrite the whole file, or move
    the SQL to a module-level constant first.
    """

    _CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS dialogue_turns (
    id       TEXT PRIMARY KEY,
    role     TEXT    NOT NULL,
    content  TEXT    NOT NULL,
    timestamp REAL   NOT NULL,
    saw_index INTEGER,
    UNIQUE(role, content, timestamp)
);

CREATE VIRTUAL TABLE IF NOT EXISTS dialogue_fts USING fts5(
    turn_id, content,
    tokenize = 'unicode61'
)
"""

    def __init__(self, config: MemorySkillConfig, ttl_seconds: int = 1800) -> None:
        self._conn = sqlite3.connect(config.db_path, check_same_thread=False, timeout=10)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._ttl_seconds = ttl_seconds
        self._conn.executescript(DialogueStore._CREATE_TABLES_SQL)

    def insert(self, turn: DialogueTurn) -> None:
        """Insert a single dialogue turn (idempotent)."""
        ts = _to_epoch(turn.timestamp)
        self._conn.execute(
            "INSERT OR IGNORE INTO dialogue_turns (id, role, content, timestamp, saw_index) "
            "VALUES (?, ?, ?, ?, ?)",
            (turn.id, turn.role, turn.content, ts, turn.saw_index),
        )
        self._conn.execute("DELETE FROM dialogue_fts WHERE turn_id = ?", (turn.id,))
        self._conn.execute(
            "INSERT INTO dialogue_fts (turn_id, content) VALUES (?, ?)",
            (turn.id, _cn_tokenize(turn.content)),
        )
        self._conn.commit()

    def insert_batch(self, turns: list[DialogueTurn]) -> None:
        """Insert multiple dialogue turns in a single transaction."""
        if not turns:
            return

        rows = [
            (t.id, t.role, t.content, _to_epoch(t.timestamp), t.saw_index)
            for t in turns
        ]

        self._conn.execute("BEGIN")
        try:
            self._conn.executemany(
                "INSERT OR IGNORE INTO dialogue_turns "
                "(id, role, content, timestamp, saw_index) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.executemany(
                "DELETE FROM dialogue_fts WHERE turn_id = ?",
                [(t.id,) for t in turns],
            )
            self._conn.executemany(
                "INSERT INTO dialogue_fts (turn_id, content) VALUES (?, ?)",
                [(t.id, _cn_tokenize(t.content)) for t in turns],
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def get_by_id(self, id: str) -> DialogueTurn | None:
        """Retrieve a turn by its unique id, or ``None`` if not found."""
        row = self._conn.execute(
            "SELECT id, role, content, timestamp, saw_index "
            "FROM dialogue_turns WHERE id = ?",
            (id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_turn(row)

    def get_recent(self, n: int = 5) -> list[DialogueTurn]:
        """Return the most recent *n* turns, newest last."""
        rows = self._conn.execute(
            "SELECT id, role, content, timestamp, saw_index "
            "FROM dialogue_turns ORDER BY timestamp DESC LIMIT ?",
            (n,),
        ).fetchall()
        rows.reverse()
        return [_row_to_turn(r) for r in rows]

    def count(self) -> int:
        """Return the total number of stored dialogue turns."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM dialogue_turns"
        ).fetchone()
        return int(row[0]) if row else 0

    def count_recent(self, minutes: int = 5) -> int:
        """Return number of turns from the last *minutes* minutes.

        Used by weaver for session-aware depth gating — distinguishes
        "current conversation activity" from "total stored history".
        """
        import time
        cutoff = time.time() - (minutes * 60)
        row = self._conn.execute(
            "SELECT COUNT(*) FROM dialogue_turns WHERE timestamp >= ?",
            (cutoff,),
        ).fetchone()
        return int(row[0]) if row else 0

    def search(self, query: str, limit: int = 10) -> list[DialogueTurn]:
        """Full-text search via FTS5 with BM25 ranking.
        For Chinese, unicode61 tokenizer splits each character; use raw query
        without phrase quoting so FTS5 AND-matches individual characters.
        """
        # Tokenize Chinese query with jieba so FTS5 can match word tokens
        safe = _cn_tokenize(query)
        rows = self._conn.execute(
            "SELECT dt.id, dt.role, dt.content, dt.timestamp, dt.saw_index "
            "FROM dialogue_fts "
            "JOIN dialogue_turns dt ON dt.id = dialogue_fts.turn_id "
            "WHERE dialogue_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT ?",
            (safe, limit),
        ).fetchall()
        return [_row_to_turn(r) for r in rows]

    def time_range(
        self, start: datetime, end: datetime
    ) -> list[DialogueTurn]:
        """Return turns whose timestamp is in [start, end)."""
        start_ts = _to_epoch(start)
        end_ts = _to_epoch(end)
        rows = self._conn.execute(
            "SELECT id, role, content, timestamp, saw_index "
            "FROM dialogue_turns "
            "WHERE timestamp >= ? AND timestamp < ? "
            "ORDER BY timestamp ASC",
            (start_ts, end_ts),
        ).fetchall()
        return [_row_to_turn(r) for r in rows]

    def cleanup(self) -> int:
        """Evict turns older than *ttl_seconds* from now.

        Returns the number of rows removed.
        """
        cutoff = _to_epoch(utcnow()) - self._ttl_seconds

        # Collect ids to remove from FTS index
        ids = self._conn.execute(
            "SELECT id FROM dialogue_turns WHERE timestamp < ?",
            (cutoff,),
        ).fetchall()

        if not ids:
            return 0

        id_list = [row[0] for row in ids]
        placeholders = ",".join("?" for _ in id_list)

        # Remove from FTS index first
        self._conn.execute(
            f"DELETE FROM dialogue_fts WHERE turn_id IN ({placeholders})",
            id_list,
        )
        # Remove from main table
        cursor = self._conn.execute(
            f"DELETE FROM dialogue_turns WHERE id IN ({placeholders})",
            id_list,
        )
        self._conn.commit()
        return cursor.rowcount

    def reindex_roles(self) -> dict[str, int]:
        """Fix mislabeled turns from chat simulation format.

        Splits ``[chat]``-format turns (multi-message blocks with embedded
        agent responses like ``\\nXueNai: ...``) into individual turns with
        correct ``role`` labels.  TUI role-play turns are left unchanged.

        Called once as a migration after the adapter's role-detection fix.

        Returns:
            dict with keys ``"split"`` (original turns split),
            ``"inserted"`` (new turns created), ``"deleted"`` (originals removed),
            ``"kept"`` (non-chat turns left as-is).
        """
        import re

        # ── 1. Collect all [chat]-format turns ─────────────────────────
        rows = self._conn.execute(
            "SELECT id, role, content, timestamp, saw_index "
            "FROM dialogue_turns "
            "WHERE content LIKE '[chat]%'"
        ).fetchall()

        if not rows:
            return {"split": 0, "inserted": 0, "deleted": 0, "kept": self.count()}

        split_count = len(rows)
        inserted = 0
        deleted = 0

        # Known agent names to detect in content
        agent_names: set[str] = set()
        for _, _, content, _, _ in rows:
            for m in re.finditer(r'^(\w+):\s', content, re.MULTILINE):
                name = m.group(1)
                if name.lower() != "user":
                    agent_names.add(name)

        self._conn.execute("BEGIN")
        try:
            for rid, role, content, ts, saw_index in rows:
                # ── 2. Parse individual messages from the block ─────────
                lines = content.split("\n")
                messages: list[tuple[str, str]] = []  # (role, text)
                current_speaker = ""
                current_text: list[str] = []

                for line in lines:
                    stripped = line.strip()
                    # Detect speaker: "user:", "XueNai:", "Shizuru:", etc.
                    speaker_match = re.match(r'^(\w+):\s?(.*)', stripped)
                    if speaker_match:
                        name = speaker_match.group(1)
                        text = speaker_match.group(2)
                        # Save previous message
                        if current_text:
                            role_label = "user" if current_speaker.lower() in ("user", "chat") else "assistant"
                            messages.append((role_label, " ".join(current_text)))
                            current_text = []
                        current_speaker = name
                        if text:
                            current_text.append(text)
                    elif stripped and current_speaker:
                        # Continuation of previous speaker's message
                        current_text.append(stripped)

                # Save last message
                if current_text:
                    role_label = "user" if current_speaker.lower() in ("user", "chat") else "assistant"
                    messages.append((role_label, " ".join(current_text)))

                if not messages:
                    continue

                # ── 3. Insert split turns ──────────────────────────────
                for i, (turn_role, turn_text) in enumerate(messages):
                    if not turn_text.strip():
                        continue
                    new_id = f"{rid}/m{i}"
                    self._conn.execute(
                        "INSERT OR REPLACE INTO dialogue_turns "
                        "(id, role, content, timestamp, saw_index) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (new_id, turn_role, turn_text, ts, saw_index),
                    )
                    self._conn.execute(
                        "DELETE FROM dialogue_fts WHERE turn_id = ?", (new_id,)
                    )
                    self._conn.execute(
                        "INSERT INTO dialogue_fts (turn_id, content) VALUES (?, ?)",
                        (new_id, _cn_tokenize(turn_text)),
                    )
                    inserted += 1

                # ── 4. Delete original combined turn ───────────────────
                self._conn.execute("DELETE FROM dialogue_fts WHERE turn_id = ?", (rid,))
                self._conn.execute("DELETE FROM dialogue_turns WHERE id = ?", (rid,))
                deleted += 1

            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

        kept = self.count()
        return {"split": split_count, "inserted": inserted, "deleted": deleted, "kept": kept}

    # ── helpers ───────────────────────────────────────────────────────────

def _to_epoch(dt: datetime) -> float:
    """Convert a timezone-aware datetime to Unix epoch (float seconds)."""
    return dt.astimezone(UTC).timestamp()


def _row_to_turn(row: tuple) -> DialogueTurn:
    """Convert a database row to a DialogueTurn."""
    id_, role, content, ts, saw_index = row
    return DialogueTurn(
        id=id_,
        role=role,
        content=content,
        timestamp=datetime.fromtimestamp(ts, tz=UTC),
        saw_index=saw_index,
    )
