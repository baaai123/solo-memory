"""MissionStore — structured mission process records for Memory Skill.

A mission is no longer just a bare line in the learning queue; it becomes a
structured object with:

  - a ChromaDB entry (category="mission") holding the mission content
  - a steps list stored in entry metadata under ``ui_steps``
    (JSON array of ``{text, skill_id, skill_title, done}``)
  - a status stored in entry metadata under ``ui_status`` ("open" | "done")

All writes go through ``LearnedStore.update()`` so the entry is re-embedded
atomically and metadata is merged — the module's own behaviour is untouched.

The mission also mirrors its lifecycle into the learning queue so the
existing ``[待拆解]`` weaver directives keep working: creating a mission
enqueues an open mission item; marking it done closes the item.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("memory_skill.mission")

# metadata keys (namespaced with ui_ to avoid clashing with module-owned keys)
META_STEPS = "ui_steps"
META_STATUS = "ui_status"

VALID_STATUS = ("open", "done")


class MissionError(Exception):
    """Raised for invalid mission operations."""


@dataclass
class MissionStep:
    """One step of a mission."""
    text: str
    skill_id: str = ""
    skill_title: str = ""
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "skill_id": self.skill_id,
            "skill_title": self.skill_title,
            "done": self.done,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MissionStep":
        return cls(
            text=str(d.get("text", "")),
            skill_id=str(d.get("skill_id", "") or ""),
            skill_title=str(d.get("skill_title", "") or ""),
            done=bool(d.get("done", False)),
        )


@dataclass
class Mission:
    """A structured mission record."""
    id: str
    content: str
    status: str = "open"
    steps: list[MissionStep] = field(default_factory=list)
    weight: float = 0.5
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "status": self.status,
            "steps": [s.to_dict() for s in self.steps],
            "weight": self.weight,
            "created_at": _ts(self.created_at),
            "updated_at": _ts(self.updated_at),
        }


def _ts(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _now() -> datetime:
    return datetime.now(UTC)


class MissionStore:
    """Structured mission records backed by LearnedStore + LearningQueue.

    ``learned_store`` must expose ``insert`` / ``update`` / ``get_entry`` /
    ``list_by_category``; ``learning_queue`` must expose ``enqueue`` /
    ``mark`` / ``all`` (may be None when tree is disabled — mission still
    works, just without queue mirroring).
    """

    def __init__(self, learned_store, learning_queue) -> None:
        self._learned = learned_store
        self._queue = learning_queue

    # ── Public API ──────────────────────────────────────────────────────

    def create(self, content: str, title: str = "") -> Mission:
        """Create a new open mission and mirror it into the learning queue.

        The mission is stored as a category="mission" ChromaDB entry whose
        content is the mission description; steps/status live in metadata.
        """
        content = (content or "").strip()
        if not content:
            raise MissionError("mission content is required")

        from memory_skill.contracts import MemoryEntry

        mission_id = self._new_mission_id()
        now = _now()
        entry = MemoryEntry(
            id=mission_id,
            content=content,
            created_at=now,
            updated_at=now,
            weight=0.5,
            category="mission",
            tags=[],
            metadata={
                "category": "mission",
                "title": (title or "").strip(),
                META_STEPS: json.dumps([], ensure_ascii=False),
                META_STATUS: "open",
            },
        )
        self._learned.insert(entry)

        if self._queue is not None:
            try:
                self._queue.enqueue("mission", content, detail=f"mission_id={mission_id}")
            except Exception as exc:
                logger.warning("Mission %s queue mirror failed: %s", mission_id, exc)

        return Mission(id=mission_id, content=content, status="open",
                       steps=[], created_at=now, updated_at=now)

    def get(self, mission_id: str) -> Mission | None:
        """Return the full mission (steps + status), or None when missing."""
        entry = self._learned.get_entry(mission_id)
        if entry is None:
            return None
        return self._entry_to_mission(entry)

    def add_step(self, mission_id: str, text: str,
                 skill_id: str = "") -> Mission:
        """Append a step to a mission. Returns the updated mission."""
        text = (text or "").strip()
        if not text:
            raise MissionError("step text is required")

        entry = self._require_entry(mission_id)
        steps = self._read_steps(entry.metadata)
        skill_title = self._resolve_skill_title(skill_id)
        steps.append(MissionStep(text=text, skill_id=skill_id,
                                 skill_title=skill_title))
        self._write_mission(entry, steps=steps,
                            status=self._read_status(entry.metadata))
        return self._reload_mission(mission_id)

    def update_step(self, mission_id: str, index: int, *,
                    text: str | None = None,
                    skill_id: str | None = None,
                    done: bool | None = None) -> Mission:
        """Modify one step by index. ``None`` fields are left unchanged."""
        entry = self._require_entry(mission_id)
        steps = self._read_steps(entry.metadata)
        if index < 0 or index >= len(steps):
            raise MissionError(f"step index {index} out of range (0..{len(steps) - 1})")

        step = steps[index]
        if text is not None:
            step.text = str(text).strip() or step.text
        if skill_id is not None:
            step.skill_id = skill_id
            step.skill_title = self._resolve_skill_title(skill_id)
        if done is not None:
            step.done = bool(done)

        self._write_mission(entry, steps=steps,
                            status=self._read_status(entry.metadata))
        return self._reload_mission(mission_id)

    def remove_step(self, mission_id: str, index: int) -> Mission:
        """Delete a step by index. Returns the updated mission."""
        entry = self._require_entry(mission_id)
        steps = self._read_steps(entry.metadata)
        if index < 0 or index >= len(steps):
            raise MissionError(f"step index {index} out of range (0..{len(steps) - 1})")
        steps.pop(index)
        self._write_mission(entry, steps=steps,
                            status=self._read_status(entry.metadata))
        return self._reload_mission(mission_id)

    def set_status(self, mission_id: str, status: str) -> Mission:
        """Set mission status to "open" or "done", mirroring the queue."""
        if status not in VALID_STATUS:
            raise MissionError(f"status must be one of {VALID_STATUS}, got {status!r}")
        entry = self._require_entry(mission_id)
        self._write_mission(entry,
                            steps=self._read_steps(entry.metadata),
                            status=status)
        self._mirror_queue_status(mission_id, status)
        return self._reload_mission(mission_id)

    def list_missions(self, status: str | None = None,
             limit: int = 100) -> list[Mission]:
        """List missions, optionally filtered by status, newest first."""
        try:
            ids = [e.id for e in self._learned.list_by_category("mission", limit=0)]
        except Exception as exc:
            logger.warning("MissionStore.list failed: %s", exc)
            return []

        missions: list[Mission] = []
        for mission_id in ids:
            entry = self._learned.get_entry(mission_id)
            if entry is None:
                continue
            mission = self._entry_to_mission(entry)
            if status is None or mission.status == status:
                missions.append(mission)
        missions.sort(key=lambda m: (m.updated_at or _now()).timestamp(),
                      reverse=True)
        return missions[:limit] if limit > 0 else missions

    # ── Internals ───────────────────────────────────────────────────────

    def _new_mission_id(self) -> str:
        ts = _now().strftime("%Y%m%d_%H%M%S")
        return f"dialogue:mission_{ts}_{uuid.uuid4().hex[:6]}"

    def _require_entry(self, mission_id: str):
        entry = self._learned.get_entry(mission_id)
        if entry is None:
            raise MissionError(f"mission {mission_id!r} not found")
        return entry

    def _reload_mission(self, mission_id: str) -> Mission:
        entry = self._learned.get_entry(mission_id)
        if entry is None:
            raise MissionError(f"mission {mission_id} disappeared during update")
        return self._entry_to_mission(entry)

    @staticmethod
    def _read_steps(metadata: dict) -> list[MissionStep]:
        raw = metadata.get(META_STEPS, "[]")
        try:
            data = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            data = []
        return [MissionStep.from_dict(d) for d in (data or []) if isinstance(d, dict)]

    @staticmethod
    def _read_status(metadata: dict) -> str:
        status = metadata.get(META_STATUS, "open")
        return status if status in VALID_STATUS else "open"

    def _resolve_skill_title(self, skill_id: str) -> str:
        """Look up a skill entry's title by id. Empty string when unknown."""
        if not skill_id:
            return ""
        try:
            entry = self._learned.get_entry(skill_id)
            if entry is None:
                return ""
            title = entry.metadata.get("title", "")
            if title:
                return title
            return entry.content[:60]
        except Exception:
            return ""

    def _write_mission(self, entry, *,
                       steps: list[MissionStep], status: str) -> None:
        """Persist steps + status via LearnedStore.update (atomic upsert)."""
        new_meta = {
            **entry.metadata,
            META_STEPS: json.dumps([s.to_dict() for s in steps],
                                   ensure_ascii=False),
            META_STATUS: status,
        }
        self._learned.update(entry.id, entry.content, metadata=new_meta)

    def _mirror_queue_status(self, mission_id: str, status: str) -> None:
        """Close/open the mirrored learning-queue item."""
        if self._queue is None:
            return
        try:
            for item in self._queue.all(limit=500):
                if f"mission_id={mission_id}" in getattr(item, "detail", ""):
                    if status == "done":
                        self._queue.mark(item.id, "done")
                    break
        except Exception as exc:
            logger.warning("Queue status mirror failed for %s: %s",
                           mission_id, exc)

    def _entry_to_mission(self, entry) -> Mission:
        return Mission(
            id=entry.id,
            content=entry.content,
            status=self._read_status(entry.metadata),
            steps=self._read_steps(entry.metadata),
            weight=entry.weight,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
        )
