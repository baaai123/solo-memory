"""ProtocolState — owns the active-learning protocol state.

The three fields that used to live on ``MemorySystem`` as magic attributes
(``_classify_pending`` / ``_pending_gaps`` / ``_mission_pending_check``) now
live here, with the write operations that mutate them. ProtocolGate reads
these through the state; tools/skill_writer mutate them through the API
methods instead of poking private fields. MemorySystem keeps a single
``protocol`` reference.

The state machine (see ProtocolGate for enforcement):
  classify(mission)         -> mission_pending_check=True, pending_gaps={gaps}
  classify(other)           -> mission_pending_check=False, pending_gaps={}
  check_skill() / search()  -> mission_pending_check=False
  mark_weave(user_message)  -> classify_pending=user_message (next turn must
                               classify), weave_count += 1
  MemorySystem.weave()      -> after weaving, ``_maybe_arm_todo_gates`` arms
                               archive_pending / queue_pending from store
                               counts (Unit 2, scheme A: the trigger logic
                               lives on MemorySystem because ProtocolState
                               cannot see the stores)
  review_default/reclassify -> clear_archive_pending()
  learning_mark             -> clear_queue_pending()

Unit 2 fields (archive + queue hard gates):
  weave_count    — monotonic counter of successful weaves. The archive
                   gate arms when weave_count % archive_interval == 0 and
                   default-category entries remain; the modulo check gives
                   an ``interval``-long grace period after each response.
  archive_pending — armed gate: next weave() raises ArchiveRequired.
  queue_pending   — armed gate: next weave() raises QueueRequired.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProtocolState:
    """Protocol state owned by a single module, mutated only through its API."""

    classify_pending: str | None = None
    pending_gaps: set = field(default_factory=set)
    mission_pending_check: bool = False

    # ── Todo hard gates (Unit 2) ─────────────────────────────────────────
    weave_count: int = 0
    archive_pending: bool = False
    queue_pending: bool = False

    def mark_classified(self, category: str, gaps=None) -> None:
        """Record a classification. A mission arms the skill-check gate."""
        self.classify_pending = None
        if category == "mission":
            self.mission_pending_check = True
            self.pending_gaps = {str(g).strip() for g in (gaps or []) if str(g).strip()}
        else:
            self.mission_pending_check = False
            self.pending_gaps.clear()

    def mark_skill_checked(self) -> None:
        """A mission has consulted existing skills (check_skill or search)."""
        self.mission_pending_check = False

    def mark_gap_filled(self, skill_name: str) -> None:
        """teach_skill succeeded: drop the skill from pending gaps."""
        if skill_name in self.pending_gaps:
            self.pending_gaps.discard(skill_name)

    def mark_weave(self, user_message: str) -> None:
        """Record that this turn was woven (next turn must classify it)."""
        self.classify_pending = user_message
        self.weave_count += 1

    def open_gaps(self) -> set:
        """Snapshot of the still-unfilled skill gaps."""
        return set(self.pending_gaps)

    # ── Todo hard gates (Unit 2) ─────────────────────────────────────────

    def arm_archive(self) -> None:
        """Arm the archive gate (MemorySystem._maybe_arm_todo_gates)."""
        self.archive_pending = True

    def arm_queue(self) -> None:
        """Arm the queue gate (MemorySystem._maybe_arm_todo_gates)."""
        self.queue_pending = True

    def clear_archive_pending(self) -> None:
        """An archive tool responded (review_default/reclassify) — disarm.

        Anti-deadlock rule (R8): any archive-tool call proves the agent
        responded, so the gate clears even when zero entries moved.
        """
        self.archive_pending = False

    def clear_queue_pending(self) -> None:
        """A queue tool responded (learning_mark) — disarm."""
        self.queue_pending = False
