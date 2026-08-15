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
  mark_weave(user_message)  -> classify_pending=user_message (next turn must classify)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ProtocolState:
    """Protocol state owned by a single module, mutated only through its API."""

    classify_pending: str | None = None
    pending_gaps: set = field(default_factory=set)
    mission_pending_check: bool = False

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

    def open_gaps(self) -> set:
        """Snapshot of the still-unfilled skill gaps."""
        return set(self.pending_gaps)
