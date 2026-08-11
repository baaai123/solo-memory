"""Protocol Gate — enforces the active-learning protocol before weave().

Extracted from _compose.MemorySystem.weave() so the gate logic is
testable in isolation and the weave() method is pure context assembly.

Gate rules (checked in order):
  1. ClassificationRequired — the previous turn was not classified.
  2. GapRequired — a mission has unfulfilled skill gaps.
"""

from __future__ import annotations


class ProtocolGate:
    """Checks protocol compliance before allowing weave() to proceed."""

    def __init__(self, system):
        self._system = system

    def check(self, user_message: str) -> None:
        """Raise ClassificationRequired or GapRequired, or pass cleanly."""
        from memory_skill._compose import ClassificationRequired, GapRequired

        if self._system._classify_pending and user_message != self._system._classify_pending:
            raise ClassificationRequired(
                f"上一轮未分类 (pending: {self._system._classify_pending[:60]}). "
                "请先调用 memory_classify 分类上一轮的消息，"
                "chat/skill/mission/pref/pers 都可以，不能跳过。"
            )

        if self._system._pending_gaps:
            raise GapRequired(
                f"以下技能尚未补齐: {', '.join(sorted(self._system._pending_gaps)[:8])}. "
                "必须先 websearch → memory_teach_skill 补齐所有技能，"
                "再继续对话。"
            )

    def after_weave(self, user_message: str) -> None:
        """Record that this turn was woven (sets classify_pending for next round)."""
        self._system._classify_pending = user_message
