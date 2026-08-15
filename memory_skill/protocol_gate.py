"""Protocol Gate — enforces the active-learning protocol before weave().

Extracted from _compose.MemorySystem.weave() so the gate logic is
testable in isolation and the weave() method is pure context assembly.

The gate reads and writes protocol state through a single
``ProtocolState`` owner (``system.protocol``), never through magic
attributes on MemorySystem. Tools and the skill writer mutate the same
state via the API methods on ``ProtocolState``.

Gate rules (checked in order):
  1. ClassificationRequired — the previous turn was not classified.
  2. SkillCheckRequired — a classified mission has not checked existing skills.
  3. GapRequired — a mission has unfulfilled skill gaps.
"""

from __future__ import annotations


class ProtocolGate:
    """Checks protocol compliance before allowing weave() to proceed."""

    def __init__(self, system):
        self._state = system.protocol

    def check(self, user_message: str) -> None:
        """Raise ClassificationRequired, SkillCheckRequired, or GapRequired."""
        from memory_skill._compose import (
            ClassificationRequired,
            GapRequired,
            SkillCheckRequired,
        )

        if self._state.classify_pending and user_message != self._state.classify_pending:
            raise ClassificationRequired(
                f"上一轮未分类 (pending: {self._state.classify_pending[:60]}). "
                "请先调用 memory_classify 分类上一轮的消息，"
                "chat/skill/mission/pref/pers 都可以，不能跳过。"
            )

        if self._state.mission_pending_check:
            raise SkillCheckRequired(
                "已分类为 mission，但尚未检查已有技能。"
                "必须先调用 memory_check_skill 或 memory_search 确认是否已有相关技能，"
                "缺技能时先 websearch → memory_teach_skill 补齐，再开始实现。"
            )

        if self._state.pending_gaps:
            raise GapRequired(
                f"以下技能尚未补齐: {', '.join(sorted(self._state.pending_gaps)[:8])}. "
                "必须先 websearch → memory_teach_skill 补齐所有技能，"
                "再继续对话。"
            )

    def after_weave(self, user_message: str) -> None:
        """Record that this turn was woven (sets classify_pending for next round)."""
        self._state.mark_weave(user_message)
