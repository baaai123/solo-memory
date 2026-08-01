"""Gap Detector — identifies knowledge gaps from user queries.

When the system encounters a question it cannot answer well (low recall
confidence), a ``Gap`` is recorded.  Gaps are passive — they don't trigger
crawling or alerts by themselves.  They are stored as signals for the
future learning-task scheduler.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from memory_skill.capability_registry import CapabilityRegistry

logger = logging.getLogger("memory_skill.gap")


_GAP_SEVERITY_THRESHOLDS = [
    ("critical", 0.0),
    ("major",    0.3),
    ("minor",    0.5),
]


@dataclass(frozen=True)
class Gap:
    """A detected knowledge gap."""
    query: str
    branch: str
    confidence: float           # 0.0 = completely unknown
    severity: str               # "critical" | "major" | "minor"
    detected_at: datetime = field(default_factory=lambda: datetime.now(UTC))


def _is_question(content: str) -> bool:
    """Heuristic: does *content* look like a question?
    """
    stripped = content.strip()
    return (
        "?" in stripped
        or stripped.endswith("吗")
        or stripped.endswith("呢")
        or any(kw in stripped for kw in ("什么", "怎么", "如何", "哪", "谁", "多少", "在哪"))
    )


class GapDetector:
    """Detects knowledge gaps in user queries.

    Parameters
    ----------
    registry: CapabilityRegistry
        Used to check whether the system can answer a query.
    min_confidence: float
        Below this confidence threshold, a gap is flagged.
    """

    def __init__(
        self,
        registry: CapabilityRegistry,
        min_confidence: float = 0.5,
        db_path: str | None = None,
    ):
        self._registry = registry
        self._min = min_confidence
        self._gaps: list[Gap] = []
        self._store = None
        if db_path:
            try:
                from memory_skill.gap_store import GapStore
                self._store = GapStore(db_path)
                self._gaps = self._store.load()
            except Exception as exc:
                logger.warning("Gap persistence unavailable: %s", exc)

    @property
    def gaps(self) -> list[Gap]:
        return list(self._gaps)

    def detect(self, query: str) -> Gap | None:
        """Check a single query for a knowledge gap.

        Returns ``None`` when the system is confident it can answer.
        """
        if not query or not query.strip():
            return None

        can, confidence = self._registry.can_answer(query)

        if can and confidence >= self._min:
            return None

        severity = self._classify_severity(confidence)

        # Determine branch
        try:
            tree_info = self._registry.classify_query(query)
            branch = tree_info.get("branch", "mem")
        except Exception:
            branch = "mem"

        gap = Gap(
            query=query,
            branch=branch,
            confidence=confidence,
            severity=severity,
        )
        self._gaps.append(gap)
        if self._store is not None:
            self._store.save(gap)
        logger.info("Gap detected [%s]: %s (conf=%.2f)", severity, query[:60], confidence)
        return gap

    def detect_many(self, queries: list[str]) -> list[Gap]:
        """Check multiple queries; returns all detected gaps."""
        gaps: list[Gap] = []
        for q in queries:
            g = self.detect(q)
            if g is not None:
                gaps.append(g)
        return gaps

    def set_decider(self, decider) -> None:
        """Attach a LearningDecider for automatic skip/ask/learn evaluation."""
        self._decider = decider

    def detect_and_decide(self, query: str):
        """Detect gap + let the agent decide whether to learn.

        Returns ``(Gap | None, Decision | None)``.
        """
        gap = self.detect(query)
        if gap is None:
            return None, None

        decider = getattr(self, "_decider", None)
        if decider is None:
            return gap, None

        decision = decider.evaluate(
            query=gap.query,
            branch=gap.branch,
            severity=gap.severity,
            history_count=len(self._gaps),
        )
        return gap, decision

    def clear(self) -> None:
        self._gaps.clear()

    @staticmethod
    def _classify_severity(confidence: float) -> str:
        for label, threshold in reversed(_GAP_SEVERITY_THRESHOLDS):
            if confidence >= threshold:
                return label
        return "critical"
