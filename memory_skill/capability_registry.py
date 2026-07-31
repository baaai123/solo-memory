"""Capability Registry — maps tree branches to answerable domains.

Each tree branch is treated as a *capability domain*.  The registry
answers one question: "can the system answer this query?"

Uses the existing retriever + tree to estimate answerability without
calling an LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger("memory_skill.capability")


@dataclass(frozen=True)
class Capability:
    """A capability domain — one tree branch."""
    branch_id: str
    label: str
    root: str
    entry_count: int
    date_count: int
    avg_weight: float
    last_updated: datetime | None


class CapabilityRegistry:
    """Answers whether the system can handle a given query.

    Parameters
    ----------
    tree: TreeManager
        The memory tree (branch labels, counts).
    retriever: Retriever
        The hybrid retriever for scoring how well we know a topic.
    """

    def __init__(self, tree, retriever):
        self._tree = tree
        self._retriever = retriever

    def list_capabilities(self) -> list[Capability]:
        """Return all capability domains with health stats."""
        from memory_skill.tree import _BASE_BRANCHES, _BRANCH_LABEL_MAP

        caps: list[Capability] = []
        branch_ids = [b["id"] for b in _BASE_BRANCHES]

        for bid in branch_ids:
            label = _BRANCH_LABEL_MAP.get(bid, bid)
            root = bid.split("_")[0]
            count, date_count = self._branch_counts(bid)
            avg_w = self._branch_avg_weight(bid)
            last_up = self._branch_last_updated(bid)

            caps.append(Capability(
                branch_id=bid,
                label=label,
                root=root,
                entry_count=count,
                date_count=date_count,
                avg_weight=avg_w,
                last_updated=last_up,
            ))

        return caps

    def classify_query(self, query: str) -> dict[str, str]:
        """Return ``{"root": ..., "branch": ...}`` for a query using TreeManager."""
        try:
            return self._tree.classify(query)
        except Exception:
            return {"root": "user", "branch": "mem"}

    def can_answer(self, query: str) -> tuple[bool, float]:
        """Return ``(can_answer, confidence)`` for a query.

        Confidence ranges from 0 (no knowledge) to 1 (highly confident).
        """
        if not query or not query.strip():
            return False, 0.0

        # 1. Determine which branch the query likely belongs to
        try:
            tree_info = self._tree.classify(query)
            branch = tree_info.get("branch", "mem")
        except Exception:
            branch = "mem"

        # 2. Search for related entries
        results = self._retriever.retrieve(query, limit=5)
        if hasattr(results, "entries"):
            entries = results.entries

        if not entries:
            return False, 0.0

        # 3. Score from result count, weights, and freshness
        weights = [getattr(e, "weight", 0.5) for e in entries]
        avg_weight = sum(weights) / len(weights) if weights else 0.5
        count_bonus = min(len(entries) / 5.0, 1.0)

        confidence = avg_weight * count_bonus
        return confidence >= 0.2, round(confidence, 3)

    def get_confidence(self, branch: str) -> float:
        """Overall health of a capability domain (0-1)."""
        count, _ = self._branch_counts(branch)
        avg_w = self._branch_avg_weight(branch)
        if count == 0:
            return 0.0
        # Scale: more entries + higher avg weight → higher confidence
        entry_score = min(count / 20.0, 1.0)
        return round(entry_score * avg_w, 3)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _branch_counts(self, branch_id: str) -> tuple[int, int]:
        try:
            return self._tree.branch_counts(branch_id)
        except Exception:
            return 0, 0

    def _branch_avg_weight(self, branch_id: str) -> float:
        try:
            return self._tree.branch_avg_weight(branch_id)
        except Exception:
            return 0.5

    def _branch_last_updated(self, branch_id: str) -> datetime | None:
        try:
            ts = self._tree.branch_last_updated(branch_id)
            return datetime.fromisoformat(ts) if ts else None
        except Exception:
            return None
