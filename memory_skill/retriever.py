"""
Memory Skill — Retriever (3-signal fusion + RRF)

Combines three retrieval signals — semantic (vector), BM25 (FTS5), and
temporal (recency decay) — into a single ranked result set via Reciprocal
Rank Fusion (RRF k=60).

Algorithm:
  1. Semantic:  ``learned_store.search(query, top_k=50)``
  2. BM25:      ``dialogue_store.search(query, limit=50)``
   3. Temporal:  ``weight × exp(-λ × age_hours)`` on all candidates
  4. RRF k=60:  weighted combination across all 3 ranked lists
  5. Top-N:     take the highest-scoring entries → MemoryEnvelope

Weights (per Task 4 calibration):
  semantic = 1.5,  BM25 = 1.5,  temporal = 0.5

Usage::

    from memory_skill.contracts import MemorySkillConfig
    from memory_skill.dialogue_store import DialogueStore
    from memory_skill.learned_store import LearnedStore
    from memory_skill.retriever import Retriever

    cfg = MemorySkillConfig()
    ds  = DialogueStore(cfg)
    ls  = LearnedStore(cfg, emb)
    ret = Retriever(cfg, ds, ls)

    envelope = ret.retrieve("What does the user prefer?")
    for entry in envelope.entries:
        print(entry.content)
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING, Any

from memory_skill.contracts import (
    DialogueStoreProtocol,
    LearnedStoreProtocol,
    utcnow,
)

if TYPE_CHECKING:
    from memory_skill.contracts import (
        DialogueTurn,
        MemoryEntry,
        MemoryEnvelope,
        MemorySkillConfig,
    )


_logger = logging.getLogger(__name__)

# Broad-search fetch size before RRF fusion
_BROAD_TOP_K: int = 50


# ═══════════════════════════════════════════════════════════════════════════════
# Retriever
# ═══════════════════════════════════════════════════════════════════════════════


class Retriever:
    """Three-signal memory retriever with Reciprocal Rank Fusion (RRF k=60).

    Combines semantic (ChromaDB), BM25 (SQLite FTS5), and temporal (recency
    decay) signals into a single relevance-ranked result set.

    Parameters
    ----------
    config:
        ``MemorySkillConfig`` — drives ``rrf_k`` and ``temporal_decay_lambda``.
    dialogue_store:
        ``DialogueStore`` for FTS5 BM25 full-text search.
    learned_store:
        ``LearnedStore`` for vector-based semantic search.
    """

    def __init__(
        self,
        config: MemorySkillConfig,
        dialogue_store: DialogueStoreProtocol,
        learned_store: LearnedStoreProtocol,
    ) -> None:
        self._config = config
        self._dialogue_store = dialogue_store
        self._learned_store = learned_store

        self._rrf_k: int = config.rrf_k
        self._temporal_decay_lambda: float = config.temporal_decay_lambda

        # RRF fusion weights — config-driven so Evolution Loop can adjust
        self._semantic_weight: float = config.rrf_semantic_weight
        self._bm25_weight: float = config.rrf_bm25_weight
        self._temporal_weight: float = config.rrf_temporal_weight

    # ── Public API ────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> MemoryEnvelope:
        """Retrieve memories by fusing 3 signals via RRF.

        Parameters
        ----------
        query:
            Natural language search query.
        limit:
            Maximum number of entries to return in the envelope.
        filters:
            Optional ChromaDB metadata filters (passed to ``learned_store``
            only; ``dialogue_store`` does not support metadata filtering).

        Returns
        -------
        MemoryEnvelope
            Envelope with entries sorted by combined RRF score (descending).
            If no results, returns an empty envelope.
        """
        from memory_skill.contracts import MemoryEnvelope

        if not query:
            return MemoryEnvelope(
                type="recall",
                entries=[],
                truncated=False,
                total_candidates=0,
                timestamp=utcnow(),
            )

        # ── 1. Semantic (ChromaDB vector search) ──────────────────────────
        semantic_entries: list[MemoryEntry] = self._learned_store.search(
            query,
            limit=_BROAD_TOP_K,
            filters=filters,
        )

        # ── 2. BM25 (SQLite FTS5 full-text search) ────────────────────────
        # Use truncated query: FTS5 ANDs all tokens, long queries never match.
        # 40 chars captures 1-2 key terms while keeping match requirements loose.
        bm25_query = query[:120] if len(query) > 120 else query
        try:
            dialogue_turns: list[DialogueTurn] = self._dialogue_store.search(
                bm25_query,
                limit=_BROAD_TOP_K,
            )
        except Exception:
            dialogue_turns = []

        # Convert DialogueTurn → MemoryEntry for unified ranking
        bm25_entries: list[MemoryEntry] = [
            self._turn_to_entry(t) for t in dialogue_turns
        ]

        # ── 3. Build candidate set (union, dedup by id) ───────────────────
        now = utcnow()
        candidates: dict[str, MemoryEntry] = OrderedDict()

        for e in semantic_entries:
            if e.id not in candidates:
                candidates[e.id] = e
        for e in bm25_entries:
            if e.id not in candidates:
                candidates[e.id] = e

        if not candidates:
            return MemoryEnvelope(
                type="recall",
                entries=[],
                truncated=False,
                total_candidates=0,
                timestamp=now,
            )

        candidate_list: list[MemoryEntry] = list(candidates.values())

        # ── 4. Build ranked lists for RRF ─────────────────────────────────

        # Semantic rank map: entry_id → 0-based rank in semantic results
        semantic_rank: dict[str, int] = {}
        for rank, e in enumerate(semantic_entries):
            if e.id not in semantic_rank:
                semantic_rank[e.id] = rank

        # BM25 rank map: entry_id → 0-based rank in BM25 results
        bm25_rank: dict[str, int] = {}
        for rank, e in enumerate(bm25_entries):
            if e.id not in bm25_rank:
                bm25_rank[e.id] = rank

        # Temporal ranking: sort all candidates by time_decay (descending)
        temporal_scores: list[tuple[str, float]] = [
            (e.id, self._time_decay(e, now)) for e in candidate_list
        ]
        temporal_scores.sort(key=lambda x: x[1], reverse=True)
        temporal_rank: dict[str, int] = {
            eid: rank for rank, (eid, _) in enumerate(temporal_scores)
        }

        # ── 5. Compute weighted RRF scores ────────────────────────────────
        k = float(self._rrf_k)

        scored: list[tuple[MemoryEntry, float]] = []
        for entry in candidate_list:
            eid = entry.id
            score = 0.0

            if eid in semantic_rank:
                score += self._semantic_weight / (k + semantic_rank[eid])
            if eid in bm25_rank:
                score += self._bm25_weight / (k + bm25_rank[eid])
            if eid in temporal_rank:
                score += self._temporal_weight / (k + temporal_rank[eid])

            # ── Per-entry evolution weight boost ──────────────────────────
            score *= (0.5 + entry.weight)

            scored.append((entry, score))

        # Sort by RRF score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        total_candidates = len(scored)
        top_n = scored[:limit]
        truncated = total_candidates > limit

        # ── 6. Build envelope ─────────────────────────────────────────────
        return MemoryEnvelope(
            type="recall",
            entries=[entry for entry, _ in top_n],
            truncated=truncated,
            total_candidates=total_candidates,
            timestamp=now,
        )

    # ── Temporal scoring ──────────────────────────────────────────────────

    def _time_decay(self, entry: MemoryEntry, now: datetime) -> float:
        """Compute exponential recency decay: ``effective = weight × exp(-λ × hours)``.

        Uses ``updated_at`` as the reference timestamp.  Older entries
        (higher age) with lower weight yield lower scores.
        λ is set via ``temporal_decay_lambda`` (default 0.01).
        """
        age_seconds = (now - entry.updated_at).total_seconds()
        age_hours = max(age_seconds / 3600.0, 0.0)

        λ = self._temporal_decay_lambda
        if λ <= 0:
            return entry.weight

        return entry.weight * math.exp(-λ * age_hours)

    # ── DialogueTurn → MemoryEntry conversion ─────────────────────────────

    @staticmethod
    def _turn_to_entry(turn: DialogueTurn) -> MemoryEntry:
        """Convert a ``DialogueTurn`` into a ``MemoryEntry`` for unified ranking.

        The entry id is prefixed with ``"dialogue_"`` to avoid collisions
        with learned-store entries.
        """
        from memory_skill.contracts import MemoryEntry

        return MemoryEntry(
            id=f"dialogue_{turn.id}",
            content=turn.content,
            created_at=turn.timestamp,
            updated_at=turn.timestamp,
            weight=0.5,
            category="dialogue",
            tags=[],
            metadata={"role": turn.role, "source": "dialogue"},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════



