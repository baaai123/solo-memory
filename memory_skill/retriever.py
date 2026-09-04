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
# Skill memories are authoritative knowledge; this boost lifts them above
# similarly-scored dialogue turns when both match a query.
_SKILL_BOOST: float = 2.0


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
        role_memory_ids: set[str] | None = None,
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
        role_memory_ids:
            Optional whitelist of memory ids visible to a bound character
            role.  When provided, both legs are filtered so only entries
            whose id is in the set reach the candidate pool — the rest of
            global memory is invisible (R10).  An empty set yields an empty
            envelope.  ``None`` (default) keeps the unbound behavior
            unchanged (R11).

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

        # Apply category filter to BM25 entries too. filters only reach the
        # semantic (ChromaDB) leg; dialogue turns are tagged "dialogue" and
        # would otherwise leak into pref/pers/skill results regardless of
        # the requested category.
        if filters and filters.get("category"):
            wanted = filters["category"]
            if isinstance(wanted, dict) and wanted.get("$ne"):
                # Exclusion filter (e.g. {"$ne": "default"}): BM25 leg only
                # carries raw dialogue turns, which never carry the excluded
                # category — keep them all.
                pass
            elif wanted == "default":
                bm25_entries = [
                    e for e in bm25_entries
                    if e.category == "dialogue" or e.category == "default"
                ]
            else:
                bm25_entries = [
                    e for e in bm25_entries if e.category == wanted
                ]

        # ── 2b. Role whitelist (character-bound retrieval) ───────────────
        # A bound character role makes only its referenced memories visible.
        # The whitelist holds learned-store ids; BM25 entries mirror the
        # ``dialogue:<id>`` learned-id format, so a referenced memory that
        # also matched BM25 survives while unreferenced turns are dropped.
        # Applied to both legs before candidate merging so RRF ranks only
        # visible memories.
        if role_memory_ids is not None:
            semantic_entries = [
                e for e in semantic_entries if e.id in role_memory_ids
            ]
            bm25_entries = [
                e for e in bm25_entries if e.id in role_memory_ids
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

        # ── 2c. Role whitelist candidate completion ──────────────────────
        # Semantic top-N often misses a role's high-weight *skill* memories
        # (crowded out by many similar dialogue units). When a whitelist is
        # active its size is small (≤~200), so pulling every referenced
        # memory into the candidate pool guarantees such skills are ranked —
        # the skill scoring boost below then lifts them above the dialogue
        # noise. Unknown ids (deleted entries) are skipped.
        if role_memory_ids is not None and len(role_memory_ids) <= 200:
            for mid in role_memory_ids:
                try:
                    learned_e = self._learned_store.get_entry(mid)
                except Exception:
                    learned_e = None
                if learned_e is None:
                    continue
                existing = candidates.get(mid)
                # BM25-leg entries only carry dialogue metadata (no real
                # category / turn_id); prefer the learned copy so skill
                # weighting and turn expansion see the true attributes.
                if existing is None or existing.category == "dialogue" and not (
                        existing.metadata or {}).get("turn_id"):
                    candidates[mid] = learned_e

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

        # Query terms for the keyword lift below: meaningful ascii words
        # (len>=3) plus CJK character bi-grams (coarse but cheap).
        _query_terms: set[str] = set()
        if query:
            import re as _re
            for w in _re.findall(r"[A-Za-z0-9]{3,}", query):
                _query_terms.add(w.lower())
            cjk = _re.findall(r"[\u4e00-\u9fff]", query)
            for i in range(len(cjk) - 1):
                _query_terms.add(cjk[i] + cjk[i + 1])

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
            # Skill memories are authoritative: lift them above dialogue
            # noise when they surface (their weight already encodes reuse).
            if entry.category == "skill":
                score *= _SKILL_BOOST
            # Keyword lift: when a whitelist pins skills that semantic
            # ranking under-scores (bge is weak on Chinese technical docs),
            # reward entries whose content actually matches query terms.
            if role_memory_ids is not None and _query_terms:
                content = entry.content or ""
                hits = sum(1 for t in _query_terms if t in content)
                if hits:
                    score *= (1.0 + 0.5 * hits)

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

    def best_semantic_match(
        self,
        query: str,
        limit: int = 5,
    ) -> tuple[MemoryEntry | None, float]:
        """Return the entry with the highest semantic score for the query.

        Answerability checks use the semantic leg directly because RRF-fused
        ordering is recency-biased — the most relevant entry is often not the
        first envelope entry (KNOWN-ISSUES #1).

        Returns
        -------
        ``(entry, semantic_score)`` — ``(None, 0.0)`` when nothing matched.
        """
        if not query:
            return None, 0.0

        entries: list[MemoryEntry] = self._learned_store.search(
            query,
            limit=limit,
        )
        if not entries:
            return None, 0.0

        best = max(entries, key=lambda e: (e.semantic_score or 0.0, len(e.content)))
        return best, best.semantic_score or 0.0

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

        The entry id mirrors the learned-store format (``dialogue:<turn.id>``)
        so feedback/boost_weight lookups by id resolve to the same chroma
        entry.  A divergent prefix (e.g. ``dialogue_``) would make boosts
        silently no-op.
        """
        from memory_skill.contracts import MemoryEntry

        return MemoryEntry(
            id=f"dialogue:{turn.id}",
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



