"""
Memory Skill — LearnedStore (ChromaDB-backed vector storage)

Provides the ``LearnedStore`` class for persistent vector-based memory
storage using ChromaDB as the backend (chosen over LanceDB — spike
showed 7× faster search).

Key design:
- ChromaDB PersistentClient with SQLite for metadata
- HNSW index with cosine distance
- Metadata stored as flat dict (complex types JSON-serialized)
- Max-entries eviction: oldest entries evicted when count exceeds limit
- Corruption fallback: search gracefully returns empty on storage errors
- EUA-style update: upsert replaces embeddings + metadata atomically

Usage::

    from memory_skill.contracts import MemorySkillConfig
    from memory_skill.embedder import Embedder
    from memory_skill.learned_store import LearnedStore

    cfg = MemorySkillConfig()
    emb = Embedder(cfg)
    store = LearnedStore(cfg, emb)
    store.insert(MemoryEntry(...))
    results = store.search("query", limit=10)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_skill.contracts import MemoryEntry, MemorySkillConfig
    from memory_skill.embedder import Embedder

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# LearnedStore
# ═══════════════════════════════════════════════════════════════════════════════


class LearnedStore:
    """ChromaDB-backed vector store for learned agent memories.

    Parameters
    ----------
    config:
        ``MemorySkillConfig`` instance — drives db_path, namespace,
        and max_learned_entries.
    embedder:
        ``Embedder`` instance for generating text embeddings.
    chroma_path:
        Optional override for the ChromaDB persistence directory.
        If not set, ``config.db_path`` is used.
    """

    _COLLECTION_PREFIX = "learned"

    def __init__(
        self,
        config: MemorySkillConfig,
        embedder: Embedder,
        chroma_path: str | None = None,
    ) -> None:
        import chromadb

        self._config: MemorySkillConfig = config
        self._embedder: Embedder = embedder

        persist_dir = chroma_path if chroma_path is not None else config.db_path
        os.makedirs(persist_dir, exist_ok=True)

        # ChromaDB settings: use cosine distance (L2-normalized vectors)
        self._client = chromadb.PersistentClient(path=persist_dir)

        collection_name = f"{self._COLLECTION_PREFIX}_{config.namespace.replace('/', '_').replace(':', '_')}"
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Public API ────────────────────────────────────────────────────────

    def insert(self, entry: MemoryEntry) -> None:
        """Insert a memory entry into the vector store.

        The entry's content is embedded, and both the vector and metadata
        are persisted.  After insertion, the max-entries constraint is enforced.

        Parameters
        ----------
        entry:
            The ``MemoryEntry`` to store.
        """
        vector = self._embedder.embed(entry.content)
        metadata = self._entry_to_chroma_meta(entry)

        self._collection.add(
            ids=[entry.id],
            embeddings=[vector],
            documents=[entry.content],
            metadatas=[metadata],
        )

        self._enforce_max_entries()

    def insert_batch(
        self,
        entries: list[MemoryEntry],
        embeddings: list[list[float]],
    ) -> None:
        """Insert multiple memory entries in a single ChromaDB batch call.

        This uses ChromaDB's native ``collection.add()`` with parallel list
        arguments (ids, embeddings, documents, metadatas), which is 10-50×
        faster than per-entry ``add()`` calls.

        After insertion, the max-entries constraint is enforced once.

        Parameters
        ----------
        entries:
            List of ``MemoryEntry`` objects to store.
        embeddings:
            Pre-computed embedding vectors, one per entry (same order).
            Each vector must have length ``config.embedding_dim``.
        """
        if not entries:
            return

        ids = [e.id for e in entries]
        documents = [e.content for e in entries]
        metadatas = [self._entry_to_chroma_meta(e) for e in entries]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        self._enforce_max_entries()

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[MemoryEntry]:
        """Search for memories most similar to the query.

        Supports optional ChromaDB ``where`` metadata filters for hybrid
        (semantic + scalar) search.

        On storage corruption, the error is caught internally and an
        empty list is returned (graceful degradation).

        Parameters
        ----------
        query:
            Natural language query string.
        limit:
            Maximum number of results to return.
        filters:
            ChromaDB metadata filter dict (e.g. ``{"category": "work"}``
            or ``{"weight": {"$gt": 0.5}}``).

        Returns
        -------
        list[MemoryEntry]
            Matching entries ordered by cosine similarity (closest first).
        """
        try:
            return self._search_impl(query, limit, filters)
        except Exception as exc:

            logger.error(
                "LearnedStore search failed (corruption suspected): %s",
                exc,
            )
            # Graceful degradation — caller can continue with empty results.
            return []

    def update(self, entry_id: str, content: str, **kwargs) -> None:
        """Update a memory entry's content (EUA-style efficient update).

        The content is re-embedded and the old vector + document are
        replaced atomically via ChromaDB upsert.  The ``updated_at``
        timestamp is refreshed.

        Parameters
        ----------
        entry_id:
            The ID of the entry to update.
        content:
            The new content string.
        **kwargs:
            Additional metadata to merge into the entry's metadata dict.
            Use ``metadata={...}`` to pass a dict of metadata key-values.

        Raises
        ------
        KeyError
            If the entry_id does not exist in the collection.
        """
        # Check existence
        existing = self._collection.get(ids=[entry_id])
        if not existing["ids"]:
            raise KeyError(
                f"Entry '{entry_id}' not found in LearnedStore"
            )

        old_meta = existing["metadatas"][0] if existing["metadatas"] else {}

        vector = self._embedder.embed(content)
        new_meta = {
            **old_meta,
            **kwargs.get("metadata", {}),
            "updated_at": time.time(),
        }

        self._collection.upsert(
            ids=[entry_id],
            embeddings=[vector],
            documents=[content],
            metadatas=[new_meta],
        )

    def delete(self, entry_id: str) -> None:
        """Delete a memory entry from the store.

        Idempotent — deleting a non-existent entry does not raise.

        Parameters
        ----------
        entry_id:
            The ID of the entry to delete.
        """
        self._collection.delete(ids=[entry_id])

    def find_duplicate(
        self,
        embedding: list[float],
        threshold: float = 0.85,
    ) -> dict | None:
        """Search for a near-duplicate memory by embedding cosine distance.

        Finds the single nearest neighbor in the collection and returns it
        if the cosine distance exceeds the threshold, indicating a match.

        Parameters
        ----------
        embedding:
            The embedding vector to compare against stored entries.
        threshold:
            Cosine distance threshold (default 0.85).  Only results with
            ``distance > threshold`` are considered duplicates.

        Returns
        -------
        dict | None
            ``{entry_id, content, weight, distance}`` if a match is found,
            ``None`` otherwise.
        """
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
        )

        if not results.get("ids") or not results["ids"][0]:
            return None

        entry_id = results["ids"][0][0]
        distance = (
            results["distances"][0][0] if results.get("distances") else 0.0
        )
        content = (
            results["documents"][0][0] if results.get("documents") else ""
        )
        meta = results["metadatas"][0][0] if results.get("metadatas") else {}
        weight = float(meta.get("weight", 0.5))

        if distance > threshold:
            return {
                "entry_id": entry_id,
                "content": content,
                "weight": weight,
                "distance": distance,
            }

        return None

    def get_weight(self, entry_id: str) -> float | None:
        """Get the current weight for a memory entry.

        Public API for ``EvolutionLoop`` — avoids direct ``_collection``
        access from outside the store.

        Parameters
        ----------
        entry_id:
            The memory entry ID to look up.

        Returns
        -------
        float | None
            The current weight, or ``None`` if the entry is not found
            or on storage error.
        """
        try:
            result = self._collection.get(
                ids=[entry_id],
                include=["metadatas"],
            )
        except Exception:
            logger.warning(
                "LearnedStore.get_weight: failed for %s", entry_id
            )
            return None

        if (
            result
            and result.get("metadatas")
            and result["metadatas"][0]
        ):
            return float(result["metadatas"][0].get("weight", 0.5))

        return None

    def get_all_weights(self) -> dict[str, float]:
        """Return a dict of {entry_id: weight} for all stored entries.

        Returns
        -------
        dict[str, float]
            Map of entry_id to current weight.  Returns empty dict on
            storage error.
        """
        try:
            result = self._collection.get(include=["metadatas"])
        except Exception:
            logger.warning("LearnedStore.get_all_weights: collection.get failed")
            return {}

        weights: dict[str, float] = {}
        for eid, meta in zip(result.get("ids", []), result.get("metadatas", [])):
            weights[eid] = float(meta.get("weight", 0.5)) if meta else 0.5
        return weights

    def set_weight(self, entry_id: str, weight: float) -> None:
        """Update the weight metadata for a memory entry.

        Public API for ``EvolutionLoop`` — avoids direct ``_collection``
        access from outside the store.  Idempotent: does nothing if the
        entry does not exist.

        Parameters
        ----------
        entry_id:
            The memory entry ID to update.
        weight:
            The new weight value (clamped by the caller).
        """
        try:
            existing = self._collection.get(
                ids=[entry_id],
                include=["metadatas"],
            )
        except Exception:
            logger.warning(
                "LearnedStore.set_weight: failed for %s", entry_id
            )
            return

        if not existing or not existing.get("ids"):
            return

        meta = dict(existing["metadatas"][0]) if existing["metadatas"][0] else {}
        meta["weight"] = weight
        self._collection.update(ids=[entry_id], metadatas=[meta])

    def set_title(self, entry_id: str, title: str) -> None:
        try:
            existing = self._collection.get(ids=[entry_id], include=["metadatas"])
        except Exception:
            return
        if not existing or not existing.get("ids"):
            return
        meta = dict(existing["metadatas"][0]) if existing["metadatas"][0] else {}
        meta["title"] = title
        self._collection.update(ids=[entry_id], metadatas=[meta])

    def boost_weight(self, entry_id: str, delta: float = 0.05, cap: float = 0.95) -> float:
        """Increase an entry's weight by *delta*, capped at *cap*.

        Returns the new weight, or 0.5 if the entry doesn't exist.
        """
        current = self.get_weight(entry_id)
        if current is None:
            return 0.5
        new = min(current + delta, cap)
        self.set_weight(entry_id, new)
        return new

    def health(self) -> dict[str, Any]:
        """Return a health/diagnostics summary of the vector store.

        Returns
        -------
        dict
            Keys: ``status``, ``collection``, ``entry_count``,
            ``backend``, ``embedding_dim``.
        """
        try:
            count = self._collection.count()
            status = "healthy"
        except Exception as exc:
            count = -1
            status = f"degraded: {exc}"

        return {
            "status": status,
            "collection": f"{self._COLLECTION_PREFIX}_{self._config.namespace}",
            "entry_count": count,
            "backend": "chromadb",
            "embedding_dim": self._config.embedding_dim,
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _search_impl(
        self,
        query: str,
        limit: int,
        filters: dict[str, Any] | None,
    ) -> list[MemoryEntry]:
        """Core search logic (no corruption handling — wrapped by search())."""
        vector = self._embedder.embed(query) if query else [0.0] * self._config.embedding_dim

        kwargs: dict[str, Any] = dict(
            query_embeddings=[vector],
            n_results=limit,
        )
        if filters:
            kwargs["where"] = filters

        results = self._collection.query(**kwargs)
        return self._parse_query_results(results)

    def _enforce_max_entries(self) -> None:
        """Evict oldest entries if collection exceeds config.max_learned_entries."""
        max_entries = self._config.max_learned_entries
        if max_entries <= 0:
            return

        current = self._collection.count()
        if current <= max_entries:
            return

        # Fetch all entries with their metadata timestamps
        all_data = self._collection.get(
            include=["metadatas"],
        )

        # Pair (id, created_at) and sort by oldest first
        pairs: list[tuple[str, float]] = []
        for eid, meta in zip(all_data["ids"], all_data["metadatas"]):
            created_at = meta.get("created_at", 0.0) if meta else 0.0
            pairs.append((eid, created_at))

        pairs.sort(key=lambda x: x[1])  # oldest first

        to_evict = current - max_entries
        evict_ids = [eid for eid, _ in pairs[:to_evict]]

        if evict_ids:
            self._collection.delete(ids=evict_ids)
            logger.debug(
                "Evicted %d oldest entries (count=%d → %d)",
                len(evict_ids),
                current,
                current - len(evict_ids),
            )

    def _parse_query_results(
        self,
        results: dict[str, Any],
    ) -> list[MemoryEntry]:
        """Convert ChromaDB query results into ``MemoryEntry`` objects.

        Attaches ``semantic_score`` (cosine similarity to the query, derived
        from the ChromaDB cosine distance) so downstream consumers such as
        ``CapabilityRegistry.can_answer`` can judge genuine relevance instead
        of raw result counts.
        """
        entries: list[MemoryEntry] = []

        ids_list = results.get("ids")
        if not ids_list or not ids_list[0]:
            return entries

        documents = results.get("documents")
        metadatas = results.get("metadatas")
        distances = results.get("distances")

        for i, eid in enumerate(ids_list[0]):
            doc = documents[0][i] if documents and documents[0] else None
            meta = metadatas[0][i] if metadatas and metadatas[0] else {}

            semantic_score: float | None = None
            if distances and distances[0]:
                # ChromaDB cosine distance ∈ [0, 2]; similarity = 1 - distance.
                semantic_score = round(max(0.0, 1.0 - float(distances[0][i])), 4)

            entries.append(
                self._chroma_meta_to_entry(eid, doc, meta, semantic_score)
            )

        return entries

    # ── Metadata conversion ───────────────────────────────────────────────

    @staticmethod
    def _entry_to_chroma_meta(entry: MemoryEntry) -> dict[str, Any]:
        """Convert a ``MemoryEntry`` to a ChromaDB-safe metadata dict."""
        return {
            "created_at": entry.created_at.timestamp(),
            "updated_at": entry.updated_at.timestamp(),
            "weight": entry.weight,
            "category": entry.category,
            "tags_json": json.dumps(entry.tags),
            "ext_metadata_json": json.dumps(entry.metadata),
        }

    @staticmethod
    def _chroma_meta_to_entry(
        entry_id: str,
        document: str | None,
        meta: dict[str, Any],
        semantic_score: float | None = None,
    ) -> MemoryEntry:
        """Reconstruct a ``MemoryEntry`` from ChromaDB metadata."""
        # Late import to avoid circular dependency at module level
        from memory_skill.contracts import MemoryEntry

        def _safe_json_load(raw: Any, default: Any) -> Any:
            """Parse JSON string, falling back to default on failure."""
            if isinstance(raw, str):
                try:
                    return json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    return default
            return default

        created_at = datetime.fromtimestamp(
            meta.get("created_at", 0.0), tz=UTC
        )
        updated_at = datetime.fromtimestamp(
            meta.get("updated_at", 0.0), tz=UTC
        )

        ext_meta = _safe_json_load(meta.get("ext_metadata_json"), {})
        if "title" in meta:
            ext_meta["title"] = meta["title"]
        return MemoryEntry(
            id=entry_id,
            content=document if document else "",
            created_at=created_at,
            updated_at=updated_at,
            weight=float(meta.get("weight", 0.0)),
            category=str(meta.get("category", "default")),
            tags=_safe_json_load(meta.get("tags_json"), []),
            metadata=ext_meta,
            semantic_score=semantic_score,
        )
