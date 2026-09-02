"""In-memory fake stores for fast tests (no ONNX / chroma / SQLite / LLM).

Implements the store protocols from ``memory_skill.contracts`` so a
``MemorySystem`` can be composed entirely from fakes.  Enables the fast
test suite (seconds instead of minutes) without real infrastructure.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime

from memory_skill.contracts import (
    DialogueTurn,
    MemoryEntry,
    MemoryEnvelope,
    SawEntry,
)
from memory_skill.protocol_state import ProtocolState


def _as_utc(dt) -> datetime:
    """Normalize a possibly-naive datetime to aware UTC."""
    if dt is None:
        return datetime.now(UTC)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class FakeEmbedder:
    """Deterministic hash-based embeddings — no ONNX model needed.

    Produces a stable 1024-dim vector per text so semantic search over
    fakes is reproducible (identical text → identical vector).
    """

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim
        self._vectors: dict[str, list[float]] = {}

    def _vector(self, text: str) -> list[float]:
        if text not in self._vectors:
            v = []
            h = int.from_bytes(uuid.uuid5(uuid.NAMESPACE_URL, text).bytes, "big")
            for i in range(self._dim):
                h = (h * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
                v.append((h >> 24) / 0xFFFFFF - 0.5)
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            self._vectors[text] = [x / norm for x in v]
        return self._vectors[text]

    def embed(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    @property
    def mode(self) -> str:
        return "fake"

    @property
    def degraded(self) -> bool:
        return False

    @property
    def reason(self) -> str | None:
        return None


class InMemoryDialogueStore:
    """Dict-backed dialogue store — no SQLite/FTS."""

    def __init__(self) -> None:
        self._turns: dict[str, DialogueTurn] = {}
        self._fts: dict[str, set[str]] = {}

    def _index(self, turn: DialogueTurn) -> None:
        words = {w for w in turn.content.split() if len(w) > 1}
        for w in words:
            self._fts.setdefault(w, set()).add(turn.id)

    def insert(self, turn: DialogueTurn) -> None:
        # Mirror real DialogueStore: timestamp roundtrips through epoch → aware UTC
        self._turns[turn.id] = DialogueTurn(
            id=turn.id,
            role=turn.role,
            content=turn.content,
            timestamp=_as_utc(turn.timestamp),
            partner=turn.partner,
            saw_index=turn.saw_index,
        )
        self._index(self._turns[turn.id])

    def insert_batch(self, turns: list[DialogueTurn]) -> None:
        for t in turns:
            self.insert(t)

    def get_by_id(self, turn_id: str) -> DialogueTurn | None:
        return self._turns.get(turn_id)

    def get_recent(self, n: int = 5) -> list[DialogueTurn]:
        # Mirror real DialogueStore: newest last (oldest→newest order)
        return sorted(self._turns.values(), key=lambda t: t.timestamp)[-n:]

    def count(self) -> int:
        return len(self._turns)

    def count_recent(self, minutes: int = 5) -> int:
        cutoff = datetime.now().timestamp() - minutes * 60
        return sum(1 for t in self._turns.values() if t.timestamp.timestamp() >= cutoff)

    def search(self, query: str, limit: int = 10) -> list[DialogueTurn]:
        # Mirror real DialogueStore OR semantics: any word match surfaces,
        # not just documents containing ALL query words.
        words = [w for w in query.split() if len(w) > 1]
        if not words:
            return []
        ids: set[str] = set()
        for w in words:
            ids |= self._fts.get(w, set())
        if not ids:
            return []
        turns = [self._turns[i] for i in ids if i in self._turns]
        return sorted(turns, key=lambda t: t.timestamp, reverse=True)[:limit]

    def time_range(self, start: str, end: str, limit: int = 100) -> list[DialogueTurn]:
        return []

    def cleanup(self) -> int:
        return 0

    def reindex_roles(self) -> dict[str, int]:
        return {"user": 0, "assistant": 0, "system": 0}


class FakeLearnedStore:
    """List-backed learned store — no chromadb."""

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._embedder: FakeEmbedder | None = None

    def attach_embedder(self, embedder: FakeEmbedder) -> None:
        self._embedder = embedder

    def _embed(self, text: str) -> list[float]:
        if self._embedder is None:
            return []
        return self._embedder.embed(text)

    def insert(self, entry: MemoryEntry) -> None:
        # Mirror real LearnedStore: datetimes roundtrip through epoch →
        # aware UTC. Naive datetimes from ingestor become aware here.
        stored = MemoryEntry(
            id=entry.id,
            content=entry.content,
            created_at=_as_utc(entry.created_at),
            updated_at=_as_utc(entry.updated_at),
            weight=entry.weight,
            category=entry.category,
            tags=list(getattr(entry, "tags", []) or []),
            metadata=dict(getattr(entry, "metadata", {}) or {}),
            is_system=getattr(entry, "is_system", False),
        )
        self._entries[entry.id] = stored
        if self._embedder is not None:
            self._embeddings[entry.id] = self._embedder.embed(entry.content)

    def insert_batch(self, entries: list[MemoryEntry]) -> None:
        for e in entries:
            self.insert(e)

    def update(self, entry_id: str, content: str, **kwargs) -> None:
        entry = self._entries.get(entry_id)
        if entry is None:
            raise KeyError(f"Entry '{entry_id}' not found in LearnedStore")
        new_meta = dict(entry.metadata or {})
        meta = kwargs.get("metadata")
        if isinstance(meta, dict):
            # Must mirror real _merge_update_metadata: system keys live on
            # entry fields, everything else merges into metadata dict.
            weight = meta.get("weight", entry.weight)
            category = meta.get("category", entry.category)
            tags_raw = meta.get("tags_json", None)
            tags = tags_raw if tags_raw is None else list(json.loads(tags_raw))
            for key, value in meta.items():
                if key not in ("weight", "category", "tags_json",
                               "created_at", "updated_at"):
                    new_meta[key] = value
        else:
            weight = entry.weight
            category = entry.category
            tags = None
        weight = kwargs.get("weight", weight)
        self._entries[entry_id] = MemoryEntry(
            id=entry.id,
            content=content,
            created_at=entry.created_at,
            updated_at=kwargs.get("updated_at", datetime.now(UTC)),
            weight=weight,
            category=category,
            tags=list(tags if tags is not None else (entry.tags or [])),
            metadata=new_meta,
            is_system=entry.is_system,
        )
        if self._embedder is not None:
            self._embeddings[entry_id] = self._embedder.embed(content)

    def search(self, query: str, limit: int = 10, filters=None) -> list[MemoryEntry]:
        qv = self._embed(query)
        scored: list[tuple[float, MemoryEntry]] = []
        for e in self._entries.values():
            if filters:
                cat_filter = filters.get("category")
                if isinstance(cat_filter, dict) and cat_filter.get("$ne"):
                    if e.category == cat_filter["$ne"]:
                        continue
                elif cat_filter and e.category != cat_filter:
                    continue
                weight_filter = filters.get("weight")
                if isinstance(weight_filter, dict) and weight_filter.get("$gte"):
                    if e.weight < weight_filter["$gte"]:
                        continue
            ev = self._embeddings.get(e.id, [])
            if qv and ev:
                sim = sum(a * b for a, b in zip(qv, ev))
            else:
                sim = 0.5 if query in e.content else 0.0
            scored.append((sim, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Mirror real LearnedStore: attach the cosine as semantic_score.
        return [
            MemoryEntry(
                id=e.id,
                content=e.content,
                created_at=e.created_at,
                updated_at=e.updated_at,
                weight=e.weight,
                category=e.category,
                tags=list(e.tags or []),
                metadata=dict(e.metadata or {}),
                is_system=e.is_system,
                semantic_score=round(max(0.0, sim), 4),
            )
            for sim, e in scored[:limit]
        ]

    def get_weight(self, entry_id: str) -> float | None:
        e = self._entries.get(entry_id)
        return e.weight if e else None

    def list_by_category(self, category: str, limit: int = 10,
                         sort_by: str = "created_at",
                         descending: bool = True) -> list[MemoryEntry]:
        matches = [e for e in self._entries.values() if e.category == category]
        key = "updated_at" if sort_by == "updated_at" else "created_at"
        matches.sort(key=lambda e: getattr(e, key).timestamp()
                     if getattr(e, key) else 0.0,
                     reverse=descending)
        return matches if limit <= 0 else matches[:limit]

    def get_entry(self, entry_id: str):
        return self._entries.get(entry_id)

    def get_raw(self, entry_id: str) -> dict | None:
        entry = self._entries.get(entry_id)
        if entry is None:
            return None
        return {
            "document": entry.content,
            "metadata": dict(entry.metadata or {}),
        }

    def set_weight(self, entry_id: str, weight: float) -> None:
        entry = self._entries.get(entry_id)
        if entry is None:
            return
        self._entries[entry_id] = MemoryEntry(
            id=entry.id,
            content=entry.content,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
            weight=weight,
            category=entry.category,
            tags=list(entry.tags or []),
            metadata=dict(entry.metadata or {}),
            is_system=entry.is_system,
        )

    def boost_weight(self, entry_id: str, delta: float = 0.05, cap: float = 0.95) -> float:
        e = self._entries.get(entry_id)
        if e is None:
            return 0.5
        new = min(e.weight + delta, cap)
        self.set_weight(entry_id, new)
        return new

    def set_title(self, entry_id: str, title: str) -> None:
        pass

    def find_duplicate(self, embedding: list[float], threshold: float = 0.85) -> dict | None:
        """Mirror real LearnedStore: nearest neighbor by cosine similarity.

        The real store uses ``hnsw:space=cosine`` — chroma returns cosine
        *distance* (1 - similarity), and a duplicate is a small distance
        (high similarity), matched when ``distance <= 1 - threshold``.
        Identical text has distance 0.0 and always matches.
        """
        best: tuple[float, str] | None = None
        for entry_id, ev in self._embeddings.items():
            if not embedding or not ev:
                continue
            sim = sum(a * b for a, b in zip(embedding, ev))
            distance = 1.0 - sim
            if best is None or distance < best[0]:
                best = (distance, entry_id)
        if best is None:
            return None
        distance, entry_id = best
        if distance <= (1.0 - threshold):
            entry = self._entries[entry_id]
            return {
                "entry_id": entry_id,
                "content": entry.content,
                "weight": entry.weight,
                "distance": distance,
                "category": entry.category,
            }
        return None

    def health(self) -> dict:
        return {"status": "healthy", "entry_count": len(self._entries), "backend": "memory"}


class FakeSawBuffer:
    def __init__(self) -> None:
        self._entries: list[SawEntry] = []
        self._next_index = 0

    def put(self, entry: SawEntry) -> None:
        self._entries.append(entry)

    def get_all(self) -> list[SawEntry]:
        return list(self._entries)

    def get_at_offset(self, heartbeat_index: int) -> SawEntry | None:
        for e in self._entries:
            if e.heartbeat_index >= heartbeat_index:
                return e
        return None


class FakeLearningQueue:
    """In-memory learning queue — mirrors LearningQueue's public API."""

    def __init__(self) -> None:
        self._items: dict[str, object] = {}
        self._next_id = 0

    def enqueue(self, kind: str, query: str, detail: str = "") -> object | None:
        query = (query or "").strip()
        if not query or kind not in ("skill", "mission"):
            return None
        if self._has_open(kind, query):
            return None
        from datetime import UTC, datetime
        from memory_skill.learning_queue import QueueItem
        self._next_id += 1
        item = QueueItem(
            id=f"lq_fake_{self._next_id}", kind=kind, query=query,
            status="open", detail=detail, created_at=datetime.now(UTC),
        )
        self._items[item.id] = item
        return item

    def open_items(self, kind: str | None = None) -> list:
        items = [i for i in self._items.values()
                 if getattr(i, "status", "") == "open"]
        if kind:
            items = [i for i in items if i.kind == kind]
        return sorted(items, key=lambda i: i.created_at)

    def mark(self, item_id: str, status: str) -> bool:
        item = self._items.get(item_id)
        if item is None or getattr(item, "status", "") != "open":
            return False
        if status not in ("done", "skipped"):
            return False
        from memory_skill.learning_queue import QueueItem
        self._items[item_id] = QueueItem(
            id=item.id, kind=item.kind, query=item.query,
            status=status, detail=item.detail, created_at=item.created_at,
        )
        return True

    def count_open(self) -> int:
        return sum(1 for i in self._items.values()
                   if getattr(i, "status", "") == "open")

    def all(self, limit: int = 50) -> list:
        items = sorted(self._items.values(),
                       key=lambda i: getattr(i, "created_at", None) or 0,
                       reverse=True)
        return items[:limit]

    def _has_open(self, kind: str, query: str) -> bool:
        return any(
            getattr(i, "kind", "") == kind
            and getattr(i, "query", "") == query
            and getattr(i, "status", "") == "open"
            for i in self._items.values()
        )

    def close(self) -> None:
        pass


class FakeTreeClassifier:
    """Deterministic classify/navigate — no LLM, no network.

    Keyword-based routing mirrors the real classifier's intent so
    tree tests run offline and fast.
    """

    _BRANCH_KEYWORDS = {
        "pref": ("喜欢", "讨厌", "习惯", "偏好"),
        "pers": ("人格", "性格", "风格", "语气"),
        "task": ("项目", "bug", "todo", "进展", "任务"),
        "skill": ("学", "API", "框架", "怎么用", "技能", "FastAPI"),
    }

    def classify(self, content: str) -> dict:
        text = content or ""
        for branch, kws in self._BRANCH_KEYWORDS.items():
            if any(kw in text for kw in kws):
                root = "assistant" if branch in ("pers", "task", "skill") else "user"
                return {"root": root, "branch": branch}
        return {"root": "user", "branch": "mem"}

    def navigate(self, query: str, max_tokens: int = 512) -> str:
        if not query:
            return ""
        # Mirror real TreeManager.navigate: returns formatted context string
        return f"[树导航] 检索分支: user_mem (3天) — {query[:30]}..."


class FakeKnowledgeSynth:
    """Deterministic knowledge synthesis — no LLM, no network.

    Wraps crawled chunks into a plain markdown document so the learning
    loop (crawl → synthesize → ingest → verify) runs fully offline.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def synthesize_markdown(self, topic: str,
                            chunks_by_url: dict[str, list]) -> str:
        self.call_count += 1
        if not chunks_by_url:
            return ""
        blocks = []
        for i, (url, chunks) in enumerate(chunks_by_url.items()):
            combined = "\n\n".join(c.text[:200] for c in chunks)
            blocks.append(f"[Source {i} — {url}]\n{combined}")
        return f"# {topic}\n\n" + "\n\n---\n\n".join(blocks)


class FakeCrawler:
    """Deterministic crawler — returns fixed chunks, no network."""

    def __init__(self, text: str = "crawled content about {topic}") -> None:
        self._text = text
        self.call_count = 0

    def crawl(self, url: str) -> list:
        from datetime import UTC, datetime
        from memory_skill.web_crawler import CrawledChunk
        self.call_count += 1
        topic = url.split("/")[-1] or "topic"
        return [
            CrawledChunk(
                source_url=url,
                text=self._text.format(topic=topic),
                index=0,
                title=topic,
                crawled_at=datetime.now(UTC),
            )
        ]


def build_fast_system(config=None):
    """Compose a MemorySystem entirely from in-memory fakes.

    Mirrors ``memory_skill._compose._build_system`` but with no ONNX,
    chroma, SQLite, tree, or LLM — everything is in-memory.
    """
    from memory_skill.contracts import MemorySkillConfig
    from memory_skill._compose import MemorySystem
    from memory_skill.ingestor import Ingestor
    from memory_skill.retriever import Retriever

    if config is None:
        config = MemorySkillConfig(db_path=":memory:", agent_name="test")

    embedder = FakeEmbedder(dim=config.embedding_dim)
    saw_buffer = FakeSawBuffer()
    dialogue_store = InMemoryDialogueStore()
    learned_store = FakeLearnedStore()
    learned_store.attach_embedder(embedder)

    retriever = Retriever(
        config=config,
        dialogue_store=dialogue_store,
        learned_store=learned_store,
    )

    ingestor = Ingestor(
        config=config,
        saw_buffer=saw_buffer,
        dialogue_store=dialogue_store,
        learned_store=learned_store,
        embedder=embedder,
        tree=None,
        learning_queue=None,
    )

    from memory_skill.pending_store import PendingStore
    pending_store = PendingStore(config.db_path)

    ms = object.__new__(MemorySystem)
    ms.__dict__.update(
        config=config,
        embedder=embedder,
        saw_buffer=saw_buffer,
        dialogue_store=dialogue_store,
        learned_store=learned_store,
        tree=None,
        ingestor=ingestor,
        retriever=retriever,
        learning_queue=None,
        pending_store=pending_store,
        protocol=ProtocolState(),
        _composed_at=datetime.now(UTC),
    )
    return ms
