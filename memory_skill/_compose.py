"""Composition factory — assembles all stores into a MemorySystem.

Replaces the god-object MemorySkill class with a plain dataclass
whose store attributes are public.  Entry points can access stores
directly; helper methods remain for convenience.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime

from memory_skill.contracts import (
    DialogueStoreProtocol,
    EmbedderProtocol,
    LearnedStoreProtocol,
    MemorySkillConfig,
    SawBufferProtocol,
    TreeManagerProtocol,
)

_logger = logging.getLogger(__name__)


@dataclass
class MemorySystem:
    """All memory stores in one place — no hidden state."""

    # All fields have defaults — the real constructor is __init__
    config: MemorySkillConfig | None = None
    embedder: object = None
    saw_buffer: object = None
    dialogue_store: object = None
    learned_store: object = None
    tree: object = None
    ingestor: object = None
    retriever: object = None
    _composed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __init__(self, config: MemorySkillConfig | None = None, **overrides):
        """Create via ``MemorySystem(config)`` or ``MemorySystem(field=value).``

        When called with a single config, auto-composes all stores.
        Use keyword args to override specific fields (for testing).
        """
        if config is not None:
            ms = _build_system(config)
            self.__dict__.update(ms.__dict__)
        # Apply overrides (from dataclass field defaults)
        for k, v in overrides.items():
            setattr(self, k, v)
        if "_composed_at" not in self.__dict__:
            self._composed_at = datetime.now(UTC)

    # ── Convenience methods (thin delegates, no new logic) ──────────────

    def ingest(self, turn) -> object:
        result = self.ingestor.ingest_dialogue(turn)
        from memory_skill.memory_extract import extract_structured, tag_title
        tag_title(self, turn)
        extract_structured(self, turn)
        return result

    def _tag_title(self, turn) -> None:
        if not self.tree:
            return
        title = self._generate_title(turn.content)
        if not title:
            return
        try:
            self.learned_store.set_title(f"dialogue:{turn.id}", title)
        except Exception:
            pass






    def retrieve(self, query: str, limit: int = 10, filters=None):
        return self.retriever.retrieve(query, limit=limit, filters=filters)

    def ingest_screen(self, entry) -> object:
        return self.ingestor.ingest_screen(
            entry.text if hasattr(entry, 'text') else str(entry),
        )

    def weave(self, user_message: str, scene_summary: str = "",
              partner: str | None = None):
        """Assemble layered memory context (convenience wrapper)."""
        from memory_skill.weaver import WeaverStores, weave

        stores = WeaverStores(
            saw_buffer=self.saw_buffer,
            dialogue_store=self.dialogue_store,
            learned_store=self.learned_store,
            retriever=self.retriever,
            agent_name=self.config.agent_name,
            namespace=self.config.namespace,
            emotion_outcomes=getattr(self, '_emotion_outcomes', []),
            tree=self.tree,
            gaps=self.gaps,
        )
        return weave(stores, user_message, scene_summary, partner=partner)

    def health(self) -> dict:
        self.embedder.embed("health")  # trigger lazy-load
        return {
            "status": "healthy",
            "embedder": {
                "loaded": getattr(self.embedder, '_load_attempted', True),
                "mode": "onnx" if getattr(self.embedder, '_session', None) else "fallback",
                "dim": self.config.embedding_dim,
            },
            "saw_buffer": {
                "entry_count": len(self.saw_buffer.get_all()),
                "capacity": self.config.saw_buffer_capacity,
            },
            "learned_store": self.learned_store.health(),
            "config": {
                "namespace": self.config.namespace,
                "db_path": self.config.db_path,
            },
        }

    def consolidate(self) -> int:
        from memory_skill.observation import ObservationConsolidator
        return ObservationConsolidator(
            self.dialogue_store, self.learned_store,
        ).run()

    def clean(self) -> dict:
        from memory_skill.cleaner import MemoryCleaner
        return MemoryCleaner(
            self.learned_store, self.embedder, self.config,
        ).run()

    def reflect(self):
        from memory_skill.reflect import reflect as _reflect
        return _reflect(self.dialogue_store, self.learned_store)

    def ensure_embedder_loaded(self) -> str:
        emb = self.embedder
        emb._load_attempted = False
        emb._ensure_model_loaded()
        return "onnx" if emb._session else "fallback"

    def auto_context(self, messages: list) -> list:
        """Inject memory context before the last user message."""
        if not messages:
            return messages

        try:
            ctx = self.weave(
                user_message=(
                    messages[-1]["content"]
                    if messages and messages[-1].get("role") == "user"
                    else ""
                ),
                scene_summary="ongoing conversation",
            )
            if not ctx.is_empty:
                block = ctx.to_prompt_block()
                if block and messages and messages[-1].get("role") == "user":
                    messages = list(messages)
                    messages[-1] = dict(messages[-1])
                    messages[-1]["content"] = f"{block}\n\n{messages[-1]['content']}"
        except Exception as exc:
            _logger.warning("auto_context failed: %s", exc)

        return messages

    def auto_ingest(self, user_msg: str, assistant_msg: str) -> None:
        """Persist both sides of a conversation turn."""
        from memory_skill.contracts import DialogueTurn

        now = datetime.now(UTC)
        for role, content in [("user", user_msg), ("assistant", assistant_msg)]:
            if not content:
                continue
            try:
                turn = DialogueTurn(
                    id=f"auto_{now:%Y%m%d_%H%M%S}_{hash(content) & 0xFFFF:04x}",
                    role=role,
                    content=content[:2000],
                    timestamp=now,
                )
                self.ingestor.ingest_dialogue(turn)
            except Exception as exc:
                _logger.warning("auto_ingest failed for %s: %s", role, exc)

    @property
    def gaps(self) -> list:
        detector = getattr(self.ingestor, '_gap_detector', None)
        return detector.gaps if detector else []


    def ingest_skill(self, title: str, content: str, source_urls: list[str]) -> object:
        from memory_skill.memory_extract import ingest_skill_ex
        return ingest_skill_ex(self, title, content)

    def count_turns(self) -> int:
        return self.dialogue_store.count()

    def get_turn(self, turn_id: str):
        return self.dialogue_store.get_by_id(turn_id)

    def get_recent_turns(self, n: int = 10):
        return self.dialogue_store.get_recent(n)

    def expand(self, message: str, limit: int = 5) -> str:
        """Expand a memory reference into its full context.

        Searches for matching titles, then returns the matched entry
        plus temporally nearby entries (same time window).
        """
        if not message:
            return ""
        results = self.retriever.retrieve(message, limit=2)
        if not results.entries:
            return ""
        match = results.entries[0]
        m = getattr(match, "metadata", {}) or {}
        title = m.get("title", match.content[:40])

        from datetime import timedelta
        ts = getattr(match, "created_at", None) or getattr(match, "updated_at", None)
        if ts:
            window_start = ts - timedelta(hours=2)
            window_end = ts + timedelta(hours=2)
            nearby = self.dialogue_store.time_range(window_start, window_end)
            if nearby:
                lines = [f"[扩展记忆 — {title}]"]
                for t in nearby[:limit]:
                    lines.append(f"  · {t.content[:100]}")
                return "\n".join(lines)
        return f"[扩展记忆 — {title}]\n  · {match.content[:200]}"
        return self.learned_store.boost_weight(entry_id, delta)

    def learn(self, topic: str, urls: list) -> object:
        from memory_skill.capability_registry import CapabilityRegistry
        from memory_skill.gap_detector import Gap
        from memory_skill.learning_task import LearningTaskManager
        from memory_skill.web_crawler import WebCrawler

        registry = CapabilityRegistry(self.tree, self.retriever)
        crawler = WebCrawler(timeout=30)
        mgr = LearningTaskManager(crawler, registry, self)
        task = mgr.create_from_gap(
            Gap(query=topic, branch="assistant_skill", confidence=0.0,
                severity="major"), urls)
        return mgr.run(task)

    @property
    def _dialogue_store(self):
        return self.dialogue_store

    @property
    def _learned_store(self):
        return self.learned_store

    @property
    def _embedder(self):
        return self.embedder

    @property
    def _tree(self):
        return self.tree

    @property
    def _saw_buffer(self):
        return self.saw_buffer

    @property
    def _retriever(self):
        return self.retriever

    @property
    def _config(self):
        return self.config

    def _ns_for(self, partner: str | None = None) -> str:
        return self.config.agent_name


# ── Factory ─────────────────────────────────────────────────────────────


def _build_system(config: MemorySkillConfig) -> MemorySystem:
    """Create a fully assembled MemorySystem from config."""
    from memory_skill.dialogue_store import DialogueStore
    from memory_skill.embedder import Embedder
    from memory_skill.ingestor import Ingestor
    from memory_skill.learned_store import LearnedStore
    from memory_skill.retriever import Retriever
    from memory_skill.saw_buffer import SawRingBuffer
    from memory_skill.tree import TreeManager

    embedder = Embedder(config)
    saw_buffer = SawRingBuffer(capacity=config.saw_buffer_capacity)
    dialogue_store = DialogueStore(config)
    learned_store = LearnedStore(
        config, embedder,
        chroma_path=config.db_path + "_chroma",
    )

    tree = None
    if config.tree_enabled:
        api_base = os.getenv("IMPORTANCE_API_BASE", "https://api.deepseek.com/v1")
        api_key = os.getenv("IMPORTANCE_API_KEY", "")
        model = os.getenv("IMPORTANCE_MODEL", "deepseek-v4-flash")
        tree = TreeManager(
            db_path=config.db_path,
            api_base=api_base,
            api_key=api_key,
            model=model,
        )

    retriever = Retriever(
        config=config,
        dialogue_store=dialogue_store,
        learned_store=learned_store,
    )

    from memory_skill.capability_registry import CapabilityRegistry
    from memory_skill.gap_detector import GapDetector
    gap_detector = None
    if tree:
        reg = CapabilityRegistry(tree, retriever)
        gap_detector = GapDetector(reg)

    ingestor = Ingestor(
        config=config,
        saw_buffer=saw_buffer,
        dialogue_store=dialogue_store,
        learned_store=learned_store,
        embedder=embedder,
        tree=tree,
        gap_detector=gap_detector,
    )

    ms = object.__new__(MemorySystem)
    ms.__dict__.update(
        config=config,
        embedder=embedder,
        saw_buffer=saw_buffer,
        dialogue_store=dialogue_store,
        learned_store=learned_store,
        tree=tree,
        ingestor=ingestor,
        retriever=retriever,
        _composed_at=datetime.now(UTC),
    )
    return ms


create = _build_system
