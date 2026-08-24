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
from memory_skill.protocol_state import ProtocolState

_logger = logging.getLogger(__name__)


class ClassificationRequired(Exception):
    """Raised by weave() when the previous turn has not been classified."""


class GapRequired(Exception):
    """Raised by weave() when classified mission has unfulfilled skill gaps."""


class SkillCheckRequired(Exception):
    """Raised by weave() when a classified mission has not yet checked existing skills."""


@dataclass(eq=False)
class MemorySystem:
    """All memory stores in one place — no hidden state.

    ``eq=False`` keeps identity semantics (and a working ``__hash__``):
    instances are used as ``WeakKeyDictionary`` keys in ``tools.py``, and
    weakrefs delegate hashing to the referent.  Value equality across the
    composite stores is meaningless and was never relied upon.
    """

    # All fields have defaults — the real constructor is __init__
    config: MemorySkillConfig | None = None
    embedder: object = None
    saw_buffer: object = None
    dialogue_store: object = None
    learned_store: object = None
    tree: object = None
    ingestor: object = None
    retriever: object = None
    learning_queue: object = None
    pending_store: object = None
    protocol: object = None
    _composed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Backward-compatible view of the protocol state. New code should go
    # through ``self.protocol``; these properties keep existing callers and
    # tests working while the migration lands.
    @property
    def _classify_pending(self):
        return self.protocol.classify_pending if self.protocol else None

    @_classify_pending.setter
    def _classify_pending(self, value):
        if self.protocol:
            self.protocol.classify_pending = value

    @property
    def _pending_gaps(self):
        return self.protocol.pending_gaps if self.protocol else set()

    @_pending_gaps.setter
    def _pending_gaps(self, value):
        if self.protocol:
            self.protocol.pending_gaps = value

    @property
    def _mission_pending_check(self):
        return self.protocol.mission_pending_check if self.protocol else False

    @_mission_pending_check.setter
    def _mission_pending_check(self, value):
        if self.protocol:
            self.protocol.mission_pending_check = value

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

    def ingest(self, turn, *, enrich: bool = True, report: bool = False) -> object:
        """Persist a dialogue turn.  Pure storage — no LLM stages.

        Classification, title generation, conclusion extraction, and
        mission decomposition are the main agent's responsibility.
        This method only writes the turn to both stores (dialogue + learned).
        """
        return self.ingestor.ingest_dialogue(turn)

    def retrieve(self, query: str, limit: int = 10, filters=None,
                  partner: str | None = None):
        if partner and not filters:
            ns = self._ns_for(partner)
            filters = {"category": ns}
        return self.retriever.retrieve(query, limit=limit, filters=filters)

    def boost_weight(self, entry_id: str, delta: float = 0.05, cap: float = 0.95) -> float:
        """Boost a memory entry's weight (delegates to learned_store)."""
        return self.learned_store.boost_weight(entry_id, delta, cap)

    def ingest_screen(self, entry) -> object:
        return self.ingestor.ingest_screen(
            entry.text if hasattr(entry, 'text') else str(entry),
        )

    def weave(self, user_message: str, scene_summary: str = "",
              partner: str | None = None):
        """Assemble layered memory context (convenience wrapper)."""
        from memory_skill.protocol_gate import ProtocolGate
        from memory_skill.weaver import WeaverStores, weave

        ProtocolGate(self).check(user_message)

        stores = WeaverStores(
            saw_buffer=self.saw_buffer,
            dialogue_store=self.dialogue_store,
            learned_store=self.learned_store,
            retriever=self.retriever,
            agent_name=self.config.agent_name,
            namespace=self.config.namespace,
            tree=self.tree,
            gaps=self.gaps,
            pending_store=getattr(self, 'pending_store', None),
            degraded=self.embedder.degraded,
            degraded_reason=self.embedder.reason,
        )
        ctx = weave(stores, user_message, scene_summary, partner=partner)
        ProtocolGate(self).after_weave(user_message)
        return ctx

    def health(self) -> dict:
        self.embedder.embed("health")  # trigger lazy-load
        return {
            "status": "healthy",
            "model_ok": not self.embedder.degraded,
            "embedder": {
                "loaded": getattr(self.embedder, '_load_attempted', True),
                "mode": self.embedder.mode,
                "dim": self.config.embedding_dim,
                "degraded": self.embedder.degraded,
                "reason": self.embedder.reason,
                "model_path": self.config.model_path,
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
        """Persist both sides of a conversation turn via the ingest pipeline.

        Uses ``enrich=False`` (no LLM stages) — this is the high-frequency
        path (transparent proxy). Truncation follows the unified head+tail
        policy (see ADR-0001).
        """
        from memory_skill.contracts import DialogueTurn, clip_auto_ingest

        now = datetime.now(UTC)
        for role, content in [("user", user_msg), ("assistant", assistant_msg)]:
            if not content:
                continue
            try:
                turn = DialogueTurn(
                    id=f"auto_{now:%Y%m%d_%H%M%S}_{hash(content) & 0xFFFF:04x}",
                    role=role,
                    content=clip_auto_ingest(content),
                    timestamp=now,
                )
                self.ingest(turn, enrich=False)
            except Exception as exc:
                _logger.warning("auto_ingest failed for %s: %s", role, exc)

    @property
    def gaps(self) -> list:
        return self.ingestor.gaps

    def ingest_skill(self, title: str, content: str, source_urls: list[str] | None = None) -> object:
        from memory_skill.memory_extract import ingest_skill_ex
        return ingest_skill_ex(self, title, content, source_urls=source_urls)

    def ingest_pers(self, trait: str) -> object:
        from memory_skill.memory_extract import ingest_pers as _ip
        return _ip(self, trait)

    def ingest_pref(self, key: str, value: str) -> object:
        from memory_skill.memory_extract import ingest_pref as _ipr
        return _ipr(self, key, value)

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

    def _ns_for(self, partner: str | None = None) -> str:
        return self.config.agent_name


# ── Factory ─────────────────────────────────────────────────────────────


def _build_system(config: MemorySkillConfig) -> MemorySystem:
    """Create a fully assembled MemorySystem from config."""
    from memory_skill.dialogue_store import DialogueStore
    from memory_skill.embedder import Embedder
    from memory_skill.ingestor import Ingestor
    from memory_skill.learned_store import LearnedStore
    from memory_skill.pending_store import PendingStore
    from memory_skill.retriever import Retriever
    from memory_skill.saw_buffer import SawRingBuffer

    embedder = Embedder(config)
    saw_buffer = SawRingBuffer(capacity=config.saw_buffer_capacity)
    dialogue_store = DialogueStore(config)
    learned_store = LearnedStore(config, embedder,
                                 chroma_path=config.db_path + "_chroma")
    tree = _build_tree(config) if config.tree_enabled else None
    retriever = Retriever(config=config, dialogue_store=dialogue_store,
                          learned_store=learned_store)
    learning_queue = _build_learning_queue(config) if config.tree_enabled else None
    pending_store = PendingStore(config.db_path) if config.tree_enabled else None
    ingestor = Ingestor(config=config, saw_buffer=saw_buffer,
                        dialogue_store=dialogue_store,
                        learned_store=learned_store, embedder=embedder,
                        tree=tree, learning_queue=learning_queue)

    ms = object.__new__(MemorySystem)
    ms.__dict__.update(
        config=config, embedder=embedder, saw_buffer=saw_buffer,
        dialogue_store=dialogue_store, learned_store=learned_store,
        tree=tree, ingestor=ingestor, retriever=retriever,
        learning_queue=learning_queue, pending_store=pending_store,
        protocol=ProtocolState(),
        _composed_at=datetime.now(UTC),
    )
    return ms


def _tree_creds() -> tuple[str, str, str]:
    return (
        os.getenv("IMPORTANCE_API_BASE", "https://api.deepseek.com/v1"),
        os.getenv("IMPORTANCE_API_KEY", ""),
        os.getenv("IMPORTANCE_MODEL", "deepseek-v4-flash"),
    )


def _build_tree(config: MemorySkillConfig):
    from memory_skill.tree import TreeManager
    api_base, api_key, model = _tree_creds()
    return TreeManager(db_path=config.db_path,
                       api_base=api_base, api_key=api_key, model=model)


def _build_learning_queue(config: MemorySkillConfig):
    from memory_skill.learning_queue import LearningQueue
    return LearningQueue(db_path=config.db_path)


create = _build_system
