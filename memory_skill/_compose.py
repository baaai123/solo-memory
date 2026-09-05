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
    MemorySkillError,
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


class ArchiveRequired(MemorySkillError):
    """Raised by weave() when the default-category backlog awaits archiving.

    Armed every ``archive_interval`` weaves while default entries remain
    (Unit 2).  Any archive-tool call (review_default / reclassify) clears
    the gate — responding is required, emptying is not.
    """


class QueueRequired(MemorySkillError):
    """Raised by weave() when learning-queue open items exceed the threshold.

    Armed when ``count_open() > queue_threshold`` (Unit 2).  Any
    learning_mark call clears the gate — responding is required, emptying
    is not.
    """


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
    mission_store: object = None
    character: object = None
    state_store: object = None
    relation_store: object = None
    protocol: object = None
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

    def ingest(self, turn, *, enrich: bool = True, report: bool = False) -> object:
        """Persist a dialogue turn.  Pure storage — no LLM stages.

        Classification, title generation, conclusion extraction, and
        mission decomposition are the main agent's responsibility.
        This method only writes the turn to both stores (dialogue + learned).

        Unit 4: when this system's agent is bound to a character role, the
        turn is double-written — the stored entry is referenced from the
        role and its metadata carries ``source_role`` — so role-scoped
        weave can see it.  Unbound systems keep the original behaviour.
        """
        role = self._resolve_character_role()
        return self.ingestor.ingest_dialogue(
            turn,
            extra_metadata={"source_role": role} if role else None,
            character_role=role,
        )

    def retrieve(self, query: str, limit: int = 10, filters=None,
                  partner: str | None = None):
        if partner and not filters:
            ns = self._ns_for(partner)
            filters = {"category": ns}
        return self.retriever.retrieve(query, limit=limit, filters=filters)

    def boost_weight(self, entry_id: str, delta: float = 0.05, cap: float = 0.95) -> float:
        """Boost a memory entry's weight (delegates to learned_store)."""
        return self.learned_store.boost_weight(entry_id, delta, cap)

    def delete_entry(self, entry_id: str) -> None:
        """Delete a global memory entry and cascade-remove its references.

        Removes the entry from the learned store, then removes the memory's
        references from every character so no orphan reference rows remain
        (Unit 2 of the character-store backend plan).  Both steps are
        idempotent; deleting an unreferenced or unknown entry is a no-op.
        """
        self.learned_store.delete(entry_id)
        if getattr(self, "character", None) is not None:
            self.character.remove_all_memory(entry_id)

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
            mission_store=self._ensure_mission_store(),
            degraded=self.embedder.degraded,
            degraded_reason=self.embedder.reason,
            character=getattr(self, 'character', None),
            protocol=self.protocol,
            learning_queue=getattr(self, 'learning_queue', None),
            state_store=getattr(self, 'state_store', None),
            relation_store=getattr(self, 'relation_store', None),
        )
        ctx = weave(stores, user_message, scene_summary, partner=partner,
                    character_role=self._resolve_character_role())
        ProtocolGate(self).after_weave(user_message)
        self._maybe_arm_todo_gates()
        return ctx

    def weave_for_role(self, role_id: str, user_message: str,
                       scene_summary: str = "") -> object:
        """Weave context for an explicit character role (tavern bridge).

        Unlike ``weave()`` (which resolves the role from the agent binding),
        this forces ``character_role=role_id`` so a SillyTavern request can
        inject any bound character's persona/state.  No protocol gates run.
        """
        from memory_skill.weaver import WeaverStores, weave
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
            mission_store=self._ensure_mission_store(),
            degraded=self.embedder.degraded,
            degraded_reason=self.embedder.reason,
            character=getattr(self, 'character', None),
            protocol=self.protocol,
            learning_queue=getattr(self, 'learning_queue', None),
            state_store=getattr(self, 'state_store', None),
            relation_store=getattr(self, 'relation_store', None),
        )
        return weave(stores, user_message, scene_summary,
                     character_role=role_id)

    def _maybe_arm_todo_gates(self) -> None:
        """Arm the archive/queue hard gates after a successful weave.

        Unit 2 scheme A: MemorySystem owns the trigger logic because it
        alone can read store counts; ProtocolGate only checks the flags.
        The archive gate arms when ``weave_count % archive_interval == 0``
        and default entries remain — the modulo check gives an interval
        of grace after each agent response.  The queue gate arms when
        open items exceed ``queue_threshold``.  Idempotent: an already
        pending gate is never re-armed.  Never raises: store hiccups
        degrade to "no arming" so weaving is never blocked by this path.
        """
        protocol = getattr(self, "protocol", None)
        if protocol is None:
            return
        if not getattr(protocol, "archive_pending", False):
            interval = self._todo_interval()
            count = getattr(protocol, "weave_count", 0)
            if interval > 0 and count >= interval and count % interval == 0 \
                    and self._default_entry_count() > 0:
                protocol.arm_archive()
        if not getattr(protocol, "queue_pending", False):
            if self._open_queue_count() > self._todo_queue_threshold():
                protocol.arm_queue()

    def _todo_interval(self) -> int:
        """Configured archive_interval, degraded to a safe default."""
        config = getattr(self, "config", None)
        try:
            return max(1, int(getattr(config, "archive_interval", 10) or 10))
        except (TypeError, ValueError):
            return 10

    def _todo_queue_threshold(self) -> int:
        """Configured queue_threshold, degraded to a safe default."""
        config = getattr(self, "config", None)
        try:
            return int(getattr(config, "queue_threshold", 20) or 20)
        except (TypeError, ValueError):
            return 20

    def _default_entry_count(self) -> int:
        """Count default-category entries (0 on missing store / failure)."""
        store = getattr(self, "learned_store", None)
        if store is None:
            return 0
        try:
            return len(store.list_by_category("default", limit=0))
        except Exception as exc:
            _logger.debug("default-category count failed: %s", exc)
            return 0

    def _open_queue_count(self) -> int:
        """Count open learning-queue items (0 on missing store / failure)."""
        queue = getattr(self, "learning_queue", None)
        if queue is None:
            return 0
        try:
            return int(queue.count_open())
        except Exception as exc:
            _logger.debug("learning-queue count failed: %s", exc)
            return 0

    def _resolve_character_role(self) -> str | None:
        """Resolve the character id this system's agent is bound to.

        ``None`` when there is no character store, no config, no binding,
        or the lookup fails (degrade with a log line — never block the
        call site).  Shared by ``weave`` and ``ingest`` (Unit 4).
        """
        character = getattr(self, "character", None)
        config = getattr(self, "config", None)
        if character is None or config is None:
            return None
        try:
            return character.get_agent_role(config.agent_name)
        except Exception as exc:
            _logger.warning("Character role resolution failed for %r: %s",
                            config.agent_name, exc)
            return None

    def _ensure_mission_store(self) -> object | None:
        """Lazily build (and cache) the MissionStore on this system.

        Returns None when the stores it needs are unavailable (e.g. in
        minimal test compositions), so callers can skip mission rendering.
        """
        if getattr(self, "mission_store", None) is not None:
            return self.mission_store
        if self.learned_store is None:
            return None
        from memory_skill.mission import MissionStore
        self.mission_store = MissionStore(
            learned_store=self.learned_store,
            learning_queue=getattr(self, "learning_queue", None),
        )
        return self.mission_store

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
            "importance_llm": {
                "configured": self._importance_key_configured(),
                "note": "distill/importance LLM 评分需要 IMPORTANCE_API_KEY（~/.config/memory-skill/.env），未配置时主动学习 distill 不可用，记忆核心不受影响",
            },
            "config": {
                "namespace": self.config.namespace,
                "db_path": self.config.db_path,
            },
        }

    def _importance_key_configured(self) -> bool:
        """True when IMPORTANCE_API_KEY is available (env or the two .env locations)."""
        try:
            from pathlib import Path
            from dotenv import load_dotenv
            load_dotenv(Path.home() / ".config" / "memory-skill" / ".env", override=False)
            load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
        except Exception:
            pass
        return bool(os.environ.get("IMPORTANCE_API_KEY", ""))

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
        self._persist_turn(user_msg, assistant_msg,
                           character_role=self._resolve_character_role())

    def ingest_for_role(self, role_id: str, user_msg: str,
                        assistant_msg: str) -> None:
        """Persist a tavern/roleplay turn under an explicit character role.

        Mirrors ``auto_ingest`` but forces ``character_role=role_id``, so the
        dialogue becomes a reference of that role (visible to its weave).
        Used by the tavern chat endpoint where there is no agent binding to
        resolve the role from (the user roleplays through the web UI).
        """
        self._persist_turn(user_msg, assistant_msg, character_role=role_id)

    def remember_impression(self, role_id: str, target_id: str,
                            kind: str = "impression",
                            note: str = "") -> str | None:
        """Persist one 'about X' memory and pin it to a role's references.

        *target_id* is a role id or the reserved ``"user"``.  ``kind`` is
        ``"pref"`` (target must be ``user``: shared user habits — stored as a
        pref entry) or ``"impression"`` (subjective view/event/relationship
        note — stored as a plain general entry).  Either way the entry lands
        in the global store and is referenced by *role_id* with ``target``
        set, so weave can inject it under the right person's heading and the
        character never mistakes it for its own experience.
        Returns the new entry id, or None when the role is unknown.
        """
        from memory_skill.contracts import DialogueTurn
        if not note.strip():
            return None
        role = self.character.get_role(role_id)
        if role is None:
            return None
        kind = kind if kind in ("pref", "impression") else "impression"
        if kind == "pref" and target_id != "user":
            kind = "impression"
        category = "pref" if kind == "pref" else "pers"
        now = datetime.now(UTC)
        turn = DialogueTurn(
            id=f"imp_{now:%Y%m%d_%H%M%S}_{hash(note) & 0xFFFF:04x}",
            role="system", content=note.strip(), timestamp=now, saw_index=0,
        )
        try:
            receipt = self.ingestor.ingest_dialogue(turn, category=category)
            entry_id = getattr(receipt, "entry_id", None) or turn.id
            self.character.add_memory(role_id, entry_id,
                                      dimension="general", target=target_id)
        except Exception:
            return None
        return entry_id

    def _persist_turn(self, user_msg: str, assistant_msg: str,
                      character_role: str | None) -> None:
        """Shared body of auto_ingest / ingest_for_role."""
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
                if character_role:
                    self.ingestor.ingest_dialogue(
                        turn, character_role=character_role)
                else:
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

    def ingest_conclusion(self, title: str, content: str = "") -> object:
        from memory_skill.memory_extract import ingest_conclusion_ex as _ic
        return _ic(self, title, content)

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
    from memory_skill.character_store import CharacterStore
    from memory_skill.dialogue_store import DialogueStore
    from memory_skill.embedder import Embedder
    from memory_skill.ingestor import Ingestor
    from memory_skill.learned_store import LearnedStore
    from memory_skill.pending_store import PendingStore
    from memory_skill.relation_store import RelationStore
    from memory_skill.retriever import Retriever
    from memory_skill.saw_buffer import SawRingBuffer
    from memory_skill.state_store import StateStore

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
    character_store = CharacterStore(
        db_path=getattr(config, "character_db_path", None) or config.db_path
    )
    state_store = StateStore(config.db_path)
    relation_store = RelationStore(config.db_path)
    ingestor = Ingestor(config=config, saw_buffer=saw_buffer,
                        dialogue_store=dialogue_store,
                        learned_store=learned_store, embedder=embedder,
                        tree=tree, learning_queue=learning_queue,
                        character_store=character_store)

    ms = object.__new__(MemorySystem)
    ms.__dict__.update(
        config=config, embedder=embedder, saw_buffer=saw_buffer,
        dialogue_store=dialogue_store, learned_store=learned_store,
        tree=tree, ingestor=ingestor, retriever=retriever,
        learning_queue=learning_queue, pending_store=pending_store,
        character=character_store, state_store=state_store,
        relation_store=relation_store,
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
