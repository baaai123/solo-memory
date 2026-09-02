"""
Ingestor — write pipeline orchestrator for the Memory Skill.

Routes dialogue turns and screen observations to the appropriate stores:
- SawRingBuffer (always, for short-term context)
- DialogueStore (dialogue turns)
- LearnedStore (embed + store with entity tags)

Dialogue pipeline:
  1. Store in SawRingBuffer (always)
  2. Store in DialogueStore (always)
  3. Entity extraction via regex → tag MemoryEntry
  4. Embed + store in LearnedStore (always)

Screen pipeline:
  1. NoiseFilter check → if noise: skip (only SawRingBuffer)
  2. If kept: check for errors → tag as high importance → LearnedStore with high weight
  3. If kept: normal screen → LearnedStore with low weight
"""

from __future__ import annotations

import re
from memory_skill.classification_helpers import _looks_like_question, _ScreenNoiseFilter
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from memory_skill.contracts import (
    DialogueStoreProtocol,
    DialogueTurn,
    EmbedderProtocol,
    IngestReceipt,
    LearnedStoreProtocol,
    MemoryEntry,
    SawBufferProtocol,
    SawEntry,
    TreeManagerProtocol,
)
import numpy as np

if TYPE_CHECKING:
    from memory_skill.character_store import CharacterStore
    from memory_skill.contracts import MemorySkillConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Regex-based entity extraction (v1 — simple, no spaCy)
# ═══════════════════════════════════════════════════════════════════════════════

# Proper names: capitalized words (at least 2 letters to avoid single-letter noise)
_PATTERN_PROPER: re.Pattern[str] = re.compile(r"\b[A-Z][a-z]{1,}\b")

# Common English words that look like proper names but aren't
_STOP_WORDS: frozenset[str] = frozenset({
    "I", "A", "An", "The", "It", "He", "She", "We", "They", "You",
    "This", "That", "These", "Those", "My", "Your", "His", "Her",
    "Our", "Their", "Its", "No", "Yes", "Not", "So", "If", "Or",
    "And", "But", "For", "Nor", "Yet", "To", "In", "On", "At",
    "By", "From", "With", "Is", "Be", "Are", "Was", "Were", "Been",
    "Do", "Does", "Did", "Has", "Have", "Had", "Can", "Could",
    "Will", "Would", "Shall", "Should", "May", "Might", "Must",
    "What", "When", "Where", "Which", "Who", "Whom", "Whose",
    "Why", "How", "All", "Each", "Every", "Both", "Few", "More",
    "Most", "Other", "Some", "Such", "Only", "Own", "Same",
    "Then", "Than", "Too", "Very", "Just", "Now", "Here", "There",
    "Also", "Even", "Still", "Again", "Already", "Always", "Never",
    "Often", "Sometimes", "Usually", "One", "Two", "Three", "First",
    "Last", "New", "Old", "Good", "Great", "Big", "Small", "Large",
    "High", "Low", "Long", "Short", "Right", "Left", "Best", "Better",
    "Today", "Tomorrow", "Yesterday", "Monday", "Tuesday", "Wednesday",
    "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
    "March", "April", "June", "July", "August", "September",
    "October", "November", "December",
})


def _extract_entities(text: str) -> list[str]:
    """Extract named entity candidates via simple regex (v1).
    
    Matches capitalized words that are not common English stop words.
    Returns deduplicated, case-preserved list.
    """
    if not text:
        return []

    matches = _PATTERN_PROPER.findall(text)
    seen: set[str] = set()
    entities: list[str] = []
    for match in matches:
        if match not in _STOP_WORDS and match not in seen:
            seen.add(match)
            entities.append(match)
    return entities


# ═══════════════════════════════════════════════════════════════════════════════
# Text chunking (V3 — mitigates 512-token embedding truncation with bge-large)
# ═══════════════════════════════════════════════════════════════════════════════

# Character-based token approximation: ~4 chars per English word-token.
_CHARS_PER_TOKEN: int = 4



# Sentence-boundary patterns to prefer when splitting chunks.
# ═══════════════════════════════════════════════════════════════════════════════
# Ingestor
# ═══════════════════════════════════════════════════════════════════════════════


class Ingestor:
    """Write pipeline orchestrator.

    Routes dialogue and screen observations through stores with noise
    filtering and entity extraction.

    Parameters
    ----------
    config: MemorySkillConfig
    saw_buffer: SawBufferProtocol
    dialogue_store: DialogueStoreProtocol
    learned_store: LearnedStoreProtocol
    embedder: EmbedderProtocol
    tree: TreeManagerProtocol | None
    character_store: CharacterStore | None
        Optional store for the character double-write hook (Unit 4). When
        set, ``ingest_dialogue(character_role=...)`` references the stored
        entry from the bound character role so role-scoped weave sees it.
    """

    def __init__(
        self,
        config: MemorySkillConfig,
        saw_buffer: SawBufferProtocol,
        dialogue_store: DialogueStoreProtocol,
        learned_store: LearnedStoreProtocol,
        embedder: EmbedderProtocol,
        tree: TreeManagerProtocol | None = None,
        learning_queue=None,
        character_store: CharacterStore | None = None,
    ) -> None:
        self._config = config
        self._saw_buffer = saw_buffer
        self._dialogue_store = dialogue_store
        self._learned_store = learned_store
        self._noise_filter = _ScreenNoiseFilter()
        self._embedder = embedder
        self._tree = tree
        self._learning_queue = learning_queue
        self._character_store = character_store

        # Monotonic heartbeat counter for SawRingBuffer entries
        self._heartbeat: int = 0

        # Last-kept screen frame for noise filter comparison
        self._last_kept_screen: str | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def ingest_dialogue(self, turn: DialogueTurn, category: str | None = None,
                         extra_metadata: dict | None = None,
                         character_role: str | None = None) -> IngestReceipt:
        """Ingest a dialogue turn — all content goes to both stores.

        1. Store in SawRingBuffer (always)
        2. Store in DialogueStore (always)
        3. Embed → semantic dedup → merge (weight+0.05) or insert (weight=0.5)
        4. Store in LearnedStore (always)
        5. Character double-write (Unit 4): when *character_role* is given,
           reference the stored entry from that character role.

        The character reference is added after the learned write (insert or
        merge) so the final ``entry_id`` — including the dedup merge target —
        lands in the role's reference set.  Without *character_role* (or
        without a ``character_store``) this hook is inert and behaviour is
        unchanged (R11).

        Returns an ``IngestReceipt`` describing what happened (see ADR-0001):
        the stored entry id, whether it was deduped, the resulting weight,
        and per-stage status. Gap detection is no longer triggered here —
        the ingest pipeline (MemorySystem.ingest) drives it under ``enrich``.
        """
        now = datetime.now()

        # 1. Store in SawRingBuffer (always)
        self._saw_buffer.put(self._make_saw_entry(turn.content, now))

        # 2. Store in DialogueStore (always)
        self._dialogue_store.insert(turn)

        # 3. Entity extraction
        entities = _extract_entities(turn.content)

        # 4. Embed + semantic deduplication
        embedding = self._embedder.embed(turn.content)
        dup = self._learned_store.find_duplicate(embedding, threshold=0.85)

        entry_id = f"dialogue:{turn.id}"
        meta = {
            "role": turn.role,
            "source": "dialogue",
            "turn_id": turn.id,
            "saw_index": turn.saw_index,
        }
        if turn.partner:
            meta["partner"] = turn.partner
        if extra_metadata:
            meta.update(extra_metadata)

        if dup:
            # Cross-category merge would clobber the structured category
            # (taught skill absorbed into a dialogue fragment becomes
            # invisible to category-scoped check_skill) — dedup only
            # within the same category.
            dup_cat = dup.get("category") or self._config.namespace
            target_cat = category or self._config.namespace
            if dup_cat != target_cat:
                dup = None

        if dup:
            try:
                # Merge: update content, bump weight, refresh timestamp
                new_weight = dup["weight"] + 0.05
                self._learned_store.update(
                    dup["entry_id"],
                    content=turn.content,
                    metadata={"weight": new_weight},
                )
                deduped = True
                entry_id = dup["entry_id"]
                weight = new_weight
            except KeyError:
                # dup referenced a stale id (entry deleted from chroma but
                # still indexed elsewhere) — fall through to a fresh write.
                import logging
                logging.getLogger(__name__).warning(
                    "Dedup target %s missing; writing fresh entry", dup["entry_id"])
                dup = None
        if not dup:
            # New entry with uniform weight
            entry = MemoryEntry(
                id=entry_id,
                content=turn.content,
                created_at=now,
                updated_at=now,
                weight=0.5,
                category=category or self._config.namespace,
                tags=entities,
                metadata=meta,
            )
            self._learned_store.insert(entry)
            deduped = False
            weight = 0.5

        # 5. Character double-write (Unit 4): reference the stored entry from
        # the bound role. Degrades to a log line — ingest must never be blocked.
        if character_role and self._character_store is not None:
            try:
                self._character_store.add_memory(character_role, entry_id)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "Character double-write failed for role=%s entry=%s: %s",
                    character_role, entry_id, exc)

        return IngestReceipt(
            entry_id=entry_id,
            deduped=deduped,
            weight=weight,
            staged={},
            timestamp=now,
        )

    def ingest_screen(self, frame_text: str) -> None:
        """Ingest a screen observation frame — noise-filter then route.
        
        1. Always store in SawRingBuffer
        2. NoiseFilter check:
           - If noise: only SawRingBuffer, skip LearnedStore
           - If kept and error: LearnedStore with high weight (0.9)
           - If kept and normal: LearnedStore with low weight (0.3)
        """
        now = datetime.now(UTC)

        # 1. Always store in SawRingBuffer
        self._saw_buffer.put(self._make_saw_entry(frame_text, now))

        # 2. Noise filter check
        if self._last_kept_screen is not None:
            if not self._noise_filter.should_keep(frame_text, self._last_kept_screen):
                return MemoryEnvelope(
                    type="saw", entries=[], truncated=False,
                    total_candidates=0, timestamp=now,
                )

        # Update last-kept screen (kept frames only)
        self._last_kept_screen = frame_text

        # Determine weight based on error detection
        has_error = self._noise_filter.is_error_frame(frame_text)
        weight = 0.9 if has_error else 0.3

        entry_id = f"screen:{self._heartbeat}"
        entry = MemoryEntry(
            id=entry_id,
            content=frame_text,
            created_at=now,
            updated_at=now,
            weight=weight,
            category=self._config.namespace,
            tags=["error"] if has_error else ["observation"],
            metadata={
                "source": "screen",
                "heartbeat_index": self._heartbeat,
                "has_error": has_error,
            },
        )
        self._learned_store.insert(entry)

    @property
    def gaps(self) -> list:
        """Open learning-queue items (skills/missions awaiting agent action)."""
        if self._learning_queue is None:
            return []
        try:
            return self._learning_queue.open_items()
        except Exception:
            return []

    # ── Private helpers ───────────────────────────────────────────────────

    def _make_saw_entry(self, content: str, timestamp: datetime) -> SawEntry:
        """Create a SawEntry with a monotonic heartbeat_index."""
        self._heartbeat += 1
        return SawEntry(
            heartbeat_index=self._heartbeat,
            content=content,
            timestamp=timestamp,
        )
