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
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from memory_skill.contracts import (
    DialogueStoreProtocol,
    DialogueTurn,
    EmbedderProtocol,
    LearnedStoreProtocol,
    MemoryEntry,
    SawBufferProtocol,
    SawEntry,
    TreeManagerProtocol,
)
import numpy as np

if TYPE_CHECKING:
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


_QUESTION_KW = ("?", "吗", "呢", "什么", "怎么", "如何", "哪", "谁", "多少", "在哪")


def _looks_like_question(content: str) -> bool:
    s = content.strip()
    return "?" in s or s.endswith(("吗", "呢")) or any(kw in s for kw in _QUESTION_KW[3:])


class _ScreenNoiseFilter:
    _ERROR_PATTERN = re.compile(
        r"Error|Exception|FATAL|Traceback|TypeError|Connection refused",
        re.IGNORECASE,
    )

    def __init__(self, threshold: float = 0.85, ngram: int = 3):
        self._threshold = threshold
        self._ngram = ngram

    def is_error_frame(self, text: str) -> bool:
        return bool(self._ERROR_PATTERN.search(text))

    def should_keep(self, current: str, last: str) -> bool:
        if self._ERROR_PATTERN.search(current):
            return True
        if current == last:
            return False
        if not current and last:
            return True
        if current and not last:
            return True
        if not current and not last:
            return False
        return self._cosine(current, last) < self._threshold

    def _cosine(self, a: str, b: str) -> float:
        na = self._ngrams(a)
        nb = self._ngrams(b)
        if not na or not nb:
            return 0.0
        vocab = sorted(set(na) | set(nb))
        idx = {ng: i for i, ng in enumerate(vocab)}
        va = np.zeros(len(vocab), dtype=np.float64)
        vb = np.zeros(len(vocab), dtype=np.float64)
        for ng in na:
            va[idx[ng]] += 1.0
        for ng in nb:
            vb[idx[ng]] += 1.0
        dot = float(np.dot(va, vb))
        na_norm, nb_norm = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        return dot / (na_norm * nb_norm) if na_norm and nb_norm else 0.0

    def _ngrams(self, text: str) -> list[str]:
        t = text.lower()
        n = self._ngram
        return [t[i:i + n] for i in range(len(t) - n + 1)] if len(t) >= n else []


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
# Ordered from strongest to weakest break preference.
_SENTENCE_BOUNDARIES: tuple[str, ...] = (
    ". ", "! ", "? ", ".\n", "!\n", "?\n", "\n\n", "\n", ".", "!", "?",
)


def _chunk_text(
    text: str,
    max_tokens: int = 512,
    stride_tokens: int = 256,
) -> list[str]:
    """Split long text into overlapping chunks for embedding.

    When a message exceeds ``max_tokens`` tokens (≈ ``max_tokens * 4``
    characters), it is split into overlapping chunks with ``stride_tokens``
    stride.  Each chunk is at most ~512 tokens.  Consecutive chunks overlap
    by ~50% (256 tokens).

    Short messages (≤ max_tokens) are returned as-is — a single chunk.

    Chunk boundaries are aligned to sentence-ending punctuation when possible.

    Parameters
    ----------
    text:
        The message text to chunk.
    max_tokens:
        Maximum approximate token count per chunk (default 512).
    stride_tokens:
        Stride (step size) in approximate tokens between chunk start
        positions (default 256 → 50% overlap).

    Returns
    -------
    list[str]
        One or more text chunks.  Short messages return a single-element list.
    """
    if not text:
        return [text]

    max_chars = max_tokens * _CHARS_PER_TOKEN
    stride_chars = stride_tokens * _CHARS_PER_TOKEN

    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    pos: int = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))

        # If not at the final chunk, try to refine the boundary to a
        # sentence-ending position.
        if end < len(text):
            window = text[pos:end]
            for sep in _SENTENCE_BOUNDARIES:
                idx = window.rfind(sep)
                if idx >= 0:
                    end = pos + idx + len(sep)
                    break

        chunk = text[pos:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        pos += stride_chars

    return chunks


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
    """

    def __init__(
        self,
        config: MemorySkillConfig,
        saw_buffer: SawBufferProtocol,
        dialogue_store: DialogueStoreProtocol,
        learned_store: LearnedStoreProtocol,
        embedder: EmbedderProtocol,
        tree: TreeManagerProtocol | None = None,
        gap_detector=None,
    ) -> None:
        self._config = config
        self._saw_buffer = saw_buffer
        self._dialogue_store = dialogue_store
        self._learned_store = learned_store
        self._noise_filter = _ScreenNoiseFilter()
        self._embedder = embedder
        self._tree = tree
        self._gap_detector = gap_detector

        # Monotonic heartbeat counter for SawRingBuffer entries
        self._heartbeat: int = 0

        # Last-kept screen frame for noise filter comparison
        self._last_kept_screen: str | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def ingest_dialogue(self, turn: DialogueTurn, category: str | None = None) -> None:
        """Ingest a dialogue turn — all content goes to both stores.

        1. Store in SawRingBuffer (always)
        2. Store in DialogueStore (always)
        3. Embed → semantic dedup → merge (weight+0.05) or insert (weight=0.5)
        4. Store in LearnedStore (always)
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

        if dup:
            # Merge: update content, bump weight, refresh timestamp
            new_weight = dup["weight"] + 0.05
            self._learned_store.update(
                dup["entry_id"],
                content=turn.content,
                metadata={"weight": new_weight},
            )
        else:
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

            if (
                self._gap_detector is not None
                and turn.role in ("user",)
                and _looks_like_question(turn.content)
            ):
                try:
                    self._gap_detector.detect(turn.content)
                except Exception as e:
                    _logger.warning("Gap detection failed for %s: %s", entry_id, e)

        from memory_skill.contracts import MemoryEnvelope
        return MemoryEnvelope(
            type="ingest", entries=[], truncated=False,
            total_candidates=0, timestamp=now,
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

    # ── Private helpers ───────────────────────────────────────────────────

    def _make_saw_entry(self, content: str, timestamp: datetime) -> SawEntry:
        """Create a SawEntry with a monotonic heartbeat_index."""
        self._heartbeat += 1
        return SawEntry(
            heartbeat_index=self._heartbeat,
            content=content,
            timestamp=timestamp,
        )
