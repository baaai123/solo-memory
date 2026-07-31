"""Unit tests — no ChromaDB, no ONNX, no LLM."""

from datetime import UTC, datetime

import pytest

from memory_skill.contracts import (
    DialogueTurn,
    MemoryEntry,
    MemoryEnvelope,
    MemorySkillConfig,
    utcnow,
)


class TestContracts:
    def test_dialogue_turn(self):
        t = DialogueTurn(id="t1", role="user", content="hello", timestamp=utcnow())
        assert t.id == "t1"
        assert t.role == "user"
        assert t.content == "hello"

    def test_memory_entry(self):
        now = datetime.now(UTC)
        e = MemoryEntry(id="e1", content="test", created_at=now, updated_at=now,
                        weight=0.5, category="default", tags=[], metadata={})
        assert e.id == "e1"
        assert e.weight == 0.5
        assert e.category == "default"

    def test_memory_envelope(self):
        now = datetime.now(UTC)
        env = MemoryEnvelope(type="recall", entries=[], truncated=False,
                            total_candidates=0, timestamp=now)
        assert env.type == "recall"
        assert env.entries == []

    def test_config_defaults(self):
        cfg = MemorySkillConfig(db_path=":memory:", agent_name="test")
        assert cfg.agent_name == "test"
        assert cfg.tree_enabled is True
        assert cfg.saw_buffer_capacity == 1000


class TestSawBuffer:
    def test_put_and_get(self):
        from memory_skill.saw_buffer import SawRingBuffer
        buf = SawRingBuffer(capacity=5)
        buf.put(SawEntryHelper(0, "hello"))
        buf.put(SawEntryHelper(1, "world"))
        all_items = buf.get_all()
        assert len(all_items) == 2
        assert "hello" in str(all_items)
        assert "world" in str(all_items)

    def test_capacity_overflow(self):
        from memory_skill.saw_buffer import SawRingBuffer
        buf = SawRingBuffer(capacity=3)
        for i in range(5):
            buf.put(SawEntryHelper(i, f"msg{i}"))
        all_items = buf.get_all()
        assert len(all_items) == 3
        contents = [str(x) for x in all_items]
        assert "msg2" in contents[-1] or "msg4" in contents[-1]

    def test_empty_buffer(self):
        from memory_skill.saw_buffer import SawRingBuffer
        buf = SawRingBuffer(capacity=5)
        assert len(buf.get_all()) == 0


class TestImportance:
    def test_high_importance(self):
        from memory_skill.importance import ImportanceScorer
        scorer = ImportanceScorer()
        score, persist = scorer.evaluate("我喜欢冰美式，每天下午都要来一杯")
        assert persist is True
        assert score >= scorer.threshold

    def test_low_importance(self):
        from memory_skill.importance import ImportanceScorer
        scorer = ImportanceScorer()
        score, persist = scorer.evaluate("ok")
        assert persist is False

    def test_trivial_emoji(self):
        from memory_skill.importance import ImportanceScorer
        scorer = ImportanceScorer()
        score, persist = scorer.evaluate("哈哈哈")
        assert persist is False


class TestStructuredExtractor:
    def test_prompt_includes_examples(self):
        from memory_skill.structured_extractor import _EXTRACT_PROMPT
        assert "冰美式" in _EXTRACT_PROMPT
        assert "简洁" in _EXTRACT_PROMPT
        assert "pref" in _EXTRACT_PROMPT
        assert "skill" in _EXTRACT_PROMPT
        assert "mission" in _EXTRACT_PROMPT
        assert "pers" in _EXTRACT_PROMPT

    def test_prompt_formats_content(self):
        from memory_skill.structured_extractor import _EXTRACT_PROMPT
        prompt = _EXTRACT_PROMPT.format(content="test message")
        assert "test message" in prompt
        assert '"type":"none"' in prompt


class TestNoiseFilter:
    def test_identical_frames(self):
        from memory_skill.ingestor import _ScreenNoiseFilter
        nf = _ScreenNoiseFilter()
        assert nf.should_keep("hello world", "hello world") is False

    def test_different_frames(self):
        from memory_skill.ingestor import _ScreenNoiseFilter
        nf = _ScreenNoiseFilter()
        assert nf.should_keep("hello world", "goodbye moon") is True

    def test_error_frame_always_kept(self):
        from memory_skill.ingestor import _ScreenNoiseFilter
        nf = _ScreenNoiseFilter()
        assert nf.is_error_frame("Error: connection refused") is True
        assert nf.is_error_frame("everything is fine") is False


def SawEntryHelper(index, text):
    """Minimal SawEntry-like object for testing."""
    from dataclasses import dataclass
    @dataclass
    class _Saw:
        index: int
        text: str
    return _Saw(index, text)
