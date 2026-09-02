"""
Embedder degraded-mode signaling tests.

Verifies that a missing ONNX model no longer fails silently: the Embedder
exposes ``mode`` / ``degraded`` / ``reason`` so callers (health report,
status tool, weave banner) can surface the SHA-256 fallback to users.

No real ONNX model is required — every test here exercises the
fallback-enabled path with a bogus model path (deterministic, no network).
"""

from __future__ import annotations

import pytest

from memory_skill.contracts import MemorySkillConfig
from memory_skill.embedder import DEFAULT_QUERY_INSTRUCTION, Embedder, clear_cache


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    """Isolate singleton-cache state between tests."""
    clear_cache()
    yield
    clear_cache()


def _make_embedder(model_path: str, fallback_enabled: bool = True) -> Embedder:
    cfg = MemorySkillConfig(
        db_path=":memory:",
        model_path=model_path,
    )
    return Embedder(cfg, fallback_enabled=fallback_enabled)


class TestEmbedQuery:
    """Query-side bge instruction prefix (asymmetric retrieval)."""

    def test_embed_query_prepends_default_instruction(self) -> None:
        emb = _make_embedder("/nonexistent/model.onnx")
        assert emb.embed_query("hello") == emb.embed(
            f"{DEFAULT_QUERY_INSTRUCTION}hello"
        )

    def test_embed_query_disabled_when_instruction_empty(self) -> None:
        cfg = MemorySkillConfig(
            db_path=":memory:",
            model_path="/nonexistent/model.onnx",
            query_instruction="",
        )
        emb = Embedder(cfg)
        assert emb.embed_query("hello") == emb.embed("hello")

    def test_embed_query_custom_instruction(self) -> None:
        cfg = MemorySkillConfig(
            db_path=":memory:",
            model_path="/nonexistent/model.onnx",
            query_instruction="custom: ",
        )
        emb = Embedder(cfg)
        assert emb.embed_query("hello") == emb.embed("custom: hello")

    def test_embed_query_empty_text_is_zero_vector(self) -> None:
        emb = _make_embedder("/nonexistent/model.onnx")
        assert emb.embed_query("") == [0.0] * emb._dim

    def test_embed_query_differs_from_document_embed(self) -> None:
        emb = _make_embedder("/nonexistent/model.onnx")
        assert emb.embed_query("hello") != emb.embed("hello")


class TestDegradedMode:
    def test_fallback_signals_degraded(self) -> None:
        """Missing model + fallback enabled → mode/degraded/reason all signal it."""
        emb = _make_embedder("/nonexistent/model.onnx", fallback_enabled=True)
        assert emb.mode == "fallback"
        assert emb.degraded is True
        assert isinstance(emb.reason, str) and emb.reason.strip(), \
            f"reason should be a non-empty error string, got {emb.reason!r}"

    def test_degraded_triggers_lazy_load(self) -> None:
        """Reading `degraded` triggers the (once-only) model load attempt."""
        emb = _make_embedder("/nonexistent/model.onnx")
        assert emb._load_attempted is False
        assert emb.degraded is True
        assert emb._load_attempted is True

    def test_embed_still_works_while_degraded(self) -> None:
        """Degraded mode keeps the deterministic fallback embedding."""
        emb = _make_embedder("/nonexistent/model.onnx")
        vec = emb.embed("hello")
        assert len(vec) == emb._dim
        assert emb.degraded is True

    def test_no_fallback_raises_instead(self) -> None:
        """fallback_enabled=False still raises ModelLoadError (unchanged behavior)."""
        from memory_skill.contracts import ModelLoadError

        emb = _make_embedder("/nonexistent/model.onnx", fallback_enabled=False)
        with pytest.raises(ModelLoadError):
            emb.embed("hello")

    def test_reason_none_before_load(self) -> None:
        """A never-loaded embedder has no error recorded yet."""
        emb = _make_embedder("/nonexistent/model.onnx")
        assert emb.reason is None

    def test_singleton_cache_preserves_reason(self) -> None:
        """A cached embedder (same model_path) keeps the degraded reason."""
        first = _make_embedder("/nonexistent/shared_model.onnx")
        first.embed("trigger load")
        assert first.degraded is True

        second = _make_embedder("/nonexistent/shared_model.onnx")
        assert second.degraded is True
        assert second.mode == "fallback"
        assert second.reason == first.reason
        assert isinstance(second.reason, str) and second.reason.strip()
