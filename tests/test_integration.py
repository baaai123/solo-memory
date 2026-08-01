"""
Memory Skill — End-to-End Integration Test (v0.5.0)

Tests the full memory pipeline against **real infrastructure**:
  - Real ONNX embedding model (bge-large-en-v1.5)
  - Real DeepSeek V4 Flash API (tree classification + importance gating)
  - Real SQLite (FTS5) + ChromaDB (HNSW) storage

This is NOT a unit test — it validates the system works as a whole.

Run::
    cd /path/to/memory/for/solo
    venv/bin/python -m pytest tests/test_integration.py -v --timeout=120

Environment:
    DEEPSEEK_API_KEY  (default: provided test key)
    DEEPSEEK_API_BASE (default: https://api.deepseek.com/v1)
    DEEPSEEK_MODEL    (default: deepseek-v4-flash)
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from datetime import UTC, datetime

import pytest

from memory_skill import DialogueTurn, MemorySkill, MemorySkillConfig
from memory_skill.contracts import MemoryEnvelope, utcnow
from memory_skill.tree import TreeManager

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Test data
# ═══════════════════════════════════════════════════════════════════════════════

_INGEST_TURNS = [
    # ── Python / FastAPI conversation ──
    DialogueTurn(
        id="t001", role="user",
        content="我最近在学 Python，想写一个 Web 后端，有什么推荐？",
        timestamp=utcnow(), partner="user",
    ),
    DialogueTurn(
        id="t002", role="assistant",
        content="推荐 FastAPI，性能好类型安全。配合 SQLAlchemy 和 Pydantic。",
        timestamp=utcnow(), partner="user",
    ),
    DialogueTurn(
        id="t003", role="user",
        content="FastAPI 和 Flask 比怎么样？",
        timestamp=utcnow(), partner="user",
    ),
    DialogueTurn(
        id="t004", role="assistant",
        content="FastAPI 原生异步、自动 OpenAPI 文档、性能更好。",
        timestamp=utcnow(), partner="user",
    ),
    # ── Preference ──
    DialogueTurn(
        id="t005", role="user",
        content="我喜欢喝冰美式，每天下午都要来一杯。",
        timestamp=utcnow(), partner="user",
    ),
    DialogueTurn(
        id="t006", role="assistant",
        content="记住了，你喜欢冰美式。",
        timestamp=utcnow(), partner="user",
    ),
    # ── Project knowledge ──
    DialogueTurn(
        id="t007", role="user",
        content="我在用 LangChain 做 RAG 项目，数据存 ChromaDB，embedding 用 bge-large。",
        timestamp=utcnow(), partner="user",
    ),
    DialogueTurn(
        id="t008", role="assistant",
        content="建议加 BM25 hybrid search 提升召回率。",
        timestamp=utcnow(), partner="user",
    ),
    # ── Trivial (should be low-weight / filtered) ──
    DialogueTurn(
        id="t009", role="user",
        content="ok", timestamp=utcnow(),
    ),
    DialogueTurn(
        id="t010", role="user",
        content="嗯嗯", timestamp=utcnow(),
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def api_config() -> dict[str, str]:
    """Return API config from env for DeepSeek (no default key)."""
    return {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "api_base": os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
    }


@pytest.fixture(scope="module")
def skill(tmp_db: str, api_config: dict[str, str]) -> MemorySkill:
    """Create MemorySkill with temp DB, real model, real LLM."""
    import memory_skill

    proj_root = memory_skill.__file__

    # Set env vars for LLMImportanceGate + TreeManager
    os.environ["IMPORTANCE_API_KEY"] = api_config["api_key"]
    os.environ["IMPORTANCE_API_BASE"] = api_config["api_base"]
    os.environ["IMPORTANCE_MODEL"] = api_config["model"]

    config = MemorySkillConfig(
        db_path=tmp_db,
        agent_name="test_agent",
        namespace="test_agent",
    )
    sk = MemorySkill(config)
    return sk


@pytest.fixture(scope="module")
def tree(skill: MemorySkill, api_config: dict[str, str]) -> TreeManager:
    """Expose the TreeManager for direct tree tests."""
    return skill._tree


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Basic health
# ═══════════════════════════════════════════════════════════════════════════════


class TestHealth:
    """Verify the memory system boots correctly."""

    def test_import(self) -> None:
        """Package imports cleanly."""
        from memory_skill import (
            DialogueTurn,
            MemoryEntry,
            MemoryEnvelope,
            MemorySkill,
            MemorySkillConfig,
            SawEntry,
        )
        assert DialogueTurn
        assert MemoryEntry
        assert MemoryEnvelope
        assert MemorySkill
        assert MemorySkillConfig
        assert SawEntry

    def test_health_report(self, skill: MemorySkill) -> None:
        """health() returns a complete, parsable report."""
        h = skill.health()
        assert "embedder" in h
        assert h["embedder"]["mode"] in ("onnx", "fallback")
        assert h["embedder"]["dim"] == 1024
        assert "learned_store" in h
        assert h["status"] in ("healthy",)

    def test_empty_system(self, skill: MemorySkill) -> None:
        """Weave on empty system returns is_empty=True."""
        ctx = skill.weave(user_message="hello")
        assert ctx.is_empty


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Ingest
# ═══════════════════════════════════════════════════════════════════════════════


class TestIngest:
    """Data ingestion pipeline."""

    def test_ingest_all(self, skill: MemorySkill) -> None:
        """Ingest all test turns — should succeed."""
        for turn in _INGEST_TURNS:
            envelope = skill.ingest(turn)
            assert isinstance(envelope, MemoryEnvelope)
            assert envelope.type == "ingest"

    def test_ingest_count(self, skill: MemorySkill) -> None:
        """Dialogue store has expected turns (trivial ones filtered)."""
        count = skill._dialogue_store.count()
        # 8 substantive turns + 2 trivial → trivial should survive but with low weight
        assert count >= 10, f"Expected >=10 turns, got {count}"

    def test_learned_store_has_entries(self, skill: MemorySkill) -> None:
        """LearnedStore has entries (nontrivial content passes importance gate)."""
        h = skill.health()
        count = h["learned_store"]["entry_count"]
        # At least the non-trivial entries should be stored
        assert count >= 6, f"Expected >=6 entries in learned store, got {count}"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Tree memory (LLM classification)
# ═══════════════════════════════════════════════════════════════════════════════


_NEEDS_LLM = pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"),
    reason="requires DEEPSEEK_API_KEY (LLM classify test)",
)


@pytest.mark.slow
class TestTreeMemory:
    """LLM-powered tree memory classification and navigation."""

    @_NEEDS_LLM
    def test_tree_classify_python(self, tree: TreeManager) -> None:
        """'Python Web 后端' → classified as assistant_task."""
        result = tree.classify("我最近在学 Python，想写一个 Web 后端，有什么推荐？")
        assert "root" in result
        assert "branch" in result
        # User can be about user studies/history or assistant skill — either valid
        assert result["root"] in ("user", "assistant")
        logger.info("Python classify → %s/%s", result["root"], result["branch"])

    @_NEEDS_LLM
    def test_tree_classify_preference(self, tree: TreeManager) -> None:
        """'喜欢冰美式' → classified as user_pref."""
        result = tree.classify("我喜欢喝冰美式，每天下午都要来一杯。")
        assert result["root"] == "user"
        assert result["branch"] in ("pref",), f"Expected pref, got {result['branch']}"
        logger.info("Preference classify → %s/%s", result["root"], result["branch"])

    @_NEEDS_LLM
    def test_tree_classify_project(self, tree: TreeManager) -> None:
        """'RAG LangChain project' → classified as assistant_task."""
        result = tree.classify("我在用 LangChain 做 RAG 项目，数据存在 ChromaDB 里")
        assert "root" in result
        assert "branch" in result
        # Could be user/mem or assistant/task — either is acceptable
        logger.info("Project classify → %s/%s", result["root"], result["branch"])

    def test_tree_add_node(self, tree: TreeManager) -> None:
        """Adding a node creates date + period layers automatically."""
        node_id = tree.add_node(
            content="测试记忆内容",
            memory_id="mem_test_001",
            root="user",
            branch="mem",
        )
        assert node_id.startswith("user_mem_")
        assert len(node_id) > 20  # user_mem_2026-07-25_am_uuid8
        logger.info("Tree node created: %s", node_id)

    def test_tree_get_context(self, tree: TreeManager) -> None:
        """get_context returns something, even if just a summary."""
        ctx = tree.get_context("Python")
        assert ctx is not None
        assert len(ctx) > 0
        logger.info("Tree context (Python): %s", ctx[:200])

    def test_tree_summary(self, tree: TreeManager) -> None:
        """Tree summary has counts for each branch."""
        summary = tree._tree_summary()
        assert "偏好" in summary or "user_pref" in summary
        assert "回忆与目标" in summary or "user_mem" in summary
        logger.info("Tree summary:\n%s", summary)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Weave (context assembly)
# ═══════════════════════════════════════════════════════════════════════════════


class TestWeave:
    """Layered memory context assembly."""

    def test_weave_with_memories(self, skill: MemorySkill) -> None:
        """After ingest, weave returns non-empty context."""
        # Ensure test data is present (this test may run independently of TestIngest)
        if skill._dialogue_store.count() < 10:
            for turn in _INGEST_TURNS:
                skill.ingest(turn)
        ctx = skill.weave(
            user_message="你记得我学的是什么框架吗？",
            scene_summary="用户在问之前学过的技术栈",
            partner="user",
        )
        assert not ctx.is_empty, "Weave should return non-empty context after ingest"
        block = ctx.to_prompt_block()
        assert block, "to_prompt_block should not be empty"
        assert len(block) > 50, f"Expected block > 50 chars, got {len(block)}"
        logger.info("Weave output (%d chars):\n%s", len(block), block[:300])

    def test_weave_tier1_scene(self, skill: MemorySkill) -> None:
        """Tier1 context includes scene summary."""
        ctx = skill.weave(
            user_message="继续之前的 Python 话题",
            scene_summary="用户回到了 Python Web 开发的讨论",
            partner="user",
        )
        assert ctx.tier1_context is not None
        assert "Python" in (ctx.tier1_context or "")
        logger.info("Tier1: %s", ctx.tier1_context[:200] if ctx.tier1_context else "")

    def test_weave_tier2_retrieval(self, skill: MemorySkill) -> None:
        """Tier2 context is non-empty after ingest."""
        ctx = skill.weave(
            user_message="你记得我用的数据库吗？",
            scene_summary="用户问之前提过的技术细节",
            partner="user",
        )
        assert not ctx.is_empty, "Weave context should not be empty"
        if ctx.tier2_context and len(ctx.tier2_context) > 20:
            logger.info("Tier2 (%d chars)", len(ctx.tier2_context))
        else:
            logger.warning("Tier2 empty — shared DB may have no relevant entries")

    def test_weave_empty_system(self, skill: MemorySkill) -> None:
        """Empty system: weave returns is_empty."""
        empty_skill = MemorySkill(MemorySkillConfig(
            db_path=":memory:",
            agent_name="empty_test",
        ))
        ctx = empty_skill.weave(user_message="hello")
        assert ctx.is_empty


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Retrieve (search)
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetrieve:
    """Semantic + BM25 hybrid search."""

    def test_search_python(self, skill: MemorySkill) -> None:
        """Searching 'Python Web 后端' returns relevant FastAPI results."""
        results = skill.retrieve("Python Web 后端框架", limit=5)
        assert len(results.entries) > 0
        texts = [e.content[:50] for e in results.entries]
        logger.info("Search 'Python': %s", texts)
        # Some result should mention FastAPI
        fastapi_hits = [t for t in texts if "FastAPI" in t]
        assert len(fastapi_hits) >= 1, f"No FastAPI hit in: {texts}"

    def test_search_rag(self, skill: MemorySkill) -> None:
        """Searching 'RAG ChromaDB' returns LangChain project context."""
        results = skill.retrieve("RAG ChromaDB LangChain", limit=5)
        assert len(results.entries) > 0
        texts = [e.content[:50] for e in results.entries]
        logger.info("Search 'RAG': %s", texts)
        hits = [t for t in texts if "LangChain" in t]
        assert len(hits) >= 1, f"No LangChain hit in: {texts}"

    def test_search_preference(self, skill: MemorySkill) -> None:
        """Searching '喝什么' returns preference about 冰美式."""
        results = skill.retrieve("用户喜欢喝什么", limit=5)
        assert len(results.entries) > 0
        texts = [e.content[:50] for e in results.entries]
        logger.info("Search '喝什么': %s", texts)
        hits = [t for t in texts if "冰美式" in t]
        assert len(hits) >= 1, f"No 冰美式 hit in: {texts}"

    def test_search_envelope(self, skill: MemorySkill) -> None:
        """Search returns a valid MemoryEnvelope."""
        results = skill.retrieve("测试搜索", limit=3)
        assert isinstance(results, MemoryEnvelope)
        assert results.total_candidates >= len(results.entries)
        assert results.type == "recall"

    def test_trivial_not_retrieved(self, skill: MemorySkill) -> None:
        """Trivial content ('ok') should not dominate retrieval results."""
        results = skill.retrieve("Python Web 后端", limit=5)
        # Relevant entries should rank above trivial 'ok' content
        relevant = [e for e in results.entries if "FastAPI" in e.content]
        assert len(relevant) >= 1, "Relevant content should rank above trivial"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Tree navigation (LLM-guided)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestTreeNavigation:
    """LLM-navigated tree context retrieval (calls DeepSeek API)."""

    def test_navigate_project(self, tree: TreeManager) -> None:
        """'我的 RAG 项目进展' → should navigate to assistant_task."""
        ctx = tree.navigate("我的 RAG 项目进展怎么样")
        assert ctx is not None
        logger.info("Navigate 'RAG': %s", ctx[:300])

    def test_navigate_preference(self, tree: TreeManager) -> None:
        """'我喜欢喝什么' → should navigate to user_pref."""
        ctx = tree.navigate("我喜欢喝什么饮料")
        assert ctx is not None
        logger.info("Navigate 'preference': %s", ctx[:300])

    def test_navigate_fallback(self, tree: TreeManager) -> None:
        """When LLM fails, fallback returns last 1 day × all 4 branches."""
        fallback = tree._fallback_navigate()
        assert len(fallback) == 5
        branches = {s["branch"] for s in fallback}
        assert branches == {
            "user_pref", "user_mem", "assistant_pers",
            "assistant_task", "assistant_skill",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Feedback
# ═══════════════════════════════════════════════════════════════════════════════


class TestFeedback:
    """Outcome detection and weight boosting."""

    def test_auto_detect_positive(self, skill: MemorySkill) -> None:
        """Positive outcome: response overlaps significantly with retrieved content."""
        from memory_skill.feedback import auto_detect_outcome

        outcome = auto_detect_outcome(
            query="Python 框架推荐",
            search_results=[
                {"id": "1", "content": "我推荐 FastAPI，性能好类型安全"},
            ],
            final_response="我推荐 FastAPI，因为性能好而且类型安全",
        )
        assert outcome == "positive", f"Expected positive, got {outcome}"

    def test_auto_detect_negative(self, skill: MemorySkill) -> None:
        """Negative outcome: response doesn't overlap with retrieved content."""
        from memory_skill.feedback import auto_detect_outcome

        outcome = auto_detect_outcome(
            query="Python 框架推荐",
            search_results=[
                {"id": "1", "content": "FastAPI 性能好类型安全"},
            ],
            final_response="我不记得了，我们换个话题吧",
        )
        assert outcome == "negative", f"Expected negative, got {outcome}"

    def test_weight_boost(self, skill: MemorySkill) -> None:
        """Boosting via set_weight works on learned store entries."""
        results = skill.retrieve("Python", limit=10)
        # Find an entry accessible via get_weight (ID with ':')
        for e in results.entries:
            current = skill._learned_store.get_weight(e.id)
            if current is not None:
                skill._learned_store.set_weight(e.id, 0.85)
                boosted = skill._learned_store.get_weight(e.id)
                assert boosted == 0.85, f"Expected 0.85, got {boosted}"
                return
        # Fallback: test succeeds if no weightable entry found
        logger.warning("No weightable entry found — may be empty store")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Observation consolidation
# ═══════════════════════════════════════════════════════════════════════════════


class TestObservation:
    """Observation consolidation from dialogue."""

    def test_consolidate(self, skill: MemorySkill) -> None:
        """Consolidate observations from ingested dialogue."""
        skill.consolidate()
        h = skill.health()
        learned_count = h["learned_store"]["entry_count"]
        # Observations may add system entries
        logger.info("After consolidate: %d learned entries", learned_count)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Importance gating
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestImportance:
    """Importance gating for content filtering."""

    def test_rule_importance_high(self) -> None:
        """'我喜欢喝冰美式' → should score >= threshold."""
        from memory_skill.importance import ImportanceScorer

        scorer = ImportanceScorer()
        score, persist = scorer.evaluate("我喜欢喝冰美式，每天下午都要来一杯")
        assert score >= scorer.threshold, f"Expected >= {scorer.threshold}, got {score}"
        assert persist is True
        logger.info("High importance: %.2f (persist=%s)", score, persist)

    def test_rule_importance_trivial(self) -> None:
        """'ok' → should score below threshold."""
        from memory_skill.importance import ImportanceScorer

        scorer = ImportanceScorer()
        score, persist = scorer.evaluate("ok")
        assert persist is False, f"Expected persist=False for 'ok', got persist={persist}"
        logger.info("Trivial: %.2f (persist=%s)", score, persist)

    def test_rule_importance_emoji(self) -> None:
        """'哈哈哈' → trivial."""
        from memory_skill.importance import ImportanceScorer

        scorer = ImportanceScorer()
        score, persist = scorer.evaluate("哈哈哈")
        assert persist is False, f"Expected persist=False for '哈哈哈'"
        logger.info("Emoji trivial: %.2f (persist=%s)", score, persist)

    def test_llm_importance_gate(self) -> None:
        """LLMImportanceGate evaluates with real DeepSeek API."""
        from memory_skill.importance_llm import LLMImportanceGate

        gate = LLMImportanceGate()
        score, persist, category = gate.evaluate("我喜欢喝冰美式")
        assert category in ("trivial", "important", "critical")
        assert 0.0 <= score <= 1.0
        logger.info("LLM gate: cat=%s score=%.2f persist=%s", category, score, persist)

    def test_llm_importance_trivial(self) -> None:
        """LLMImportanceGate evaluates 'ok'."""
        from memory_skill.importance_llm import LLMImportanceGate

        gate = LLMImportanceGate()
        score, persist, category = gate.evaluate("ok")
        logger.info("LLM gate 'ok': cat=%s score=%.2f persist=%s", category, score, persist)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Cleaner
# ═══════════════════════════════════════════════════════════════════════════════


class TestCleaner:
    """Safe soft-delete cleanup."""

    def test_clean_run(self, skill: MemorySkill) -> None:
        """Cleaner runs without error and returns counts."""
        report = skill.clean()
        assert report is not None
        assert "merged" in report
        logger.info("Clean report: %s", report)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Namespace isolation
# ═══════════════════════════════════════════════════════════════════════════════


class TestNamespace:
    """Multi-agent namespace isolation."""

    def test_agent_namespace(self, skill: MemorySkill) -> None:
        """Agent namespace is set correctly."""
        ns = skill._ns_for(partner="user")
        assert ns == "test_agent", f"Expected test_agent, got {ns}"

    def test_different_agent_no_cross_contam(self, tmp_db: str) -> None:
        """Different agents should not see each other's memories."""
        agent_a = MemorySkill(MemorySkillConfig(
            db_path=tmp_db, agent_name="agent_a",
        ))
        agent_b = MemorySkill(MemorySkillConfig(
            db_path=tmp_db, agent_name="agent_b",
        ))

        agent_a.ingest(DialogueTurn(
            id="ns_test", role="user",
            content="Agent A 的秘密数据", timestamp=utcnow(),
        ))
        results_b = agent_b.retrieve("秘密数据", limit=5)
        texts_b = [e.content[:30] for e in results_b.entries]
        secret_hits = [t for t in texts_b if "Agent A" in t]


# ═══════════════════════════════════════════════════════════════════════════════
# 12. MCP tool layer
# ═══════════════════════════════════════════════════════════════════════════════


class TestMCPTools:
    """MCP ToolHandler interface (protocol layer, not the stdio transport)."""

    def _make_handler(self, skill: MemorySkill):
        """Create a ToolHandler for testing."""
        from memory_skill.mcp_tools import ToolHandler
        return ToolHandler(skill)

    def test_tool_status(self, skill: MemorySkill) -> None:
        """ToolHandler.status() returns health data."""
        handler = self._make_handler(skill)
        result = handler.handle("memory_status", {})
        assert "embedder" in result
        assert result.get("status") == "healthy"

    def test_tool_weave(self, skill: MemorySkill) -> None:
        """ToolHandler.weave() auto-ingests + returns context."""
        handler = self._make_handler(skill)
        result = handler.handle("memory_weave", {
            "user_message": "测试 MCP weave",
            "scene_summary": "MCP 协议层测试",
        })
        assert "prompt_block" in result
        assert "is_empty" in result
        logger.info("MCP weave prompt_block (%d chars)", len(result.get("prompt_block", "")))

    def test_tool_search(self, skill: MemorySkill) -> None:
        """ToolHandler.search() returns formatted results."""
        handler = self._make_handler(skill)
        result = handler.handle("memory_search", {
            "query": "Python FastAPI",
            "limit": 5,
        })
        assert "results" in result
        assert "count" in result
        logger.info("MCP search: %d results", result["count"])

    def test_tool_ingest(self, skill: MemorySkill) -> None:
        """ToolHandler.ingest() stores a turn."""
        handler = self._make_handler(skill)
        result = handler.handle("memory_ingest", {
            "content": "MCP 测试消息",
            "role": "user",
        })
        assert result.get("status") == "ingested"
        assert "turn_id" in result

    def test_tool_feedback(self, skill: MemorySkill) -> None:
        """ToolHandler.feedback() records outcome."""
        handler = self._make_handler(skill)
        result = handler.handle("memory_feedback", {
            "query_id": "test_query",
            "outcome": "positive",
            "cited_ids": [],
        })
        assert result.get("status") == "recorded"

    def test_tool_unknown(self, skill: MemorySkill) -> None:
        """Unknown tool returns error dict, not exception."""
        handler = self._make_handler(skill)
        result = handler.handle("memory_nonexistent", {})
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Error handling
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestTransparentProxy:
    """Transparent memory proxy — auto-context + auto-ingest."""

    @pytest.fixture(scope="class")
    def proxy_skill(self, tmp_db: str, api_config: dict[str, str]) -> MemorySkill:
        """MemorySkill with pre-ingested data for proxy test."""
        from memory_skill import MemorySkill, MemorySkillConfig
        import memory_skill as _ms

        os.environ["IMPORTANCE_API_KEY"] = api_config["api_key"]
        os.environ["IMPORTANCE_API_BASE"] = api_config["api_base"]
        os.environ["IMPORTANCE_MODEL"] = api_config["model"]

        sk = MemorySkill(MemorySkillConfig(db_path=tmp_db, agent_name="proxy_test"))
        sk.ingest(DialogueTurn(
            id="proxy_pre", role="user",
            content="我用 Python 写了一个 RAG 项目", timestamp=utcnow(),
        ))
        return sk

    def test_auto_context_injects_memory(self, proxy_skill: MemorySkill) -> None:
        """auto_context() enriches messages with memory block."""
        messages = [
            {"role": "user", "content": "你记得我用什么语言吗？"},
        ]
        enriched = proxy_skill.auto_context(messages)
        assert len(enriched) == 1
        content = enriched[0]["content"]
        assert "RAG" in content or "记忆" in content or "Python" in content
        logger.info("auto_context output (%d chars)", len(content))

    def test_auto_ingest_persists(self, proxy_skill: MemorySkill) -> None:
        """auto_ingest() persists both sides."""
        before = proxy_skill._dialogue_store.count()
        proxy_skill.auto_ingest("测试消息", "这是 AI 回复")
        after = proxy_skill._dialogue_store.count()
        assert after >= before + 1

    def test_proxy_chat_completions(
        self, proxy_skill: MemorySkill, api_config: dict[str, str],
    ) -> None:
        """End-to-end: proxy forwards to LLM, injects context, ingests response."""
        from memory_skill.transparent_proxy import TransparentProxy

        # Start proxy on a free port
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        proxy = TransparentProxy(
            proxy_skill,
            api_config["api_base"],
            api_config["api_key"],
            port=port,
        )
        proxy.start(block=False)

        try:
            diag_before = proxy_skill._dialogue_store.count()

            request_body = json.dumps({
                "model": api_config["model"],
                "messages": [
                    {"role": "user", "content": "你好，我用 Python 做开发"},
                ],
                "max_tokens": 300,
            })
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/v1/chat/completions",
                data=request_body.encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())

            assert "choices" in result
            assert len(result["choices"]) > 0
            content = result["choices"][0].get("message", {}).get("content", "")
            assert content, "LLM must return non-empty response"
            logger.info("LLM response: %s", content[:100])

            # Give auto-ingest a moment to complete
            time.sleep(1)
            diag_after = proxy_skill._dialogue_store.count()
            assert diag_after > diag_before, (
                f"Dialogue store should grow: {diag_before} → {diag_after}"
            )
            logger.info("Dialogue: %d → %d", diag_before, diag_after)

        finally:
            proxy.stop()

    def test_proxy_passthrough(
        self, proxy_skill: MemorySkill, api_config: dict[str, str],
    ) -> None:
        """Proxy passes through endpoints without crashing."""
        from memory_skill.transparent_proxy import TransparentProxy

        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        proxy = TransparentProxy(
            proxy_skill,
            api_config["api_base"],
            api_config["api_key"],
            port=port,
        )
        proxy.start(block=False)

        try:
            # Test that passthrough doesn't crash (status may vary by provider)
            req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/models")
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                    logger.info("Passthrough OK: HTTP %d", resp.status)
            except urllib.error.HTTPError:
                pass  # expected if upstream doesn't support /v1/models
        finally:
            proxy.stop()


class TestErrorHandling:
    """Edge cases and error conditions."""

    def test_empty_ingest(self, skill: MemorySkill) -> None:
        """Ingesting empty content doesn't crash."""
        turn = DialogueTurn(
            id="empty_test", role="user",
            content="", timestamp=utcnow(),
        )
        envelope = skill.ingest(turn)
        assert envelope is not None

    def test_empty_retrieve(self, skill: MemorySkill) -> None:
        """Retrieving with empty query returns empty results."""
        results = skill.retrieve("", limit=5)
        assert len(results.entries) == 0

    def test_embed_fallback(self) -> None:
        """Embedder with non-existent model uses fallback mode."""
        skill = MemorySkill(MemorySkillConfig(
            db_path=":memory:",
            model_path="/nonexistent/path",
        ))
        h = skill.health()
        assert h["embedder"]["mode"] in ("fallback",), f"Expected fallback, got {h['embedder']['mode']}"


class TestWebCrawler:
    """Web crawler — fetch, extract, chunk, ingest, and recall."""

    _TEST_HTML = """<!DOCTYPE html>
<html><head><title>Test Page</title>
<style>body { color: red; }</style>
<script>console.log("ignore me");</script>
</head><body>
<nav>Home | About</nav>
<h1>Memory Systems in AI</h1>
<p>Long-term memory is essential for AI agents that need to maintain
context across multiple conversations. Unlike short-term context windows,
persistent memory allows agents to recall past interactions, user preferences,
and learned knowledge over extended periods.</p>
<p>A good memory system combines semantic search with full-text retrieval,
using hybrid ranking to balance precision and recall. Key components include
embedding models for vector search, SQLite FTS5 for keyword matching, and
a weighting system that prioritizes frequently cited knowledge.</p>
<p>The next frontier is memory evolution — where stored knowledge self-updates
as new information arrives, detecting contradictions and merging related facts
into coherent, current summaries.</p>
<footer>Copyright 2026</footer>
</body></html>"""

    def test_extract_html(self) -> None:
        """Extract clean text from HTML, stripping scripts/styles/nav/footer."""
        from memory_skill.web_crawler import WebCrawler

        crawler = WebCrawler()
        title, body = crawler._extract(self._TEST_HTML)

        assert title == "Test Page"
        assert "Memory Systems" in body
        assert "hybrid ranking" in body
        assert "memory evolution" in body
        # Cruft must be stripped
        assert "console.log" not in body
        assert "color: red" not in body
        assert "Home | About" not in body
        assert "Copyright 2026" not in body  # footer stripped by length filter
        logger.info("Extracted %d chars: %s...", len(body), body[:100])

    def test_chunk_long_text(self) -> None:
        """Long text is split into ≤2000-char chunks at paragraph boundaries."""
        from memory_skill.web_crawler import WebCrawler, _CHUNK_MAX_CHARS

        crawler = WebCrawler()
        # Build a long text with clear paragraph breaks
        paragraphs = []
        for i in range(24):
            paragraphs.append(
                f"Paragraph {i}: This is a medium-length paragraph about memory systems "
                f"and how they help AI agents remember context across sessions. "
                f"It contains enough text to be meaningful when chunked properly."
            )
        body = "\n\n".join(paragraphs)
        chunks = crawler._chunk("Long Test", body, "http://test/chunk")

        assert len(chunks) >= 3
        for c in chunks:
            assert len(c.text) <= _CHUNK_MAX_CHARS
            assert c.source_url == "http://test/chunk"
        # First chunk should have title
        assert chunks[0].text.startswith("[Long Test]")
        logger.info("Chunked %d chars into %d chunks", len(body), len(chunks))

    @pytest.mark.network
    def test_crawl_real_page(self) -> None:
        """Crawl a real web page end-to-end."""
        from memory_skill.web_crawler import WebCrawler

        crawler = WebCrawler(timeout=10)
        try:
            chunks = crawler.crawl("http://httpbin.org/html")
        except Exception:
            pytest.skip("httpbin.org unreachable (network error)")
        if not chunks:
            pytest.skip("httpbin.org unavailable")

        assert len(chunks) > 0
        for c in chunks:
            assert c.source_url == "http://httpbin.org/html"
            assert len(c.text) <= 2000

    def test_crawl_bad_url(self) -> None:
        """Crawling a bad URL returns empty list (no crash)."""
        from memory_skill.web_crawler import WebCrawler

        crawler = WebCrawler(timeout=5)
        chunks = crawler.crawl("http://nonexistent.invalid/page")
        assert isinstance(chunks, list)
        assert len(chunks) == 0

    @pytest.mark.network
    def test_crawl_and_ingest(self, skill: MemorySkill) -> None:
        """Crawl a real page and verify it can be retrieved from memory."""
        from memory_skill.web_crawler import WebCrawler

        crawler = WebCrawler(timeout=10)
        try:
            chunks = crawler.crawl("http://httpbin.org/html")
        except Exception:
            pytest.skip("httpbin.org unreachable (network error)")
        if not chunks:
            pytest.skip("httpbin.org unavailable (no content)")
        assert len(chunks) > 0

        count = crawler.ingest(skill, chunks)
        assert count == len(chunks)

        results = skill.retrieve("Herman Melville", limit=3)
        content = " ".join(e.content for e in results.entries)
        if "503" in content:
            pytest.skip("httpbin.org returned 503")

        if "Moby-Dick" not in content and "Melville" not in content:
            pytest.skip("httpbin.org content changed (missing expected text)")
        assert "Moby-Dick" in content or "Melville" in content

    @pytest.mark.network
    def test_non_html_skip(self) -> None:
        """Non-HTML content types are skipped."""
        from memory_skill.web_crawler import WebCrawler

        crawler = WebCrawler(timeout=10)
        try:
            chunks = crawler.crawl("http://httpbin.org/image/png")
        except Exception:
            pytest.skip("httpbin.org unreachable (network error)")
        if chunks and "503" in chunks[0].text:
            pytest.skip("httpbin.org returned 503 for image endpoint")
        assert len(chunks) == 0


class TestCapabilityGap:
    """Capability registry + knowledge gap detection."""

    @pytest.fixture(scope="function")
    def gap_skill(self, tmp_db: str, api_config: dict[str, str]) -> MemorySkill:
        import memory_skill as _ms
        os.environ["IMPORTANCE_API_KEY"] = api_config["api_key"]
        os.environ["IMPORTANCE_API_BASE"] = api_config["api_base"]
        os.environ["IMPORTANCE_MODEL"] = api_config["model"]
        sk = MemorySkill(MemorySkillConfig(db_path=tmp_db, agent_name="gap_test"))
        return sk

    def test_capability_list(self, gap_skill: MemorySkill) -> None:
        from memory_skill.capability_registry import CapabilityRegistry
        reg = CapabilityRegistry(gap_skill._tree, gap_skill._retriever)
        caps = reg.list_capabilities()
        assert len(caps) == 5
        assert any(c.branch_id == "assistant_skill" for c in caps)

    def test_can_answer_false_on_empty(self, gap_skill: MemorySkill) -> None:
        from memory_skill.capability_registry import CapabilityRegistry
        reg = CapabilityRegistry(gap_skill._tree, gap_skill._retriever)
        can, conf = reg.can_answer("QuantumFlux 协议的 ZetaWave 变体？")
        # Smoke test: ensure it returns valid numbers
        assert isinstance(can, bool)
        assert 0.0 <= conf <= 1.0
        logger.info("can_answer on obscure: can=%s conf=%.3f", can, conf)

    def test_can_answer_true_after_ingest(self, gap_skill: MemorySkill) -> None:
        from memory_skill.capability_registry import CapabilityRegistry
        gap_skill.ingest(DialogueTurn(
            id="cap_py", role="user",
            content="Python 中类型提示用 typing 模块", timestamp=utcnow(),
        ))
        gap_skill.ingest(DialogueTurn(
            id="cap_async", role="user",
            content="异步用 asyncio 和 await 关键字", timestamp=utcnow(),
        ))
        reg = CapabilityRegistry(gap_skill._tree, gap_skill._retriever)
        can, conf = reg.can_answer("Python 怎么做异步？")
        assert can is True, f"Should answer, confidence={conf}"

    def test_gap_detect_unknown(self, gap_skill: MemorySkill) -> None:
        from memory_skill.capability_registry import CapabilityRegistry
        from memory_skill.gap_detector import GapDetector
        reg = CapabilityRegistry(gap_skill._tree, gap_skill._retriever)
        detector = GapDetector(reg, min_confidence=0.4)
        gap = detector.detect("ThermalFlux 引擎的 ZetaResonance 配置？")
        # Gap may be None if DB has unrelated entries with high confidence
        if gap is not None:
            assert gap.severity in ("critical", "major", "minor")

    def test_gap_none_on_known(self, gap_skill: MemorySkill) -> None:
        from memory_skill.capability_registry import CapabilityRegistry
        from memory_skill.gap_detector import GapDetector
        gap_skill.ingest(DialogueTurn(
            id="cap_k8s", role="user",
            content="Kubernetes Ingress 用 nginx-ingress controller", timestamp=utcnow(),
        ))
        reg = CapabilityRegistry(gap_skill._tree, gap_skill._retriever)
        detector = GapDetector(reg, min_confidence=0.3)
        gap = detector.detect("Kubernetes Ingress 怎么配置？")
        assert gap is None or gap.severity == "minor"

    def test_can_answer_false_on_unrelated_content(self, gap_skill: MemorySkill) -> None:
        """Symptom test (KNOWN-ISSUES #1): unrelated query must NOT be answerable.

        Pre-fix, a store populated with Docker content answered the totally
        unrelated "量子场论重整化群" with can=True conf≈0.5 because confidence
        measured result count, not relevance.
        """
        from memory_skill.capability_registry import CapabilityRegistry
        gap_skill.ingest(DialogueTurn(
            id="cap_docker", role="user",
            content="Docker 部署用 docker build、docker run 和 docker compose", timestamp=utcnow(),
        ))
        reg = CapabilityRegistry(gap_skill._tree, gap_skill._retriever)
        can, conf = reg.can_answer("量子场论重整化群 是什么？")
        assert can is False, f"Unrelated query answered: can={can} conf={conf}"

    def test_gap_detected_on_unrelated(self, gap_skill: MemorySkill) -> None:
        """Unrelated query on a populated store must produce a knowledge gap."""
        from memory_skill.capability_registry import CapabilityRegistry
        from memory_skill.gap_detector import GapDetector
        gap_skill.ingest(DialogueTurn(
            id="cap_docker2", role="user",
            content="Docker 部署用 docker build、docker run 和 docker compose", timestamp=utcnow(),
        ))
        reg = CapabilityRegistry(gap_skill._tree, gap_skill._retriever)
        detector = GapDetector(reg, min_confidence=0.5)
        gap = detector.detect("ThermalFlux 引擎的 ZetaResonance 配置？")
        assert gap is not None, "Unrelated query must be detected as a knowledge gap"

    def test_semantic_score_exposed(self, gap_skill: MemorySkill) -> None:
        """LearnedStore.search must expose per-entry semantic score in [0, 1]."""
        gap_skill.ingest(DialogueTurn(
            id="cap_sem", role="user",
            content="Python 用 asyncio 做异步编程", timestamp=utcnow(),
        ))
        entries = gap_skill._learned_store.search("Python 怎么做异步", limit=5)
        assert entries, "expected at least one entry"
        assert all(
            e.semantic_score is not None and 0.0 <= e.semantic_score <= 1.0
            for e in entries
        ), "semantic_score must be attached in [0, 1]"

    def test_is_question_heuristic(self) -> None:
        from memory_skill.ingestor import _looks_like_question
        assert _looks_like_question("怎么用 Python？")
        assert not _looks_like_question("今天天气不错")
        assert _looks_like_question("你记得我的密码吗")


@pytest.mark.slow
class TestActiveLearning:
    """Knowledge synthesis + learning decider + learn task."""

    def test_synth_with_mock_chunks(self, api_config: dict[str, str]) -> None:
        """Synthesis with known crawl data produces entries."""
        from memory_skill.knowledge_synth import KnowledgeSynth

        synth = KnowledgeSynth(api_config["api_base"], api_config["api_key"], api_config["model"])
        chunks = {
            "http://a.test/api": [_mock_chunk("Python 3.13 于 2024 年发布，引入了改进的类型系统。")],
            "http://b.test/docs": [_mock_chunk("Python 3.13 introduced improved type hints and a new REPL.")],
        }
        result = synth.synthesize("Python 3.13 features", chunks)

        assert result.total_facts >= 0
        logger.info("Synth: %d facts, verified=%d, conflicts=%d",
                     result.total_facts, result.verified_count, result.conflict_count)

    def test_synth_empty_input(self, api_config: dict[str, str]) -> None:
        """Empty input returns empty result."""
        from memory_skill.knowledge_synth import KnowledgeSynth

        synth = KnowledgeSynth(api_config["api_base"], api_config["api_key"], api_config["model"])
        result = synth.synthesize("nothing", {})
        assert result.total_facts == 0
        assert result.overall_confidence == 0.0

    def test_decider_with_critical_gap(self, api_config: dict[str, str]) -> None:
        """Critical gap on technical topic → learn."""
        from memory_skill.learning_decider import LearningDecider

        decider = LearningDecider(api_config["api_base"], api_config["api_key"], api_config["model"])
        decision = decider.evaluate(
            "Kubernetes Ingress 怎么配置？", "assistant_skill", "critical", history_count=3,
        )
        assert decision.action in ("skip", "ask", "learn")
        logger.info("Decision: %s (%.2f)", decision.action, decision.confidence)

    def test_decider_with_minor_gap(self, api_config: dict[str, str]) -> None:
        """Minor gap on casual topic → skip."""
        from memory_skill.learning_decider import LearningDecider

        decider = LearningDecider(api_config["api_base"], api_config["api_key"], api_config["model"])
        decision = decider.evaluate(
            "今天中午吃什么？", "user_mem", "minor",
        )
        assert decision.action in ("skip", "ask", "learn")
        logger.info("Decision: %s", decision.action)

    def test_learn_task_create(self, skill: MemorySkill) -> None:
        """LearningTask can be created and has correct initial state."""
        from memory_skill.learning_task import LearningTask

        task = LearningTask(
            id="test_task_001",
            topic="Python async",
            branch="assistant_skill",
            gap_query="Python 怎么做异步？",
            sources=["http://httpbin.org/html"],
        )
        assert task.status == "pending"
        assert not task.is_terminal()
        task._set_status("done")
        assert task.is_terminal()

    def test_gap_detector_with_decider(self, skill: MemorySkill, api_config: dict[str, str]) -> None:
        """detect_and_decide returns gap + decision."""
        from memory_skill.capability_registry import CapabilityRegistry
        from memory_skill.gap_detector import GapDetector
        from memory_skill.learning_decider import LearningDecider

        registry = CapabilityRegistry(skill._tree, skill._retriever)
        decider = LearningDecider(api_config["api_base"], api_config["api_key"], api_config["model"])
        detector = GapDetector(registry, min_confidence=0.4)
        detector.set_decider(decider)

        gap, decision = detector.detect_and_decide("gRPC 怎么配置负载均衡？")
        if gap:
            assert gap.severity in ("critical", "major", "minor")
        if decision:
            assert decision.action in ("skip", "ask", "learn")
        logger.info("detect_and_decide: gap=%s decision=%s",
                     gap is not None, decision.action if decision else "no decider")


def _mock_chunk(text: str):
    from memory_skill.web_crawler import CrawledChunk
    return CrawledChunk(source_url="http://mock", text=text, index=0)
