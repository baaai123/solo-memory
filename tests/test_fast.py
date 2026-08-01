"""Fast-path tests using in-memory fakes — no ONNX/chroma/LLM.

These run in seconds (vs minutes for the real-infrastructure suite).
They exercise ingest → retrieve → weave → feedback through the real
MemorySystem interface with fake stores behind it.
"""

from __future__ import annotations

from datetime import datetime

from memory_skill.contracts import DialogueTurn, MemorySkillConfig
from tests.fakes import build_fast_system


class TestFastIngest:
    def test_ingest_and_count(self):
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingest(DialogueTurn(id="t1", role="user", content="你好", timestamp=datetime.now()))
        ms.ingest(DialogueTurn(id="t2", role="assistant", content="你好！有什么可以帮你？", timestamp=datetime.now()))
        assert ms.count_turns() == 2

    def test_ingest_writes_both_stores(self):
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingest(DialogueTurn(id="t1", role="user", content="测试内容", timestamp=datetime.now()))
        assert len(ms.learned_store._entries) == 1
        assert ms.dialogue_store.count() == 1

    def test_ingest_idempotent_by_unique_constraint(self):
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        turn = DialogueTurn(id="t1", role="user", content="内容", timestamp=datetime.now())
        ms.ingest(turn)
        ms.ingest(turn)
        assert ms.count_turns() == 1


class TestFastRetrieve:
    def test_retrieve_hits_semantic(self):
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingest(DialogueTurn(id="t1", role="user", content="我推荐使用 FastAPI 框架", timestamp=datetime.now()))
        r = ms.retrieve("FastAPI", limit=3)
        assert r.total_candidates >= 1

    def test_retrieve_empty_db(self):
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        r = ms.retrieve("anything", limit=3)
        assert r.total_candidates == 0


class TestFastFeedback:
    def test_boost_weight_works(self):
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingest(DialogueTurn(id="t1", role="user", content="学习 Python", timestamp=datetime.now()))
        entry_id = "dialogue:t1"
        w0 = ms.learned_store.get_weight(entry_id)
        new = ms.boost_weight(entry_id)
        assert new == (w0 or 0.5) + 0.05


class TestFastWeave:
    def test_weave_nonempty_with_data(self):
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingest(DialogueTurn(id="t1", role="user", content="最近在学什么？", timestamp=datetime.now()))
        ctx = ms.weave(user_message="学什么", scene_summary="测试")
        assert ctx.time_context  # time always present


class TestFastLearning:
    def test_learning_loop_runs_offline(self):
        """Learning loop executes crawl→synth→ingest→verify with fakes."""
        from datetime import datetime
        from tests.fakes import FakeKnowledgeSynth, FakeCrawler
        from memory_skill.capability_registry import CapabilityRegistry
        from memory_skill.gap_detector import Gap
        from memory_skill.learning_task import LearningTaskManager

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        registry = CapabilityRegistry(None, ms.retriever)
        crawler = FakeCrawler(text="FastAPI 教程内容: pip install fastapi, 创建 main.py")
        synth = FakeKnowledgeSynth()
        mgr = LearningTaskManager(crawler, registry, ms, synth=synth)

        task = mgr.create_from_gap(
            Gap(query="FastAPI 怎么入门", branch="assistant_skill",
                confidence=0.0, severity="major"),
            urls=["https://example.com/fastapi"],
        )
        result = mgr.run(task)

        # 注入生效: crawler + synth 被调用
        assert crawler.call_count >= 1
        assert synth.call_count >= 1
        # 闭环阶段执行: crawling + verifying 出现在日志
        stages = {s for s, _, _ in result.status_log}
        assert "crawling" in stages
        assert "verifying" in stages
        # 记忆已 ingest (skill 条目)
        assert ms.dialogue_store.count() >= 1

    def test_learning_loop_no_sources_fails_cleanly(self):
        """Empty sources → failed without crawling."""
        from tests.fakes import FakeKnowledgeSynth, FakeCrawler
        from memory_skill.capability_registry import CapabilityRegistry
        from memory_skill.gap_detector import Gap
        from memory_skill.learning_task import LearningTaskManager

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        registry = CapabilityRegistry(None, ms.retriever)
        mgr = LearningTaskManager(FakeCrawler(), registry, ms, synth=FakeKnowledgeSynth())
        task = mgr.create_from_gap(
            Gap(query="topic", branch="assistant_skill", confidence=0.0, severity="minor"),
            urls=[],
        )
        result = mgr.run(task)
        assert result.status == "failed"
