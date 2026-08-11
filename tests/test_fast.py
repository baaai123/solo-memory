"""Fast-path tests using in-memory fakes — no ONNX/chroma/LLM.

These run in seconds (vs minutes for the real-infrastructure suite).
They exercise ingest → retrieve → weave → feedback through the real
MemorySystem interface with fake stores behind it.
"""

from __future__ import annotations

import pytest
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
        assert ctx.time_context

    def test_nudge_gated_on_message_relevance(self):
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="n1", role="system",
            content="pip install failed: ERROR", timestamp=datetime.now(),
        ))
        entry = list(ms.learned_store._entries.values())[0]
        ms.learned_store.set_weight(entry.id, 0.9)

        related = ms.weave("how do I fix the pip install error", scene_summary="")
        assert "pip install" in related.memory_nudge

        ms._classify_pending = None  # simulate classification completed
        unrelated = ms.weave("what is the weather like", scene_summary="")
        assert unrelated.memory_nudge == ""

    def test_nudge_unfiltered_when_no_user_message(self):
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="n2", role="system", content="critical deploy note",
            timestamp=datetime.now(),
        ))
        entry = list(ms.learned_store._entries.values())[0]
        ms.learned_store.set_weight(entry.id, 0.9)

        ctx = ms.weave(user_message="", scene_summary="")
        assert "critical deploy note" in ctx.memory_nudge

    def test_weave_blocks_when_prev_not_classified(self):
        from memory_skill._compose import ClassificationRequired

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.weave("first message", scene_summary="")
        # Now _classify_pending = "first message"
        with pytest.raises(ClassificationRequired):
            ms.weave("second message", scene_summary="")

    def test_weave_allows_when_classified(self):
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.weave("first", scene_summary="")
        ms._classify_pending = None  # simulate classify()
        ctx = ms.weave("second", scene_summary="")
        assert ctx.time_context


class TestFastLearning:
    def test_teach_skill_rejects_no_sources(self):
        from tests.fakes import FakeLearningQueue
        from memory_skill.skill_writer import SkillWriter

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        writer = SkillWriter(ms)
        result = writer.teach_skill("Docker", "# Docker", source_urls=[])
        assert result["status"] == "error"
        assert "source_urls" in result["reason"]

    def test_teach_skill_stores_and_marks_done(self):
        from tests.fakes import FakeLearningQueue
        from memory_skill.skill_writer import SkillWriter

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        queue = FakeLearningQueue()
        ms.learning_queue = queue
        ms.ingestor._learning_queue = queue
        queue.enqueue("skill", "Docker Compose", "a")

        writer = SkillWriter(ms)
        result = writer.teach_skill("Docker Compose", "# Docker Compose\n\n多容器编排",
                                    source_urls=["https://docs.docker.com/compose/"])
        assert result["status"] == "stored"
        assert queue.count_open() == 0

    def test_update_skill_rewrites(self):
        from memory_skill.skill_writer import SkillWriter

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        writer = SkillWriter(ms)
        result = writer.teach_skill("Docker", "# Docker\n\n旧",
                                    source_urls=["https://example.com/docker"])
        entry_id = result["entry_id"]

        upd = writer.update_skill(entry_id, "# Docker\n\n新")
        assert upd["status"] == "updated"
        entries = ms.learned_store.search("Docker 新", limit=3)
        assert entries and "新" in entries[0].content

    def test_update_skill_missing_errors(self):
        from memory_skill.skill_writer import SkillWriter

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        writer = SkillWriter(ms)
        result = writer.update_skill("nope", "x")
        assert result["status"] == "error"

    def test_weave_renders_queue_directives(self):
        from tests.fakes import FakeLearningQueue

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        queue = FakeLearningQueue()
        ms.learning_queue = queue
        ms.ingestor._learning_queue = queue
        ms.ingest(DialogueTurn(id="seed", role="user", content="hello",
                               timestamp=datetime.now()))
        queue.enqueue("skill", "Docker Compose", "需要掌握")

        ctx = ms.weave(user_message="学什么", scene_summary="")
        block = ctx.to_prompt_block()
        assert "先 websearch" in block
        assert "Docker Compose" in block