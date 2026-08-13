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
        ), category="conclusion", extra_metadata={"title": "pip 安装失败"})
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
        ), category="conclusion", extra_metadata={"title": "部署提醒"})
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

    def test_tier2_isolates_default_fragments(self):
        """tier2 excludes unclassified fragments; structured skill ranks."""
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        # Unclassified dialogue fragments (category = namespace = "default")
        for i in range(5):
            ms.ingestor.ingest_dialogue(DialogueTurn(
                id=f"frag{i}", role="user",
                content=f"随口聊的内容{i}，没有实质信息",
                timestamp=datetime.now(),
            ))
        # A structured skill entry about FastAPI
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="skill1", role="system",
            content="我学会了 FastAPI，性能好类型安全",
            timestamp=datetime.now(),
        ), category="skill", extra_metadata={"title": "FastAPI"})

        ms._classify_pending = None
        ctx = ms.weave("FastAPI 性能", scene_summary="")
        # Structured skill content should surface in tier2, not fragment noise
        assert "FastAPI" in (ctx.tier2_context or "")

    def test_nudge_excludes_default_fragments(self):
        """High-weight default fragments must not consume nudge slots."""
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        # High-weight fragment (would previously win a nudge slot)
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="noisy", role="system",
            content="pip install failed: ERROR connecting",
            timestamp=datetime.now(),
        ))
        noisy = list(ms.learned_store._entries.values())[0]
        ms.learned_store.set_weight(noisy.id, 0.9)
        # Structured pref entry with lower weight
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="pref1", role="system",
            content="用户偏好晚上工作",
            timestamp=datetime.now(),
        ), category="pref", extra_metadata={"title": "工作偏好"})
        pref = [e for e in ms.learned_store._entries.values()
                if e.category == "pref"][0]
        ms.learned_store.set_weight(pref.id, 0.85)

        ms._classify_pending = None
        ctx = ms.weave("晚上工作", scene_summary="")
        nudge = ctx.memory_nudge or ""
        # Nudge must surface the structured pref, not the noisy fragment
        assert "晚上工作" in nudge
        assert "pip install" not in nudge

    def test_fragments_still_searchable_via_memory_search(self):
        """Isolation hides fragments from tier2, not from explicit search."""
        from memory_skill.tools import handle_search

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="frag", role="user",
            content="我昨天聊过 QuantumFlux 协议",
            timestamp=datetime.now(),
        ))
        ms._classify_pending = None
        ctx = ms.weave("QuantumFlux", scene_summary="")
        # Not injected into weave context (it's a default fragment)
        assert "QuantumFlux" not in (ctx.tier2_context or "")
        # But explicit search still finds it
        result = handle_search(ms, {"query": "QuantumFlux", "limit": 5})
        texts = [r["content"] for r in result["results"]]
        assert any("QuantumFlux" in t for t in texts)


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

    def test_classify_enqueues_skill_and_mission(self):
        from memory_skill.tools import handle_classify, handle_learning_queue
        from tests.fakes import FakeLearningQueue

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        queue = FakeLearningQueue()
        ms.learning_queue = queue
        ms.ingestor._learning_queue = queue
        ms._classify_pending = None
        ms._pending_gaps = set()

        handle_classify(ms, {"category": "skill", "note": "Kubernetes"})
        items = handle_learning_queue(ms, {})["items"]
        assert any(i["kind"] == "skill" and i["query"] == "Kubernetes" for i in items)

        handle_classify(ms, {"category": "mission", "note": "部署微服务",
                             "gaps": ["Docker", "K8s"]})
        items = handle_learning_queue(ms, {})["items"]
        mission = [i for i in items if i["query"] == "部署微服务"]
        assert mission and mission[0]["kind"] == "mission"
        assert "Docker" in mission[0]["detail"]

    def test_classify_does_not_enqueue_chat(self):
        from memory_skill.tools import handle_classify, handle_learning_queue
        from tests.fakes import FakeLearningQueue

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        queue = FakeLearningQueue()
        ms.learning_queue = queue
        ms.ingestor._learning_queue = queue
        ms._classify_pending = None

        handle_classify(ms, {"category": "chat", "note": "随便聊聊"})
        assert handle_learning_queue(ms, {})["count"] == 0

    def test_learning_mark_closes_mission(self):
        from memory_skill.tools import handle_learning_mark, handle_learning_queue
        from tests.fakes import FakeLearningQueue

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        queue = FakeLearningQueue()
        ms.learning_queue = queue
        ms.ingestor._learning_queue = queue
        item = queue.enqueue("mission", "检查主动学习实现逻辑")

        result = handle_learning_mark(ms, {"item_id": item.id, "status": "done"})
        assert result["status"] == "marked"
        assert handle_learning_queue(ms, {})["count"] == 0

    def test_learning_mark_requires_item_id(self):
        from memory_skill.tools import handle_learning_mark

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        result = handle_learning_mark(ms, {})
        assert result.get("error")

    def test_conclusions_uses_list_by_category(self):
        from memory_skill.contracts import MemoryEntry
        from memory_skill.tools import handle_conclusions

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.learned_store.insert(MemoryEntry(
            id="conclusion:1", content="## 结论\n学习要 websearch",
            created_at=datetime.now(), updated_at=datetime.now(),
            weight=0.9, category="conclusion", tags=[], metadata={},
        ))
        result = handle_conclusions(ms, {"limit": 5})
        assert "error" not in result
        assert any(c["id"] == "conclusion:1" for c in result["conclusions"])


class TestFastDistill:
    def test_pending_rejects_unverified_evidence(self):
        """Anti-hallucination guard: evidence ids must exist in dialogue."""
        from memory_skill.distill import DistillCandidate

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="real1", role="user", content="我用了 FastAPI", timestamp=datetime.now()))
        cand = DistillCandidate(
            topic="FastAPI", summary="用户使用 FastAPI",
            evidence=["real1", "fake999"], suggested="skill", confidence=0.8)
        ok, msg = ms.pending_store.add_candidate(cand, ms.dialogue_store)
        assert not ok
        assert "fake999" in msg
        assert ms.pending_store.count_open() == 0

    def test_pending_add_and_list(self):
        from memory_skill.distill import DistillCandidate

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="d1", role="user", content="我偏好晚上工作", timestamp=datetime.now()))
        ms.pending_store.add_candidate(DistillCandidate(
            topic="工作偏好", summary="用户偏好晚上工作",
            evidence=["d1"], suggested="pref", confidence=0.7),
            ms.dialogue_store)
        items = ms.pending_store.list_open()
        assert len(items) == 1
        assert items[0].topic == "工作偏好"

    def test_candidates_never_leak_into_retrieval(self):
        """Pending candidates must not appear in memory_search results."""
        from memory_skill.distill import DistillCandidate
        from memory_skill.tools import handle_search

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="d1", role="user", content="量子引擎配置", timestamp=datetime.now()))
        ms.pending_store.add_candidate(DistillCandidate(
            topic="量子引擎", summary="量子引擎配置说明",
            evidence=["d1"], suggested="skill", confidence=0.9),
            ms.dialogue_store)

        result = handle_search(ms, {"query": "量子引擎", "limit": 10})
        for r in result["results"]:
            assert "量子引擎" not in r["content"] or r["entry_id"].startswith("dialogue:")

    def test_pending_mark_accept_reject(self):
        from memory_skill.distill import DistillCandidate

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="d1", role="user", content="内容", timestamp=datetime.now()))
        _, cand_id = ms.pending_store.add_candidate(DistillCandidate(
            topic="T", summary="S", evidence=["d1"], suggested="chat", confidence=0.5),
            ms.dialogue_store)
        assert ms.pending_store.mark(cand_id, "accepted")
        assert not ms.pending_store.mark(cand_id, "accepted")  # already closed
        assert ms.pending_store.count_open() == 0
        assert not ms.pending_store.mark(cand_id, "bogus")

    def test_weave_injects_pending_reminder(self):
        from memory_skill.distill import DistillCandidate

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="d1", role="user", content="我学了 Rust", timestamp=datetime.now()))
        ms.pending_store.add_candidate(DistillCandidate(
            topic="Rust", summary="用户学了 Rust", evidence=["d1"],
            suggested="skill", confidence=0.8), ms.dialogue_store)

        ms._classify_pending = None
        ctx = ms.weave(user_message="继续", scene_summary="")
        block = ctx.to_prompt_block()
        assert "待审核提炼" in block
        assert "Rust" in block

    def test_weave_no_reminder_when_pending_empty(self):
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="d1", role="user", content="普通对话", timestamp=datetime.now()))
        ms._classify_pending = None
        ctx = ms.weave(user_message="继续", scene_summary="")
        assert "待审核提炼" not in ctx.to_prompt_block()

    def test_accept_conclusion_promotes_to_structured(self):
        """Accepting a conclusion candidate writes it to the learned store."""
        from memory_skill.distill import DistillCandidate
        from memory_skill.tools import handle_pending_mark

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="d1", role="user", content="根因是编码问题", timestamp=datetime.now()))
        _, cid = ms.pending_store.add_candidate(DistillCandidate(
            topic="编码问题根因", summary="中文乱码是编码问题",
            evidence=["d1"], suggested="conclusion", confidence=0.9),
            ms.dialogue_store)

        result = handle_pending_mark(ms, {"candidate_id": cid, "status": "accepted"})
        assert result["status"] == "marked"
        assert "promoted_to" in result
        # Now exists as structured conclusion entry
        concls = ms.learned_store.list_by_category("conclusion", limit=5)
        assert any("编码问题" in c.content for c in concls)

    def test_accept_skill_requires_manual_teach(self):
        """Skill candidates are accepted but NOT auto-promoted (source_urls gate)."""
        from memory_skill.distill import DistillCandidate
        from memory_skill.tools import handle_pending_mark

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="d1", role="user", content="我会用 Docker", timestamp=datetime.now()))
        _, cid = ms.pending_store.add_candidate(DistillCandidate(
            topic="Docker 使用", summary="用户会 Docker",
            evidence=["d1"], suggested="skill", confidence=0.9),
            ms.dialogue_store)

        result = handle_pending_mark(ms, {"candidate_id": cid, "status": "accepted"})
        assert result["status"] == "marked"
        assert "promoted_to" not in result
        # Skill NOT auto-written (teach_skill with source_urls is the gate)
        skills = ms.learned_store.list_by_category("skill", limit=5)
        assert not any("Docker" in c.content for c in skills)

    def test_conclusion_dedicated_injection(self):
        """[历史结论] block surfaces conclusion entries in weave."""
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="c1", role="system",
            content="## 结论\n根因是编码问题,应该转 UTF-8",
            timestamp=datetime.now(),
        ), category="conclusion", extra_metadata={"title": "编码根因"})

        ms._classify_pending = None
        ctx = ms.weave(user_message="编码", scene_summary="")
        assert "历史结论" in ctx.to_prompt_block()
        assert "编码根因" in ctx.to_prompt_block()

    def test_title_preview_excludes_default_fragments(self):
        """[近期记忆] no longer shows unclassified fragments."""
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        # A fragment with high weight (would previously win the slot)
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="frag", role="user",
            content="[SYSTEM DIRECTIVE] 很长的系统噪音内容占位", timestamp=datetime.now(),
        ))
        frag = list(ms.learned_store._entries.values())[0]
        ms.learned_store.set_weight(frag.id, 0.95)
        # A structured skill
        ms.ingestor.ingest_dialogue(DialogueTurn(
            id="sk1", role="system",
            content="# FastAPI\n\n快速 API 框架", timestamp=datetime.now(),
        ), category="skill", extra_metadata={"title": "FastAPI 技能"})

        ms._classify_pending = None
        ctx = ms.weave(user_message="FastAPI", scene_summary="")
        block = ctx.to_prompt_block()
        # The [近期记忆] slot must show the structured skill, not the fragment
        # (fragments may still appear in tier1 — that's the raw recent dialogue).
        preview = block.split("[近期记忆]")[1] if "[近期记忆]" in block else ""
        assert "SYSTEM DIRECTIVE" not in preview
        assert "FastAPI" in preview
