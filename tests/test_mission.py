"""Tests for MissionStore — structured mission process records."""

from __future__ import annotations

import json

import pytest
from datetime import datetime

from memory_skill.contracts import DialogueTurn
from memory_skill.mission import (
    MissionError,
    MissionStore,
    MissionStep,
)

from tests.fakes import build_fast_system


@pytest.fixture
def store():
    from tests.fakes import FakeLearningQueue
    ms = build_fast_system()
    store = MissionStore(
        learned_store=ms.learned_store,
        learning_queue=FakeLearningQueue(),
    )
    return store


@pytest.fixture
def store_no_queue():
    ms = build_fast_system()
    return MissionStore(learned_store=ms.learned_store, learning_queue=None)


class TestCreate:
    def test_create_inserts_mission_entry(self, store):
        mission = store.create("调研上古卷轴5 mod 整合包")
        assert mission.id.startswith("dialogue:mission_")
        assert mission.status == "open"
        assert mission.steps == []

        entry = store._learned.get_entry(mission.id)
        assert entry is not None
        assert entry.content == "调研上古卷轴5 mod 整合包"

    def test_create_mirrors_queue_item(self, store):
        mission = store.create("调研任务")
        items = store._queue.all(limit=100)
        matched = [i for i in items if f"mission_id={mission.id}" in i.detail]
        assert len(matched) == 1
        assert matched[0].kind == "mission"
        assert matched[0].status == "open"

    def test_create_without_queue_still_works(self, store_no_queue):
        mission = store_no_queue.create("无队列任务")
        entry = store_no_queue._learned.get_entry(mission.id)
        assert entry is not None

    def test_create_empty_content_rejected(self, store):
        with pytest.raises(MissionError):
            store.create("   ")


class TestSteps:
    def test_add_step_appends_and_persists(self, store):
        mission = store.create("任务")
        mission = store.add_step(mission.id, "第一步")
        mission = store.add_step(mission.id, "第二步")
        assert len(mission.steps) == 2
        assert mission.steps[0].text == "第一步"
        assert mission.steps[0].done is False

        reloaded = store.get(mission.id)
        assert len(reloaded.steps) == 2

    def test_add_step_with_skill_link(self, store):
        # skill entry must exist in learned store
        mission = store.create("任务")
        from memory_skill.contracts import MemoryEntry
        from datetime import UTC, datetime

        now = datetime.now(UTC)
        skill = MemoryEntry(
            id="dialogue:skill_test_000001",
            content="Docker Compose 多容器部署",
            created_at=now, updated_at=now, weight=0.5,
            category="skill", tags=["docker"],
            metadata={"title": "Docker Compose 多容器部署"},
        )
        store._learned.insert(skill)

        mission = store.add_step(mission.id, "部署容器", skill_id=skill.id)
        assert mission.steps[0].skill_id == skill.id
        assert mission.steps[0].skill_title == "Docker Compose 多容器部署"

    def test_add_step_unknown_skill_gets_empty_title(self, store):
        mission = store.create("任务")
        mission = store.add_step(mission.id, "某步", skill_id="dialogue:skill_nonexistent")
        assert mission.steps[0].skill_title == ""

    def test_add_step_empty_text_rejected(self, store):
        mission = store.create("任务")
        with pytest.raises(MissionError):
            store.add_step(mission.id, "")

    def test_update_step_text_and_done(self, store):
        mission = store.create("任务")
        store.add_step(mission.id, "原始文本")
        mission = store.update_step(mission.id, 0, text="新文本", done=True)
        assert mission.steps[0].text == "新文本"
        assert mission.steps[0].done is True

    def test_update_step_out_of_range(self, store):
        mission = store.create("任务")
        with pytest.raises(MissionError):
            store.update_step(mission.id, 5, done=True)

    def test_remove_step(self, store):
        mission = store.create("任务")
        store.add_step(mission.id, "步骤1")
        store.add_step(mission.id, "步骤2")
        mission = store.remove_step(mission.id, 0)
        assert len(mission.steps) == 1
        assert mission.steps[0].text == "步骤2"

    def test_remove_step_out_of_range(self, store):
        mission = store.create("任务")
        with pytest.raises(MissionError):
            store.remove_step(mission.id, 0)


class TestStatus:
    def test_set_status_done(self, store):
        mission = store.create("任务")
        mission = store.set_status(mission.id, "done")
        assert mission.status == "done"
        reloaded = store.get(mission.id)
        assert reloaded.status == "done"

    def test_set_status_mirrors_queue(self, store):
        mission = store.create("任务")
        store.set_status(mission.id, "done")
        items = store._queue.all(limit=100)
        matched = [i for i in items if f"mission_id={mission.id}" in i.detail]
        assert len(matched) == 1
        assert matched[0].status == "done"

    def test_set_status_invalid(self, store):
        mission = store.create("任务")
        with pytest.raises(MissionError):
            store.set_status(mission.id, "in_progress")


class TestRead:
    def test_get_missing_returns_none(self, store):
        assert store.get("dialogue:mission_does_not_exist") is None
    def test_get_roundtrip_preserves_steps(self, store):
        mission = store.create("任务")
        store.add_step(mission.id, "步骤A")
        store.add_step(mission.id, "步骤B")
        store.update_step(mission.id, 1, done=True)
        store.set_status(mission.id, "done")

        reloaded = store.get(mission.id)
        assert reloaded.status == "done"
        assert len(reloaded.steps) == 2
        assert reloaded.steps[0].text == "步骤A"
        assert reloaded.steps[1].done is True

    def test_list_filters_by_status(self, store):
        m1 = store.create("任务一")
        m2 = store.create("任务二")
        store.set_status(m1.id, "done")

        done = store.list_missions(status="done")
        open_ = store.list_missions(status="open")
        assert any(x.id == m1.id for x in done)
        assert not any(x.id == m2.id for x in done)
        assert any(x.id == m2.id for x in open_)

    def test_metadata_uses_ui_namespaced_keys(self, store):
        mission = store.create("任务")
        store.add_step(mission.id, "步骤")
        entry = store._learned.get_entry(mission.id)
        assert "ui_steps" in entry.metadata
        assert "ui_status" in entry.metadata
        assert entry.metadata["ui_status"] == "open"


class TestMissionWeaveIntegration:
    """Weaver renders mission steps from the structured MissionStore.

    Guards against the old dual-schema drift: steps must come from
    ``ui_steps`` metadata (MissionStore), never from regex-parsing content.
    """

    def _fast_system(self):
        from memory_skill.contracts import MemorySkillConfig
        from memory_skill.mission import MissionStore
        from tests.fakes import build_fast_system
        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        ms.mission_store = MissionStore(
            learned_store=ms.learned_store,
            learning_queue=None,
        )
        return ms

    def test_weave_renders_structured_steps(self):
        ms = self._fast_system()
        ms.ingest(DialogueTurn(id="d1", role="user", content="开始做一个任务",
                               timestamp=datetime.now()))
        ms.protocol.mark_classified("chat")

        mission = ms.mission_store.create("部署服务")
        ms.mission_store.add_step(mission.id, "写后端")
        ms.mission_store.add_step(mission.id, "写前端")
        ms.mission_store.update_step(mission.id, 0, done=True)

        ctx = ms.weave(user_message="部署", scene_summary="")
        # steps render with ✓ for done, ○ for open
        assert "写后端" in ctx.mission_context
        assert "写前端" in ctx.mission_context
        assert "✓" in ctx.mission_context
        assert "○" in ctx.mission_context

    def test_weave_renders_skill_hint_from_metadata(self):
        ms = self._fast_system()
        ms.ingest(DialogueTurn(id="d2", role="user", content="任务",
                               timestamp=datetime.now()))
        ms.protocol.mark_classified("chat")

        mission = ms.mission_store.create("任务")
        # skill_title is resolved at add_step time and stored in ui_steps
        ms.mission_store.add_step(mission.id, "步骤")
        mission = ms.mission_store.get(mission.id)
        assert mission.steps[0].skill_title == ""  # no skill linked → no hint

        ctx = ms.weave(user_message="任务", scene_summary="")
        assert "→" in ctx.mission_context

    def test_weave_without_mission_store_skips_block(self):
        from memory_skill.contracts import MemorySkillConfig
        from memory_skill.weaver import WeaverStores, weave
        from tests.fakes import build_fast_system

        ms = build_fast_system(MemorySkillConfig(db_path=":memory:"))
        stores = WeaverStores(
            saw_buffer=ms.saw_buffer,
            dialogue_store=ms.dialogue_store,
            learned_store=ms.learned_store,
            retriever=ms.retriever,
            agent_name="agent",
            namespace="assistant",
            mission_store=None,  # explicitly absent
        )
        ctx = weave(stores, user_message="任务", scene_summary="")
        assert ctx.mission_context == ""
