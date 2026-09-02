"""Unit 3 tests — mission done auto-closes its mirrored learning-queue item."""

from __future__ import annotations

import pytest

from memory_skill.mission import MissionStore

from tests.fakes import FakeLearningQueue, build_fast_system


@pytest.fixture
def store():
    ms = build_fast_system()
    return MissionStore(
        learned_store=ms.learned_store,
        learning_queue=FakeLearningQueue(),
    )


class TestMissionAutoClose:
    def test_done_auto_closes_queue_item(self, store):
        """Given a mission with a mirrored queue item, When status → done,
        Then the queue item is marked done."""
        mission = store.create("调研任务")
        assert store._queue.count_open() == 1

        store.set_status(mission.id, "done")

        open_items = store._queue.open_items(kind="mission")
        assert all(f"mission_id={mission.id}" not in i.detail
                   for i in open_items)
        closed = [i for i in store._queue.all(limit=100)
                  if f"mission_id={mission.id}" in i.detail]
        assert len(closed) == 1
        assert closed[0].status == "done"

    def test_done_without_queue_item_is_silent(self, store):
        """Given a mission whose queue item is absent, When status → done,
        Then the update succeeds and nothing is raised."""
        mission = store.create("无队列关联任务")

        orphan = MissionStore(
            learned_store=store._learned,
            learning_queue=FakeLearningQueue(),
        )
        orphan.set_status(mission.id, "done")

        reloaded = orphan.get(mission.id)
        assert reloaded.status == "done"
        assert orphan._queue.count_open() == 0

    def test_done_on_already_done_item_is_idempotent(self, store):
        """Given a queue item already closed, When mission set done again,
        Then no error and the item stays done."""
        mission = store.create("幂等任务")
        store.set_status(mission.id, "done")
        assert store._queue.count_open() == 0

        store.set_status(mission.id, "done")

        assert store.get(mission.id).status == "done"
        assert store._queue.count_open() == 0

    def test_mark_failure_degrades_to_log(self, store):
        """Given a queue whose mark raises, When mission set done,
        Then the mission update still succeeds (failure logged only)."""

        class BrokenQueue(FakeLearningQueue):
            def mark(self, item_id, status):
                raise RuntimeError("queue unavailable")

        mission = store.create("降级任务")
        broken = MissionStore(
            learned_store=store._learned,
            learning_queue=BrokenQueue(),
        )
        broken._queue.enqueue("mission", mission.content,
                              detail=f"mission_id={mission.id}")

        broken.set_status(mission.id, "done")

        assert broken.get(mission.id).status == "done"

    def test_lifecycle_done_reduces_open_count(self, store):
        """Given two open missions, When one finishes its lifecycle (steps →
        done), Then the queue open count decreases by exactly one."""
        m1 = store.create("生命周期任务一")
        store.create("生命周期任务二")
        assert store._queue.count_open() == 2

        store.add_step(m1.id, "拆解步骤")
        store.update_step(m1.id, 0, done=True)
        store.set_status(m1.id, "done")

        assert store._queue.count_open() == 1
        remaining = store._queue.open_items(kind="mission")
        assert len(remaining) == 1
        assert f"mission_id={m1.id}" not in remaining[0].detail

    def test_reopen_does_not_resurrect_queue_item(self, store):
        """Given a done mission, When reopened, Then the queue item stays
        closed (queue API has no reopen transition)."""
        mission = store.create("重开任务")
        store.set_status(mission.id, "done")
        store.set_status(mission.id, "open")

        assert store.get(mission.id).status == "open"
        assert store._queue.count_open() == 0
