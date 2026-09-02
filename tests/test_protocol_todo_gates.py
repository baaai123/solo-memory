"""Unit 2 hard gates — archive + queue enforcement.

Exercises the todo enforcement loop end-to-end:
  weave → MemorySystem._maybe_arm_todo_gates arms archive_pending /
  queue_pending from store counts → next weave's ProtocolGate.check
  raises ArchiveRequired / QueueRequired → an archive/queue tool call
  clears the flag → weave recovers.

Fast fakes cover the arming/clearing state machine; the reclassify
integration paths use a real LearnedStore (ChromaDB in a tmp dir) so
``set_category`` / ``get_entry`` behave exactly as in production.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from memory_skill._compose import (
    ArchiveRequired,
    MemorySystem,
    QueueRequired,
)
from memory_skill.contracts import MemoryEntry, MemorySkillConfig
from memory_skill.learned_store import LearnedStore
from memory_skill.tools import (
    handle_learning_mark,
    handle_reclassify,
    handle_review_default,
)
from tests.fakes import FakeEmbedder, FakeLearningQueue, build_fast_system


def _config(**overrides) -> MemorySkillConfig:
    return MemorySkillConfig(db_path=":memory:", **overrides)


def _entry(entry_id: str, content: str, category: str = "default") -> MemoryEntry:
    now = datetime.now(UTC)
    return MemoryEntry(
        id=entry_id,
        content=content,
        created_at=now,
        updated_at=now,
        weight=0.5,
        category=category,
        tags=[],
        metadata={},
    )


def _system(**config_overrides) -> MemorySystem:
    """Fast system with an empty fake learning queue mounted."""
    system = build_fast_system(_config(**config_overrides))
    system.learning_queue = FakeLearningQueue()
    return system


def _weave_n(system: MemorySystem, n: int) -> None:
    """Weave n turns, classifying each so the classify gate stays clear."""
    for i in range(n):
        system.weave(f"message {i}", scene_summary="")
        system.protocol.mark_classified("chat")


# ── Archive gate ───────────────────────────────────────────────────────────


def test_archive_gate_arms_at_interval_and_blocks_weave():
    """Given a non-empty default backlog and interval=3, when the 3rd
    weave completes, then archive_pending arms and the next weave raises
    ArchiveRequired."""
    system = _system(archive_interval=3)
    system.learned_store.insert(_entry("d1", "unclassified default entry"))

    _weave_n(system, 3)

    assert system.protocol.weave_count == 3
    assert system.protocol.archive_pending is True
    with pytest.raises(ArchiveRequired):
        system.weave("blocked", scene_summary="")


def test_review_default_clears_archive_gate():
    """Given an armed archive gate, when the agent calls review_default,
    then the gate clears and the next weave passes."""
    system = _system(archive_interval=3)
    system.learned_store.insert(_entry("d1", "still default"))
    _weave_n(system, 3)
    with pytest.raises(ArchiveRequired):
        system.weave("blocked", scene_summary="")

    result = handle_review_default(system, {})

    assert result["total"] == 1
    assert system.protocol.archive_pending is False
    ctx = system.weave("after review", scene_summary="")
    assert ctx.time_context


def test_archive_gate_stays_off_when_default_empty():
    """Given an empty default category, when weave_count hits the
    interval, then the archive gate does not arm."""
    system = _system(archive_interval=3)

    _weave_n(system, 3)

    assert system.protocol.weave_count == 3
    assert system.protocol.archive_pending is False
    ctx = system.weave("still fine", scene_summary="")
    assert ctx.time_context


def test_archive_gate_not_rearmed_until_next_cycle():
    """Given an armed gate the agent responds to, when the following
    weaves run, then the gate does not re-arm until the next interval
    boundary (armed protection — one trigger per cycle)."""
    system = _system(archive_interval=3)
    system.learned_store.insert(_entry("d1", "left in default"))
    _weave_n(system, 3)
    with pytest.raises(ArchiveRequired):
        system.weave("blocked", scene_summary="")

    system.protocol.clear_archive_pending()

    # Counts 4 and 5: inside the grace period — no re-arm.
    for i in range(2):
        system.weave(f"free {i}", scene_summary="")
        assert system.protocol.archive_pending is False
        system.protocol.mark_classified("chat")
    # Count 6: next cycle boundary with default still non-empty → re-arms.
    system.weave("sixth", scene_summary="")
    assert system.protocol.archive_pending is True
    system.protocol.mark_classified("chat")
    with pytest.raises(ArchiveRequired):
        system.weave("blocked again", scene_summary="")


# ── Queue gate ─────────────────────────────────────────────────────────────


def test_queue_gate_arms_above_threshold_and_blocks():
    """Given 21 open items and threshold=20, when a weave completes,
    then queue_pending arms and the next weave raises QueueRequired."""
    system = _system(queue_threshold=20)
    for i in range(21):
        system.learning_queue.enqueue("skill", f"learn thing {i}")

    system.weave("first", scene_summary="")
    system.protocol.mark_classified("chat")

    assert system.protocol.queue_pending is True
    with pytest.raises(QueueRequired):
        system.weave("blocked", scene_summary="")


def test_learning_mark_clears_queue_gate():
    """Given an armed queue gate, when the agent marks one item done
    (21 → 20 open), then the gate clears and stays cleared because
    20 is not above threshold (respond, don't empty)."""
    system = _system(queue_threshold=20)
    for i in range(21):
        system.learning_queue.enqueue("skill", f"learn thing {i}")
    system.weave("first", scene_summary="")
    system.protocol.mark_classified("chat")
    with pytest.raises(QueueRequired):
        system.weave("blocked", scene_summary="")

    item_id = system.learning_queue.open_items()[0].id
    result = handle_learning_mark(system, {"item_id": item_id, "status": "done"})

    assert result["status"] == "marked"
    assert system.protocol.queue_pending is False
    ctx = system.weave("after mark", scene_summary="")
    assert ctx.time_context
    assert system.protocol.queue_pending is False


def test_queue_gate_stays_off_at_or_below_threshold():
    """Given exactly threshold open items, when weaving, then the queue
    gate does not arm (strictly greater comparison)."""
    system = _system(queue_threshold=20)
    for i in range(20):
        system.learning_queue.enqueue("skill", f"learn {i}")

    system.weave("first", scene_summary="")
    assert system.protocol.queue_pending is False
    system.protocol.mark_classified("chat")
    ctx = system.weave("second", scene_summary="")
    assert ctx.time_context
    assert system.protocol.queue_pending is False


# ── todo_context injection ─────────────────────────────────────────────────


def test_todo_context_renders_when_gates_pending():
    """Given armed gates with live store counts, when the weaver
    assembles context, then todo_context names the backlog and is
    rendered inside the prompt-block directive section."""
    from memory_skill.weaver import WeaverStores, weave as weave_raw

    system = _system()
    system.learned_store.insert(_entry("d1", "unclassified"))
    system.learning_queue.enqueue("skill", "learn X")
    system.protocol.archive_pending = True
    system.protocol.queue_pending = True

    stores = WeaverStores(
        saw_buffer=system.saw_buffer,
        dialogue_store=system.dialogue_store,
        learned_store=system.learned_store,
        retriever=system.retriever,
        agent_name=system.config.agent_name,
        namespace=system.config.namespace,
        protocol=system.protocol,
        learning_queue=system.learning_queue,
    )
    ctx = weave_raw(stores, "hello", "scene")

    assert "default 分类 1 条未归档" in ctx.todo_context
    assert "学习队列 1 条 open" in ctx.todo_context
    assert ctx.todo_context in ctx.to_prompt_block()

    system.protocol.archive_pending = False
    system.protocol.queue_pending = False
    empty = weave_raw(stores, "hello", "scene")
    assert empty.todo_context == ""


# ── Defensive: shells without a protocol reference ─────────────────────────


def test_archive_handlers_survive_missing_protocol(tmp_path):
    """Given a minimal shell with no protocol attribute, when archive
    tools run, then they succeed without crashing (getattr defence)."""
    config = MemorySkillConfig(db_path=str(tmp_path / "chroma"))
    store = LearnedStore(config, FakeEmbedder(dim=config.embedding_dim))
    ms = object.__new__(MemorySystem)
    ms.__dict__["learned_store"] = store
    store.insert(_entry("d1", "some default content"))

    review = handle_review_default(ms, {})
    reclassified = handle_reclassify(ms, {"entry_id": "d1", "category": "pref"})

    assert review["total"] == 1
    assert reclassified["ok"] is True
    assert getattr(ms, "protocol", None) is None


def test_learning_mark_survives_missing_protocol():
    """Given a minimal shell with no protocol attribute, when
    learning_mark runs, then it succeeds without crashing."""
    ms = object.__new__(MemorySystem)
    ms.__dict__["learning_queue"] = FakeLearningQueue()
    ms.learning_queue.enqueue("skill", "learn something")

    result = handle_learning_mark(ms, {"item_id": "lq_fake_1", "status": "done"})

    assert result["status"] == "marked"
    assert getattr(ms, "protocol", None) is None


# ── Integration: full trigger → respond → recover loops ────────────────────


def test_integration_archive_trigger_response_recovery(tmp_path):
    """Given a real store with one default entry, when the archive gate
    triggers, then review + reclassify recover the weave and the gate
    stays disarmed once default is empty."""
    config = MemorySkillConfig(
        db_path=str(tmp_path / "m.db"), archive_interval=3, queue_threshold=20,
    )
    system = build_fast_system(config)
    system.learning_queue = FakeLearningQueue()
    real = LearnedStore(
        config, FakeEmbedder(dim=config.embedding_dim),
        chroma_path=str(tmp_path / "chroma"),
    )
    system.learned_store = real
    real.insert(_entry("d1", "把这条归到 pref"))

    _weave_n(system, 3)
    with pytest.raises(ArchiveRequired):
        system.weave("blocked", scene_summary="")

    review = handle_review_default(system, {})
    assert review["total"] == 1
    assert system.protocol.archive_pending is False

    system.weave("recovered", scene_summary="")  # count 4 → grace, no re-arm
    assert system.protocol.archive_pending is False
    system.protocol.mark_classified("chat")

    handle_reclassify(system, {"entry_id": "d1", "category": "pref"})
    system.weave("five", scene_summary="")       # count 5
    system.protocol.mark_classified("chat")
    system.weave("six", scene_summary="")        # count 6 = cycle, default empty
    assert system.protocol.archive_pending is False
    system.protocol.mark_classified("chat")

    ctx = system.weave("seven", scene_summary="")
    assert ctx.time_context
    assert system.protocol.archive_pending is False


def test_integration_queue_trigger_response_recovery():
    """Given 3 open items with threshold=2, when the queue gate
    triggers, then marking down below the threshold recovers the weave."""
    system = _system(queue_threshold=2)
    for i in range(3):
        system.learning_queue.enqueue("skill", f"learn {i}")

    system.weave("first", scene_summary="")
    system.protocol.mark_classified("chat")
    with pytest.raises(QueueRequired):
        system.weave("blocked", scene_summary="")

    for item in system.learning_queue.open_items()[:2]:
        handle_learning_mark(system, {"item_id": item.id, "status": "done"})

    assert system.learning_queue.count_open() == 1
    assert system.protocol.queue_pending is False

    ctx = system.weave("recovered", scene_summary="")
    assert ctx.time_context
    assert system.protocol.queue_pending is False
