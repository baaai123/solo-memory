"""
Memory Skill — Weaver: automatic context injection for agents.

V9: Conversation-unit tier2 with surrounding dialogue.
  tier1: Scene perception + recent dialogue turns (~80t, always injected)
  tier2: Retrieved conversation units (user→agent→...) with context (~300t)
  emotion: Derived affinity from partner feedback history (1 line)
  nudge: High-weight reminders with behavioral intensity (💡/⚠)

Depth auto-selection:
  compact (<3 turns, no stored history):    tier1 only
  standard (3-10 turns OR has stored data): tier1 + tier2 + emotion + nudge
  deep (>10 turns):                         same + heartbeat check

Key V9 changes from V8:
  - Tier2 format: flat fact list → conversation units with surrounding dialogue
  - Each unit shows: User→Agent→Agent→... (full turn context)
  - Unit boundary: starts at preceding user message, ends at next user message
  - V8 features retained: depth gate, emotion context, anti-hallucination guard
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, TYPE_CHECKING

from memory_skill.contracts import (
    DialogueStoreProtocol,
    LearnedStoreProtocol,
    SawBufferProtocol,
    TreeManagerProtocol,
)
from memory_skill.capability_registry import (
    _SEM_CORROBORATED,
    token_overlap,
)

if TYPE_CHECKING:
    from memory_skill.retriever import Retriever

_logger = logging.getLogger(__name__)


# ── Narrow protocols: each builder takes only what it needs ──

class RetrievalSource(Protocol):
    def retrieve(self, query: str, limit: int = 10, filters=None): ...


class DialogueSource(Protocol):
    def get_recent(self, n: int = 5): ...
    def get_by_id(self, turn_id: str): ...
    def count(self) -> int: ...
    def count_recent(self, minutes: int = 5) -> int: ...


class MemorySource(Protocol):
    def search(self, query: str, limit: int = 10, filters=None): ...


@dataclass
class WeaverStores:
    saw_buffer: SawBufferProtocol
    dialogue_store: DialogueStoreProtocol
    learned_store: LearnedStoreProtocol
    retriever: Retriever
    agent_name: str
    namespace: str
    tree: TreeManagerProtocol | None = None
    gaps: list = field(default_factory=list)
    pending_store: object = None
    degraded: bool = False
    degraded_reason: str | None = None


@dataclass
class WeaveContext:
    """Layered memory context ready for agent system prompt injection.

    tier1_context: Always injected (~80 tokens). Scene perception.
    tier2_context: Deep memory retrieval (~150 tokens). Gated by depth.
    memory_nudge: High-weight reminders. Weight >= 0.85 threshold.
    needs_second_pass: True when high-weight memories warrant a follow-up
        injection after the agent's next response (heartbeat-style).
    """

    time_context: str = ""
    tier1_context: str = ""
    tier2_context: str = ""
    memory_nudge: str = ""
    tree_context: str = ""
    tree_nav: str = ""
    skill_context: str = ""
    mission_context: str = ""
    pref_context: str = ""
    pers_context: str = ""
    conclusion_context: str = ""
    gap_context: str = ""
    title_preview: str = ""
    historic_hint: str = ""
    pending_context: str = ""
    embedder_banner: str = ""
    needs_second_pass: bool = False

    def to_prompt_block(self) -> str:
        # MANDATORY directive block — rendered first and visually separated
        # from background memory so the agent cannot mistake directives for
        # passive context. Classification is required every turn; pending
        # learning/decomposition directives are enforced by ProtocolGate.
        directives: list[str] = []
        directives.append(
            "[分类指令] 必须对上一轮用户消息分类 — "
            "memory_classify(category=chat|skill|mission|pref|pers)"
            " — 不可跳过，否则下一轮 weave 拒绝服务"
        )
        if self.gap_context:
            directives.append(self.gap_context)

        parts: list[str] = []
        if directives:
            parts.append("═══ 必须执行指令（protocol_gate 强制）═══\n"
                         + "\n".join(directives)
                         + "\n═══ 指令区结束（以下为背景记忆，仅供参考）═══")
        if self.embedder_banner:
            parts.append(self.embedder_banner)
        if self.time_context:
            parts.append(self.time_context)
        # tier2 first (if available), tier1 as fallback
        if self.tier2_context:
            parts.append(self.tier2_context)
        elif self.tier1_context:
            parts.append(self.tier1_context)
        if self.memory_nudge:
            parts.append(self.memory_nudge)
        if self.skill_context:
            parts.append(self.skill_context)
        if self.mission_context:
            parts.append(self.mission_context)
        if self.pref_context:
            parts.append(self.pref_context)
        if self.pers_context:
            parts.append(self.pers_context)
        if self.conclusion_context:
            parts.append(self.conclusion_context)
        if self.pending_context:
            parts.append(self.pending_context)
        if self.title_preview:
            parts.append(self.title_preview)
        if self.historic_hint:
            parts.append(self.historic_hint)
        if self.tree_context:
            parts.append(self.tree_context)
        if self.tree_nav:
            parts.append(self.tree_nav)
        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        return not (self.tier1_context or self.tier2_context
                    or self.memory_nudge)


# ── Constants ────────────────────────────────────────

_NUDGE_WEIGHT_THRESHOLD: float = 0.85
_NUDGE_CRITICAL_THRESHOLD: float = 0.95
_NUDGE_MAX_ITEMS: int = 3
_CHARS_PER_TOKEN: int = 4
_TIER1_MAX_CHARS: int = 80 * 4
_TIER2_MAX_CHARS: int = 300 * 4  # was 150→300 chars per memory
_NUDGE_MAX_CHARS: int = 30 * 4
_MAX_RECENT_TURNS: int = 3


# ── weave() ──────────────────────────────────────────

def weave(
    stores: WeaverStores,
    user_message: str = "",
    scene_summary: str = "",
    *,
    partner: str | None = None,
) -> WeaveContext:
    """Assemble layered memory context from all stores.

    Depth is automatically chosen:
      compact (recent <3 turns AND 0 total stored): tier1 only (~80 tokens)
      standard (3-10 recent OR has stored history): tier1 + tier2 (~300 tokens)
      deep (>10 recent): tier1 + tier2 + heartbeat check

    Note: "recent" = last 5 minutes (via count_recent(), not total stored).
    "total stored" = all turns ever (via count()).  These are now DIFFERENT,
    fixing the bug where a new session with old stored data jumped straight
    to deep depth.
    """
    now = datetime.now()
    ctx = WeaveContext(time_context=f"[现在时间] {now.strftime('%Y-%m-%d %H:%M:%S')}")
    ns = _resolve_namespace(stores.agent_name, stores.namespace, partner)
    turn_count = _count_recent_turns(stores.dialogue_store)

    ctx.tier1_context = _build_tier1(
        stores.dialogue_store, stores.saw_buffer, stores.agent_name, scene_summary)

    # Directives must render even for brand-new sessions: pending learning /
    # decomposition items are the whole point of a fresh mission, and a
    # no-history early return would silently drop them.
    ctx.gap_context = _build_gap_context(stores.gaps)
    ctx.pending_context = _build_pending_context(stores.pending_store)

    if stores.degraded:
        ctx.embedder_banner = (
            "⚠ [EMBEDDER DEGRADED] 语义检索/去重/学习判定已降级为 SHA-256 hash（无 ONNX 模型）"
            + (f" — {stores.degraded_reason}" if stores.degraded_reason else "")
            + "。修复: 运行 ./download_model.sh 或 export MEMORY_MODEL_PATH=… 后重启。"
        )

    total_stored = _count_all_dialogue(stores.dialogue_store)
    if turn_count < 3 and total_stored == 0:
        return ctx  # no history at all — tier1 only

    if turn_count < 3:
        turn_count = 3

    _apply_standard_blocks(ctx, stores, user_message, ns, partner)

    # deep depth adds the heartbeat check: surface a second-pass nudge when a
    # high-weight memory is relevant to the current message.
    if turn_count > 10 and _has_high_weight(stores.learned_store, user_message):
        ctx.needs_second_pass = True
    return ctx


def _apply_standard_blocks(
    ctx: WeaveContext,
    stores: WeaverStores,
    user_message: str,
    ns: str,
    partner: str | None,
) -> None:
    """Populate the standard context slots shared by every non-compact weave.

    One call site instead of two copy-pasted branches; adding a context slot
    now means editing a single line here rather than both branches.
    """
    if user_message:
        ctx.tier2_context = _build_tier2(
            stores.retriever, stores.dialogue_store, stores.agent_name, user_message, ns)
        ctx.historic_hint = _build_historic_hint(stores.retriever, user_message)
    ctx.memory_nudge = _build_nudge(stores.learned_store, user_message)
    ctx.skill_context = _build_skill_context(stores.retriever, user_message)
    ctx.mission_context = _build_mission_context(stores.retriever, user_message)
    ctx.pref_context = _build_pref_context(stores.retriever)
    ctx.pers_context = _build_pers_context(stores.retriever)
    ctx.conclusion_context = _build_conclusion_context(stores.retriever)
    ctx.title_preview = _build_title_preview(stores.retriever)
    ctx.tree_context = _build_tree_context(stores.tree, user_message)
    ctx.tree_nav = _build_tree_nav(stores.tree, user_message)


# ── Tier builders ────────────────────────────────────

def _build_tier1(dialogue: DialogueSource, saw: SawBufferProtocol,
                 agent_name: str, scene_summary: str) -> str:
    parts: list[str] = []
    if scene_summary:
        parts.append(f"[当前场景] {scene_summary}")
    else:
        entries = saw.get_all()
        if entries:
            parts.append(f"[当前感知] {entries[-1].content[:_TIER1_MAX_CHARS]}")
    turns = dialogue.get_recent(_MAX_RECENT_TURNS)
    if turns:
        agent_label = agent_name or "助手"
        lines = [f"[{t.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] {'用户' if t.role == 'user' else agent_label}: {t.content[:80]}"
                 for t in turns[-_MAX_RECENT_TURNS:]]
        parts.append("[最近对话]\n" + "\n".join(lines))
    result = "\n".join(parts)
    return result[:_TIER1_MAX_CHARS + 100]


def _build_tier2(retriever: RetrievalSource, dialogue: DialogueSource,
                 agent_name: str, user_message: str, ns: str = "") -> str:
    """Build tier2 context — conversation units with surrounding dialogue.

    V9: Each retrieved fact is expanded into a conversation unit that shows
    the surrounding dialogue context (user → agent → agent → ...), not just
    the isolated fact.  This gives the agent the full conversational context
    of the memory, not a decontextualized snippet.
    """
    if not user_message:
        return ""
    try:
        if ns and ns != "default":
            envelope = retriever.retrieve(
                user_message, limit=4, filters={"category": ns})
        else:
            # Isolate unclassified fragments: semantic leg searches
            # structured memory only; BM25 leg still supplies the raw
            # dialogue units needed for context expansion.
            envelope = retriever.retrieve(
                user_message, limit=4,
                filters={"category": {"$ne": "default"}})
    except Exception as e:
        _logger.warning("Tier2 retrieval failed: %s", e)
        return ""
    if not envelope.entries:
        return ""

    units: list[str] = []
    seen_units: set[str] = set()
    ds = dialogue

    for e in envelope.entries:  # skip non-dialogue entries (e.g. persona card) to find a buildable unit
        turn_id = e.metadata.get("turn_id", "") if e.metadata else ""
        if not turn_id:
            continue

        # Get the target turn and surrounding dialogue
        target_turn = ds.get_by_id(turn_id)
        if not target_turn:
            continue

        # Find surrounding turns (±2 turns from target)
        recent = ds.get_recent(500)  # fetch enough for surrounding context
        target_idx = None
        for i, t in enumerate(recent):
            if t.id == turn_id:
                target_idx = i
                break
        if target_idx is None:
            continue

        # Build unit: always start with a user message, end with the next user
        # Walk backward to find the preceding user turn
        unit_start = target_idx
        while unit_start > 0:
            if recent[unit_start - 1].role == "user":
                unit_start -= 1
                break
            unit_start -= 1

        # Walk forward to find the next user turn (inclusive)
        unit_end = target_idx + 1
        while unit_end < len(recent):
            if recent[unit_end].role == "user":
                if unit_end > target_idx:
                    unit_end += 1  # include ending user turn
                break
            unit_end += 1
        unit_end = min(unit_end, len(recent))

        # Build unit lines (cap: 3 turns, 50 chars each)
        unit_lines: list[str] = []
        max_turns = min(unit_end, len(recent))
        for i in range(unit_start, min(unit_start + 3, max_turns)):
            t = recent[i]
            if t.role == "user":
                label = "User"
            elif t.role == "assistant":
                # Infer speaker from turn ID prefix (namespace)
                # e.g. "st_agent_a_123" → agent_a, "st_user_456" → user
                ns_match = t.id.split("_")[1] if t.id.startswith("st_") and t.id.count("_") >= 2 else ""
                label = ns_match if ns_match else (agent_name or "助手")
            else:
                label = t.role
            snippet = t.content[:50]
            unit_lines.append(f"{label}: {snippet}")

        unit_text = "\n".join(unit_lines)
        unit_key = unit_text[:80]
        if unit_key in seen_units:
            continue
        seen_units.add(unit_key)

        ts = e.created_at.strftime("%Y-%m-%d %H:%M:%S") if e.created_at else "????-??-?? ??:??:??"
        units.append(f"[{ts}] [记忆片段]\n{unit_text}")

    if units:
        header = "以下是你经历的对话片段。你只能使用这些信息，不得编造:"
        return header + "\n\n" + "\n\n".join(units) + \
               "\n\n（如果你不确定某条信息是否在上面列出——就不要提及它。）"
    return ""


def _build_nudge(memory: MemorySource, user_message: str = "") -> str:
    """Build nudge with behavioral intensity based on weight.

    V7 upgrade:
      weight >= 0.95 → "⚠ 务必...——这件事很重要"
      weight >= 0.85 → "💡 可以...——让对话更自然"

    V9 (ADR-0001 candidate 8): nudge entries are gated on relevance to the
    current ``user_message`` via distinctive-token overlap — an unrelated
    high-weight entry (e.g. a stale "pip install failed" from a past
    session) is no longer injected every turn as noise.  When no
    ``user_message`` is given, the gate is skipped (keep historical
    behaviour so empty-context weaves still surface critical items).
    """
    try:
        entries = memory.search(
            "", limit=_NUDGE_MAX_ITEMS * 3,
            filters={"weight": {"$gte": _NUDGE_WEIGHT_THRESHOLD},
                     "category": {"$ne": "default"}})
    except Exception as e:
        _logger.debug("Nudge retrieval failed: %s", e)
        return ""

    # Isolate unclassified fragments: nudge should surface structured
    # memory (skills/prefs/pers), not raw dialogue noise.
    high = [e for e in entries
            if e.weight >= _NUDGE_WEIGHT_THRESHOLD
            and e.category != "default"]
    if not high:
        return ""

    if user_message.strip():
        high = [e for e in high if token_overlap(user_message, e.content)]
        if not high:
            return ""

    # Sort by weight descending
    high.sort(key=lambda e: e.weight, reverse=True)

    lines: list[str] = []
    for e in high[:_NUDGE_MAX_ITEMS]:
        snippet = e.content[:_NUDGE_MAX_CHARS]
        ts = e.created_at.strftime("%Y-%m-%d %H:%M:%S") if e.created_at else "????-??-?? ??:??:??"
        if e.weight >= _NUDGE_CRITICAL_THRESHOLD:
            lines.append(f"[{ts}] ⚠ {snippet}——务必主动提及，这件事很重要")
        else:
            lines.append(f"[{ts}] 💡 {snippet}——可以自然提到，让对话更顺畅")
    return " | ".join(lines) if lines else ""


# ── Helpers ──────────────────────────────────────────

def _resolve_namespace(agent_name: str, namespace: str, partner: str | None) -> str:
    if agent_name:
        return agent_name
    return namespace


def _count_all_dialogue(dialogue: DialogueSource) -> int:
    """Count total stored dialogue turns (across all sessions)."""
    try:
        return dialogue.count()
    except Exception as e:
        _logger.debug("Total dialogue count failed: %s", e)
        return 0


def _count_recent_turns(dialogue: DialogueSource) -> int:
    """Count turns from the last 5 minutes — approximates "current session"."""
    try:
        return dialogue.count_recent(minutes=5)
    except Exception as e:
        _logger.debug("Recent turn count failed: %s", e)
        return 0


def _has_high_weight(memory: MemorySource, user_message: str = "") -> bool:
    """True when a relevant high-weight entry exists (needs_second_pass gate).

    Mirrors the candidate-8 relevance gate of ``_build_nudge`` so the deep
    branch's second-pass flag agrees with what nudge would actually surface.
    """
    try:
        entries = memory.search(
            "", limit=5,
            filters={"weight": {"$gte": _NUDGE_WEIGHT_THRESHOLD},
                     "category": {"$ne": "default"}})
        high = [e for e in entries
                if e.weight >= _NUDGE_WEIGHT_THRESHOLD
                and e.category != "default"]
        if not high:
            return False
        if user_message.strip():
            return any(token_overlap(user_message, e.content) for e in high)
        return True
    except Exception as e:
        _logger.debug("High-weight check failed: %s", e)
        return False


def _build_tree_context(tree, user_message: str) -> str:
    """Build tree context from the memory tree for agent injection.

    Delegates to TreeManager.get_context() which finds relevant branches
    by label similarity to the query and returns siblings + parent labels.
    """
    if not tree or not user_message:
        return ""
    try:
        return tree.get_context(user_message)
    except Exception as e:
        _logger.debug("Tree context retrieval failed: %s", e)
        return ""


def _build_tree_nav(tree, user_message: str) -> str:
    """Build deterministic tree context in parallel with RRF retrieval.

    Uses the LLM-free navigation path so weave() never triggers an internal
    LLM call (ADR-0002: weave is pure context assembly). Agents that want
    LLM-selected branches call TreeManager.navigate() explicitly via a tool.
    """
    if not tree or not user_message:
        return ""
    try:
        return tree.navigate_without_llm()
    except Exception as e:
        _logger.debug("Tree navigation failed: %s", e)
        return ""


def _safe_retrieve(
    retriever: RetrievalSource,
    query: str,
    category: object,
    limit: int,
):
    """Retrieve with the category filter, tolerating backend failures.

    The category-slot builders share this scaffold (guard + filter +
    empty-check) and differ only in how they render their hits. One
    implementation, six call sites.
    """
    try:
        result = retriever.retrieve(query, limit=limit, filters={"category": category})
    except Exception:
        return []
    return result.entries if result.entries else []


def _build_skill_context(retriever: RetrievalSource, user_message: str) -> str:
    """Retrieve relevant skill titles for agent awareness."""
    if not user_message:
        return ""
    entries = _safe_retrieve(retriever, user_message, "skill", 5)
    titles = []
    for e in entries[:5]:
        t = e.content.split('\n')[0].lstrip('# ')[:40]
        if t and t not in titles:
            titles.append(t)
    if not titles:
        return ""
    return "[已掌握的技能]\n  · " + "\n  · ".join(titles)


def _build_mission_context(retriever: RetrievalSource, user_message: str) -> str:
    entries = _safe_retrieve(retriever, user_message, "mission", 2)
    if not entries:
        return ""
    import re
    lines = ["[当前任务]"]
    for e in entries[:2]:
        title = e.content.split('\n')[0].lstrip('# ')[:50]
        lines.append(f"\n→ {title}")
        steps = re.split(r'\n##\s*', '\n' + e.content)
        for step_block in steps[1:]:
            step_lines = step_block.strip().split('\n')
            header = step_lines[0]
            m = re.match(r'(.+?)\s*\[(done|doing|pending)\]', header)
            if not m:
                continue
            step_name, step_status = m.group(1).strip(), m.group(2)
            marks = {'done': '✓', 'doing': '●', 'pending': '○'}
            mark = marks.get(step_status, '○')
            skill_name = ""
            for sl in step_lines:
                if sl.strip().startswith('skill:'):
                    skill_name = sl.split(':', 1)[1].strip()
                    break
            skill_hint = ""
            if skill_name:
                has = retriever.retrieve(skill_name, limit=5, filters={"category": "skill"})
                found = False
                for sk in has.entries:
                    if skill_name.lower() in sk.content.lower()[:100]:
                        found = True
                        break
                skill_hint = f" → {skill_name} {'✅' if found else '⚠'}"
            lines.append(f"  {mark} {step_name}{skill_hint}")
    return "\n".join(lines)


def _build_gap_context(gaps: list) -> str:
    """Render open learning-queue items as explicit agent directives.

    Unlike the legacy passive gap display, these are actionable commands
    for the main agent: learn an unknown skill, or decompose a mission
    into steps before executing it.  The agent is expected to act on them
    (check_skill → web search → teach_skill; mission_decompose) and to
    confirm the result with the user.
    """
    if not gaps:
        return ""
    skill_items = [g for g in gaps if getattr(g, "kind", "") == "skill"]
    mission_items = [g for g in gaps if getattr(g, "kind", "") == "mission"]
    other = [g for g in gaps if getattr(g, "kind", "") not in ("skill", "mission")]

    lines: list[str] = []

    def _query(g) -> str:
        return getattr(g, "query", "") or str(g)[:80]

    if skill_items:
        lines.append("[待学习] 先 websearch 搜索 → 找到权威来源 → "
                     "memory_teach_skill(必须带 source_urls) → 反馈用户确认：")
        for g in skill_items[:5]:
            lines.append(f"  📚 {_query(g)}")

    if mission_items:
        lines.append("[待拆解] 以下任务尚未分解为步骤（由主 agent 自行分析拆解）：")
        for g in mission_items[:5]:
            lines.append(f"  📋 {_query(g)}")

    if other:
        lines.append("[待处理] " + "; ".join(_query(g) for g in other[:3]))

    # These directives are enforced: skipping them causes the next weave to
    # be rejected by ProtocolGate, so the agent must act before continuing.
    lines.append("⚠ 上述指令由协议硬门强制执行——不执行则下一轮记忆注入被拒绝。")

    return "\n".join(lines)


def _build_pending_context(pending_store) -> str:
    """Surface distill candidates awaiting agent review (fixes agent drift)."""
    if pending_store is None:
        return ""
    try:
        items = pending_store.list_open(limit=5)
    except Exception:
        return ""
    if not items:
        return ""
    lines = ["[待审核提炼] 以下对话碎片候选待主 agent 判断是否值得沉淀："]
    for c in items:
        lines.append(f"  · {c.topic}（建议 {c.suggested}）"
                     f"——memory_pending 查看证据")
    return "\n".join(lines)


def _build_pref_context(retriever: RetrievalSource) -> str:
    entries = _safe_retrieve(retriever, "all", "pref", 10)
    if not entries:
        return ""
    lines = ["[用户偏好]"]
    for e in entries[-10:]:
        lines.append(f"  · {e.content}")
    return "\n".join(lines)


def _build_pers_context(retriever: RetrievalSource) -> str:
    entries = _safe_retrieve(retriever, "all", "pers", 10)
    if not entries:
        return ""
    cards = [e for e in entries if e.content.startswith('# ')]
    latest = max(cards, key=lambda e: len(e.content)) if cards else entries[-1]
    return "[人格特征]\n  · " + latest.content


def _build_conclusion_context(retriever: RetrievalSource) -> str:
    """Dedicated injection channel for conclusion entries.

    Conclusions (knowledge/root-cause judgments) previously shared the
    unfiltered [近期记忆] slot and had no guaranteed exposure.  This
    gives them their own slot, newest first.
    """
    entries = _safe_retrieve(retriever, "all", "conclusion", 10)
    if not entries:
        return ""
    lines = ["[历史结论]"]
    for e in entries[-3:]:
        title = e.metadata.get("title", "") if hasattr(e, "metadata") and e.metadata else ""
        content = title if title else e.content[:60].replace("\n", " ")
        lines.append(f"  · {content}")
    return "\n".join(lines)


def _build_title_preview(retriever: RetrievalSource) -> str:
    try:
        result = retriever.retrieve(
            "all", limit=5, filters={"category": {"$ne": "default"}})
    except Exception:
        return ""
    entries = result.entries if result.entries else []
    if not entries:
        return ""
    lines = ["[近期记忆]"]
    for e in entries[:5]:
        title = e.metadata.get("title", "") if hasattr(e, "metadata") and e.metadata else ""
        content = title if title else e.content[:50].replace("\n", " ")
        lines.append(f"  · {content}")
    return "\n".join(lines)


# Entries ingested in the last N minutes are this session's own turns —
# hinting at them would tell the agent "you just said that". Skip them.
_HISTORIC_HINT_RECENT_SKIP_MINUTES: int = 10

# Cap hint content — a trigger, not a dump.
_HISTORIC_HINT_MAX_TITLE: int = 40


def _build_historic_hint(retriever: RetrievalSource, user_message: str) -> str:
    """Detect a strongly-related past memory and hint the agent to
    actively `memory_search` it — WITHOUT injecting its content.

    Passive weave context is noise when the agent has no need for it;
    a hint that *named* past work (title + date) creates a genuine
    information need, so the agent's own memory_search result is
    treated as relevant rather than ignored.

    Relevance requires BOTH a strong semantic score AND a shared
    distinctive token — same double-corroboration as KNOWN-ISSUES #1.
    Pure semantic thresholds misfire on short Chinese sentences (bge
    inflates cosine for token-sparse queries), so a bare 0.72 cosine
    would hint at unrelated recent turns ("今天天气怎么样" → "重启了").
    """
    if not user_message:
        return ""
    try:
        result = retriever.retrieve(user_message, limit=6)
    except Exception as e:
        _logger.debug("Historic hint retrieval failed: %s", e)
        return ""
    if not result.entries:
        return ""

    now = datetime.now(UTC)
    for e in result.entries:
        if not e.semantic_score or e.semantic_score < _SEM_CORROBORATED:
            continue
        if e.created_at and (now - e.created_at).total_seconds() < (
            _HISTORIC_HINT_RECENT_SKIP_MINUTES * 60
        ):
            continue
        if not token_overlap(user_message, e.content):
            continue
        title = e.content.split("\n")[0].lstrip("# ")[:_HISTORIC_HINT_MAX_TITLE]
        date = e.created_at.strftime("%m-%d") if e.created_at else "过去"
        return (
            f"[历史相关] 你在 {date} 处理过「{title}」——"
            "需要当时的细节/结论吗?用 memory_search 主动检索"
        )
    return ""
