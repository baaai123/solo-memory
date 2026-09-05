"""Tavern state extraction — dialogue → 8-dim state via the LLM bridge.

Explicit, agent-invoked only (never in ingest/weave — ADR-0002).  The LLM
call goes through the configured IMPORTANCE endpoint (ds-web bridge), and
the result is a strict JSON of the 8 fixed dimensions, null-preserving.
"""

from __future__ import annotations

import json
import os

from memory_skill.state_store import _STATE_DIMENSIONS


def extract_state(conversation: str, current: dict | None = None) -> dict | None:
    """Extract an 8-dim state JSON from *conversation*. Returns None on failure."""
    out = extract_state_and_impressions(conversation, current, cast=())
    return out.get("state") if out else None


def extract_state_and_impressions(
    conversation: str, current: dict | None = None,
    cast: tuple | list = (),
) -> dict | None:
    """One bridge call returning both the 8-dim state and impressions.

    *cast* is a sequence of ``{"id": ..., "name": ...}`` for every other
    party present in the scene (roles plus the special ``"user"``).  The
    extractor only writes impressions about someone actually present.
    Returns ``{"state": ...|None, "impressions": [...]}`` or None on
    failure — callers must treat failure as best-effort no-op.
    """
    from memory_skill._llm_utils import call_llm

    api_base = os.getenv("IMPORTANCE_API_BASE", "http://127.0.0.1:8456/v1")
    api_key = os.getenv("IMPORTANCE_API_KEY", "sk-bridge")
    model = os.getenv("IMPORTANCE_MODEL", "deepseek-chat")
    raw = call_llm(
        api_base=api_base, api_key=api_key, model=model,
        prompt=_build_extract_prompt(conversation, current, cast),
        max_tokens=1000, temperature=0.0, timeout=120.0, retries=1,
    )
    if not raw:
        return None
    from memory_skill._llm_utils import parse_json_response

    parsed = parse_json_response(raw)
    if not isinstance(parsed, dict):
        return None
    state = parsed.get("state")
    state = {k: v for k, v in state.items() if k in _STATE_DIMENSIONS} \
        if isinstance(state, dict) else None
    impressions = parsed.get("impressions")
    cleaned = []
    if isinstance(impressions, list):
        cast_ids = {c.get("id") for c in cast}
        for it in impressions:
            if not isinstance(it, dict):
                continue
            tid = it.get("target_id") or ""
            note = str(it.get("note") or "").strip()
            kind = it.get("kind") if it.get("kind") in ("pref", "impression") \
                else "impression"
            if tid in cast_ids and note:
                cleaned.append({"target_id": tid, "kind": kind,
                                "note": note[:120]})
    return {"state": state, "impressions": cleaned}


_CAST_LABELS = """在场者:
{cast_lines}"""

_EXTRACT_TMPL = """你是角色扮演记忆引擎的提取器。根据一段对话，为【发言角色】提取：
A. 当前状态：对话末尾该角色 {dims} 8 个维度的最新状态。
B. 本回合中，该角色对在场其他人产生的、值得长期记住的印象或关系认知。

{cast_block}
规则（严格）：
- 状态：没提到的维度填 null；每项一句话以内。
- impressions 只在出现实质内容时输出（具体发生的事 / 明确评价 / 关系变化 / 对方告知的习惯或规则）。对方随口寒暄不要记。
- 若内容是【用户告知的习惯/规则/选择】(如"我调图绝不超分"、"我晚上工作") → 必须输出 kind="pref"，target_id="user"，即使该角色只在状态 action 里写了"记住"——习惯必须进入 impressions，不能只留在状态里。
- 若是该角色对某人【主观的看法、经历、或两人之间的相处】→ kind="impression"。
- 只能针对在场者，target_id 必须来自在场者 id 清单；note 一句话 ≤ 60 字。
- 只输出一个 JSON 对象，不要任何其他文字。

示例 1：
对话：
用户:跟你说个我的习惯,我每次调图都画风优先,绝不超分。
测试1:好,记住了,以后不主动提超分。
输出：{{"state": {{"mood": "专注", "need": null, "health": null, "clothing": null, "item": null, "action": "记住用户习惯", "scene": null, "weather": null}}, "impressions": [{{"target_id": "user", "kind": "pref", "note": "用户调图画风优先,默认不超分"}}]}}

示例 2：
对话：
测试2:这杯威士忌温了不好喝,我习惯加冰。
测试1:(接过话题)加冰确实更利落。
输出：{{"state": {{"mood": "平静", "need": null, "health": null, "clothing": null, "item": null, "action": null, "scene": null, "weather": null}}, "impressions": [{{"target_id": "测试2", "kind": "impression", "note": "测试2 喝威士忌喜欢加冰"}}]}}

{current_block}对话：
{conversation}"""


def _build_extract_prompt(conversation: str, current: dict | None,
                          cast: tuple | list = ()) -> str:
    current_block = ""
    if current:
        items = [f"- {k}: {v}" for k, v in current.items()
                 if k in _STATE_DIMENSIONS and v]
        if items:
            current_block = "【当前状态】\n" + "\n".join(items) + "\n\n"
    cast_lines = "\n".join(
        f"- {c.get('name', c.get('id', ''))} (id: {c.get('id', '')})"
        for c in cast) or "- (只有用户, id: user)"
    return _EXTRACT_TMPL.format(
        dims="/".join(_STATE_DIMENSIONS),
        cast_block=_CAST_LABELS.format(cast_lines=cast_lines),
        current_block=current_block,
        conversation=conversation,
    )
