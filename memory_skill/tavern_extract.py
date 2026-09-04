"""Tavern state extraction — dialogue → 8-dim state via the LLM bridge.

Explicit, agent-invoked only (never in ingest/weave — ADR-0002).  The LLM
call goes through the configured IMPORTANCE endpoint (ds-web bridge), and
the result is a strict JSON of the 8 fixed dimensions, null-preserving.
"""

from __future__ import annotations

import json
import os

from memory_skill.state_store import _STATE_DIMENSIONS

_PROMPT_TMPL = """你是角色扮演记忆系统的状态提取器。根据对话，提取 AI 角色「当前」（对话末尾）的状态。只输出一个 JSON 对象，字段固定：{dims}。

规则：只取对话末尾的最新状态（不是历史）；没提到的维度填 null；每项一句话以内；只输出 JSON，不要任何其他文字。

{current_block}对话：
{conversation}"""


def _build_prompt(conversation: str, current: dict | None) -> str:
    current_block = ""
    if current:
        items = [f"- {k}: {v}" for k, v in current.items()
                 if k in _STATE_DIMENSIONS and v]
        if items:
            current_block = "【当前状态】\n" + "\n".join(items) + "\n\n"
    return _PROMPT_TMPL.format(
        dims="/".join(_STATE_DIMENSIONS),
        current_block=current_block,
        conversation=conversation,
    )


def extract_state(conversation: str, current: dict | None = None) -> dict | None:
    """Extract an 8-dim state JSON from *conversation*. Returns None on failure."""
    from memory_skill._llm_utils import call_llm

    api_base = os.getenv("IMPORTANCE_API_BASE", "http://127.0.0.1:8456/v1")
    api_key = os.getenv("IMPORTANCE_API_KEY", "sk-bridge")
    model = os.getenv("IMPORTANCE_MODEL", "deepseek-chat")
    raw = call_llm(
        api_base=api_base, api_key=api_key, model=model,
        prompt=_build_prompt(conversation, current),
        max_tokens=800, temperature=0.0, timeout=120.0, retries=1,
    )
    if not raw:
        return None
    from memory_skill._llm_utils import parse_json_response

    parsed = parse_json_response(raw)
    if not isinstance(parsed, dict):
        return None
    return {k: v for k, v in parsed.items() if k in _STATE_DIMENSIONS}
