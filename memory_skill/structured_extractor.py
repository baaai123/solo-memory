"""Unified classify + extract — one LLM call for all structured branches.

Given a piece of content, returns a JSON dict with ``type`` and 
branch-specific fields, or ``{"type": "none"}`` for irrelevant content.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """\
Content: "{content}"

Classify AND extract. Return JSON:

Examples:
"I like iced coffee every afternoon" → {{"type":"pref","key":"饮品","value":"冰美式"}}
"Learning FastAPI deployment with uvicorn" → {{"type":"skill","title":"FastAPI 部署","goal":"掌握 uvicorn 和 nginx 部署"}}
"You're too verbose, be brief" → {{"type":"pers","trait":"简洁"}}
"Need to finish the report by Friday" → {{"type":"mission","title":"写周报","deadline":"周五","priority":"high"}}
"Nice weather today" → {{"type":"none"}}

Rules:
- skill: A topic the user is learning. goal=what they want to master.
- mission: Task with deadline and priority.
- pref: Topic is generic (饮品/工作), value is the specific preference.
- pers: Single word or short phrase. Agent style feedback.
- none: Everything else.
"""


def classify_and_extract(api_base: str, api_key: str, model: str,
                         content: str) -> dict:
    from memory_skill._llm_utils import call_llm, parse_json_response

    prompt = _EXTRACT_PROMPT.format(content=content[:500])
    for _ in range(3):
        raw = call_llm(api_base, api_key, model, prompt, max_tokens=1024, temperature=0.0)
        if raw:
            result = parse_json_response(raw)
            if isinstance(result, dict) and "type" in result:
                return result
    return {"type": "none"}
