"""Tree Classifier — LLM-based content-to-branch routing.

Separated from TreeManager so the tree remains a pure index.
Uses ``_llm_utils.call_llm`` for API calls.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)

_TREE_CLASSIFY_PROMPT = """\
Content: "{content}"

Classify into one branch. Return JSON only:
{{"root": "user"|"assistant", "branch": "pref"|"mem"|"pers"|"task"|"skill"}}

- user/pref: preferences, habits, likes/dislikes, routines
- user/mem: memories, life events, casual updates, observations
- assistant/pers: personality traits, tone, style, behavior rules
- assistant/task: specific projects in progress, bugs to fix, deadlines, todos
- assistant/skill: reusable knowledge, tools mastered, techniques learned, APIs, frameworks
"""


class TreeClassifier:
    """LLM-based content classifier that determines which tree branch a
    piece of content belongs to.

    Parameters
    ----------
    api_base: DeepSeek API base URL.
    api_key: DeepSeek API key.
    model: Model name (e.g. ``deepseek-v4-flash``).
    """

    def __init__(self, api_base: str, api_key: str, model: str):
        self._api_base = api_base
        self._api_key = api_key
        self._model = model

    def classify(self, content: str) -> dict:
        """Return ``{"root": ..., "branch": ...}`` for *content*.

        On API failure, returns ``{"root": "user", "branch": "mem"}``.
        """
        prompt = _TREE_CLASSIFY_PROMPT.format(content=content[:500])
        try:
            result = _call_llm(
                self._api_base, self._api_key, self._model, prompt,
            )
            if result and "branch" in result:
                _logger.debug("Tree classify: %s → %s", content[:60], result)
                return result
        except Exception as exc:
            _logger.warning("Tree classify API failed: %s, using fallback", exc)
        return _fallback_classify()


def _call_llm(api_base: str, api_key: str, model: str, prompt: str) -> dict | None:
    from memory_skill._llm_utils import call_llm, parse_json_response
    import logging
    raw = call_llm(api_base, api_key, model, prompt, max_tokens=256, temperature=0.0)
    if raw:
        result = parse_json_response(raw)
        return result if isinstance(result, dict) else None
    logging.getLogger(__name__).debug("LLM returned empty for prompt: %.100s", prompt)
    return None


def _fallback_classify() -> dict:
    return {"root": "user", "branch": "mem"}


_SKILL_PATH_PROMPT = """\
Existing: {existing_tree}
New skill: {title}
Assign this skill to a path in the tree. Output ONLY a JSON path list. Start new branch if no fit."""


def classify_skill_path(api_base: str, api_key: str, model: str,
                        title: str, content: str,
                        existing_tree: str) -> list[str]:
    parts = title.strip().split()
    if not parts:
        return []
    first = parts[0]
    rest = " ".join(parts[1:]) if len(parts) > 1 else None
    path = [first]
    if rest:
        path.append(rest)
    return path
