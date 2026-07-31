"""Learning Decider — LLM decides whether to learn from a knowledge gap.

Given a detected ``Gap`` (from ``gap_detector``), the decider evaluates:
  - severity of the gap
  - historical frequency of similar gaps
  - whether the topic is technical / learnable
  - estimated cost vs value of learning

Returns a ``Decision``: skip | ask | learn.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

logger = logging.getLogger("memory_skill.decider")

_TIMEOUT = 30


# ── Prompt ────────────────────────────────────────────────────────────────────


_DECIDE_PROMPT = """\
A knowledge gap was detected during a conversation.

Gap severity: {severity}
Topic: {query}
Related branch: {branch}
Historical gap count for this topic: {history_count}
Existing capability score for this branch: {capability_score:.2f}

Decide whether to learn this topic. Options:
- "skip":  casual chat / emotional / one-time / not technical. Not worth learning.
- "ask":   could be useful but unclear. Ask the user before proceeding.
- "learn": important technical knowledge the agent should master.

Return JSON only:
{{"action": "skip"|"ask"|"learn", "reasoning": "one sentence in Chinese", "confidence": 0.0-1.0}}

Rules:
- Prefer "learn" for critical severity + technical topics
- Prefer "skip" for minor severity + non-technical
- Prefer "ask" when uncertain
"""


# ── Data ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Decision:
    action: str           # "skip" | "ask" | "learn"
    reasoning: str        # LLM's explanation
    confidence: float     # decision confidence 0-1
    suggested_urls: list[str] = field(default_factory=list)


# ── Engine ────────────────────────────────────────────────────────────────────


class LearningDecider:
    """LLM evaluates whether a knowledge gap is worth learning."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        capability_score: float = 0.0,
    ):
        self._api_base = api_base
        self._api_key = api_key
        self._model = model
        self._capability_score = capability_score

    def set_capability_score(self, score: float) -> None:
        self._capability_score = score

    def evaluate(self, query: str, branch: str, severity: str,
                 history_count: int = 0) -> Decision:
        """Decide whether to learn from a gap.

        Parameters
        ----------
        query:
            The original user question that caused the gap.
        branch:
            Tree branch the gap belongs to.
        severity:
            Gap severity: "critical" | "major" | "minor".
        history_count:
            How many similar gaps have been seen before.
        """
        prompt = _DECIDE_PROMPT.format(
            severity=severity,
            query=query,
            branch=branch,
            history_count=history_count,
            capability_score=self._capability_score,
        )

        try:
            result = self._call_llm(prompt)
            decision = Decision(
                action=result.get("action", "skip"),
                reasoning=result.get("reasoning", ""),
                confidence=float(result.get("confidence", 0.5)),
            )
            logger.info(
                "Decider: action=%s conf=%.2f reason=%s",
                decision.action, decision.confidence, decision.reasoning,
            )
            return decision
        except Exception as exc:
            logger.warning("Decider LLM failed: %s, defaulting to skip", exc)
            return Decision(
                action="skip",
                reasoning=f"决策失败，默认跳过: {exc}",
                confidence=0.0,
            )

    # ── LLM ──────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> dict:
        from memory_skill._llm_utils import call_llm, parse_json_response
        import json
        raw = call_llm(self._api_base, self._api_key, self._model, prompt, max_tokens=128, temperature=0.1)
        if raw:
            result = parse_json_response(raw)
            if isinstance(result, dict):
                return result
        return {"action": "skip", "reasoning": "LLM调用失败", "confidence": 0.0}
