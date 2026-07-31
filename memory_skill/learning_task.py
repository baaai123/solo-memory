"""Learning Task — closed-loop knowledge acquisition.

Orchestrates web crawling, multi-source synthesis, and verification
to autonomously fill a knowledge gap.  Verifies completion by retesting
the original gap query — "done" means "can now answer the question."
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from memory_skill.capability_registry import CapabilityRegistry
from memory_skill.gap_detector import Gap
from memory_skill.web_crawler import WebCrawler

logger = logging.getLogger("memory_skill.learning")

_MAX_ATTEMPTS = 3
_MIN_CONFIDENCE = 0.5

_STATUS_ORDER = ("pending", "crawling", "synthesizing", "verifying", "done", "failed")


@dataclass
class LearningTask:
    id: str
    topic: str
    branch: str
    gap_query: str
    sources: list[str] = field(default_factory=list)
    status: str = "pending"
    attempts: int = 0
    result_entry_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    status_log: list[tuple[str, str, datetime]] = field(default_factory=list)

    def _set_status(self, status: str, detail: str = "") -> None:
        if status not in _STATUS_ORDER:
            return
        self.status = status
        self.status_log.append((status, detail, datetime.now(UTC)))
        if status == "done":
            self.completed_at = datetime.now(UTC)

    def is_terminal(self) -> bool:
        return self.status in ("done", "failed")


class LearningTaskManager:
    """Orchestrate crawl → synthesize → verify learning loops."""

    def __init__(
        self,
        crawler: WebCrawler,
        registry: CapabilityRegistry,
        skill,  # MemorySkill
    ):
        self._crawler = crawler
        self._registry = registry
        self._skill = skill

    def run(self, task: LearningTask) -> LearningTask:
        """Run the full closed-loop learning cycle.

        1. Crawl sources
        2. Synthesize knowledge
        3. Ingest into memory
        4. Verify: can we answer the original gap query?
        5. If not → retry up to 3 attempts with broader search
        """
        if task.is_terminal():
            return task

        while task.attempts < _MAX_ATTEMPTS:
            task.attempts += 1
            task._set_status("crawling")

            if not task.sources:
                task._set_status("failed", "no sources provided")
                return task

            # 1. Crawl
            chunks_by_url: dict[str, list] = {}
            for url in task.sources:
                try:
                    chunks = self._crawler.crawl(url)
                    if chunks:
                        chunks_by_url[url] = chunks
                except Exception as exc:
                    logger.warning("Crawl failed for %s: %s", url, exc)

            if not chunks_by_url:
                task._set_status("failed", "all sources failed to crawl")
                return task

            # 2. Synthesize markdown from crawled content
            try:
                from memory_skill.knowledge_synth import KnowledgeSynth
                api_base = os.getenv("IMPORTANCE_API_BASE", "https://api.deepseek.com/v1")
                api_key = os.getenv("IMPORTANCE_API_KEY", "")
                model = os.getenv("IMPORTANCE_MODEL", "deepseek-v4-flash")
                synth = KnowledgeSynth(api_base, api_key, model)
                markdown = synth.synthesize_markdown(task.topic, chunks_by_url)
            except Exception:
                markdown = ""

            # 3. Ingest as skill entry
            if markdown:
                try:
                    self._skill.ingest_skill(
                        task.topic, markdown, task.sources,
                    )
                except Exception as exc:
                    logger.warning("Skill ingest failed: %s", exc)

            # 4. Verify — closed loop
            task._set_status("verifying")
            can, confidence = self._registry.can_answer(task.gap_query)

            if can and confidence >= _MIN_CONFIDENCE:
                task._set_status("done", f"verified (conf={confidence:.2f})")
                return task

            # Not done — try broader search next iteration
            logger.info("Attempt %d failed (conf=%.2f). Retrying...", task.attempts, confidence)
            if task.attempts < _MAX_ATTEMPTS:
                task.sources = self._expand_sources(task.topic, task.sources)

        task._set_status("failed", f"exceeded {_MAX_ATTEMPTS} attempts")
        return task

    def create_from_gap(self, gap: Gap, urls: list[str]) -> LearningTask:
        task_id = f"lt_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        return LearningTask(
            id=task_id,
            topic=gap.query,
            branch="assistant_skill",
            gap_query=gap.query,
            sources=urls,
        )

    def _expand_sources(self, topic: str, existing: list[str]) -> list[str]:
        """Expand source list for retry — placeholder for future search integration."""
        return existing  # Currently reuse same sources; future: search + append
