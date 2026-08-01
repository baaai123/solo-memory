"""Shared fixtures for Memory Skill integration tests.

Uses a temporary SQLite database to avoid polluting production data.
Requires ``DEEPSEEK_API_KEY`` env var (or uses the default test key).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from memory_skill import DialogueTurn, MemorySkill, MemorySkillConfig
from memory_skill.contracts import utcnow

# ── API defaults ─────────────────────────────────────────────────────────────
# API keys are read from environment only — never commit real keys.
# LLM-dependent tests skip when DEEPSEEK_API_KEY is unset.
_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# ── Test data ────────────────────────────────────────────────────────────────

_TEST_TURNS: list[DialogueTurn] = [
    DialogueTurn(
        id="test_001",
        role="user",
        content="我最近在学 Python，想写一个 Web 后端，有什么推荐？",
        timestamp=utcnow(),
        partner="user",
    ),
    DialogueTurn(
        id="test_002",
        role="assistant",
        content="推荐 FastAPI，性能好、类型安全、文档自动生成。配合 SQLAlchemy 和 Pydantic 很强大。",
        timestamp=utcnow(),
        partner="user",
    ),
    DialogueTurn(
        id="test_003",
        role="user",
        content="FastAPI 和 Flask 比怎么样？我之前用过 Flask。",
        timestamp=utcnow(),
        partner="user",
    ),
    DialogueTurn(
        id="test_004",
        role="assistant",
        content="FastAPI 比 Flask 快很多，原生支持异步，自动生成 OpenAPI 文档。Flask 更灵活但需要自己搭很多东西。",
        timestamp=utcnow(),
        partner="user",
    ),
    DialogueTurn(
        id="test_005",
        role="user",
        content="好的，那我用 FastAPI 了。顺便问一下，你记得我中午吃了什么吗？",
        timestamp=utcnow(),
        partner="user",
    ),
    DialogueTurn(
        id="test_006",
        role="assistant",
        content="我不记得你中午吃了什么，我们之前聊的是 Python Web 框架的话题。",
        timestamp=utcnow(),
        partner="user",
    ),
    # ── Preference signals ──────────────────────────────────────────────
    DialogueTurn(
        id="test_007",
        role="user",
        content="我喜欢喝冰美式，每天下午都要来一杯。",
        timestamp=utcnow(),
        partner="user",
    ),
    DialogueTurn(
        id="test_008",
        role="assistant",
        content="记住了，你喜欢冰美式，每天下午喝。",
        timestamp=utcnow(),
        partner="user",
    ),
    # ── Task / project knowledge ────────────────────────────────────────
    DialogueTurn(
        id="test_009",
        role="user",
        content="我在用 LangChain 做一个 RAG 项目，数据存在 ChromaDB 里，embedding 用的 bge-large-en-v1.5。",
        timestamp=utcnow(),
        partner="user",
    ),
    DialogueTurn(
        id="test_010",
        role="assistant",
        content="这个方案很成熟。建议你加上 BM25 全文检索做 hybrid search，能显著提升召回率。",
        timestamp=utcnow(),
        partner="user",
    ),
    # ── Trivial content (should be filtered by importance gate) ────────
    DialogueTurn(
        id="test_011",
        role="user",
        content="ok",
        timestamp=utcnow(),
    ),
    DialogueTurn(
        id="test_012",
        role="user",
        content="嗯嗯",
        timestamp=utcnow(),
    ),
]


@pytest.fixture(scope="module")
def tmp_db() -> Iterator[str]:
    """Create a temporary SQLite database and Chroma dir, yield the db path."""
    tmp = tempfile.mkdtemp(prefix="memory_skill_test_")
    db_path = os.path.join(tmp, "test_memory.db")
    chroma_path = os.path.join(tmp, "chroma")
    yield db_path
    # Cleanup
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def skill(tmp_db: str) -> Iterator[MemorySkill]:
    """Create a fully initialized MemorySkill instance with temp DB + real LLM."""
    proj_root = Path(__file__).resolve().parent.parent
    config = MemorySkillConfig(
        db_path=tmp_db,
        model_path=str(proj_root / "models" / "bge-large-en-v1.5"),
        agent_name="test_agent",
        # API credentials passed through env (read by importance_llm + tree)
    )
    # Set env vars so that TreeManager and LLMImportanceGate pick them up
    os.environ["IMPORTANCE_API_KEY"] = _API_KEY
    os.environ["IMPORTANCE_API_BASE"] = _API_BASE
    os.environ["IMPORTANCE_MODEL"] = _MODEL

    sk = MemorySkill(config)
    yield sk
    sk._conn.close()
