# Changelog

All notable changes to Memory Skill are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.5.0] — 2026-06-04

### Added
- **Transparent memory context**: weave output now uses natural language ("关于user，你记得:") instead of system labels ("[相关记忆]"). Agent perceives memories as its own, not as injected data.
- **Role-based initial weighting**: user messages (facts from others) get weight=0.6, assistant messages (agent's own words) get weight=0.4 — user-provided facts rank higher in retrieval.
- **Evaluation framework**: `evaluation/duel.py` with automated agent_duel scoring (keyword recall + length quality) and JSON report output.
- **Cross-session evolution**: `EvolutionLoop` now persists outcomes to `{db}_outcomes.json`. Evolution threshold lowered from 20 → 5 samples so every conversation triggers learning.
- **Room Agent integration**: `MemorySkillAdapter` in `agent/src/room/memory_adapter.py` with direct Python import (no subprocess overhead, <50ms/call). Auto-weave in `Agent._build_messages()`, auto-ingest in `Room.say()`.
- **Structured logging**: replaced all `print()`/`warnings.warn()` with Python `logging` module, configurable via `MEMORY_SKILL_LOG_LEVEL` env var.

### Changed
- **Weave context now labels memories by partner**: tier2 entries include `[与{partner}]` prefix when partner metadata exists.
- **Namespace model simplified**: `_ns_for()` returns agent name (e.g. `"agent_a"`) for all partners — memories are scoped to the agent, not partitioned by partner. Cross-partner recall is intentional.
- **Bridge weave order fixed**: `weave` now retrieves first, then ingests — fixing the auto-ingest pollution where the current message dominated retrieval results.

### Fixed
- **Nudge threshold**: `$gt` 0.85 → `$gte` 0.85 so weight=0.85 entries trigger nudge reminders.
- **Bridge search**: `memory_search` now passes `partner` parameter for object-aware retrieval.
- **Tier2 gate test**: corrected expectation — `_build_targeted_recall` is designed to work even when the gate is closed.

---

## [0.6.0] — 2026-07-25

### Added
- **Transparent Proxy** (`transparent_proxy.py`): OpenAI-compatible HTTP proxy. Agent points API base at it → memory auto-injects context before every request and auto-ingests after every response. Zero code changes needed.
- **Web Crawler** (`web_crawler.py`): curl-driven URL fetcher with text extraction and 2000-char chunking. Integrates with MemorySkill for knowledge ingestion from web sources.
- **Active Learning — Phase 1**: knowledge gap detection with CapabilityRegistry and GapDetector. System can now answer "do I know this?" with confidence scores.
- **Active Learning — Phase 2**: autonomous learning pipeline. Gap → LLM decide skip/ask/learn → crawl → store raw content → closed-loop verification (retest `can_answer`). `skill.learn(topic, urls)` API.
- **Knowledge Synth** (`knowledge_synth.py`): multi-source LLM cross-validation and fact extraction. Standalone utility.
- **Tree branch split**: `assistant_task` (任务与技能) → `assistant_task` (任务) + `assistant_skill` (技能). 5 branches now.
- **CLI proxy command**: `memory proxy --port 8888` starts the transparent proxy.

### Changed
- **21 private-access violations eliminated**: 6 modules no longer access `_`-prefixed internals.
- **observation/cleaner/reflect decoupled from MemorySkill**: receive concrete stores directly instead of the full skill object. Eliminates 3 TYPE_CHECKING cycles.
- **noise_filter merged into ingestor**: `noise_filter.py` deleted, `_ScreenNoiseFilter` inlined. Fixes split-state bug.
- **TreeManager +3 public APIs**: `branch_counts()`, `branch_avg_weight()`, `branch_last_updated()`.
- **LearnedStore.boost_weight()**: eliminates duplicated get/set weight pattern in mcp_tools and room_adapter.
- **MemorySkill +5 public APIs**: `ensure_embedder_loaded()`, `count_turns()`, `get_turn()`, `get_recent_turns()`, `boost_weight()`.
- **Ingestor gap detection**: auto-detects user questions and runs gap detector on ingest (non-blocking).

### Removed
- **`noise_filter.py`**: merged into `ingestor.py`.

### Post-0.6.0 additions

- **Structured Extractor** (`structured_extractor.py`): unified classify_and_extract — one LLM call classifies content into pref/pers/skill/mission/none and extracts structured fields.
- **Auto-extraction pipeline**: `MemorySystem.ingest()` automatically runs extraction on user messages. No manual ingest calls needed.
- **Pref branch**: key-value extraction and storage. Always injected into weave as `[用户偏好]`.
- **Pers branch**: character card format with 设定/风格/规则 sections. Traits accumulated automatically. Always injected into weave as `[人格设定]`.
- **Skill goal extraction**: classify returns learning goal, not full document. Full knowledge comes from learning pipeline (crawl → synth).
- **Mission branch**: structured task entries with deadline/priority. Injected into weave as `[当前任务]`.
- **Pers card format**: three-section markdown (设定/风格/规则). New traits auto-inserted into 风格 section.
- **Weave seven sections**: persona · preferences · scene · tier1 · tier2 · skills · missions · gaps.

---

## [0.4.0] — 2026-06-01

### Added
- **MCP Server**: `memory_skill/mcp_server.py` with 4 tools (search, ingest, status, feedback) via stdio JSON-RPC.
- **Feedback auto-detection**: `memory_skill/feedback.py` with LLM (MiniCPM-5) + rule-based fallback.
- **ONNX positional encoding fix**: `min(max_len, 512)` truncation in `_onnx_embed_batch()`, reducing embed failures from 11.2% to 0%.
- **OpenCode plugin**: `.opencode/plugins/memory-skill.ts` with 5 native tools.

### Changed
- Evolution now uses `evolution_min_samples=20` and `evolution_max_delta=0.05`.
- RRF weights configurable via `MemorySkillConfig` (was hardcoded in `retriever.py`).

---

## [0.3.0] — 2026-06-01

### Added
- **bge-large-en-v1.5** embedding (1024-dim, 512-token) replacing all-MiniLM-L6-v2 (384-dim, 256-token).
- **MiniCPM5Rewriter**: query rewriting via `llama-cpp-python`, lazy-load on first `rewrite()`.
- CUDAExecutionProvider support with CPU fallback.

### Changed
- Vector recall@10 improved from 0% to 90% (ablation result).
- `_chunk_text` threshold relaxed from 256 to 512 tokens.

---

## [0.2.0] — 2026-06-01

### Added
- **Batch ingest**: `Ingestor.ingest_batch()` with `IngestProfile` timing breakdown (≥500 msg/s target).
- **Text chunking**: `_chunk_text()` for messages exceeding 256 tokens, with sentence-boundary alignment.
- **Ablation experiment**: vector-only vs BM25-only vs RRF fusion, revealing 256-token root cause.
- **Query rewriting stub**: `QueryRewriter` Protocol + `_rewrite_query()` passthrough.
- **RRF calibration**: grid search (240 combinations), weights adjusted to 1.5/1.5/0.5.

### Changed
- Ingest speed from 27 msg/s to 42 msg/s (sequential), batch path ≥500 msg/s.

---

## [0.1.0] — 2026-06-01

### Added
- Initial implementation with 15+ modules, 174 tests, TDD.
- Core: `MemorySkill`, `Embedder` (all-MiniLM-L6-v2 ONNX), `Retriever` (3-signal RRF), `Ingestor`, `EvolutionLoop`.
- Storage: SQLite FTS5 (`DialogueStore`), ChromaDB (`LearnedStore`), `SawRingBuffer`.
- Weaver: `WeaveContext` with tier1/tier2/nudge layered context injection.
- `MemorySkillConfig` frozen dataclass with sensible defaults.
