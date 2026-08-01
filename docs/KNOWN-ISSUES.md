# Known Issues

Known defects and architectural debts, tracked for dedicated fixes.
Each entry: impact, root cause, and the fix direction — so a future
session can pick it up without re-diagnosing.

---

## 1. `can_answer` confidence is polluted by result count — gap detection is broken

**Status**: Resolved · **Severity**: High · **Found**: 2026-08-02 · **Fixed**: 2026-08-02

### Impact
`memory_gaps` always returned 0. The learning loop's gap-detection leg was
effectively dead — the system never discovered *what it doesn't know*, so
`gap → decide → learn` never fired automatically (manual `memory_learn`
still worked).

### Symptom
```
can_answer("量子场论重整化群 是什么") → can=True, conf=0.5   # system knows NOTHING about this
can_answer("Docker 怎么部署")        → can=True, conf=0.53  # genuinely knows
```
Both unknown and known queries returned `can=True` with conf ≈ 0.5.

### Root cause
`CapabilityRegistry.can_answer` computed:
```python
confidence = avg_weight * count_bonus   # count_bonus = min(len(entries)/5, 1.0)
return confidence >= 0.2, ...
```
RRF retrieval returns dialogue entries all weighted 0.5, so *any* query
that hits a few results saturates conf to ~0.5 — regardless of whether the
results are actually relevant. Confidence measured "had results", not
"can answer".

### Fix (implemented — deviates from the original direction, see below)
`can_answer` now scores the **top semantic hit** (via `Retriever.best_semantic_match`,
which uses the semantic leg directly — RRF-fused ordering is recency-biased and
surfaces unrelated recent entries first), corroborated by query↔content token overlap:

- `semantic_score` (cosine similarity) is now attached to every `MemoryEntry`
  returned by `LearnedStore.search` (derived from ChromaDB distance).
- `can_answer` returns True when: `sem ≥ 0.85` (overwhelming semantic match),
  **or** `sem ≥ 0.72` **and** the query shares a distinctive token with the hit
  (ASCII word or non-stop CJK bigram — stops like 什么/怎么 excluded).
- Uncorroborated hits are damped (`conf = sem × 0.5`) so gap detection fires.

### Why the original fix direction was NOT followed (calibration, 2026-08-02)
The documented direction was "threshold on the top result's semantic score
(e.g. cosine > 0.6)". Empirical calibration on bge-large-en-v1.5 with the real
store proved a **pure semantic threshold is not viable** for Chinese:

| query | top-1 hit | cosine |
|---|---|---|
| "Python 怎么做异步" (KNOWN) | Python venv doc | 0.786 |
| "量子场论重整化群 是什么" (UNKNOWN) | "工作: 晚上工作" | 0.780 |
| "Docker 怎么部署" (KNOWN) | Docker doc | 0.853 |
| "QuantumFlux 协议的 ZetaWave 变体？" (UNKNOWN) | fallout task | 0.654 |

Unrelated hits reach 0.62–0.78, overlapping the 0.74–0.89 range of genuine
matches — no single cosine threshold separates them (KNOWN 0.786 vs UNKNOWN
0.780 differ by 0.006). RRF ranking was additionally dominated by recency
(see #6 for the BM25 leg — once fixed, relevance dominated again). The
token-overlap corroboration is what actually discriminates: Docker/Python/
Fallout hits share query tokens; the garbage hits share none. Constants
(`_SEM_STRONG=0.85`, `_SEM_CORROBORATED=0.72`) are calibrated on the table
above — re-calibrate if the embedder changes.

### Files
- `memory_skill/contracts.py` — `MemoryEntry.semantic_score` field
- `memory_skill/learned_store.py` — attach score from ChromaDB distance
- `memory_skill/retriever.py` — `best_semantic_match()` for answerability
- `memory_skill/capability_registry.py` — new `can_answer` + token overlap
- `tests/test_integration.py` — regression tests (symptom + gap + score exposure)

---

## 2. OpenCode plugin loading is unreliable — auto-ingest falls back to protocol

**Status**: Workaround in place · **Severity**: Medium · **Found**: 2026-08-01

### Impact
`opencode-auto-memory` plugin (event-hook auto-ingest of assistant
replies) does not reliably load in OpenCode. User turns are auto-ingested
by `memory_weave`'s V10 path, but assistant replies are only stored when
the agent passes `assistant_content` per the enforced protocol.

### Root cause
OpenCode loads local plugins only from `~/.config/opencode/plugins/`
(or `.opencode/plugins/`); the `plugin` array in `opencode.json` accepts
npm package names only. Even after installing via `opencode plugin`, the
plugin did not load in this environment (silent failure; path with spaces
+ ESM suspected).

### Workaround (chosen)
Enforced per-turn protocol: `memory_weave(user_message, assistant_content)`
auto-ingests both sides. Protocol lives in `SKILL.md` and the opencode
`prompt_append`. Agent must pass `assistant_content` or half the
conversation is lost.

### Fix direction
Publish the plugin to npm (`opencode-solo-memory`) so OpenCode installs
it natively, or debug OpenCode's local-plugin loading (path-with-spaces /
ESM import). Low priority while the protocol workaround holds.

---

## 3. WSL `/mnt/d` is a 9P filesystem — `os.path.isdir` is unreliable

**Status**: Known gotcha · **Severity**: Low · **Found**: 2026-08-02

### Impact
Scripts checking mod/game directories under `/mnt/d` (Windows drives)
may get wrong `os.path.isdir` results (cached/case issues), producing
false "missing mod" reports.

### Workaround
Use `os.listdir(dir)` + set membership for exact matching, not
`os.path.isdir` per candidate.

---

## 4. Historical: `boost_weight` id format drift (fixed)

**Status**: Resolved · **Found**: 2026-08-01 · **Fixed**: commit `2634e61`

`_turn_to_entry` generated ids as `dialogue_<turn.id>` but chroma stores
`dialogue:<turn.id>`. `boost_weight`/`memory_feedback` looked up ids that
never matched → weight boosts silently no-op'd (stayed 0.5). Fixed by
mirroring the learned-store id format. Kept here as a historical record
of the failure pattern (id-format drift between retrieval and storage).

---

## 5. Historical: learning loop fake-done on anti-bot/empty pages (fixed)

**Status**: Resolved · **Found**: 2026-08-01 · **Fixed**: commits `c2b9cbf`, `c94e120`

Cloudflare challenge pages and navigation-only pages were ingested and
reported `done` with no real content. Fixed via `_is_anti_bot_or_error`
(marker + near-empty detection) at crawl time. Kept as historical record.

---

## 6. BM25 multi-word queries never matched — FTS5 AND-token semantics (fixed)

**Status**: Resolved · **Severity**: High (memory_search + RRF ranking) ·
**Found**: 2026-08-02 (during #1 calibration) · **Fixed**: 2026-08-02

### Impact
`DialogueStore.search` built the FTS5 `MATCH` expression by jieba-tokenizing
the raw query and passing the token sequence verbatim. FTS5 **AND-joins**
bare tokens, so a natural-language query tokenized into N words only matched
documents containing **all** N — in practice always 0 hits. The BM25 leg
(MISSION's "primary signal", RRF weight 2.5) was dead for multi-word queries,
so RRF ranking was dominated by the temporal/recency leg and `memory_search`
returned unrelated recent entries first (e.g. "Python 怎么做异步" → top hit
was an unrelated "Agent 人物卡").

### Diagnosis note
The failure is **not** the Chinese tokenizer — jieba tokenization works
(corpus indexed as clean words) and BM25 ranking works (single-token queries
match; `MATCH 'Python OR 异步'` ranks the venv doc first). It is purely the
default-AND MATCH semantics applied to long queries. (An earlier calibration
note in #1 briefly mislabeled this as "BM25 dead for Chinese" — corrected.)

### Fix
`DialogueStore.search` now builds a partial-match expression via
`_bm25_match_expr`: jieba tokens are filtered through a function-word
stoplist (`_CN_STOP_WORDS`), quoted (FTS5 syntax safety), and OR-joined.
BM25 `rank` naturally up-ranks documents matching more query terms, so
relevance order is preserved. InMemoryDialogueStore (tests/fakes) mirrored
the OR semantics for contract parity.

### Files
- `memory_skill/dialogue_store.py` — `_bm25_match_expr`, `search` OR-join
- `tests/fakes.py` — `InMemoryDialogueStore.search` OR semantics
- `tests/test_contract.py` — multi-word partial-match contract test

### Residual (not fixed)
The temporal leg still contributes 0.5 RRF weight; a very recent
semantically-noisy dialogue entry can outrank a genuinely relevant older
entry (e.g. "Python 怎么做异步" top-1 is the just-ingested weave turn,
venv doc is #2). Acceptable — relevance now dominates the ordering.
