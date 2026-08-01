# Known Issues

Known defects and architectural debts, tracked for dedicated fixes.
Each entry: impact, root cause, and the fix direction — so a future
session can pick it up without re-diagnosing.

---

## 1. `can_answer` confidence is polluted by result count — gap detection is broken

**Status**: Open · **Severity**: High · **Found**: 2026-08-02

### Impact
`memory_gaps` always returns 0. The learning loop's gap-detection leg is
effectively dead — the system never discovers *what it doesn't know*, so
`gap → decide → learn` never fires automatically (manual `memory_learn`
still works).

### Symptom
```
can_answer("量子场论重整化群 是什么") → can=True, conf=0.5   # system knows NOTHING about this
can_answer("Docker 怎么部署")        → can=True, conf=0.53  # genuinely knows
```
Both unknown and known queries return `can=True` with conf ≈ 0.5.

### Root cause
`CapabilityRegistry.can_answer` computes:
```python
confidence = avg_weight * count_bonus   # count_bonus = min(len(entries)/5, 1.0)
return confidence >= 0.2, ...
```
RRF retrieval returns dialogue entries all weighted 0.5, so *any* query
that hits a few results saturates conf to ~0.5 — regardless of whether the
results are actually relevant. Confidence measures "had results", not
"can answer".

### Fix direction
Give retrieval a **semantic relevance score** and use it in `can_answer`
(and gap detection), instead of weight × count:
- LearnedStore already embeds entries; compute query-vs-result cosine in
  `retriever` and expose it on `MemoryEntry` (e.g. `semantic_score`).
- `can_answer` should threshold on the **top result's semantic score**
  (e.g. cosine > 0.6), not aggregate weight.
- GapDetector.detect then correctly produces a Gap when the top hit is
  irrelevant.

### Files
- `memory_skill/capability_registry.py` — `can_answer`
- `memory_skill/retriever.py` — expose per-entry semantic score
- `memory_skill/gap_detector.py` — may need threshold tuning

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
