# Known Issues

Known defects and architectural debts, tracked for dedicated fixes.
Each entry: impact, root cause, and the fix direction — so a future
session can pick it up without re-diagnosing.

> 编号说明：`#7` 缺失为历史遗留（早期条目被移除后未重排）。
> 2026-08-07 曾出现两个 `#8`（中文+英文），英文条目已重编号为 `#11`。

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

**Status**: Resolved · **Severity**: Medium · **Found**: 2026-08-01 · **Fixed**: 2026-08-02

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

### Hook-layer hard enforcement — feasible (researched 2026-08-02)
The protocol workaround is unreliable because it depends on the agent
*choosing* to call `memory_weave` every turn (live failure: whole rounds
skipped it). A framework-level hard constraint is achievable — verified
against the local `@opencode-ai/plugin` types and the working
`@code-yeongyu/comment-checker` plugin (oh-my-openagent):

- **Verified working — tool-chain injection**: `tool.execute.before` /
  `tool.execute.after` hooks receive `{tool, sessionID, callID}` and can
  rewrite the tool's `args`/`output`. comment-checker uses exactly this to
  force a response to every flagged comment (live in this session). A
  memory-enforcement hook reuses the pattern: on `tool.execute.before`,
  read `client.session.messages({path:{id:sessionID}})` for the latest user
  message, check whether `memory_weave` was called since it arrived, and if
  not, inject "YOU MUST CALL memory_weave (user_message, assistant_content)
  BEFORE any other tool" via the output. This is a *hard* constraint —
  framework-injected at tool-call time, not prompt advice the agent can
  ignore.
- **Unverified — transparent injection (experimental API)**: 
  `experimental.chat.messages.transform` (rewrite the message array before
  the LLM call) and `experimental.chat.system.transform` (rewrite system
  prompt) could make memory context arrive *without* the agent calling
  weave at all — MISSION's "transparent to the agent" goal. `chat.message`
  part-pushing was ruled out (EventV2 branded ids); these transform hooks
  operate on the existing message array and may bypass that restriction.
  **Not tested on this opencode build — verify before relying on it**
  (experimental API, may change).

### Implementation (2026-08-02, chosen path A — tool-chain enforcement)
`opencode-auto-memory/index.js` now registers a `tool.execute.after` hook:
- Fetches `client.session.messages()`, finds the latest user message,
  checks whether a `memory_weave` tool call exists after it.
- If not (and the executed tool isn't itself `memory_weave`, and the tool
  didn't fail), appends the ENFORCEMENT_WARNING to the tool result.
- Warns at most once per user turn (per-session `warnedBySession` map).
- Offline-verified 7/7 scenarios (warn / no-warn / self / dedup / reset /
  failure-skip). Deployed to `~/.config/opencode/plugins/solo-memory.js`.
- The `experimental.*` transform route (path B) was deliberately not used:
  experimental API may break on the next opencode update.

### Live verification — two real bugs found & fixed (2026-08-02)
Live verify (intentional violation: run a tool without calling memory_weave)
initially showed the hook never fired. Debug instrumentation (console.log +
`/tmp/solo-memory-debug.log` — console.log IS captured into opencode's log,
proven by Systematic's "initialized" line) traced two real defects, both in
opencode's plugin contract vs. this plugin's code:

1. **Export format — plugin was never loaded**. opencode's loader (verified
   by reverse-engineering the compiled binary) reads `q.default` and requires
   file/path plugins to `export id` (`rQ`/`tQ` in the loader source). The
   original `export const plugin = ...` + `export { x as server }` was silently
   ignored. Fixed to `export const id = "solo-memory"; export default { id,
   server: opencodeAutoMemory }` — matching oh-my-openagent's proven shape
   (`return { id, server }` + `export { x as default }`).
2. **SDK response shape — hook fired but silently bailed**. `client.session.messages()`
   returns `{ data: [...] }` (SDK client wrapper), not a bare array; the code's
   `if (!msgs || !msgs.length) return;` saw `msgs.length === undefined` and
   always exited. Fixed with comment-checker's `normalizeSDKResponse` pattern:
   `const msgs = Array.isArray(resp) ? resp : (resp?.data ?? [])`.

Final live verify PASSED (2026-08-02 01:46): a deliberate no-weave tool call
returned `[MEMORY PROTOCOL ENFORCEMENT — ACTION REQUIRED]` in the tool result;
debug log shows `msgs: 128 → lastUser → ENFORCEMENT_WARNING appended`. #2 closed.

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

---

## 8. 强制 weave 硬阻塞在高频工具会话中是净负担（超时 + 开销）

**Status**: Resolved · **Severity**: High · **Found**: 2026-08-07（4天 Skyrim mod 搬运会话）· **Fixed**: 2026-08-07

### Impact
每次工具调用前被硬 block 强制要求 `memory_weave`（未调用则拒绝执行工具）。
在长会话高频交互下（上百轮、每轮多次工具调用），该机制：
- 显著打断流程（每轮都要先 weave 才能干活）
- 多次出现 `MCP error -32001: Request timed out`（传超长 `assistant_content` 时向量化慢 → 超时）
- 用户亲自提问"为什么 weave 失败"、"记忆你有正常使用吗"——体验受损

### Symptom
```
[工具调用被拒] Tool execution rejected: memory_weave has not been called...
[MCP error] -32001: Request timed out（连续3次）
```
超时集中在 `assistant_content` 传超长文本时（如整段总结/分析结论）。

### Root cause
- 每次调用前强制 weave → 高频会话中 weave 调用频率 = 工具调用频率，开销放大
- `assistant_content` 全文传入 → embedding 推理耗时 → 超时
- weave 返回值在多数轮次未被实际使用（注入质量低，见 #9）

### Fix (implemented 2026-08-07)
1. **硬阻塞降频**（部署版 `~/.config/opencode/plugins/solo-memory.js`）：`tool.execute.before`
   从"每个工具调用前都硬阻塞"改为**每 user 轮次至多硬阻塞一次**（`hardBlockedBySession`
   Map 记录该轮已阻塞；同一轮内后续工具放行）。高频会话不再每轮被打断，
   而协议合规仍被首轮强制兜底。
2. **超时根因消除**（`memory_skill/mcp_tools.py`）：`_weave` 自动 ingest 前将
   `user_message`/`assistant_content` 通过 `_clip_auto_ingest` 截断（**头尾保留**：
   前 490 + 后 300 字符，中段省略；总长 ≤ 800）。中文长回复的结论常在结尾，
   从头截断会丢结论（见 #9），头尾保留同时保住上下文和结论，长文本 embedding
   不再触发 `-32001`。嵌入层本就只有 512 token 位置编码（bge-large-en-v1.5），
   800 字符（≈802 token）已超限，截断对检索质量无损失。
3. 仓库版 `opencode-auto-memory/index.js` 已同步（此前部署版含 HARD BLOCK、
   仓库版只有软警告——两版漂移）。

### Files
- `~/.config/opencode/plugins/solo-memory.js`（部署版）与 `opencode-auto-memory/index.js`（仓库版）
- `memory_skill/mcp_tools.py` — `_MAX_AUTO_INGEST_CHARS` 截断

---

## 9. 记忆存储"过程"而非"知识"——大量对话残片、结论未结构化

**Status**: Partially resolved · **Severity**: Medium · **Found**: 2026-08-07 · **Fixed (dedup + cleanup)**: 2026-08-07

### Impact
4 天会话后 learned_store 从 551 → 863 条（+312），但检索发现绝大多数是
`dialogue:mcp_*` 对话残片（user 原话 + assistant 推理过程 + mission 重复三份），
**真正可复用的结论型知识没有被单独提炼**。用户问"之前怎么解决的"时，
仍需搜索 + 拼凑 + 重新推理，而不是直接命中一条干净结论。

### Symptom
```
# 检索"镜像排序规则" → 命中5条 dialogue:mcp_assistant_*（过程描述），无一条是结论
# 检索"PinkieRose 依赖" → 横跨5+条从"废案"到"必须装"的过程，结论散落
```
863 条里可复用知识估计仅 50-80 条，本次 +312 条大部分是噪音。

### Root cause
- 所有记忆以 `dialogue:` 前缀 + `default` 分类存储，无结构化知识类型
- 无"结论提炼"环节——重要决策（装/跳过/原因）没有独立成条
- 同一话题存 user+assistant+mission 三条重复，relevance 全是 0.5（无重要性筛选）

### Fix (dedup part implemented 2026-08-07)
`opencode-auto-memory/index.js`（部署版 `solo-memory.js`）的 `event` hook 现在检测
该 user 轮次是否已调用 `memory_weave`（weave 已自动 ingest 双份）——若已 weave 则
跳过 `ingest_pair`，消除同一轮被存两次的重复。`_weave` 的自动 ingest 截断到 800 字符
同时减少了过程残片的冗余长度。

### Fix (cleanup implemented 2026-08-07)
`scripts/cleanup_fragments.py` 启发式重权（零 LLM 成本，软操作不删除）：
- **碎片降权**：`dialogue:mcp_*` 中 <40 字符的闲聊残片（279 条，如"今天中午吃什么"）
  → weight 0.5 → 0.3 + `metadata.fragment=True`——仍可检索但排最后
- **结论提权**：≥200 字符且含结论关键词（结论/原因/决定/修复/验证/依赖…）的
  120 条 → weight 0.5 → 0.6 + `metadata.knowledge=True`——排最前
- 幂等可重跑；`--apply` 才写库（默认 dry-run）
- 分类归档到 pref/pers/skill/mission 由 `scripts/reclassify_memories.py`（LLM）
  完成，独立于本脚本

### Remaining (结论沉淀/分类, 未实现)
1. **结论沉淀**：在关键决策点（装了/跳过了/为什么）显式 `memory_ingest` 一条干净的结论条目
   ——已通过 SKILL.md Search-First Discipline 与 prompt_append 引导（见 #11/#10）
2. **历史残片清理**：既有 `dialogue:mcp_*` 残片已降权（fragment=True），
   定期清理可调用 `cleanup_fragments.py` 幂等执行

---

## 10. 记忆跨会话连续性有效，但推理链未保留

**Status**: Resolved (documented) · **Severity**: Low · **Found**: 2026-08-07 · **Fixed**: 2026-08-07

### Impact
记忆模块最有价值的一次使用：用户问"之前怎么解决的"时，memory_search
找到了 8 月 4 日的"MO2 界面顺序 = 文件顺序镜像"锚点验证记录，直接修正了
当天的排序误判。**跨会话的"事实"记住了，但"为什么"（推理链）丢失**——
每次都要重新推演。

### Symptom
```
# 有价值（存了事实）: "界面最后=文件第1行=高优先级"（8/4 锚点）
# 丢失（没存推理）: 该结论基于"用户提供锚点"验证，而非凭空推出
```

### Root cause
记忆只存"当时发生了什么/说了什么"，不存"为什么这是对的"（依据、验证方法）。

### Fix (implemented 2026-08-07 — behavior-layer, documented)
SKILL.md 新增 **Preserve the Reasoning Chain** 段落 + `prompt_append` 注入
REASONING CHAIN 规则：结论条目必须附带依据/验证方法（"由用户锚点 X 验证"），
附 GOOD/BAD 示例。这是行为引导——未来会话能否复用完整推理取决于 agent 是否遵守。


---

## 11. Memory module usage pattern in a marathon session (2026-08-01 ~ 08-07) — passive-inject rich, active-use nil

**Status**: Resolved (documented) · **Severity**: Medium · **Found**: 2026-08-07 · **Fixed**: 2026-08-07

### Impact
A ~6-day, 100+ turn session (LL mod research → TuLED modpack triage → ENB swap → mod migration →
new modpack review) used the memory module heavily in *passive* mode but almost never *actively*:
`memory_search` was called ~2 times (both after the user pointed it out), `memory_ingest` 0 times
(depends entirely on system auto-ingest), and the `memory-skill`/`memory-protocol` skills were never
loaded. The module itself worked correctly; the agent treated `memory_weave` as a compliance ritual
instead of an information source.

### Symptom
- `/tmp` scratch data was wiped mid-session; agent re-crawled 302 pages instead of searching memory
  for prior findings (hours wasted).
- Cross-modpack comparisons and the crash triage chain relied on conversation context, not retrieval;
  several evidence links (e.g. "dxgi.dll is elderroll's ReShade") survived only because weave
  re-injected them.
- User explicitly called this out mid-session: "you basically did not use the memory module's skill,
  mission, and proactive retrieval".

### Root cause
1. The agent's behavior model defines "memory" as *call weave once per turn* (compliance), not
   *search first when hitting an information gap* (capability).
2. Long, tool-dense sessions create an illusion of "context is enough" — so search never fires.
3. Deep immersion in the execution flow turns weave into muscle memory.

### One related incident
The hard-block deadlock (2026-08-04): the plugin's `MEMORY_TOOL_PREFIX = "memory_"` whitelist did
not match the MCP tool's real name `opencode-memory_memory_weave`, so `memory_weave` blocked itself.
All tools were rejected until the user hand-edited the prefix (and `p.tool.includes("memory_weave")`).

### Fix direction
- Train the agent to call `memory_search` before re-doing any crawl/computation ("has this been
  researched before?"), and before starting a new sub-topic.
- Actively `memory_ingest` key decisions (e.g. "TuLED and elderroll share D:\game\SkyrimSE") instead
  of relying on auto-ingest.
- Load `memory-skill` at least once to internalize the protocol; consider a "search-first" nudge
  in the weave output when the current topic matches an existing memory branch.
- Whitelist check should use `tool.includes("memory_weave")` (not prefix equality) to avoid the
  self-deadlock class of bug.

### Fix (implemented 2026-08-07 — behavior-layer, documented)
SKILL.md 新增 **Search-First Discipline** 段落 + `prompt_append` 注入 SEARCH-FIRST 规则：
信息缺口 → 先 `memory_search`（"以前研究过吗？"）再重爬/重算；新子主题前检索旧工作；
关键决策显式 `memory_ingest`。附失败案例（/tmp 被清后重爬 302 页）作为反面教材。
这是行为引导——agent 是否主动用检索取决于其遵守程度。
