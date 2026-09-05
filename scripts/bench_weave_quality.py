#!/usr/bin/env python3
"""Read-only weave-quality benchmark against the REAL memory DB.

Runs multiple realistic queries through MemorySystem.weave() without
writing anything (no ingest, no classify — weave itself is read-only).
For each query it captures per-block content and entry ids, then
computes:

  - cross-block entry duplication (same entry id in >1 block)
  - per-block char sizes and total weave size
  - tier2 unit overlap with nudge/title/skill blocks (structural dup)
  - nudge relevance (token overlap with the query)
  - directive overhead (non-memory tokens)

Output: JSON report to stdout (or --out file).

Usage:
    python3 scripts/bench_weave_quality.py --queries 8
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memory_skill.contracts import MemorySkillConfig  # noqa: E402
from memory_skill._compose import _build_system  # noqa: E402
from memory_skill.protocol_state import ProtocolState  # noqa: E402
from memory_skill.weaver import WeaverStores, weave as weaver_weave  # noqa: E402


def gateless_weave(system, user_message: str):
    """Call the weaver directly, bypassing ProtocolGate — benchmark only.

    The gate exists to discipline *agents* (classify before acting); the
    benchmark must not mutate protocol state (weave_count, gates) or the
    real conversation is perturbed. weaver.weave() itself is read-only.
    """
    stores = WeaverStores(
        saw_buffer=system.saw_buffer,
        dialogue_store=system.dialogue_store,
        learned_store=system.learned_store,
        retriever=system.retriever,
        agent_name=system.config.agent_name,
        namespace=system.config.namespace,
        tree=system.tree,
        gaps=system.gaps,
        pending_store=getattr(system, "pending_store", None),
        mission_store=system._ensure_mission_store(),
        degraded=system.embedder.degraded,
        degraded_reason=system.embedder.reason,
        character=getattr(system, "character", None),
        protocol=ProtocolState(),
        learning_queue=getattr(system, "learning_queue", None),
        state_store=getattr(system, "state_store", None),
        relation_store=getattr(system, "relation_store", None),
    )
    return weaver_weave(
        stores, user_message, "",
        character_role=system._resolve_character_role())

# Realistic queries drawn from this user's actual work streams.
DEFAULT_QUERIES = [
    "记忆模块 weaver 上下文注入 重复 截断",
    "酒馆角色 状态维 人设 relation store",
    "solo-memory 备份 迁移 整库快照",
    "dsh 插件 autoBackup turn-stopping",
    "RimWorld mod packageId 重复",
    "docker memorier 模型就绪 embedder 降级",
    "图像生成 超分 画风 分辨率",
    "memory-ui 多用户 账号 注册",
]

BLOCK_ATTRS = [
    "tier1_context", "tier2_context", "memory_nudge", "skill_context",
    "mission_context", "pref_context", "pers_context", "conclusion_context",
    "title_preview", "historic_hint", "gap_context", "todo_context",
    "pending_context", "tree_context", "tree_nav", "tavern_context",
]

_STOP = set("的了和是在我你他它这个那个有没不就都也很什么怎么为什么可以能要".split())


def tokens(text: str) -> set[str]:
    """Cheap CJK bigram + latin word tokenizer for overlap checks."""
    text = re.sub(r"\s+", " ", text.lower())
    toks: set[str] = set()
    for w in re.findall(r"[a-z0-9_]{2,}", text):
        toks.add(w)
    for i in range(len(text) - 1):
        a, b = text[i], text[i + 1]
        if a not in _STOP and b not in _STOP and not a.isspace() and not b.isspace():
            toks.add(a + b)
    return toks


def entry_ids_in(block_text: str) -> list[str]:
    """Extract dialogue entry ids referenced by a block (turn_id forms)."""
    return re.findall(r"dialogue:[A-Za-z0-9_]+", block_text or "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = MemorySkillConfig(db_path=str(REPO / "opencode_memory.db"))
    t0 = time.time()
    system = _build_system(cfg)
    load_s = time.time() - t0

    queries = DEFAULT_QUERIES[: args.queries]
    per_query = []
    dup_totals: dict[str, int] = {}
    for q in queries:
        ctx = gateless_weave(system, q)
        blocks = {}
        for attr in BLOCK_ATTRS:
            val = getattr(ctx, attr, "") or ""
            if val.strip():
                blocks[attr] = val
        total_chars = sum(len(v) for v in blocks.values())

        # nudge relevance: overlap of query tokens with nudge entry tokens
        nudge = blocks.get("memory_nudge", "")
        nudge_rel = []
        if nudge:
            qt = tokens(q)
            for part in nudge.split(" | "):
                ov = len(qt & tokens(part)) / max(1, len(qt))
                nudge_rel.append(round(ov, 3))

        # tier2 vs other-block content duplication (same 50-char snippet lines)
        tier2 = blocks.get("tier2_context", "")
        tier2_lines = {ln for ln in tier2.splitlines() if len(ln) > 20}
        overlap_lines = []
        for name, val in blocks.items():
            if name == "tier2_context":
                continue
            for ln in val.splitlines():
                if len(ln) > 20 and ln in tier2_lines:
                    overlap_lines.append((name, ln[:60]))

        per_query.append({
            "query": q,
            "block_sizes": {k: len(v) for k, v in blocks.items()},
            "total_chars": total_chars,
            "nudge_count": len(nudge_rel),
            "nudge_relevance": nudge_rel,
            "tier2_cross_block_line_dups": overlap_lines,
            "blocks_present": list(blocks.keys()),
        })

    # Aggregate: directive overhead share
    dir_chars = sum(
        pq["block_sizes"].get("gap_context", 0) +
        pq["block_sizes"].get("todo_context", 0) +
        120  # fixed [分类指令] boilerplate
        for pq in per_query)
    all_chars = sum(pq["total_chars"] for pq in per_query)

    nudge_all = [r for pq in per_query for r in pq["nudge_relevance"]]
    low_rel = sum(1 for r in nudge_all if r < 0.05)

    summary = {
        "db_path": cfg.db_path,
        "model_load_seconds": round(load_s, 2),
        "queries": len(per_query),
        "avg_total_chars": round(all_chars / max(1, len(per_query))),
        "avg_directive_chars": round(dir_chars / max(1, len(per_query))),
        "nudge_items_total": len(nudge_all),
        "nudge_items_low_relevance(<0.05 overlap)": low_rel,
        "nudge_items_low_relevance_pct": round(100 * low_rel / max(1, len(nudge_all))),
        "queries_with_tier2_line_dups": sum(1 for pq in per_query if pq["tier2_cross_block_line_dups"]),
    }

    report = {"summary": summary, "per_query": per_query}
    out = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"report → {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
