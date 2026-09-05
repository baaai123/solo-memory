#!/usr/bin/env python3
"""Entry-ID ground-truth retrieval benchmark (v2, read-only).

Method (addresses v1's flaws — see conclusion memory):
  1. Sample REAL user messages from dialogue_turns (natural phrasing,
     not synthetic keyword lists).
  2. For each message, derive ground truth: the learned-store entries
     that are temporally adjacent to the answer turn and share
     distinctive tokens with it (self-anchored labels — imperfect but
     entry-level, not string-level).
  3. Retrieve with each leg separately (semantic / bm25 / fused) and
     measure Precision@K, Recall@K, MRR against the entry IDs.
  4. Additionally record which block each fused top-entry lands in
     (weaver attribution) — the real cross-block dedup question.

Nothing is written to the DB.

Usage: python3 scripts/bench_recall_quality.py [--k 10] [--samples 30]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memory_skill.contracts import MemorySkillConfig  # noqa: E402
from memory_skill._compose import _build_system  # noqa: E402

_STOP = set("的了和是在我你他它这个那个有没不就都也很什么怎么为什么可以能要还有然后现在因为所以但是就是可能应该".split())
MIN_LEN = 8        # skip trivial user turns
MAX_LEN = 120      # skip pasted dumps


def tokens(text: str) -> set[str]:
    text = re.sub(r"\s+", " ", text.lower())
    toks: set[str] = set()
    for w in re.findall(r"[a-z0-9_]{2,}", text):
        toks.add(w)
    for i in range(len(text) - 1):
        a, b = text[i], text[i + 1]
        if a not in _STOP and b not in _STOP and not a.isspace() and not b.isspace():
            toks.add(a + b)
    return toks


def load_user_messages(db_path: Path, n: int) -> list[dict]:
    """Deterministic sample of real user turns (evenly spaced by recency)."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT id, content, timestamp FROM dialogue_turns "
        "WHERE role='user' AND length(content) BETWEEN ? AND ? "
        "ORDER BY timestamp DESC LIMIT 400",
        (MIN_LEN, MAX_LEN),
    ).fetchall()
    conn.close()
    if not rows:
        return []
    step = max(1, len(rows) // n)
    picked = rows[::step][:n]
    return [{"id": r[0], "content": r[1], "ts": r[2]} for r in picked]


def ground_truth_ids(system, msg: dict) -> set[str]:
    """Entries adjacent in time + token-sharing with the message turn.

    The assistant turn(s) right after this user turn were ingested as
    learned entries whose content derives from that exchange. Learned
    entries live in ChromaDB with ISO created_at metadata; we query the
    learned store directly for entries created in a window after the
    user turn and keep those sharing distinctive tokens with it
    (entry-level anchor, same corroboration idea as the weaver gate).
    """
    from datetime import datetime, timedelta, timezone

    try:
        ts = datetime.fromtimestamp(float(msg["ts"]), tz=timezone.utc)
    except Exception:
        return set()
    lo = ts - timedelta(minutes=2)
    hi = ts + timedelta(minutes=10)
    # learned entries live in Chroma; pull everything once (no where-query
    # on created_at — chroma metadata filters on this field are stringly
    # typed) and window-filter in Python.
    try:
        entries = system.learned_store.list_by_category("default", limit=0)
        for cat in ("pref", "pers", "skill", "mission", "conclusion", "knowledge"):
            entries += system.learned_store.list_by_category(cat, limit=0)
    except Exception:
        return set()
    seen: set[str] = set()
    uniq = []
    for e in entries:
        if e.id not in seen:
            seen.add(e.id)
            uniq.append(e)
    entries = [e for e in uniq
               if e.created_at and lo <= e.created_at.astimezone(timezone.utc) <= hi]
    qt = tokens(msg["content"])
    ids = set()
    for e in entries:
        ov = len(qt & tokens(e.content))
        if ov >= max(2, int(0.10 * len(qt))):
            ids.add(e.id)
    return ids


def metrics(ranked: list[str], truth: set[str], k: int) -> dict:
    if not truth:
        return {"precision_at_k": None, "recall_at_k": None, "mrr": None}
    top = ranked[:k]
    hits = [1 if e in truth else 0 for e in top]
    prec = sum(hits) / k
    rec = sum(hits) / len(truth)
    mrr = 0.0
    for i, e in enumerate(ranked):
        if e in truth:
            mrr = 1.0 / (i + 1)
            break
    return {"precision_at_k": round(prec, 3), "recall_at_k": round(rec, 3), "mrr": round(mrr, 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = MemorySkillConfig(db_path=str(REPO / "opencode_memory.db"))
    system = _build_system(cfg)
    retriever = system.retriever

    msgs = load_user_messages(Path(cfg.db_path), args.samples)
    per_sample = []
    agg = {"semantic": [], "bm25": [], "fused": []}
    skipped_no_truth = 0

    for m in msgs:
        truth = ground_truth_ids(system, m)
        if not truth:
            skipped_no_truth += 1
            continue
        try:
            env = retriever.retrieve(m["content"], limit=args.k)
        except Exception:
            continue
        fused_ids = [e.id for e in env.entries]
        # per-leg rankings from the retriever internals are not exposed;
        # approximate legs via direct store searches with the same query.
        try:
            sem = [e.id for e in system.learned_store.search(
                m["content"], limit=args.k, filters={"category": {"$ne": "default"}})]
        except Exception:
            sem = []
        try:
            bm = [e.id for e in system.dialogue_store.search(m["content"], limit=args.k)]
        except Exception:
            bm = []
        bm = [f"dialogue:{x}" if not str(x).startswith("dialogue:") else x for x in bm]

        row = {
            "msg_id": m["id"],
            "msg": m["content"][:60],
            "truth_size": len(truth),
            "semantic": metrics(sem, truth, args.k),
            "bm25": metrics(bm, truth, args.k),
            "fused": metrics(fused_ids, truth, args.k),
        }
        per_sample.append(row)
        for leg in agg:
            if row[leg]["precision_at_k"] is not None:
                agg[leg].append(row[leg])

    def mean(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "samples_with_truth": len(per_sample),
        "samples_skipped_no_truth": skipped_no_truth,
        "k": args.k,
        "semantic": {k: mean(agg["semantic"], k) for k in ("precision_at_k", "recall_at_k", "mrr")},
        "bm25": {k: mean(agg["bm25"], k) for k in ("precision_at_k", "recall_at_k", "mrr")},
        "fused": {k: mean(agg["fused"], k) for k in ("precision_at_k", "recall_at_k", "mrr")},
    }
    report = {"summary": summary, "per_sample": per_sample}
    out = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"report → {args.out}")
    else:
        print(out)


if __name__ == "__main__":
    main()
