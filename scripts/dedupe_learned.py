"""Merge exact-duplicate entries in learned_store (KNOWN-ISSUES #9).

Phase 1 of dedup: exact-text duplicates only.  No embedding, no LLM —
groups entries by identical content and keeps the highest-weight survivor,
deleting the rest.  Safe, fast (seconds), re-runnable.

Exact-text duplicates are unambiguous (same content = same memory), so
this phase needs no similarity threshold and cannot misfire.  Semantic
near-duplicates (similar but not identical text) are NOT touched here —
that requires embedding comparison and is a separate, riskier pass.

Usage::

    python scripts/dedupe_learned.py --dry-run   # report only (default)
    python scripts/dedupe_learned.py --apply     # backup + delete duplicates
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from memory_skill.contracts import MemorySkillConfig
from memory_skill._compose import MemorySystem

_MERGE_BUMP = 0.05
_CAP = 0.95
_TAGGED = "dedupe_exact"


def _backup(db_path: str) -> bool:
    tag = time.strftime("%Y%m%d_%H%M%S")
    targets = [db_path]
    chroma = db_path + "_chroma"
    if os.path.isdir(chroma):
        targets.append(chroma)
    done: list[str] = []
    for t in targets:
        bak = f"{t}.bak-{tag}"
        try:
            if os.path.isdir(t):
                shutil.copytree(t, bak)
            else:
                shutil.copy2(t, bak)
            done.append(bak)
        except Exception as exc:
            print(f"  backup FAILED for {t}: {exc}")
            for d in done:
                shutil.rmtree(d, ignore_errors=True)
            return False
    print(f"  backup -> {', '.join(done)}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Merge exact-duplicate learned entries (#9)")
    ap.add_argument("--apply", action="store_true", help="backup + delete duplicates (default: dry-run)")
    ap.add_argument("--db", default=os.path.join(ROOT, "opencode_memory.db"),
                    help="path to the memory db")
    args = ap.parse_args()

    ms = MemorySystem(MemorySkillConfig(db_path=args.db))
    store = ms.learned_store

    entries = store.search("", limit=100000)
    print(f"entries: {len(entries)}")

    if args.apply:
        if not _backup(args.db):
            print("aborting: backup failed")
            return 1
        print("running with backups in place")

    groups: dict[str, list] = defaultdict(list)
    for e in entries:
        content = (e.content or "").strip()
        if not content:
            continue
        if (e.metadata or {}).get(_TAGGED):
            continue
        groups[content].append(e)

    dup_groups = {c: es for c, es in groups.items() if len(es) > 1}
    print(f"exact-duplicate groups: {len(dup_groups)}")

    merged = 0
    deleted = 0
    survivors = 0
    for content, es in sorted(dup_groups.items(), key=lambda kv: -len(kv[1])):
        keeper = max(es, key=lambda e: e.weight)
        losers = [e for e in es if e.id != keeper.id]
        merged += 1
        if args.apply:
            bumped = min(keeper.weight + _MERGE_BUMP * len(losers), _CAP)
            ext = {**(keeper.metadata or {}), _TAGGED: True}
            store.update(
                keeper.id, keeper.content,
                metadata={
                    "ext_metadata_json": json.dumps(ext, ensure_ascii=False),
                    "weight": bumped,
                },
            )
            for loser in losers:
                store.delete(loser.id)
            deleted += len(losers)
            survivors += 1

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] merged groups: {merged}")
    print(f"  deleted duplicates: {deleted}")
    print(f"  survivors bumped: {survivors}")
    if not args.apply:
        print("  (dry-run — rerun with --apply to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
