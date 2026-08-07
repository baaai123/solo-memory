"""Clean up dialogue fragment noise in learned_store (KNOWN-ISSUES #9).

The marathon-session auto-ingest left ~772 `dialogue:mcp_*` entries in
category `default` with uniform weight 0.5 — mostly process fragments
(short chit-chat turns, reasoning noise) with a minority of real
conclusions. This script re-weights them heuristically (zero LLM cost):

  - SHORT fragments (<40 chars, weight <= 0.5): drop to 0.3, tag
    metadata["fragment"]=True — still retrievable, but rank last.
  - KNOWLEDGE entries (>=200 chars containing conclusion keywords):
    raise to 0.6, tag metadata["knowledge"]=True — rank first.

Soft operations only (weight/metadata), never deletes. Re-runnable —
already-tagged entries are skipped.

Usage::

    python scripts/cleanup_fragments.py --dry-run   # report only (default)
    python scripts/cleanup_fragments.py --apply     # write changes

For LLM-based reclassification (default -> pref/pers/skill/mission),
use the separate scripts/reclassify_memories.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from memory_skill.contracts import MemorySkillConfig
from memory_skill._compose import MemorySystem

_FRAGMENT_MAX_CHARS = 40
_FRAGMENT_WEIGHT = 0.3
_KNOWLEDGE_MIN_CHARS = 200
_KNOWLEDGE_WEIGHT = 0.6
_CONCLUSION_KEYWORDS = (
    "结论", "原因", "决定", "修复", "改为", "验证", "成功", "完成",
    "规则", "依赖", "推荐", "应该", "必须", "不兼容", "冲突",
)


def _classify_len(content: str) -> int:
    if len(content) < _FRAGMENT_MAX_CHARS:
        return "fragment"
    if len(content) >= _KNOWLEDGE_MIN_CHARS and any(
        kw in content for kw in _CONCLUSION_KEYWORDS
    ):
        return "knowledge"
    return "keep"


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-weight dialogue fragments (#9)")
    ap.add_argument("--apply", action="store_true", help="write changes (default: dry-run)")
    ap.add_argument("--db", default=os.path.join(ROOT, "opencode_memory.db"),
                    help="path to the memory db")
    args = ap.parse_args()

    ms = MemorySystem(MemorySkillConfig(db_path=args.db))
    entries = ms.learned_store.search("", limit=10000)

    mcp = [
        e for e in entries
        if e.id.startswith("dialogue:")
        and (e.metadata or {}).get("turn_id", "").startswith("mcp_")
    ]

    counts = {"fragment": 0, "knowledge": 0, "already": 0, "keep": 0}
    for e in mcp:
        meta = e.metadata or {}
        if meta.get("fragment") or meta.get("knowledge"):
            counts["already"] += 1
            continue
        kind = _classify_len(e.content)
        counts[kind] = counts.get(kind, 0) + 1
        if not args.apply or kind == "keep":
            continue
        # NOTE: LearnedStore.update() writes `metadata` as flat Chroma keys,
        # but entry.metadata is round-tripped via `ext_metadata_json` — so
        # flags must live inside ext_metadata_json to be readable back.
        ext = {**meta, kind: True}
        if kind == "fragment":
            ms.learned_store.update(
                e.id, e.content,
                metadata={
                    "ext_metadata_json": json.dumps(ext, ensure_ascii=False),
                    "weight": _FRAGMENT_WEIGHT,
                },
            )
        else:
            ms.learned_store.update(
                e.id, e.content,
                metadata={
                    "ext_metadata_json": json.dumps(ext, ensure_ascii=False),
                    "weight": _KNOWLEDGE_WEIGHT,
                },
            )

    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"[{mode}] dialogue:mcp_* entries: {len(mcp)}")
    print(f"  fragment (<{_FRAGMENT_MAX_CHARS}ch → w={_FRAGMENT_WEIGHT}): {counts['fragment']}")
    print(f"  knowledge (>={_KNOWLEDGE_MIN_CHARS}ch + conclusion kw → w={_KNOWLEDGE_WEIGHT}): {counts['knowledge']}")
    print(f"  already tagged: {counts['already']}")
    print(f"  keep: {counts['keep']}")
    if not args.apply:
        print("  (dry-run — rerun with --apply to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
