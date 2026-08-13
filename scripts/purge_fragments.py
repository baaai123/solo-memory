"""Purge unclassified fragments from learned_store (C plan, phase 4).

Phase 1 (fragment isolation) already stopped tier2/nudge from being
polluted by ``default`` fragments.  This script removes the *historical*
fragments that isolation can no longer hide behind: they still sit in
the chroma collection and dilute explicit ``memory_search``.

What it deletes (category == "default"):
  - every fragment with weight < 0.85 (the uniform auto-ingest weight was
    0.5; weight >= 0.85 means the entry was promoted/boosted = keep)
  - noise fragments regardless of weight: one-token replies ("是", "ok"),
    "[internal]"/"[SYSTEM DIRECTIVE]" scaffolding, "loaded: ..." echoes

What it KEEPS:
  - weight >= 0.85 with meaningful content (e.g. a distilled doc chunk)
  - dialogue_store originals are untouched (separate SQLite store — BM25
    full-text search of raw turns keeps working)

Safety: backs up the chroma dir before applying; dry-run by default.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from memory_skill.contracts import MemorySkillConfig
from memory_skill._compose import MemorySystem

_WEIGHT_KEEP = 0.85
_CONCLUSION_KEYWORDS = (
    "结论", "原因", "决定", "修复", "改为", "验证", "成功", "完成",
    "规则", "依赖", "推荐", "应该", "必须", "不兼容", "冲突",
    "关键", "发现", "教训", "根因", "方案", "建议",
)
_NOISE_PATTERNS = (
    re.compile(r"^\s*(是|好|嗯|ok|好的|可以|继续|test|loaded|done|完成)\s*$", re.I),
    re.compile(r"\[internal\]", re.I),
    re.compile(r"\[SYSTEM DIRECTIVE", re.I),
    re.compile(r"^loaded: ", re.I),
    re.compile(r"^(running|finished|exit(ed)?) (code )?\d+", re.I),
)


def _is_noise(content: str) -> bool:
    return any(p.search(content) for p in _NOISE_PATTERNS)


def _has_conclusion(content: str) -> bool:
    return any(k in content for k in _CONCLUSION_KEYWORDS)


def _collect(skill) -> tuple[list, list, list]:
    """Return (to_delete, conclusion_keep, high_weight_keep)."""
    all_def = skill.learned_store.list_by_category("default", limit=0)
    to_delete: list[tuple[str, str, float]] = []
    conclusion_keep: list[tuple[str, str, float]] = []
    high_keep: list[tuple[str, str, float]] = []
    for e in all_def:
        if _has_conclusion(e.content):
            # Likely real knowledge from pre-rewrite distillation — keep
            # (demote to 0.6) so a later distill pass can promote it.
            conclusion_keep.append((e.id, e.content, e.weight))
        elif e.weight >= _WEIGHT_KEEP:
            high_keep.append((e.id, e.content, e.weight))
        elif _is_noise(e.content):
            to_delete.append((e.id, e.content, e.weight))
        else:
            # Ordinary low-weight fragment: no conclusion markers, no
            # promotion — safe to purge (dialogue originals survive).
            to_delete.append((e.id, e.content, e.weight))
    return to_delete, conclusion_keep, high_keep


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", default="opencode_memory.db",
                        help="SQLite db path (chroma is <db>_chroma)")
    parser.add_argument("--apply", action="store_true",
                        help="backup + actually delete (default: dry-run)")
    args = parser.parse_args()

    config = MemorySkillConfig(db_path=args.db)
    skill = MemorySystem(config)

    to_delete, conclusion_keep, high_keep = _collect(skill)
    total = len(to_delete)
    print(f"default 碎片总数: {total + len(conclusion_keep) + len(high_keep)}")
    print(f"  待删除(低权重纯噪音): {total}")
    print(f"  保留-含结论关键词(降权待提炼): {len(conclusion_keep)}")
    print(f"  保留-高权重真实内容: {len(high_keep)}")

    if total == 0:
        print("无需清理。")
        return 0

    print("\n待删除样例(前5):")
    for eid, content, w in to_delete[:5]:
        print(f"  [{w:.2f}] {content[:60]!r}")

    if not args.apply:
        print("\n(dry-run) 未做任何修改。加 --apply 执行。")
        return 0

    # Backup chroma dir before destructive op
    chroma_dir = args.db + "_chroma"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = f"{chroma_dir}.bak-{stamp}"
    if os.path.isdir(chroma_dir):
        shutil.copytree(chroma_dir, backup)
        print(f"\n已备份 chroma 到 {backup}")

    for eid, _content, _w in to_delete:
        skill.learned_store.delete(eid)

    # Demote kept conclusion fragments (weight 0.5-ish) to 0.6 so a later
    # distill pass can promote them; they stay retrievable meanwhile.
    for eid, content, w in conclusion_keep:
        if w < 0.6:
            skill.learned_store.update(
                eid, content, metadata={"weight": 0.6},
            )

    remaining = len(skill.learned_store.list_by_category("default", limit=0))
    print(f"删除完成。default 剩余: {remaining} "
          f"(含 {len(conclusion_keep)} 条降权待提炼)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
