"""Re-classify default-category memories into structured branches.

Fixes the max_tokens=256 bug fallout: every ingested turn landed in
category 'default' because LLM classification returned 'none'. This
re-runs the fixed classifier over existing default memories and routes
them to pref/pers/skill/mission where appropriate. Memorable content
that classifies as 'none' stays in default (it's still retrievable).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("IMPORTANCE_API_BASE", "https://api.deepseek.com/v1")
os.environ.setdefault("IMPORTANCE_MODEL", "deepseek-v4-flash")
KEY = os.environ.get("IMPORTANCE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
if not KEY:
    sys.exit("IMPORTANCE_API_KEY or DEEPSEEK_API_KEY required")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory_skill.contracts import MemorySkillConfig
from memory_skill._compose import MemorySystem
from memory_skill.structured_extractor import classify_and_extract
from memory_skill.memory_extract import ingest_pref, ingest_pers, ingest_skill_ex, ingest_mission_ex

API_BASE = os.environ["IMPORTANCE_API_BASE"]
MODEL = os.environ["IMPORTANCE_MODEL"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "opencode_memory.db")
CHROMA = os.path.join(ROOT, "opencode_memory.db_chroma")

ms = MemorySystem(MemorySkillConfig(db_path=DB))

client = __import__("chromadb").PersistentClient(path=CHROMA)
col = client.get_collection("learned_default")
data = col.get(include=["metadatas", "documents"])

routed = {"pref": 0, "pers": 0, "skill": 0, "mission": 0, "none": 0}
errors = []
for eid, meta, doc in zip(data["ids"], data["metadatas"], data["documents"]):
    if meta.get("category") != "default":
        continue
    content = doc or ""
    if len(content) < 8:  # skip near-empty
        continue
    try:
        result = classify_and_extract(API_BASE, KEY, MODEL, content)
    except Exception as exc:
        errors.append((eid, str(exc)))
        continue
    t = result.get("type", "none")
    if t == "pref":
        ingest_pref(ms, result.get("key", ""), result.get("value", ""))
    elif t == "pers":
        ingest_pers(ms, result.get("trait", ""))
    elif t == "skill":
        ingest_skill_ex(ms, result.get("title", ""),
                        f"# {result.get('title', '')}\n\n学习目标: {result.get('goal', '')}")
    elif t == "mission":
        ingest_mission_ex(ms, result.get("title", ""), content)
    routed[t] = routed.get(t, 0) + 1
    if t != "none":
        col.delete(ids=[eid])
        print(f"  [{t}] {content[:45]}")

print()
print(f"重分类完成: pref={routed['pref']} pers={routed['pers']} skill={routed['skill']} "
      f"mission={routed['mission']} none={routed['none']} errors={len(errors)}")
