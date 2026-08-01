import json
import sys

from memory_skill import MemorySkill, MemorySkillConfig
from memory_skill.contracts import DialogueTurn
from datetime import datetime, timezone


def _skill(db):
    return MemorySkill(MemorySkillConfig(db_path=db))


def cmd_weave(db, message, scene=""):
    sk = _skill(db)
    ctx = sk.weave(user_message=message, scene_summary=scene)
    print(json.dumps({"block": ctx.to_prompt_block(), "ok": True}, ensure_ascii=False))


def cmd_ingest(db, role, content):
    sk = _skill(db)
    turn = DialogueTurn(
        id=f"auto_{datetime.now(timezone.utc).timestamp():.0f}_{abs(hash(content)) & 0xFFFF:04x}",
        role=role,
        content=content[:2000],
        timestamp=datetime.now(timezone.utc),
    )
    sk.ingest(turn)
    print(json.dumps({"ok": True, "role": role, "len": len(content)}, ensure_ascii=False))


def cmd_ingest_pair(db, user_msg, assistant_msg):
    sk = _skill(db)
    now = datetime.now(timezone.utc)
    for role, content in (("user", user_msg), ("assistant", assistant_msg)):
        if not content:
            continue
        turn = DialogueTurn(
            id=f"auto_{now.timestamp():.0f}_{abs(hash(content)) & 0xFFFF:04x}",
            role=role,
            content=content[:2000],
            timestamp=now,
        )
        sk.ingest(turn)
    print(json.dumps({"ok": True}, ensure_ascii=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    db = sys.argv[2]
    if cmd == "weave":
        msg = sys.stdin.read()
        cmd_weave(db, msg)
    elif cmd == "ingest_pair":
        payload = json.loads(sys.stdin.read())
        cmd_ingest_pair(db, payload.get("user", ""), payload.get("assistant", ""))
