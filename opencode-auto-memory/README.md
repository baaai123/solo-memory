# opencode-solo-memory

**Transparent long-term memory for OpenCode** — auto weave/ingest, no agent discipline required.

Part of the [solo-memory](https://github.com/baaai123/solo-memory) project. This plugin makes the agent's memory **fully automatic**:

- `chat.message` hook → auto-injects memory context (weave) before every turn
- `event` hook → auto-stores user+assistant pair (ingest) after each reply

The agent never needs to remember to call `memory_weave`/`memory_ingest` — it's invisible.

## Install

```bash
# 1. Clone solo-memory (contains the memory core + this plugin)
git clone https://github.com/baaai123/solo-memory
cd solo-memory

# 2. Set up core (venv + deps + embedding model)
./setup.sh          # or: python3 -m venv venv && pip install -r requirements.txt
./download_model.sh # bge-large-en-v1.5 → models/

# 3. Configure key (for LLM classification / learning)
cp .env.example .env   # fill IMPORTANCE_API_KEY
```

## Configure OpenCode

Add the plugin path to `~/.config/opencode/opencode.json`:

```json
{
  "plugin": [
    "/abs/path/to/solo-memory/opencode-auto-memory"
  ]
}
```

That's it. Restart OpenCode — memory is now automatic.

## Env overrides

| Env var | Default | Purpose |
|---|---|---|
| `MEMORY_SKILL_PROJECT` | plugin dir's parent | solo-memory project root |
| `MEMORY_SKILL_DB_PATH` | `<project>/opencode_memory.db` | memory database |
| `MEMORY_SKILL_PYTHON` | `<project>/venv/bin/python` | venv python |

On first use, the plugin auto-creates the venv and installs `requirements.txt` if missing.

## How it works

```
user message → chat.message hook → bridge.py weave → [Memory Context] injected
assistant reply → event hook → bridge.py ingest_pair → both turns stored
```

`bridge.py` runs the real `MemorySkill` API in-process — no MCP, no protocol overhead. All failures are non-fatal: memory never breaks chat.

## License

Apache-2.0
