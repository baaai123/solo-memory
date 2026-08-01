# opencode-solo-memory

**Transparent long-term memory for OpenCode** — auto weave/ingest, no agent discipline required.

Part of the [solo-memory](https://github.com/baaai123/solo-memory) project. This plugin makes the agent's memory **fully automatic**:

- `chat.message` hook → auto-injects memory context (weave) before every turn
- `event` hook → auto-stores user+assistant pair (ingest) after each reply

The agent never needs to remember to call `memory_weave`/`memory_ingest` — it's invisible.

## Language support

Language-agnostic plugin: works identically for Chinese or English conversations.
Memory quality differs by signal (see solo-memory README): Chinese is served mainly
by BM25 (jieba), English mainly by semantic vectors (bge-large-en-v1.5). Both
languages are stored and retrievable in the same database.

## Install

### Option A: let your AI agent install it (fastest)

Give your AI agent this instruction:

```
Install https://github.com/baaai123/solo-memory and hook it into my OpenCode.

Steps:
1. git clone https://github.com/baaai123/solo-memory
2. Run ./setup.sh (creates venv + installs deps + configures embedding model)
3. Register the opencode-auto-memory plugin in opencode.json
4. Ask me for IMPORTANCE_API_KEY when needed

Note: ./setup.sh handles the environment in one command; the
opencode-auto-memory plugin auto-injects memory context and auto-stores
dialogue — the agent never calls memory tools manually.
```

The AI clones, sets up, and registers the plugin on its own. The only manual
step is providing `IMPORTANCE_API_KEY` (your private credential — no memory
system can hold it for you).

### Option B: manual install

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
