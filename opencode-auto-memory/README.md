# opencode-solo-memory

**Automatic long-term memory for OpenCode** — auto-ingest on every turn, injection via memory protocol.

Part of the [solo-memory](https://github.com/baaai123/solo-memory) project. This plugin:

- `event` hook → **auto-stores** user+assistant pairs after each reply (fully automatic, no side effects)
- `chat.message` hook → captures pending user text for pairing (does **not** modify the message)

**Memory injection** is done the clean way — through the memory protocol, not message mutation:
opencode's `prompt_append` tells the agent to call `memory_weave` (MCP tool) before responding.
The agent retrieves and weaves the context into its own reasoning. This avoids the EventV2
branded-part schema issue entirely (see below) and keeps user messages untouched.

## Why not inject in the plugin?

OpenCode v1.18 EventV2 persistence requires every part to carry branded aggregate ids
(`prt_` for part, `ses_` for session, `msg_` for message). At `chat.message` hook time:
- a manually constructed part can never satisfy the schema (id must be `prt_`-prefixed),
- `messageID` is assigned inside `createUserMessage` — it's `undefined` during the hook,
- appending to the user's text part pollutes their message.

So message-level injection is impossible/ugly in the current plugin API. The protocol
approach (system prompt + MCP tool) is the clean equivalent.

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
4. Fill in .env with my own IMPORTANCE_API_KEY (my personal LLM API key)

Note: ./setup.sh handles the environment in one command; the
opencode-auto-memory plugin auto-injects memory context and auto-stores
dialogue — the agent never calls memory tools manually.
```

The AI clones, sets up, and registers the plugin on its own. The only manual
step is filling `.env` with **your own LLM API key** (used for classification /
synthesis / learning, billed to your own API account). Any OpenAI-compatible
model works — the default is DeepSeek V4 Flash; set `IMPORTANCE_API_BASE` /
`IMPORTANCE_MODEL` in `.env` to switch (OpenAI, Qwen, GLM, local vLLM, …).

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
injection:  prompt_append protocol → agent calls memory_weave (MCP) → context in its reasoning
storage:    user message → chat.message hook (captures text)
            assistant reply → event hook → bridge.py ingest_pair → both turns stored
```

`bridge.py` runs the real `MemorySkill` API in-process for storage — no MCP, no protocol overhead.
Injection uses the MCP `memory_weave` tool per the system prompt protocol. All failures are
non-fatal: memory never breaks chat.

## License

Apache-2.0
