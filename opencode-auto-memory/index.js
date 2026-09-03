import { execFile, execFileSync, execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import os from "node:os";
import fs from "node:fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Debug trace — console.log is captured into opencode's log (proven by
// Systematic's "initialized"); dbg() file is the fallback channel.
const DEBUG_LOG = "/tmp/solo-memory-debug.log";
const dbg = (msg) => {
  console.log(`[solo-memory] ${msg}`);
  try { fs.appendFileSync(DEBUG_LOG, `${new Date().toISOString()} ${msg}\n`); } catch {}
};
// Project root: env override, else the solo-memory repo (this plugin ships
// inside it; when copied to opencode's plugins dir, __dirname is wrong).
const PROJECT = process.env.MEMORY_SKILL_PROJECT
  || "/home/pc/projects/memory/memory for solo";
const DB = process.env.MEMORY_SKILL_DB_PATH
  || path.join(PROJECT, "opencode_memory.db");
const VENV = path.join(PROJECT, "venv");
const PY = process.env.MEMORY_SKILL_PYTHON
  || path.join(VENV, "bin", "python");
const BRIDGE = path.join(__dirname, "bridge.py");
const REQUIREMENTS = path.join(PROJECT, "requirements.txt");

// ONNX embedding model (bge-large-en-v1.5) — guaranteed by the first-run
// bootstrap. Without it the Python layer falls back to SHA-256 hash-only
// semantic search (dedup/can_answer disabled) = silently degraded.
const MODEL_DIR = path.join(PROJECT, "models", "bge-large-en-v1.5");
const MODEL_ONNX = path.join(MODEL_DIR, "model.onnx");
const MODEL_ONNX_ALT = path.join(MODEL_DIR, "onnx", "model.onnx");
const MODEL_SKIP_ENV = "MEMORY_SKIP_MODEL"; // "1" = explicitly skip (== setup.sh --no-model)

// Cloud backup trigger: after each finished assistant turn we touch the
// ccmp-backup daemon signal file (~/.ccmp-backup/signal, debounced there).
// SOLO_MEMORY_BACKUP_SIGNAL overrides the path; "0" disables.
const backupSignalPath =
  process.env.SOLO_MEMORY_BACKUP_SIGNAL === "0"
    ? ""
    : process.env.SOLO_MEMORY_BACKUP_SIGNAL || path.join(os.homedir(), ".ccmp-backup", "signal");

let setupInProgress = false;
let setupDone = false;
let modelDownloadTried = false; // one download attempt per process

// First-run bootstrap: create venv + install deps if missing
function ensureSetup() {
  if (setupDone) return ensureModel();
  if (!fs.existsSync(PY)) {
    if (setupInProgress) return false;
    setupInProgress = true;
    try {
      console.error("[solo-memory] First run: creating venv + installing deps…");
      execSync(`python3 -m venv "${VENV}"`, { cwd: PROJECT });
      execSync(`"${PY}" -m pip install -q -r "${REQUIREMENTS}"`, { cwd: PROJECT });
      setupInProgress = false;
      setupDone = true;
      console.error("[solo-memory] Setup complete.");
      return ensureModel();
    } catch (e) {
      setupInProgress = false;
      console.error("[solo-memory] Setup failed:", e.message);
      return false;
    }
  }
  setupDone = true;
  return ensureModel();
}

function modelOk() {
  return fs.existsSync(MODEL_ONNX) || fs.existsSync(MODEL_ONNX_ALT);
}

// Model-installation guarantee: after this returns, either the ONNX model
// exists, or the plugin has logged a prominent actionable error to stderr.
// NEVER throws — degraded-but-alive beats crashing opencode's startup.
function ensureModel() {
  try {
    if (modelOk()) return true;
    if (process.env[MODEL_SKIP_ENV] === "1") {
      console.error("[solo-memory] MEMORY_SKIP_MODEL=1 — running WITHOUT embedding model (DEGRADED: hash-only search). Download with: " + path.join(PROJECT, "download_model.sh"));
      return true;
    }
    if (!modelDownloadTried) {
      modelDownloadTried = true;
      console.error("[solo-memory] Embedding model missing — downloading bge-large-en-v1.5 (≈1.3GB, may take a while)…");
      const env = { ...process.env, HF_ENDPOINT: process.env.HF_ENDPOINT || "https://hf-mirror.com" };
      execFileSync("bash", [path.join(PROJECT, "download_model.sh")], { cwd: PROJECT, timeout: 900000, stdio: "inherit", env });
      console.error("[solo-memory] Model download complete.");
    }
    return true;
  } catch (e) {
    console.error("[solo-memory] ⚠ MODEL DOWNLOAD FAILED — memory will run DEGRADED (hash-only semantic search; dedup/can_answer disabled). Manual fix:\n  1) bash " + path.join(PROJECT, "download_model.sh") + "\n  2) or set HF_ENDPOINT to a reachable mirror\n  3) or export MEMORY_SKIP_MODEL=1 to acknowledge degraded mode");
    return true;
  }
}

function runBridge(args, input = "") {
  return new Promise((resolve) => {
    if (!ensureSetup()) { resolve(null); return; }
    const child = execFile(
      PY, [BRIDGE, ...args, DB],
      { timeout: 30000 },
      (err, stdout) => {
        if (err) { resolve(null); return; }
        try { resolve(JSON.parse(stdout)); }
        catch { resolve(null); }
      },
    );
    if (input) child.stdin.write(input);
    child.stdin.end();
  });
}

// Track pending user message → pair with next assistant reply
let pendingUser = "";

// Per-session id of the last user message we warned about, so each user turn
// triggers at most one enforcement warning (avoids spamming on repeat tools).
const warnedBySession = new Map();

// Parallel tool calls land in the message stream non-deterministically; if a
// non-memory tool fires `before` while weave is still executing, don't reject.
const weaveInFlight = new Map(); // sessionID -> boolean

// KNOWN-ISSUES #8: hard-blocking EVERY tool call before weave in high-frequency
// sessions is a net burden (timeouts + overhead). Downgrade: block at most ONCE
// per user turn (the first non-memory tool after a user message); subsequent
// tools in the same turn pass — the after-hook warning still nudges the agent.
const hardBlockedBySession = new Map(); // sessionID -> lastUserId (hard-blocked)

// memory_* tools must always pass, or enforcement deadlocks on itself.
const MEMORY_TOOL_PREFIX = "opencode-memory_";

// True if the latest user message has a memory_weave call after it.
async function hasWeavedAfterLastUser(client, sessionID) {
  const resp = await client.session.messages({ path: { id: sessionID } });
  const msgs = Array.isArray(resp) ? resp : (resp?.data ?? []);
  if (!msgs || !msgs.length) return { weaved: true, lastUserId: null };

  let lastUserIdx = -1;
  for (let i = msgs.length - 1; i >= 0; i--) {
    if (msgs[i].info?.role === "user") { lastUserIdx = i; break; }
  }
  if (lastUserIdx < 0) return { weaved: true, lastUserId: null };

  const lastUserId = msgs[lastUserIdx].info.id;
  for (let i = lastUserIdx + 1; i < msgs.length; i++) {
    for (const p of msgs[i].parts || []) {
      if (p.type === "tool" && p.tool.includes("memory_weave")) return { weaved: true, lastUserId };
    }
  }
  return { weaved: false, lastUserId };
}

// Enforcement message appended to a tool result when the agent acted before
// calling memory_weave for the current user turn.
const ENFORCEMENT_WARNING =
  "\n\n[MEMORY PROTOCOL ENFORCEMENT — ACTION REQUIRED]\n" +
  "You executed a tool before calling memory_weave for the current user message.\n" +
  "Per SKILL.md, EVERY turn MUST begin with:\n" +
  "  memory_weave(user_message=<current user message>, assistant_content=<your previous reply, verbatim>)\n" +
  "Call memory_weave NOW before any further tool calls — otherwise this exchange is never persisted.";

export const opencodeAutoMemory = async ({ client }) => {
  dbg("server() invoked — plugin loaded");
  return {
    // Memory injection is NOT done here. We cannot push a new part at
    // chat.message time: EventV2 requires branded ids (prt_/ses_/msg_) and
    // messageID is assigned inside createUserMessage — a manually built
    // part can never satisfy the schema, and appending to the user's text
    // part pollutes their message. Instead, injection happens through the
    // memory protocol: the system prompt (prompt_append) tells the agent to
    // call memory_weave (MCP tool) before responding, and the agent weaves
    // the retrieved context into its own reasoning. This plugin only
    // captures the pending user text so the event hook can auto-ingest.

    // HARD enforcement (KNOWN-ISSUES #2): the protocol is prompt-advice the
    // agent can skip (observed: whole rounds without weave). This hook makes
    // it a framework-level constraint, same mechanism as oh-my-openagent's
    // comment-checker: on tool.execute.after, if the latest user message has
    // no memory_weave call after it, append a mandatory warning to the tool
    // result — the agent reads it before its next step and cannot ignore it.
    "tool.execute.before": async (input, _output) => {
      try {
        if (input.tool.startsWith(MEMORY_TOOL_PREFIX)) {
          if (input.tool === "memory_weave") weaveInFlight.set(input.sessionID, true);
          return;
        }
        if (weaveInFlight.get(input.sessionID)) return;
        const { weaved, lastUserId } = await hasWeavedAfterLastUser(client, input.sessionID);
        if (weaved || !lastUserId) return;
        if (hardBlockedBySession.get(input.sessionID) === lastUserId) return; // once per turn
        hardBlockedBySession.set(input.sessionID, lastUserId);
        dbg(`HARD BLOCK "${input.tool}" before memory_weave (once per turn)`);
        throw new Error(
          "[MEMORY PROTOCOL ENFORCEMENT — HARD BLOCK]\n" +
          "Tool execution rejected: memory_weave has not been called for the current user message.\n" +
          "Every turn MUST start with:\n" +
          "  memory_weave(user_message=<current user message>, assistant_content=<your previous reply, verbatim>)\n" +
          "Call memory_weave NOW, then retry this tool."
        );
      } catch (err) {
        if (err instanceof Error && err.message.startsWith("[MEMORY PROTOCOL ENFORCEMENT")) throw err;
        dbg(`before hook error: ${err?.message || err}`);
      }
    },

    "tool.execute.after": async (input, output) => {
      try {
        if (input.tool.startsWith(MEMORY_TOOL_PREFIX)) {
          if (input.tool === "memory_weave") weaveInFlight.set(input.sessionID, false);
          return;
        }
        if (String(output.output || "").slice(0, 200).match(/error:|failed to|could not/i)) return;
        if (weaveInFlight.get(input.sessionID)) return;

        const { weaved, lastUserId } = await hasWeavedAfterLastUser(client, input.sessionID);
        if (weaved || !lastUserId) return;
        if (warnedBySession.get(input.sessionID) === lastUserId) return;

        warnedBySession.set(input.sessionID, lastUserId);
        output.output += ENFORCEMENT_WARNING;
        dbg("ENFORCEMENT_WARNING appended");
      } catch (err) { dbg(`after hook error: ${err?.message || err}`); }
    },

    "chat.message": async (_input, output) => {
      try {
        const text = (output.parts || [])
          .filter((p) => p.type === "text")
          .map((p) => p.text || "")
          .join("\n")
          .trim();
        if (text) pendingUser = text;
      } catch { /* non-fatal */ }
    },

    event: async ({ event }) => {
      // After an assistant reply completes, auto-ingest the pair.
      // KNOWN-ISSUES #9: if the agent already called memory_weave for this
      // turn (weave auto-ingests both sides), skip — ingest_pair would
      // duplicate the same exchange in learned_store.
      try {
        const type = event?.type || "";
        const props = event?.properties || {};
        if (type === "message.updated" && props.info?.role === "assistant") {
          signalBackup();
          const reply = props.info.parts
            ? props.info.parts.filter((p) => p.type === "text").map((p) => p.text || "").join("\n")
            : "";
          if (reply && pendingUser) {
            const sessionID = props.sessionID || props.parentID || "";
            let weaved = false;
            if (sessionID) {
              try { ({ weaved } = await hasWeavedAfterLastUser(client, sessionID)); }
              catch { weaved = false; }
            }
            if (weaved) {
              pendingUser = "";
              dbg("skip ingest_pair — weave already ingested this turn");
              return;
            }
            await runBridge(["ingest_pair"], JSON.stringify({ user: pendingUser, assistant: reply }));
            pendingUser = "";
          }
        }
      } catch { /* non-fatal */ }
    },
  };
};

// ── Cloud backup signal (debounced upstream in ccmp-backup daemon) ──────
// Touching the file after a finished turn arms the daemon's debounce window;
// more activity resets it, so a backup fires only once the user goes idle.
function signalBackup() {
  if (!backupSignalPath) return;
  try {
    const d = new Date().toISOString();
    fs.appendFileSync(backupSignalPath, `${d}\n`);
  } catch { /* non-fatal */ }
}

// opencode plugin contract (verified against binary source, 2026-08-02):
// - the loader reads `q.default` — a file/path plugin MUST default-export
//   an object (named exports like `plugin`/`server` are silently ignored)
// - the object MUST carry an `id` (rQ/tQ: "Path plugin must export id")
// - and a `server` function (returning the hook map), not `tui`
// Matches oh-my-openagent: `return { id, server }` + `export { x as default }`.
export const id = "solo-memory";
export default { id, server: opencodeAutoMemory };
