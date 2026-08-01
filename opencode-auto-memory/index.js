import { execFile, execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
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

let setupInProgress = false;
let setupDone = false;

// First-run bootstrap: create venv + install deps if missing
function ensureSetup() {
  if (setupDone) return true;
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
      return true;
    } catch (e) {
      setupInProgress = false;
      console.error("[solo-memory] Setup failed:", e.message);
      return false;
    }
  }
  setupDone = true;
  return true;
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
    "tool.execute.after": async (input, output) => {
      try {
        if (input.tool === "memory_weave") return; // doing the right thing
        if (String(output.output || "").slice(0, 200).match(/error:|failed to|could not/i)) return;

        const resp = await client.session.messages({ path: { id: input.sessionID } });
        const msgs = Array.isArray(resp) ? resp : (resp?.data ?? []);
        if (!msgs || !msgs.length) return;

        let lastUserIdx = -1;
        for (let i = msgs.length - 1; i >= 0; i--) {
          if (msgs[i].info?.role === "user") { lastUserIdx = i; break; }
        }
        if (lastUserIdx < 0) return;
        const lastUser = msgs[lastUserIdx].info;
        if (warnedBySession.get(input.sessionID) === lastUser.id) return;

        let weaved = false;
        for (let i = lastUserIdx + 1; i < msgs.length && !weaved; i++) {
          for (const p of msgs[i].parts || []) {
            if (p.type === "tool" && p.tool === "memory_weave") { weaved = true; break; }
          }
        }
        if (weaved) return;

        warnedBySession.set(input.sessionID, lastUser.id);
        output.output += ENFORCEMENT_WARNING;
        dbg("ENFORCEMENT_WARNING appended");
      } catch (err) { dbg(`hook error: ${err?.message || err}`); }
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
      // After an assistant reply completes, auto-ingest the pair
      try {
        const type = event?.type || "";
        const props = event?.properties || {};
        if (type === "message.updated" && props.info?.role === "assistant") {
          const reply = props.info.parts
            ? props.info.parts.filter((p) => p.type === "text").map((p) => p.text || "").join("\n")
            : "";
          if (reply && pendingUser) {
            await runBridge(["ingest_pair"], JSON.stringify({ user: pendingUser, assistant: reply }));
            pendingUser = "";
          }
        }
      } catch { /* non-fatal */ }
    },
  };
};

// opencode plugin contract (verified against binary source, 2026-08-02):
// - the loader reads `q.default` — a file/path plugin MUST default-export
//   an object (named exports like `plugin`/`server` are silently ignored)
// - the object MUST carry an `id` (rQ/tQ: "Path plugin must export id")
// - and a `server` function (returning the hook map), not `tui`
// Matches oh-my-openagent: `return { id, server }` + `export { x as default }`.
export const id = "solo-memory";
export default { id, server: opencodeAutoMemory };
