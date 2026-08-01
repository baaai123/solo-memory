import { execFile, execSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Project root = parent of this plugin dir (opencode-auto-memory/../)
const PROJECT = process.env.MEMORY_SKILL_PROJECT
  || path.dirname(__dirname);
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

export const opencodeAutoMemory = async ({ client }) => {
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

export const plugin = opencodeAutoMemory;
export { opencodeAutoMemory as server };
