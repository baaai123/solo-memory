import { execFile, execSync } from "node:child_process";
import { randomUUID } from "node:crypto";
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
    "chat.message": async (input, output) => {
      // Extract user text, then push a complete TextPart carrying the
      // id/sessionID/messageID aggregate fields opencode's EventV2
      // persistence requires. A bare {type, text} part fails save with
      // SchemaError / InvalidDurableEvent.
      try {
        const text = (output.parts || [])
          .filter((p) => p.type === "text")
          .map((p) => p.text || "")
          .join("\n")
          .trim();
        if (!text) return;

        const result = await runBridge(["weave"], text);
        if (result && result.ok && result.block) {
          output.parts.push({
            id: randomUUID(),
            sessionID: input?.sessionID || "",
            messageID: input?.messageID || "",
            type: "text",
            text: `[Memory Context]\n${result.block}`,
            synthetic: true,
          });
          pendingUser = text;
        }
      } catch { /* non-fatal: memory must never break chat */ }
    },

    event: async ({ event }) => {
      // After an assistant reply completes, auto-ingest the pair
      try {
        const type = event?.type || "";
        const props = event?.properties || {};
        if (type === "message.updated" && props.message?.role === "assistant") {
          const reply = props.message.parts
            ? props.message.parts.filter((p) => p.type === "text").map((p) => p.text || "").join("\n")
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
