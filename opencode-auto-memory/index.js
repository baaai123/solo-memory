import { execFile } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Resolve: venv python + project db. Override via env.
const PROJECT = process.env.MEMORY_SKILL_PROJECT
  || "/home/pc/projects/memory/memory for solo";
const DB = process.env.MEMORY_SKILL_DB_PATH
  || path.join(PROJECT, "opencode_memory.db");
const PY = process.env.MEMORY_SKILL_PYTHON
  || path.join(PROJECT, "venv", "bin", "python");
const BRIDGE = path.join(__dirname, "bridge.py");

function runBridge(args, input = "") {
  return new Promise((resolve) => {
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
    "chat.message": async (_input, output) => {
      // Extract user text from parts; inject memory context as a system-role part
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
            type: "text",
            text: `[Memory Context]\n${result.block}`,
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
