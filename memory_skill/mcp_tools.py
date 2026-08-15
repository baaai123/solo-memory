"""MCP tool definitions and handlers for Memory Skill.

Exposes 4 tools over the MCP JSON-RPC protocol:
  - ``memory_search``   — semantic search across dialogue + learned memory
  - ``memory_ingest``   — ingest a dialogue turn into the memory system
  - ``memory_status``   — health summary (entry counts, model status)
  - ``memory_feedback`` — record retrieval outcome + boost cited memories
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from memory_skill.skill import MemorySkill


from memory_skill.tools import TOOL_SCHEMAS

logger = logging.getLogger("memory_skill.mcp")

# ═══════════════════════════════════════════════════════════════════════════════
# Protocol Injection (delivered with every status call + tool descriptions)
# ═══════════════════════════════════════════════════════════════════════════════

MEMORY_PROTOCOL = """IMPORTANT — Memory Skill Usage Protocol (v5 auto-injection):
1. ON WAKE-UP: Call memory_status to check health and model availability.
2. BEFORE RESPONDING: Call memory_weave to get auto-injected context for the
   current conversation. This is the PRIMARY way to use memory — the system
   automatically assembles relevant past conversations and learned knowledge.
3. AFTER INTERACTIONS: Call memory_ingest to save significant dialogue turns.
4. FOR DEEP QUERIES: Use memory_search when memory_weave isn't enough and you
   need to explicitly search for specific past information.
5. FOR LEARNING: Call memory_feedback with cited memory IDs to boost their weights.

The weave() paradigm means the Agent does NOT need to call memory_search before
every turn. memory_weave handles context injection automatically.

MCP servers and middleware can also use ``skill.auto_context(messages)`` and
``skill.auto_ingest(user_msg, assistant_response)`` for fully transparent
memory injection — the LLM has zero knowledge of the memory system.
These methods handle context injection and ingestion without any tool calls
from the LLM side."""

# ═══════════════════════════════════════════════════════════════════════════════
# Tool Definitions (MCP schema format)
# ═══════════════════════════════════════════════════════════════════════════════

TOOLS: list[dict[str, Any]] = [
    {
        "name": name,
        "description": meta["description"],
        "inputSchema": meta["inputSchema"],
    }
    for name, meta in sorted(TOOL_SCHEMAS.items())
]


# ToolHandler
# ═══════════════════════════════════════════════════════════════════════════════

# KNOWN-ISSUES #8: full-length assistant_content (e.g. a whole summary block)
# makes the auto-ingest embedding slow enough to trip MCP -32001 timeouts.
# Cap what gets embedded at ingest time; retrieval/weave still sees the
# full query where it matters.
#
# Head+tail retention: Chinese replies often put the conclusion at the END,
# so naive head-only truncation would drop it (KNOWN-ISSUES #9 "结论散落").
# Keep the first `_HEAD` chars + last `_TAIL` chars, ellipsis in between.

class ToolHandler:
    """Dispatches MCP tool calls to standalone handler functions in tools.py."""

    def __init__(self, skill: MemorySkill) -> None:
        self._skill = skill

    def handle(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from memory_skill.tools import DISPATCH
        handler = DISPATCH.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(self._skill, arguments)
        except Exception as exc:
            logger.exception("Tool %s failed", tool_name)
            return {"error": f"{type(exc).__name__}: {exc}"}
