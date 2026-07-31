"""Memory Skill — lightweight, real-time, self-evolving Agent memory module."""

__version__ = "0.5.0"

from memory_skill.contracts import (
    DialogueTurn,
    MemoryEntry,
    MemoryEnvelope,
    MemorySkillConfig,
    MemorySkillError,
    ModelLoadError,
    SawEntry,
    SearchStoreCorruptionError,
    StoreCorruptionError,
)
from memory_skill.skill import MemorySkill
from memory_skill.weaver import WeaveContext

# Lazy import — room_adapter depends on the Room package,
# which is NOT available in OpenCode/MCP environments.
try:
    from memory_skill.room_adapter import MemorySkillAdapter
except ImportError:
    MemorySkillAdapter = None  # type: ignore[assignment]

__all__ = [
    "MemorySkill",
    "MemorySkillConfig",
    "WeaveContext",
    "MemoryEntry",
    "MemoryEnvelope",
    "DialogueTurn",
    "SawEntry",
    "MemorySkillError",
    "StoreCorruptionError",
    "ModelLoadError",
    "SearchStoreCorruptionError",
    "MemorySkillAdapter",
]
