# Memory Skill — 项目理念

## 一句话

**一个对 Agent 完全透明的长期记忆插件——Agent 不知道自己有记忆，但行为被记忆驱动。**

## 核心设计原则

### 1. 无知觉

Agent 不应感知到记忆系统的存在。不调用 `memory_search`，不知道 weave 在运行。Agent 收到的只是一个组织得更好的 system prompt。

### 2. 逐字原文

Never summarize. Store verbatim. 摘要丢失语境、态度、细节。

### 3. BM25 主信号

中文对话场景 exact word match > vector similarity。BM25=2.5, semantic=0.5。

### 4. 行为驱动

记忆注入不只是信息——是行为指令。"你很清楚...""务必...""好感度偏高..."。

### 5. 独立插件

Memory 不知道 Persona、Emotion、TTS 的存在。桥接在 RoomAssembly 层。

## 为什么不是别的方式

| 替代方案 | 为什么没选 |
|----------|-----------|
| Letta/MemGPT | Agent 不应该感知到记忆系统 |
| CrewAI task memory | 角色对话不需要 task planning |
| MemVerse GPT 摘要 | 摘要丢失语境。verbatim+BM25 更可靠 |
| 纯向量检索 | 嵌入质量不可靠。BM25 作为主信号 |

## 已知权衡

- NSFW 数据上的 observation 是噪音——已在 weave 中禁用
- Evolution 反馈链路长——delta=0.15，需多轮
- GPU 依赖——CPU fallback (SHA-256) 仅测用
- jieba 必需——FTS5 不切中文
