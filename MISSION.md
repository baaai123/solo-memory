# Memory Skill — 项目理念

## 一句话

**一个对 Agent 透明的长期记忆系统——Agent 不需要感知记忆的存在，但行为被记忆驱动。**

## 核心设计原则

### 1. 两半记忆模型

对话块自动走两边：非结构化 (user_mem) 存原文，结构化 (pref/pers/skill/mission) 提取事实。检索时各司其职。

### 2. 逐字原文

Never summarize. Store verbatim. 摘要丢失语境、态度、细节。skill/mission 用 LLM 合成文档，但 user_mem 永远存原文。

### 3. BM25 主信号

中文对话 exact word match > vector similarity。BM25=2.5, semantic=0.5。

### 4. Agent 主动检索

Agent 有权主动控制检索——引用记忆标题，系统自动展开相关记忆。不是只能被动接收 weave 注入。

### 5. 主动学习

检测到知识缺口 → Agent 决定学不学 → crawl → synth → verify。记忆系统不只是存储，还自我成长。

## 为什么不是别的方式

| 替代方案 | 为什么没选 |
|----------|-----------|
| Letta/MemGPT | Agent 应该专注任务，记忆透明化 |
| MemVerse GPT 摘要 | 摘要丢失语境。verbatim + BM25 更可靠 |
| 纯向量检索 | 嵌入质量不可靠。BM25 主信号 |

## 已知权衡

- BGE-large-en 是英文模型，中文语义向量弱——BM25 主信号补足
- 结构化提取依赖 LLM 质量——classify prompt 用示例稳定
- DeepSeek reasoning tokens 消耗 max_tokens——需调大
- jieba 必需——FTS5 不切中文
