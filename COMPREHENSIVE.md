# Memory Skill — 技术文档

> 版本: v0.6.0 | 最后更新: 2026-07-31

---

## 1. 核心理念

Memory Skill 是长期记忆系统——Agent 感知不到它的存在，但它持续优化对话质量。

**每轮对话自动做三件事：**
1. **注入上下文** — weave 组装 8 区块记忆上下文
2. **存储记忆** — 对话进非结构化侧，重要内容进结构化侧
3. **主动检索** — Agent 引用记忆标题时自动展开

---

## 2. 两半记忆模型

```
每个对话块
  ├─ 非结构化 (user_mem): 原文存储，时间+语义检索
  └─ 结构化 (pref/skill/pers/mission): classify_and_extract → 路由
```

### 非结构化侧 (user_mem)

```
存储: DialogueStore (SQLite FTS5, BM25) + LearnedStore (ChromaDB, 向量)
索引: title / time / vector / id 四元组
检索: RRF 融合 (BM25×2.5 + semantic×0.5 + time×0.5)
```

### 结构化侧 (pref/skill/pers/mission)

| 分支 | 提取 | 存储 | weave |
|------|------|------|-------|
| skill | title + goal | 学习目标 markdown | 标题列表 |
| mission | title + deadline | 任务 markdown + 步骤 | 步骤 + 技能状态 |
| pref | key + value | 键值对 | 全量注入 |
| pers | trait | 人物卡累积 | 最新卡片 |

---

## 3. 写链 — 对话 → 持久化

```
ingest(turn)
  ├─ ① ingest_dialogue: SawBuffer + DialogueStore + LearnedStore
  ├─ ② tag_title: LLM 生成标题 → ChromaDB metadata
  └─ ③ extract_structured: classify_and_extract(LLM)
        ├─ pref → ingest_pref(key, value)
        ├─ pers → ingest_pers(trait) → 人物卡累积
        ├─ skill → ingest_skill(title, goal)
        └─ mission → ingest_mission(title, content)
```

---

## 4. 读链 — 消息 → 记忆注入

```
weave(user_message)
  ├─ [人格设定]   pers 人物卡 (始终注入)
  ├─ [用户偏好]   pref 键值对 (始终注入)
  ├─ [当前场景]   scene_summary
  ├─ [最近对话]   tier1 最近3轮
  ├─ [检索记忆]   tier2 RRF 检索
  ├─ [已掌握的技能] skill 标题列表
  ├─ [当前任务]   mission 步骤+技能状态
  ├─ [知识缺口]   gap 检测结果
  └─ [近期记忆]   title preview (Agent 检索入口)
```

---

## 5. Agent 主动检索

```
Agent 回复中提到记忆标题
  → transparent_proxy 拦截
  → expand(message): 标题匹配 → 时间窗口展开
  → 注入 [扩展记忆] 到上下文
```

---

## 6. 主动学习管道

```
gap 检测 → learning_decider (skip/ask/learn)
  → skill.learn(topic, urls)
  → crawl (web_crawler) → synth (knowledge_synth) → ingest_skill → verify
```

---

## 7. 模块地图

```
入口:     cli / mcp_server / transparent_proxy / room_adapter
组合:     _compose.py (MemorySystem)
管道:     ingestor / retriever / weaver / tree
存储:     dialogue_store / learned_store / saw_buffer / embedder
结构化:   memory_extract / structured_extractor
学习:     gap_detector / learning_decider / learning_task / web_crawler / knowledge_synth
质量:     cleaner / observation / reflect / feedback / importance
基础:     contracts / _llm_utils / tree_classifier
```

---

## 8. 指标

| 指标 | 数值 |
|------|------|
| 中文检索精度 | 93% (300条) |
| 检索延迟 | 35-100ms |
| 结构提取 | 5 类型分类正确 |
| 单元测试 | 15 (0.11s) |
| 集成测试 | 65 |
