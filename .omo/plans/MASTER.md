# Memory Skill v0.6 — 全链路复盘

## 一条用户消息的完整旅程

```
"我喜欢喝冰美式，每天下午都要来一杯"
  │
  ├─ ① ingest_dialogue                      [~1s, ONNX embed]
  │     ├→ SawRingBuffer          内存          ← 高频缓冲区
  │     ├→ DialogueStore          SQLite FTS5   ← BM25 检索 (jieba中文)
  │     └→ LearnedStore           ChromaDB      ← 语义检索 (ONNX 1024维)
  │
  ├─ ② tag_title                            [~2s, LLM]
  │     └→ LLM生成: "喝冰美式每天下午" → ChromaDB metadata
  │
  ├─ ③ extract_structured                   [~3s, LLM]
  │     └→ classify_and_extract → type="pref", key="饮品", value="冰美式"
  │        → ingest_pref → 存 category="pref"
  │
  ├─ ④ weave (下次对话时触发)
  │     ├→ tier1: 最近对话           ← 固定注入
  │     ├→ tier2: RRF检索            ← 语义+BM25+时间 (35ms, 200条@93%)
  │     ├→ skill: 技能标题列表        ← 仅标题, 不注全文
  │     ├→ mission: 步骤+技能状态     ← 解析markdown, 检查skill存在
  │     ├→ pref: 用户偏好            ← 全量注入
  │     ├→ pers: Agent人物卡          ← 最新版本
  │     ├→ gap: 知识缺口              ← 自动检测
  │     └→ preview: 近期记忆标题      ← Agent检索入口
  │
  └─ ⑤ expand (Agent引用标题时触发)
        └→ 搜标题匹配 → 时间窗口展开 → 同时段记忆列表
```

## 核心指标

| 指标 | 数值 |
|------|------|
| 检索精度 (BM25中文) | 93% (300条) |
| 检索延迟 | 102ms/q |
| 结构提取准确率 | 5/5 类型正确 |
| Agent检索展开 | 标题匹配 + 时间窗口 |
| 总模块 | 33 |
| 测试 | 15单元 + 65集成 |
