# Memory Skill — 总体规划 v0.6

## P0 接入 — 全部完成

- ✅ classify_and_extract 接入 ingest_dialogue (实测通过)
- ✅ learning_task 使用 synthesize_markdown + ingest_skill (代码就位，API 限流待重测)

## 结构侧四分支 — 全部完成

- ✅ skill: classify提取 + skill树 + weave标题列表
- ✅ mission: classify提取 + 步骤解析 + skill关联 + ⚠缺失标记
- ✅ pref: classify提取 + 键值对 + weave全量
- ✅ pers: classify提取 + 人物卡累积 + weave最新

## 非结构侧 — 完成

- ✅ user_mem索引 (title/time/vector/id)
- ✅ expand() 时间展开
- ✅ title预览 (LLM生成 + ChromaDB存储 + weave注入)
- ✅ Agent检索主动权 (透明代理 + expand)

## 管道 — 完成

- ✅ weave八区块 (人格·偏好·场景·tier1·tier2·技能·任务·缺口·记忆)
- ✅ 透明代理 (注入+存储+展开)
- ✅ 网页爬虫 + 知识合成
- ✅ 学习闭环 (gap→decider→crawl→synth→ingest→verify)

## 架构 — 完成

- ✅ 0私有违规 · 5Protocols · God-object拆解 · LLM去重 · user_mem去树化
