# Memory Skill v0.6 — 待办

## 未完成

### 设计已定、未接入
- **classify_and_extract 未接入 ingest_dialogue** — pref/pers/skill/mission 的自动提取只在手动测试中验过
- **learning_task 未接 knowledge_synth** — 当前仍存原始分块，未走 synth → markdown → ingest_skill 流程
- **mission-skill 互补** — 步骤拆解+技能关联，只记了方案

### 不可靠
- **WAL 数据库锁** — 连续 ingest 偶发 database locked
- **DeepSeek v4-flash 空响应** — 提示词超过 2 行偶发返回空（classify_skill_path 改规则驱动解决了，但 classify_and_extract 依赖它）

### 未做
- **单元测试** — 65 个全是集成测试
- **Git 初始化** — 项目无版本控制
- **pref 提取规则** — 只有 class/ext，无验证去重
- **CONTEXT.md** — scraped

### 已做但需验证
- **端到端 skill 闭环** — learn → crawl → synth → ingest → weave（过一次，不够稳）
- **透明代理** — 偶发 LLM 空响应
