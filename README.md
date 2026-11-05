# 法律顾问 Agent（LangGraph + LlamaIndex）

一个面向法律顾问场景的智能体：在给出法律回答时严格满足
**引用准确性、文件有效性、位阶效应、两层结构** 四项约束，并在最终输出前提供
**执业律师人工审核（人在环）** 闸门。

## 整体架构（融合 ks-lma-chat 工业级 RAG & Trace 升级版）

```
用户输入（可选「法律事实发生日 act_date」）
  └─ 事实抽取与合规    （PII 脱敏 + 律所利益冲突检索 + 事实时间/地域提取）
  └─ 问题拆解          （复杂问题 → 独立子问题）
  └─ 三维检索式构建    （ks-lma-chat：最精准 focus / 找例外 exception / 查后果 consequence）
  └─ 多路 RAG 检索     （向量 + BM25 + 引致条款扩展 + 新旧法对照回填）
  └─ 有效性校验        （时间感知打标，不提前删除失效条文）      ← 约束2
  └─ 重排 Rerank       （Batch Rerank 结合法条相关度打分截断）  ← 约束1
  └─ 位阶排序          （司法解释位阶随被解释法律浮动 +0.5）    ← 约束3
  └─ 规则判定          （条文级时间定位 + 立法法位阶冲突裁决 → 四态判定）← 规则层
       ├─ HIT          → 确定性执行（置信度 A，抽检）
       ├─ PARTIAL      → 规则骨架 + LLM 补全（B，必审）
       ├─ MISS         → LLM 受限兜底（C，必审+专家）
       └─ CONFLICT     → 禁止 LLM，升级人工（立即）
  └─ 两层合成          （第一层=引用；第二层=六段式分析）        ← 约束1/4
  └─ 细粒度溯源 Trace  （ks-lma-chat：N-gram 句级对齐 + 原文高亮支撑片段）
  └─ 律师审核工作台    （人在环 interrupt：approve / revise / re_retrieve）
  └─ 输出与审计归档    （SHA-256 防篡改审计哈希链）
```

> 各约束落点：引用准确性由"真实检索结果直接作为第一层 + 重排只重排序不编造"保证，
> 有效性由"有效性校验节点"保证，位阶由"位阶排序 + 规则判定"保证，两层结构由结构化输出保证。
> 律师审核台前端见 `app/static/index.html`（访问服务根路径 `/` 即可打开）。

### 规则引擎与兜底策略（层 2）

`app/rules.py` 是确定性的规则引擎，不依赖 LLM，可离线单测：

- **条文级时间定位** `locate_article()`：判定每一条具体条文在行为发生日是否适用。
  行为日早于施行日 → 依「法不溯及既往」排除；晚于失效日 → 排除；**落在旧法有效区间内 →
  保留为「行为时旧法」**并强制标注「现已失效、不得以现行法主张」。
  > 这是关键修复：早期版本在 `validate` 阶段就删除已废止条文，导致历史纠纷无依据可用。
  > 现在 `validate` 只打标，取舍权交给时间定位。
- **法律级版本定位** `temporal_locate()`：按「行为发生日」定位该法应适用的版本（留痕用）。
- **位阶冲突裁决** `resolve_hierarchy()`：
  - 上位法优于下位法；
  - **司法解释依附被解释法律**：位阶 = 被解释法位阶 + 0.5（解释《民法典》的司法解释为 1.5，
    高于行政法规与部门规章），与被解释法律之间是「解释关系」而非冲突；
  - **仅当同一立法机关制定时**才适用「新法优于旧法 / 特别法优于一般法」（《立法法》第 92 条）——
    依《宪法》第 58 条，全国人大与其常委会视为同一立法机关，否则第 92 条在绝大多数法律之间无法适用；
  - **不可比位阶**（地方性法规 vs 部门规章）不做机械排序，依法转国务院 / 全国人大常委会裁决。
- **裁决结论落地**：裁决出的优先法会回写引用顺序，并作为硬事实注入 LLM 提示词
  （`【位阶裁决结论｜必须服从】`），避免模型自行改用法条。
- **四态判定**：`HIT`（确定性）/ `PARTIAL`（规则骨架 + LLM 补全）/ `MISS`（LLM 受限兜底）/
  `CONFLICT`（禁止 LLM，升级人工）。**规则冲突绝不用 LLM 裁决**。
- **兜底硬约束**：引用闭包、措辞降级、缺口显式化、给出核实项、禁入自动通道，兜底时自动注入提示词。

> 关键设计：`HIT` 的 `llm_allowed=False` 语义是「不需要 LLM」而非「禁止 LLM」，聚合时只有
> 显式禁止（CONFLICT / 依法需裁决）才让整体禁止兜底，避免 `HIT + MISS` 组合丢失兜底约束。
> 所有 `MISS`/`CONFLICT` 应聚合为「规则缺口清单」，作为规则库迭代的输入。

### 人在环审核台（前端）

启动服务后访问 `http://localhost:8000/` 即可打开律师审核台：提交法律咨询 → 查看
第一层引用（含位阶/有效性/相关性评分）与第二层回答 → 执业律师点击「通过 / 附意见通过 / 驳回重生成」。
前端通过 `/legal-query` 与 `/human-review` 两个接口与后端交互。

## 技术选型理由

- **LangGraph**：用有状态图把每一步变成可审计节点，`interrupt` 提供业界最干净的人在环原语，
  天然支持"修改后重跑合成"的回环。
- **LlamaIndex**：以 `llama_index.core` 的 `TextNode` 承载法规与元数据，节点级溯源让每条结论
  都可绑定到真实条文；检索层自实现 Okapi BM25（中文按字 uni/bi-gram 切分），无需嵌入、零额外依赖即可运行。
- **两层结构化输出**：第一层引用直接来自检索结果（杜绝模型编造条文），第二层才由 LLM 生成分析，
  兼顾准确性与可读性。

## 运行方式

### 1）Docker（推荐，免密钥即可演示）

```bash
docker compose up --build
# 健康检查
curl http://localhost:8000/health
```

### 2）本地

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 命令行演示（MOCK 模式）
python -m app.run

# 或启动 API 服务
uvicorn app.server:app --host 0.0.0.0 --port 8000
```

### 3）接入真实 LLM（如 DeepSeek）

在 `docker-compose.yml` 或 `.env` 中设置 `MOCK=false` 并填写
`LLM_API_KEY / LLM_API_BASE / LLM_MODEL`（OpenAI 兼容接口）。

## API

- `GET  /health` 健康检查
- `POST /legal-query`  `{query, act_date?, thread_id?}` → 触发流程；返回 `needs_review`（含待审内容）或 `done`
- `POST /human-review` `{thread_id, decision(approve|revise), comments?}` → 恢复流程

示例：

```bash
# act_date 为法律事实发生日（可省略，默认按当日）
curl -X POST http://localhost:8000/legal-query \
  -H 'Content-Type: application/json' \
  -d '{"query":"公司拖欠工资且未签劳动合同，我能否解除并要求赔偿？","act_date":"2024-03-01"}'
```

返回中还包含规则判定结果：

```json
{
  "rule_plan": { "mode": "deterministic", "confidence": "A", "review": "sample", "reason": "..." },
  "rule_notes": ["时间效力：《中华人民共和国民法典》依 2024-03-01 适用 2021-01-01。", "..."],
  "answer_confidence": "A"
}
```

返回的 `review` 即为两层结构：

```json
{
  "citations": [ { "law_name":"中华人民共和国劳动合同法", "article":"第38条", "validity":"现行有效", ... } ],
  "answer": "针对您的问题……依据劳动合同法第38条、第47条、第87条……",
  "disclaimer": "本回答由 AI 生成，不构成正式法律意见，须经执业律师审核。"
}
```

## 测试

```bash
python tests/test_flow.py
```

## 知识库（真实法规）

`app/data/sample_laws.json` 现有 **44 条**条文，覆盖四类位阶：

| 位阶 | 数量 | 说明 |
|------|:--:|------|
| 法律 | 40 | 民法典、劳动合同法、消费者权益保护法、公司法（含 2018 修正旧文本与 2023 修订新文本）、行政处罚法、行政复议法、个人信息保护法等 |
| 司法解释 | 2 | 法释〔2023〕13 号《民法典合同编通则解释》第 65 条（违约金调整）、第 3 条（合同成立），已按最高人民法院公告核验原文 |
| 部门规章 | 1 | 《工资支付暂行规定》（`trial=true`，`verified=false`） |
| 地方性法规 | 1 | 《上海市劳动合同条例》（`verified=false`） |

其中：一条**已废止**的《合同法》第107条、一条**已被修订**的公司法（2018 修正）第26条，
用于演示「行为发生日落在旧法有效区间 → 适用行为时旧法」。

每条均带完整元数据：

```
law_name / document_number / hierarchy / host_law / host_hierarchy
promulgation_date / effective_date / effective_to / validity / superseded_by
article / paragraph / content / source / category / scope
partially_invalid / invalid_parts / trial / verified / note
```

- `host_law` / `host_hierarchy`：司法解释的被解释法律，用于计算位阶（缺失则体检报错）
- `effective_to`：失效日期，时间效力判定的关键依据
- `paragraph`：款/项定位（如 `第（一）项`），支持款项级引用
- `scope`：`特别` / `一般`，用于《立法法》第 92 条「特别法优于一般法」
- `verified`：**内容是否已经权威来源核验**；`false` 的条目会被标记「待核验」，
  并在提示词中注明「不得作为正式依据」
- `trial`：试行 / 暂行规范，效力稳定性较低

接入真实库时将 `DATA_PATH` 指向你的数据文件或目录（目录下全部 `.json` 自动合并）即可，
字段保持一致即可无缝替换。

## 测试

```bash
# 离线核心算法验证（零依赖，无需安装任何包）
python verify_offline.py

# 规则引擎与兜底策略验证（零依赖，63 项断言）
python verify_rules_offline.py

# 法规库数据体检（零依赖；元数据缺失会阻断，可接入 CI）
python verify_data.py

# 完整链路冒烟测试（需先安装依赖）
python tests/test_flow.py
```

> **为什么必须有 `verify_data.py`**：规则引擎的正确性依赖元数据完整性。若全库 `hierarchy`
> 只有一个取值、`scope` 全空，规则函数本身仍会通过单测（单测用的是合成数据），但在真实库上
> **永不触发**。数据体检用于拦截这类「引擎正确、业务失效」的问题。

> 说明：示例库条文为代表性摘录，用于演示与验证，**不构成正式法律意见**；
> 正式对外须接入权威法规库并逐条核对原文与时效性。
