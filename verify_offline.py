"""离线核心算法验证（零依赖即可运行，无需安装 pydantic/langgraph/llama-index）。

验证链路：load_nodes → BM25 检索 → 有效性校验 → 重排(rerank) → 位阶裁决 → 两层结构组装
覆盖：检索召回、失效条文排除（约束2）、重排打分（约束1）、位阶排序（约束3）、两层输出（约束4）。

说明：本环境无法安装依赖（沙箱安全删除守卫拦截 wheel 安装），故此处用「鸭子类型」的轻量
对象模拟 Citation，直接调用生产代码 app.ingest / app.rerank 的真实检索与重排逻辑。
"""
import os
import sys
from datetime import date
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import settings
from app.ingest import build_retriever, build_version_timelines, load_nodes
from app.rerank import rerank
from app.rules import locate_article, rank_of

settings.mock = True  # 走启发式，无需密钥

# 位阶：数值越小位阶越高（与 models.py 保持一致，仅用于断言）
HIERARCHY_ORDER = {
    "宪法": 0, "法律": 1, "行政法规": 2, "地方性法规": 3, "部门规章": 3,
    "司法解释": 4, "规范性文件": 5,
}


def to_citation(m, score):
    """把节点元数据映射为鸭子类型对象，字段与 Citation 一致。"""
    return SimpleNamespace(
        law_name=m.get("law_name", ""),
        document_number=m.get("document_number", ""),
        hierarchy=m.get("hierarchy", ""),
        host_law=m.get("host_law", ""),
        host_hierarchy=m.get("host_hierarchy", ""),
        promulgation_date=m.get("promulgation_date", ""),
        effective_date=m.get("effective_date", ""),
        effective_to=m.get("effective_to", ""),
        validity=m.get("validity", "现行有效"),
        superseded_by=m.get("superseded_by", ""),
        article=m.get("article", ""),
        paragraph=m.get("paragraph", ""),
        content=m.get("content", ""),
        source=m.get("source", ""),
        category=m.get("category", ""),
        score=round(float(score), 4),
        rerank_score=0.0,
    )


def run_pipeline(query: str, act_date: str = None):
    nodes = load_nodes()
    retriever = build_retriever(nodes)
    queries = [query, query.replace("？", "").replace("?", "")]
    collected = {}
    for q in queries:
        for r in retriever.retrieve(q):
            node = r.node if hasattr(r, "node") else r
            m = node.metadata
            key = (m.get("law_name", ""), m.get("article", ""))
            score = getattr(r, "score", 0.0)
            if key not in collected or score > collected[key].score:
                collected[key] = to_citation(m, score)

    retrieved = sorted(collected.values(), key=lambda c: c.score, reverse=True)

    # 有效性校验（约束2）：只打标留痕，不提前删除——旧法可能是行为发生日的唯一依据
    validation_notes, validated = [], []
    for c in retrieved:
        if c.validity != "现行有效":
            validation_notes.append(f"保留待时间定位 {c.law_name}{c.article}（{c.validity}）")
        validated.append(c)

    # 重排（约束1）
    reranked = rerank(query, validated)

    # 位阶排序（约束3）：司法解释位阶随被解释法律浮动
    ranked = sorted(
        reranked,
        key=lambda c: (rank_of(c.hierarchy, getattr(c, "host_hierarchy", "")), -c.rerank_score, -c.score),
    )

    # 条文级时间定位：由「行为发生日」决定最终去留
    tl = build_version_timelines()
    ad = act_date or date.today().isoformat()
    applicable = []
    for c in ranked:
        o = locate_article(ad, c.effective_date, tl.get(c.law_name, []), c.effective_to,
                           f"{c.law_name}{c.article}")
        if o.detail.get("applicable") is False:
            validation_notes.append(f"时间定位排除 {c.law_name}{c.article}：{o.reason}")
            continue
        applicable.append(c)

    return {
        "retrieved": retrieved, "validation_notes": validation_notes,
        "validated": validated, "reranked": reranked, "ranked": ranked,
        "applicable": applicable, "act_date": ad,
    }


def assert_equal(name, cond):
    print(f"  {'✅' if cond else '❌'} {name}")
    if not cond:
        raise AssertionError(name)


def main():
    print("=== 离线核心算法验证（零依赖）===")
    print(f"知识库条文数：{len(load_nodes())}")

    print("\n[用例1] 合同违约 + 员工撞人（多子问题）")
    q = "公司员工送货途中撞伤行人，公司要赔吗？员工没签劳动合同有什么后果？"
    r = run_pipeline(q)
    ranks = [HIERARCHY_ORDER.get(c.hierarchy, 9) for c in r["ranked"]]
    assert_equal("约束3 位阶排序单调（上位法在前）",
                 all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1)))
    assert_equal("约束1 重排分数区间合法 [0,1]",
                 all(0.0 <= c.rerank_score <= 1.0 for c in r["ranked"]))
    assert_equal("约束4 第一层存在有效引用", len(r["ranked"]) > 0)

    print(f"  召回 {len(r['retrieved'])} 条 → 有效 {len(r['validated'])} 条 → 重排 {len(r['reranked'])} 条")
    print("  位阶排序结果（前5）：")
    for c in r["ranked"][:5]:
        print(f"    · {c.law_name}{c.article} [{c.hierarchy}] rel={c.rerank_score:.3f}")

    print("\n[用例2] 行政处罚（验证重排相关性）")
    q2 = "违法行为轻微没造成后果，会被行政处罚吗？"
    r2 = run_pipeline(q2)
    assert_equal("专项 行政处罚法第33条被命中",
                 any(c.law_name == "中华人民共和国行政处罚法" and c.article == "第33条" for c in r2["ranked"]))
    top1 = r2["ranked"][0]
    assert_equal("专项 重排后 top1 为最相关条文(第33条)",
                 top1.law_name == "中华人民共和国行政处罚法" and top1.article == "第33条")

    print("\n[用例3] 合同违约（约束2 时间效力：旧法不得提前删除）")
    q3 = "对方不履行合同义务，我可以要求继续履行并赔偿损失吗？"

    # 3a 行为发生在民法典施行前（2020-06-01）：应适用当时的《合同法》第107条
    r3 = run_pipeline(q3, act_date="2020-06-01")
    assert_equal("约束2 召回中包含已废止《合同法》第107条",
                 any(c.law_name == "中华人民共和国合同法" and c.article == "第107条" for c in r3["retrieved"]))
    assert_equal("约束2 已废止条文被保留至时间定位阶段（不再提前删除）",
                 any(c.law_name == "中华人民共和国合同法" and c.article == "第107条" for c in r3["validated"]))
    assert_equal("约束2 2020 年行为 → 适用行为时旧法《合同法》第107条",
                 any(c.law_name == "中华人民共和国合同法" and c.article == "第107条" for c in r3["applicable"]))
    assert_equal("约束2 2020 年行为 → 《民法典》尚未施行，不得溯及适用",
                 all(not (c.law_name == "中华人民共和国民法典" and c.article == "第577条") for c in r3["applicable"]))

    # 3b 行为发生在当下：旧法已失效，适用现行《民法典》
    r3b = run_pipeline(q3, act_date="2025-06-01")
    assert_equal("约束2 2025 年行为 → 旧法《合同法》因失效被排除",
                 all(not (c.law_name == "中华人民共和国合同法") for c in r3b["applicable"]))
    assert_equal("约束2 2025 年行为 → 现行《民法典》第577条被保留",
                 any(c.law_name == "中华人民共和国民法典" and c.article == "第577条" for c in r3b["applicable"]))
    assert_equal("约束2 排除说明已记录",
                 any("合同法" in n and "107" in n for n in r3b["validation_notes"]))

    print("\n[用例4] 司法解释位阶依附（约束3）")
    r4 = run_pipeline("违约金约定过分高于造成的损失，能否请求减少？", act_date="2025-06-01")
    jiti = [c for c in r4["ranked"] if c.hierarchy == "司法解释"]
    assert_equal("约束3 司法解释被召回", bool(jiti))
    if jiti:
        assert_equal("约束3 司法解释位阶 = 被解释法律位阶 + 0.5",
                     rank_of(jiti[0].hierarchy, jiti[0].host_hierarchy) == 1.5)
        assert_equal("约束3 司法解释排位低于法律、高于部门规章",
                     rank_of("部门规章", "") > 1.5 > rank_of("法律", ""))

    print("\n=== 全部断言通过 ✅ ===")


if __name__ == "__main__":
    main()
