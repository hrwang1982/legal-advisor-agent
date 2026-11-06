"""核心链路冒烟与 P0/P1 集成测试（MOCK 模式，免密钥）：
覆盖：事实抽取 → 问题拆解 → 法言法语改写 → 混合检索（含引致条款扩展 + 新旧法条对照回填）
→ 有效性打标 → 多子问题重排与噪声截断 → 位阶排序 → 规则判定 → 六段式合成与三重后验校验。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MOCK", "true")

from app.audit import AuditStore
from app.compliance import check_conflict_of_interest, mask_pii, verify_reviewer_permission
from app.graph import route_after_review
from app.models import Citation
from app.nodes import (
    apply_citation_edits,
    decompose,
    extract_facts_node,
    human_review,
    post_verify_answer,
    rank_hierarchy,
    rerank_node,
    retrieve,
    rewrite,
    rules_node,
    synthesize,
    validate,
)


def run_full_state(payload: dict) -> dict:
    state = dict(payload)
    for fn in (
        extract_facts_node,
        decompose,
        rewrite,
        retrieve,
        validate,
        rerank_node,
        rank_hierarchy,
        rules_node,
        synthesize,
    ):
        state.update(fn(state))
    return state


def test_pipeline():
    # 1) 未显式传 act_date 的现行咨询场景：验证事实抽取、旧法在 rules 节点排除、置信度降级为 B（必审）
    state = run_full_state({"query": "对方不履行合同义务，我可以要求继续履行并赔偿损失吗？"})
    assert state["fact_summary"]["act_date_source"] == "assumed_today", "未传日期应标记为 assumed_today"
    assert state["decomposed"], "问题拆解应产生子问题"
    assert state["rewritten_queries"], "问题改写应产生检索 query"
    assert state["retrieved"], "应检索到候选条文"

    validated_set = {(c.law_name, c.article) for c in state["validated"]}
    assert ("中华人民共和国合同法", "第107条") in validated_set, "validate 阶段应保留旧法待时间定位"

    kept = {(c.law_name, c.article) for c in state["ranked"]}
    assert ("中华人民共和国合同法", "第107条") not in kept, "现行纠纷应在 rules_node 排除已废止《合同法》第107条"
    assert ("中华人民共和国民法典", "第577条") in kept, "现行纠纷应保留《民法典》第577条"
    # P1 验证：召回《民法典》第577条时，引致图谱应自动关联补齐第584条（损害赔偿范围与可预见规则）
    assert ("中华人民共和国民法典", "第584条") in kept, "P1引致图谱应自动关联《民法典》第584条"
    assert state["ranked"][0].hierarchy == "法律", "排序后首位应为最高位阶(法律)"
    assert state["rule_plan"]["confidence"] == "B", "未提供行为发生日期时不得静默给 A 级置信度，应降级为 B"
    assert state["rule_plan"]["review"] == "must", "未提供行为发生日期时必须进入人工必审"
    assert state["final_output"].citations, "第一层引用不可为空"
    assert state["final_output"].answer, "第二层回答不可为空"
    assert state["final_output"].missing_facts, "应输出待核实关键事实清单"
    assert state["final_output"].procedural_notes, "应输出诉讼时效等程序提示"

    # 2) 历史纠纷场景（2020-06-01）：验证《合同法》第107条被保留为行为时旧法，而《民法典》因尚未施行被排除
    hist_state = run_full_state({
        "query": "对方不履行合同义务，我可以要求继续履行并赔偿损失吗？",
        "act_date": "2020-06-01",
    })
    hist_kept = {(c.law_name, c.article) for c in hist_state["ranked"]}
    assert ("中华人民共和国合同法", "第107条") in hist_kept, "2020年纠纷应适用行为时旧法《合同法》第107条"
    assert ("中华人民共和国民法典", "第577条") not in hist_kept, "2020年纠纷不得溯及适用2021年施行的《民法典》"

    # 3) P1 验证：口语化表达（「被老板无故炒鱿鱼」「买到假货能退一赔三吗」）通过法言法语扩展与引致图谱精准命中
    colloq_state = run_full_state({
        "query": "我被老板无故炒鱿鱼了，能拿多少赔偿？另外在商场买到假货能退一赔三吗？",
        "act_date": "2025-05-01",
    })
    colloq_kept = {(c.law_name, c.article) for c in colloq_state["ranked"]}
    assert ("中华人民共和国劳动合同法", "第87条") in colloq_kept, "P1口语扩展：『炒鱿鱼』应命中《劳动合同法》第87条违法解除赔偿金"
    assert ("中华人民共和国劳动合同法", "第47条") in colloq_kept, "P1引致扩展：第87条应自动引致补齐第47条经济补偿基数计算标准"
    assert ("中华人民共和国消费者权益保护法", "第55条") in colloq_kept, "P1口语扩展+多子问题重排：『退一赔三』应命中《消费者权益保护法》第55条"

    # 4) P1 验证：按新法术语提问历史公司出资纠纷（2023-05-01），新旧法条文对照表自动回填2018版《公司法》第26条
    corp_hist = run_full_state({
        "query": "有限责任公司全体股东认缴的出资额必须在五年内缴足吗？",
        "act_date": "2023-05-01",
    })
    corp_kept = {(c.law_name, c.article) for c in corp_hist["ranked"]}
    assert ("中华人民共和国公司法（2018年修正）", "第26条") in corp_kept, "P1新旧法回填：2023年公司出资问题应自动回填2018版《公司法》第26条"
    assert ("中华人民共和国公司法", "第47条") not in corp_kept, "2023年事实不得适用2024-07-01施行的《公司法》第47条"

    # 5) P1 验证：生成后三重后验校验器（引用闭包拦截 + 部分失效披露补齐 + 兜底措辞降级）
    c_partial = Citation(
        law_name="中华人民共和国民法典",
        article="第577条",
        hierarchy="法律",
        partially_invalid=True,
        invalid_parts="第2款",
        verified=False,
    )
    mock_plan = {"llm_callable": True, "mode": "llm_fallback", "confidence": "C", "review": "expert"}
    bad_ans = "依据《中华人民共和国民法典》第577条及《中华人民共和国刑法》第266条，本案必然胜诉，一定能获得全部赔偿。"
    fixed_ans, p_notes = post_verify_answer(bad_ans, [c_partial], mock_plan)
    assert any("引用闭包告警" in n and "刑法" in n for n in p_notes), "P1后验校验应拦截未召回的《刑法》第266条"
    assert "第2款" in fixed_ans and "已失效" in fixed_ans, "P1后验校验应自动补齐部分失效条款排除警示"
    assert "待核验" in fixed_ans, "P1后验校验应自动补齐未核验条文警示"
    assert "必然胜诉" not in fixed_ans and "一定能获得" not in fixed_ans, "P1后验校验应在兜底模式下消除绝对化措辞"

    # 6) P2 验证：PII 个人信息自动脱敏 + 律所利益冲突前置拦截（《律师法》第39条）
    pii_state = run_full_state({
        "query": "我叫张三，身份证110101199003071234，电话13812345678，想起诉北京澜舟科技有限公司违约。",
        "act_date": "2025-05-01",
    })
    assert "110101199003071234" not in pii_state["masked_query"], "P2合规：masked_query 中的身份证号必须被自动脱敏"
    assert "13812345678" not in pii_state["masked_query"], "P2合规：masked_query 中的手机号必须被自动脱敏"
    assert len(pii_state["compliance_info"]["pii_types"]) >= 2, "P2合规：应记录脱敏 PII 类型"
    assert pii_state["compliance_info"]["has_conflict"] is True, "P2合规：命中律所既有客户名录应触发利益冲突预警"
    assert pii_state["rule_plan"]["review"] == "expert", "P2合规：利益冲突案件必须升级为 expert 专家/合伙人复核"

    # 7) P2 验证：律师工作台增删第一层引用（apply_citation_edits）与分级权限校验（verify_reviewer_permission）
    first_cit = pii_state["final_output"].citations[0]
    rem_spec = f"{first_cit.law_name}|{first_cit.article}"
    updated_cits, edit_logs = apply_citation_edits(
        pii_state["final_output"].citations,
        add_specs=["中华人民共和国公司法|第54条"],
        remove_specs=[rem_spec],
    )
    updated_keys = {(c.law_name, c.article) for c in updated_cits}
    assert (first_cit.law_name, first_cit.article) not in updated_keys, "P2工作台：律师移除的条文不应保留在第一层引用中"
    assert ("中华人民共和国公司法", "第54条") in updated_keys, "P2工作台：律师新增的条文应自动从法条库补齐进入第一层引用"
    assert len(edit_logs) == 2, "P2工作台：应完整记录增删操作日志"

    ok_jr, _ = verify_reviewer_permission("expert", "lawyer")
    ok_sr, _ = verify_reviewer_permission("expert", "senior_lawyer")
    assert not ok_jr and ok_sr, "P2权限：expert 级复核必须拦截普通 lawyer，仅放行 senior_lawyer/partner/expert"

    # 8) P2 验证：SQLite 防篡改审计哈希链（SHA-256）+ AI 初稿与律师终稿 Diff + 规则缺口自动归集
    mem_store = AuditStore(":memory:")
    rec1 = mem_store.record_audit_event(
        thread_id="test-t1",
        event_type="human_review",
        decision="approve",
        reviewer_name="王合伙人",
        license_no="11101201510123456",
        reviewer_role="partner",
        comments="经高级合伙人复核通过，补充第509条",
        confidence="B",
        review_level="expert",
        citations=[c.model_dump() for c in updated_cits],
        draft_answer=pii_state["final_output"].answer,
        final_answer=pii_state["final_output"].answer + "\n【合伙人补充意见：建议先行发送律师函催告。】",
    )
    assert rec1["entry_hash"] and len(rec1["entry_hash"]) == 64, "P2审计：应生成64位 SHA-256 审计哈希"
    assert "合伙人补充意见" in rec1["answer_diff"], "P2审计：应自动计算 AI 初稿与律师终稿的 Unified Diff"
    chain_ok, checked_cnt, _ = mem_store.verify_hash_chain()
    assert chain_ok is True and checked_cnt == 1, "P2审计：哈希链完整性校验应通过"

    mem_store.record_rule_gap(
        thread_id="test-t1",
        plan={
            "mode": "llm_with_warning",
            "confidence": "B",
            "review": "expert",
            "reason": "时间边界或利益冲突预警",
            "verify_actions": ["核查利益冲突"],
        },
        query=pii_state["masked_query"],
    )
    assert len(mem_store.list_rule_gaps()) == 1, "P2闭环：PARTIAL/MISS/CONFLICT 应自动归集至规则缺口表"

    # 9) 融合 ks-lma-chat 专项测试：三维法律检索式构建 + 句级细粒度溯源对齐（Trace Engine）
    fusion_state = run_full_state({"query": "对方不履行合同义务，我可以要求继续履行并赔偿损失吗？"})
    sq = fusion_state.get("structured_queries") or {}
    assert "focus" in sq and "exception" in sq and "consequence" in sq, "融合特性：应输出三维法律检索式（focus/exception/consequence）"
    assert len(sq["focus"]) > 0 and len(sq["consequence"]) > 0, "三维检索式：焦点式与后果式不为空"

    fo = fusion_state.get("final_output")
    assert fo is not None, "最终两层结构回答不应为空"
    traces = getattr(fo, "sentence_traces", [])
    assert len(traces) > 0, "细粒度溯源：两层回答中应包含 sentence_traces 句级溯源条目"
    first_tr = traces[0]
    assert first_tr.sentence_idx >= 1, "溯源条目序号应正确"
    assert first_tr.matched_law and first_tr.matched_article, "溯源条目应绑定到具体法律法规与条号"
    assert first_tr.confidence_score > 0, "溯源匹配度分数应大于0"
    assert first_tr.highlight_snippet, "溯源条目应定位出法条原文的高亮支撑片段"

    # 10) 「不使用知识库」专项测试：跳过知识库检索，纯由大模型直接生成答复
    from app.graph import graph
    for no_kb_val in ("none", "不使用知识库"):
        no_kb_state = run_full_state({
            "query": "客户通过手机银行购买高风险理财产品需要做哪些额外测试",
            "kb_scope": no_kb_val,
        })
        assert no_kb_state.get("retrieved") == [], "不使用知识库时应完全跳过检索，retrieved 应为空"
        assert no_kb_state.get("ranked") == [], "不使用知识库时 ranked 应为空"
        assert no_kb_state.get("structured_queries") == {}, "不使用知识库时不应生成三维检索式"
        assert no_kb_state["rule_plan"]["mode"] == "direct_llm", "不使用知识库时模式应标记为 direct_llm"
        no_kb_fo = no_kb_state.get("final_output")
        assert no_kb_fo is not None and no_kb_fo.citations == [], "不使用知识库时第一层引用必须为空"
        assert no_kb_fo.answer and "未找到与您问题直接对应" not in no_kb_fo.answer, "不使用知识库时应由大模型直接生成答复，不得返回未找到条文提示"

    cfg_no_kb = {"configurable": {"thread_id": "test-no-kb-graph"}}
    graph.invoke({"query": "对方违约不交货怎么办？", "kb_scope": "none"}, cfg_no_kb)
    g_state = graph.get_state(cfg_no_kb).values
    assert g_state["rule_plan"]["mode"] == "direct_llm", "LangGraph 路由应在 kb_scope=none 时直达 synthesize"
    assert g_state["final_output"].citations == [], "LangGraph 路由在 kb_scope=none 时不返回知识库条文"
    assert "【一、" not in g_state["final_output"].answer, "不使用知识库时不得输出法律助手的六段式或章节框架"
    assert "【二、" not in g_state["final_output"].answer, "不使用知识库时不得输出法律助手的法理分析框架"
    assert len(g_state["final_output"].answer) > 20, "LangGraph 路由在 kb_scope=none 时应直接生成大模型答复"
    assert not g_state["final_output"].disclaimer, "不使用知识库时不得输出法律助手的免责声明"

    print("✅ 核心链路与 P0/P1/P2 全部 10 大专项场景（含 ks-lma-chat 融合特性与不使用知识库直连生成）测试通过")


if __name__ == "__main__":
    test_pipeline()
