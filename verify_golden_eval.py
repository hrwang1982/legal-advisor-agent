"""P2 黄金问答评测集（Golden QA Benchmark）与可交付合规/审计闭环门禁验证脚本。

运行方式：
    python3 verify_golden_eval.py
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MOCK", "true")

from app.audit import AuditStore, audit_store
from app.compliance import (
    build_compliance_disclaimer,
    check_conflict_of_interest,
    mask_pii,
    verify_reviewer_permission,
)
from app.graph import graph, route_after_review
from app.nodes import (
    apply_citation_edits,
    decompose,
    extract_facts_node,
    human_review,
    rank_hierarchy,
    rerank_node,
    retrieve,
    rewrite,
    rules_node,
    synthesize,
    validate,
    verify_citation_closure,
)


def run_pipeline(payload: dict) -> dict:
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


GOLDEN_CASES = [
    {
        "id": "G01_现行合同违约与引致扩展",
        "payload": {
            "query": "对方不履行合同义务造成我方损失，我可以要求继续履行并赔偿损失吗？",
            "act_date": "2025-05-01",
        },
        "must_include": [
            ("中华人民共和国民法典", "第577条"),
            ("中华人民共和国民法典", "第584条"),
        ],
        "must_exclude": [
            ("中华人民共和国合同法", "第107条"),
        ],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G02_民法典施行前历史合同纠纷",
        "payload": {
            "query": "对方不履行合同义务，我可以要求继续履行并赔偿损失吗？",
            "act_date": "2020-06-01",
        },
        "must_include": [
            ("中华人民共和国合同法", "第107条"),
        ],
        "must_exclude": [
            ("中华人民共和国民法典", "第577条"),
        ],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G03_新法词问旧案_公司五年实缴回填2018旧法",
        "payload": {
            "query": "有限责任公司全体股东认缴的出资额必须在五年内缴足吗？",
            "act_date": "2023-05-01",
        },
        "must_include": [
            ("中华人民共和国公司法（2018年修正）", "第26条"),
        ],
        "must_exclude": [
            ("中华人民共和国公司法", "第47条"),
        ],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G04_口语多争点_炒鱿鱼赔偿与买假货退一赔三",
        "payload": {
            "query": "我被老板无故炒鱿鱼了，能拿多少赔偿？另外在商场买到假货能退一赔三吗？",
            "act_date": "2025-05-01",
        },
        "must_include": [
            ("中华人民共和国劳动合同法", "第87条"),
            ("中华人民共和国劳动合同法", "第47条"),
            ("中华人民共和国消费者权益保护法", "第55条"),
        ],
        "must_exclude": [],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G05_特别法vs新一般法_机动车交通事故责任与民法典侵权",
        "payload": {
            "query": "员工开车执行工作任务发生机动车交通事故撞伤行人，侵权赔偿责任如何承担？",
            "act_date": "2025-05-01",
        },
        "must_include": [
            ("中华人民共和国道路交通安全法", "第76条"),
            ("中华人民共和国民法典", "第1191条"),
        ],
        "must_exclude": [],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G06_地域管辖核验_北京劳动纠纷排除上海地方性法规",
        "payload": {
            "query": "我在北京一家公司上班，用人单位违法解除劳动合同，劳动合同条例或劳动合同法如何规定赔偿金？",
            "act_date": "2025-05-01",
            "region": "北京",
        },
        "must_include": [
            ("中华人民共和国劳动合同法", "第87条"),
        ],
        "must_exclude": [
            ("上海市劳动合同条例", "第1条"),
        ],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G07_跨民法典施行日持续性违约事实_按结束日适用新法",
        "payload": {
            "query": "对方从2020年5月1日开始持续不履行合同义务，一直持续到2021年3月1日，我能否要求违约损害赔偿？",
        },
        "must_include": [
            ("中华人民共和国民法典", "第577条"),
        ],
        "must_exclude": [
            ("中华人民共和国合同法", "第107条"),
        ],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G08_诉讼时效期间与劳动争议仲裁时效区分",
        "payload": {
            "query": "向人民法院请求保护民事权利的三年诉讼时效期间怎么起算？劳动争议申请仲裁的时效期间是多久？",
            "act_date": "2025-05-01",
        },
        "must_include": [
            ("中华人民共和国民法典", "第188条"),
            ("中华人民共和国劳动争议调解仲裁法", "第27条"),
        ],
        "must_exclude": [],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G09_新公司法出资加速到期与引致五年认缴",
        "payload": {
            "query": "公司不能清偿到期债务，债权人能否要求未届出资期限的股东提前缴纳出资？",
            "act_date": "2025-05-01",
        },
        "must_include": [
            ("中华人民共和国公司法", "第54条"),
            ("中华人民共和国公司法", "第47条"),
        ],
        "must_exclude": [
            ("中华人民共和国公司法（2018年修正）", "第26条"),
        ],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G10_未传日期高危场景_强制降级置信度与人工必审",
        "payload": {
            "query": "对方违约不履行合同义务，我能否要求继续履行并赔偿损失？",
        },
        "must_include": [
            ("中华人民共和国民法典", "第577条"),
        ],
        "must_exclude": [
            ("中华人民共和国合同法", "第107条"),
        ],
        "expect_high_risk_calibration": True,
    },
]


def evaluate_golden_dataset() -> None:
    print("=" * 72)
    print("🏛️  开始执行黄金问答评测集（Golden QA Benchmark, N=10）")
    print("=" * 72)

    total_expected_articles = 0
    recalled_expected_articles = 0
    invalid_or_forbidden_misuse = 0
    high_risk_cases = 0
    high_risk_calibrated = 0
    closure_compliant_cases = 0

    for case in GOLDEN_CASES:
        cid = case["id"]
        state = run_pipeline(case["payload"])
        output = state["final_output"]
        plan = state["rule_plan"]
        conf = plan.get("confidence", "B")
        rev_lvl = plan.get("review", "must")
        kept_set = {(c.law_name, c.article) for c in output.citations}

        # 1) 核心法条召回统计
        for req in case["must_include"]:
            total_expected_articles += 1
            if req in kept_set:
                recalled_expected_articles += 1
            else:
                raise AssertionError(f"[{cid}] 核心法条漏召回: 期望 {req}，实际召回 {kept_set}")

        # 2) 失效/未施行/异地法条误用统计
        for forbid in case["must_exclude"]:
            if forbid in kept_set:
                invalid_or_forbidden_misuse += 1
                raise AssertionError(f"[{cid}] 严禁误用法条被保留在最终引用中: {forbid}")

        for c in output.citations:
            if not c.applicable_at:
                invalid_or_forbidden_misuse += 1
                raise AssertionError(f"[{cid}] 不适用法条混入第一层引用: {c.law_name} {c.article}")

        # 3) 高危案件置信度校准检查
        if case["expect_high_risk_calibration"] or state["fact_summary"].get("act_date_source") == "assumed_today":
            high_risk_cases += 1
            if conf != "A" and rev_lvl in ("must", "expert", "immediate"):
                high_risk_calibrated += 1
            else:
                raise AssertionError(
                    f"[{cid}] 高危场景未降级置信度: confidence={conf}, review={rev_lvl}"
                )

        # 4) 生成正文引用闭包检查（正文出现的《XX》第X条必须属于第一层 citations 或程序元法规白名单）
        closure_warns = verify_citation_closure(output.answer, output.citations)
        if not closure_warns:
            closure_compliant_cases += 1
        else:
            raise AssertionError(f"[{cid}] 违反引用闭包: {closure_warns}")

        print(
            f"  [PASS] {cid:<42} | 召回条文: {len(output.citations)} 条 | "
            f"置信度: {conf} | 复核级别: {rev_lvl}"
        )

    recall_rate = recalled_expected_articles / max(total_expected_articles, 1) * 100.0
    misuse_rate = invalid_or_forbidden_misuse / len(GOLDEN_CASES) * 100.0
    calibration_rate = high_risk_calibrated / max(high_risk_cases, 1) * 100.0
    closure_rate = closure_compliant_cases / len(GOLDEN_CASES) * 100.0

    print("-" * 72)
    print(f"📊 门禁指标 1 - 核心法条召回率 (Core Article Recall):       {recall_rate:.1f}% (目标 ≥ 95.0%)")
    print(f"📊 门禁指标 2 - 失效/异地法误用率 (Invalid Law Misuse):    {misuse_rate:.1f}% (目标 = 0.0%)")
    print(f"📊 门禁指标 3 - 高危置信度校准率 (High-Risk Calibration):  {calibration_rate:.1f}% (目标 = 100.0%)")
    print(f"📊 门禁指标 4 - 生成引用闭包合规率 (Citation Closure):     {closure_rate:.1f}% (目标 = 100.0%)")
    print("-" * 72)

    assert recall_rate >= 95.0, "核心法条召回率未达标"
    assert misuse_rate == 0.0, "失效或异地法条误用率必须为 0%"
    assert calibration_rate == 100.0, "高危案件置信度校准率必须为 100%"
    assert closure_rate == 100.0, "生成正文引用闭包合规率必须为 100%"


def verify_p2_compliance_and_audit_closed_loop() -> None:
    print("🔐 开始执行 P2 合规守卫、律师工作台与防篡改审计闭环验证")
    print("-" * 72)

    # 1. PII 脱敏测试（身份证 + 手机号 + 银行卡号）
    raw_text = "当事人李四，身份证号 310101198805123456，手机号 13987654321，银行卡 6222021001012345678，咨询同业竞争违约金。"
    masked_text, pii_types = mask_pii(raw_text)
    assert "310101198805123456" not in masked_text, "身份证号未脱敏"
    assert "13987654321" not in masked_text, "手机号未脱敏"
    assert "6222021001012345678" not in masked_text, "银行卡号未脱敏"
    assert len(pii_types) == 3, f"期望识别 3 类 PII，实际识别 {pii_types}"
    print("  [PASS] P2.1 PII 隐私脱敏（身份证/手机号/银行卡号）验证通过")

    # 2. 律所利益冲突前置检索（《律师法》第39条）
    coi_res = check_conflict_of_interest("我方拟起诉澜舟科技有限公司合同违约，索赔 500 万元。")
    assert coi_res["has_conflict"] is True, "命中律所既有客户时应触发利益冲突告警"
    assert any(c["registered_client"] == "澜舟科技有限公司" for c in coi_res["conflicts"]), "应返回命中的利益冲突客户名称"
    print("  [PASS] P2.2 律所利益冲突前置拦截（《律师法》第39条）验证通过")

    # 3. 律师工作台：分级权限拦截 -> 补充检索路由 -> 合伙人接管并增删引用/改稿
    state = run_pipeline({
        "query": "我方拟起诉北京澜舟科技有限公司不履行合同义务，身份证110101199003078888",
        "act_date": "2025-05-01",
    })
    assert state["rule_plan"]["review"] == "expert", "命中利益冲突应强制升级为 expert 复核"

    # 3.1 普通执业律师尝试直接审批 expert 级案件 -> 应被权限守卫拦截
    jr_review = human_review({
        **state,
        "_mock_review_feedback": {
            "thread_id": "test-coi-01",
            "decision": "approve",
            "reviewer_name": "张实习",
            "reviewer_role": "lawyer",
            "comments": "普通律师尝试直接放行",
        },
    })
    assert jr_review["human_approved"] is False, "普通律师审批 expert 案件必须被权限守卫拦截"
    assert jr_review["review_action"] == "revise", "权限不足时应回退至 revise"
    assert jr_review["review_audit"]["event_type"] == "permission_denied", "应记录 permission_denied 审计事件"
    print("  [PASS] P2.3 分级权限守卫（拦截普通律师越权审批 expert 级案件）验证通过")

    # 3.2 律师触发 re_retrieve 补充检索重跑 -> 路由必须回到 extract_facts
    re_ret_state = human_review({
        **state,
        "_mock_review_feedback": {
            "thread_id": "test-coi-01",
            "decision": "re_retrieve",
            "extra_query": "补充检索违约金过高请求适当减少的民法典规定",
            "act_date": "2025-05-01",
            "reviewer_name": "张合伙人",
            "reviewer_role": "partner",
            "comments": "需补充违约金调减法条",
        },
    })
    next_node = route_after_review({**state, **re_ret_state})
    assert next_node == "extract_facts", f"re_retrieve 决策应路由至 extract_facts，实际为 {next_node}"
    assert "违约金过高" in re_ret_state["query"], "re_retrieve 应将补充检索词合入 query"
    print("  [PASS] P2.4 律师工作台触发补充检索重跑（re_retrieve -> extract_facts）验证通过")

    # 3.3 合伙人增删引用 + 在线改稿审批通过
    first_c = state["final_output"].citations[0]
    rem_str = f"{first_c.law_name}|{first_c.article}"
    sr_review = human_review({
        **state,
        "_mock_review_feedback": {
            "thread_id": "test-coi-01",
            "decision": "approve",
            "reviewer_name": "张合伙人",
            "license_no": "11101201010888888",
            "reviewer_role": "partner",
            "comments": "合伙人已核查利益冲突豁免函，补充公司法第54条并修订终稿",
            "edited_answer": state["final_output"].answer + "\n\n【合伙人审定意见：已核验利益冲突豁免，同意出具。】",
            "add_citations": ["中华人民共和国公司法|第54条"],
            "remove_citations": [rem_str],
        },
    })
    assert sr_review["human_approved"] is True, "合伙人审批 expert 级案件应放行"
    final_cit_keys = {(c.law_name, c.article) for c in sr_review["final_output"].citations}
    assert ("中华人民共和国公司法", "第54条") in final_cit_keys, "合伙人新增的第54条应写入终稿引用"
    assert (first_c.law_name, first_c.article) not in final_cit_keys, "合伙人移除的首条引用应从终稿引用剔除"
    assert "合伙人审定意见" in sr_review["final_output"].answer, "合伙人在线修订正文应生效"
    assert "生成式人工智能辅助生成" in sr_review["final_output"].disclaimer, "终稿必须附带生成式AI合规声明"
    print("  [PASS] P2.5 合伙人增删第一层引用、在线改稿与合规免责标识验证通过")

    # 4. SQLite 会话持久化恢复 + SHA-256 防篡改审计账本与篡改检测验证
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_f:
        tmp_db_path = tmp_f.name

    try:
        store = AuditStore(tmp_db_path)
        # 4.1 会话快照保存与跨实例恢复
        store.save_snapshot(
            "thread-prod-001",
            status="approved",
            state_data={
                "query": state["masked_query"],
                "act_date": "2025-05-01",
                "rule_plan": state["rule_plan"],
                "human_approved": True,
            },
        )
        store_reloaded = AuditStore(tmp_db_path)
        loaded_snap = store_reloaded.get_snapshot("thread-prod-001")
        assert loaded_snap is not None, "SQLite 应成功恢复会话快照"
        assert loaded_snap["status"] == "approved", "恢复的会话状态应为 approved"
        assert loaded_snap["state"]["human_approved"] is True, "恢复的快照应保留复核状态"

        # 4.2 连续写入两条审计记录，验证哈希链
        log1 = store_reloaded.record_audit_event(
            thread_id="thread-prod-001",
            event_type="permission_denied",
            decision="reject_approve",
            reviewer_name="张实习",
            reviewer_role="lawyer",
            comments="越权审批被拒",
            confidence="B",
            review_level="expert",
            citations=[c.model_dump() for c in state["final_output"].citations],
            draft_answer=state["final_output"].answer,
            final_answer=state["final_output"].answer,
        )
        log2 = store_reloaded.record_audit_event(
            thread_id="thread-prod-001",
            event_type="human_review",
            decision="approve",
            reviewer_name="张合伙人",
            license_no="11101201010888888",
            reviewer_role="partner",
            comments="合伙人复核通过",
            confidence="B",
            review_level="expert",
            citations=[c.model_dump() for c in sr_review["final_output"].citations],
            draft_answer=state["final_output"].answer,
            final_answer=sr_review["final_output"].answer,
        )
        assert log2["prev_hash"] == log1["entry_hash"], "第二条审计日志的 prev_hash 必须等于第一条的 entry_hash"
        ok_chain, cnt_chain, _ = store_reloaded.verify_hash_chain()
        assert ok_chain is True and cnt_chain == 2, "正常审计账本哈希链校验必须通过"

        # 4.3 模拟恶意篡改数据库中的审计终稿，验证 verify_hash_chain() 能否捕获篡改
        conn = sqlite3.connect(tmp_db_path)
        conn.execute("UPDATE audit_logs SET final_answer = '被恶意篡改的法律意见终稿' WHERE id = 1")
        conn.commit()
        conn.close()

        ok_tamper, _, err_msg = store_reloaded.verify_hash_chain()
        assert ok_tamper is False, "篡改数据库记录后，SHA-256 哈希链校验必须报警 valid=False"
        assert "#1" in err_msg, f"应精准定位被篡改的记录 #1，实际返回: {err_msg}"
        print("  [PASS] P2.6 SQLite 会话持久化恢复与 SHA-256 防篡改哈希链（含篡改检测）验证通过")

        # 5. P0 漏洞封堵专项验证：条号级引用闭包核验（拦截同部法律编造条号、民法典编造条号、中文数字条号、无书名号引用）
        c_lc38 = [
            c for c in state["final_output"].citations
            if c.law_name == "中华人民共和国民法典" and c.article == "第577条"
        ]
        hallucinated_ans = (
            "依据《中华人民共和国民法典》第577条、民法典第八千八百八十八条及《劳动合同法》第九百九十九条，对方应赔偿。"
        )
        closure_alerts = verify_citation_closure(hallucinated_ans, c_lc38)
        assert any("第8888条" in a for a in closure_alerts), f"必须拦截编造的民法典第8888条: {closure_alerts}"
        assert any("第999条" in a for a in closure_alerts), f"必须拦截编造的劳动合同法第999条: {closure_alerts}"
        print("  [PASS] P3.1 条号级防幻觉引用闭包（拦截民法典/同部法编造条号、中文数字与无书名号引用）验证通过")

        # 6. P0 执业证注册表防伪造越权验证（普通律师持本人执业证伪造自报 partner 角色 -> 必须被服务端注册表拦截）
        spoof_ok, spoof_msg = verify_reviewer_permission(
            "expert",
            reviewer_role="partner",
            license_no="13101201810012345",  # 注册表中核定职级为 lawyer（张律师）
        )
        assert spoof_ok is False and "越权拦截" in spoof_msg, f"应拦截伪造 partner 角色的普通律师执业证: {spoof_msg}"
        print("  [PASS] P3.2 执业律师注册表服务端鉴权（拦截持普通律师执业证伪造 partner 越权审批）验证通过")

        # 7. RAG 外规文档动态导入 + BGE-M3 向量化解析 + 混合检索端到端召回验证
        from app.config import settings
        from app.ingest import invalidate_cache
        from app.kb import KnowledgeBaseStore

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_kb:
            tmp_kb_path = tmp_kb.name
        old_kb_db = settings.kb_db_path
        try:
            settings.kb_db_path = tmp_kb_path
            test_kb = KnowledgeBaseStore(tmp_kb_path)
            ext_doc = (
                "第二十一条 国家建立数据分类分级保护制度，对数据实行分类分级保护，重要数据目录由有关部门制定。\n"
                "第四十五条 开展数据处理活动的组织、个人不履行数据安全保护义务的，由有关主管部门责令改正，给予警告，可以并处五万元以上五十万元以下罚款。"
            )
            imp_res = test_kb.ingest_document(
                raw_text=ext_doc,
                law_name="中华人民共和国数据安全法",
                hierarchy="法律",
                effective_date="2021-09-01",
                category="数据合规",
                source="全国人民代表大会常务委员会",
                scope="特别",
            )
            assert imp_res["ingested_count"] == 2 and imp_res["vector_dim"] > 0, f"外规导入应成功解析2条并生成向量: {imp_res}"
            invalidate_cache()

            ds_state = run_pipeline({
                "query": "企业未履行数据分类分级保护义务，违反数据安全法会受到什么行政罚款？",
                "act_date": "2025-05-01",
            })
            ds_cites = {(c.law_name, c.article) for c in ds_state["final_output"].citations}
            assert ("中华人民共和国数据安全法", "第45条") in ds_cites or ("中华人民共和国数据安全法", "第21条") in ds_cites, (
                f"导入的外规《数据安全法》应通过 BGE-M3 + BM25 混合检索被成功召回: {ds_cites}"
            )
            print("  [PASS] P3.3 RAG 知识库外规文档切分导入 + BGE-M3 向量解析与混合检索召回验证通过")
        finally:
            settings.kb_db_path = old_kb_db
            invalidate_cache()
            if os.path.exists(tmp_kb_path):
                os.remove(tmp_kb_path)

        # 8. 外置知识与规则配置中心（免改代码热更新）+ 外部大模型配置（千问 Qwen / 本地 OpenAI 兼容接口）验证
        from app.knowledge import knowledge_center

        orig_syn = knowledge_center.get_synonym_map()
        try:
            new_syn = dict(orig_syn)
            new_syn["断缴五险"] = "未依法为劳动者缴纳社会保险费 解除劳动合同 经济补偿"
            knowledge_center.update_section("legal_synonym_map", new_syn)
            assert "断缴五险" in knowledge_center.get_synonym_map(), "外置同义词热更新应立即生效"
        finally:
            knowledge_center.update_section("legal_synonym_map", orig_syn)

        updated_cfg = settings.update_runtime({"provider_preset": "qwen", "mock": True})
        assert "dashscope.aliyuncs.com" in updated_cfg["llm_api_base"], "切换千问预设应自动配置 DashScope OpenAI 兼容接口"
        updated_local = settings.update_runtime({"provider_preset": "local_openai", "mock": True})
        assert updated_local["embed_model"] == "BAAI/bge-m3", "切换本地预设应默认配置 BAAI/bge-m3 向量模型"
        print("  [PASS] P3.4 外置知识规则中心热更新（免改代码）与千问/本地大模型预设切换验证通过")

        # 9. 知识飞轮与大模型兜底闭环验证（建议1~4优化落地）
        # 9.1 未收录案由动态推导（建议1）+ 补充法条候选池与闭包豁免（建议2）+ 解释学涵摄提示（建议3）
        construct_q = "2024年包工头承包厂房主体结构施工，发包人迟延支付500万工程进度款拒不审价"
        res_dyn = graph.invoke(
            {"query": construct_q, "thread_id": "thread-test-dynamic-construct"},
            {"configurable": {"thread_id": "thread-test-dynamic-construct"}},
        )
        fs_dyn = res_dyn.get("fact_summary") or {}
        fo_dyn = res_dyn.get("final_output")
        assert any("建设工程" in ct for ct in fs_dyn.get("case_types", [])), "未收录案由应动态推导为建设工程纠纷"
        assert len(fs_dyn.get("missing_facts", [])) > 0, "动态要件推导应包含缺失事实要件清单"
        assert fo_dyn is not None, "应成功生成两层咨询意见"
        assert len(fo_dyn.suggested_citations) > 0, "应生成大模型推荐补充法条候选池（建议2）"
        assert not any("引用闭包告警" in n and "建议" in n for n in res_dyn.get("rule_notes", [])), "建议补充条文章节应豁免闭包校验拦截"

        # 9.2 知识飞轮（建议4）：自动捕获提案并支持一键采纳合入 knowledge_config.json
        pending_props = audit_store.list_rule_proposals(status="PENDING", limit=20)
        assert len(pending_props) > 0, "动态推导成果应自动归集为待审规则提案（rule_proposals）"
        case_prop = next((p for p in pending_props if p["proposal_type"] == "case_type_rules"), None)
        assert case_prop is not None, "应成功归集 case_type_rules 类型的规则提案"
        
        orig_case_rules = knowledge_center.get_case_type_rules()
        try:
            ok_adopt, adopt_msg = audit_store.adopt_rule_proposal(case_prop["id"], reviewer_name="资深规则专家")
            assert ok_adopt is True, f"采纳提案应返回成功: {adopt_msg}"
            reloaded_rules = knowledge_center.get_case_type_rules()
            assert case_prop["proposal_key"] in reloaded_rules or "建设工程" in str(reloaded_rules), "采纳后新案由规则应直接热合入知识配置库"
        finally:
            knowledge_center.update_section("case_type_rules", orig_case_rules)

        print("  [PASS] P3.5 大模型动态要件推导、补充法条池与规则提案一键采纳知识飞轮验证通过")
    finally:
        if os.path.exists(tmp_db_path):
            os.remove(tmp_db_path)


if __name__ == "__main__":
    evaluate_golden_dataset()
    verify_p2_compliance_and_audit_closed_loop()
    print("=" * 72)
    print("🎉 P2 黄金问答评测集与合规/审计全链路验收 100% 通过！")
    print("=" * 72)
