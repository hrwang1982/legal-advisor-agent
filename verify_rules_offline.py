"""规则引擎与事实防错层离线验证（零第三方依赖）。

覆盖：时间效力（含实体从旧/程序从新/司法解释审理时点/从旧兼从轻/持续性事实/条文首次施行日）、
位阶裁决（含新一般 vs 旧特别/部门规章 vs 地方政府规章/成对特别一般关系）、
事实要素抽取与缺省日期降级策略、以及「无规则时的 LLM 兜底策略」是否安全。
运行：python3 verify_rules_offline.py
"""
from app.facts import analyze_facts, check_regional_applicability
from app.ingest import build_version_timelines
from app.rules import (
    Confidence,
    ReviewLevel,
    RuleStatus,
    _same_legislative_body,
    aggregate_outcomes,
    fallback_plan,
    is_incomparable,
    locate_article,
    prompt_footer,
    rank_of,
    resolve_hierarchy,
    temporal_locate,
)

PASS, FAIL = 0, 0


def check(name: str, cond: bool, extra: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} {extra}")


def section(title: str):
    print(f"\n[{title}]")


def main():
    # ————————————————————————————————————————
    section("规则一：时间效力定位")
    versions = [
        {"version": "合同法(1999)", "effective_from": "1999-10-01", "effective_to": "2020-12-31"},
        {"version": "民法典(2021)", "effective_from": "2021-01-01", "effective_to": None},
    ]

    # 1) 唯一命中
    o = temporal_locate("2019-06-01", versions)
    check("HIT：2019 年行为适用《合同法》", o.status == RuleStatus.HIT
          and o.detail.get("applicable_version") == "合同法(1999)")
    check("HIT 不调用 LLM", not fallback_plan(o)["llm_callable"])

    # 2) 法不溯及既往（早于最早生效日）
    o = temporal_locate("1998-01-01", versions)
    check("HIT：早于生效日 → 依法不溯及既往",
          o.status == RuleStatus.HIT and o.confidence == Confidence.A
          and o.detail.get("applicable") is False)

    # 3) 空档期 → 未覆盖，允许受限兜底
    gap = [{"version": "旧版", "effective_from": "2000-01-01", "effective_to": "2010-12-31"},
           {"version": "新版", "effective_from": "2012-01-01", "effective_to": None}]
    o = temporal_locate("2011-06-01", gap)
    p = fallback_plan(o)
    check("MISS：空档期 → LLM 受限兜底", o.status == RuleStatus.MISS and p["mode"] == "llm_fallback")
    check("MISS 必审+专家", o.review == ReviewLevel.EXPERT)
    check("MISS 携带硬约束", len(p["prompt_constraints"]) >= 5)
    check("MISS 给出核实事项", len(p["verify_actions"]) >= 1)

    # 4) 版本重叠 → 规则冲突，禁止 LLM
    overlap = [{"version": "A", "effective_from": "2020-01-01", "effective_to": None},
               {"version": "B", "effective_from": "2020-06-01", "effective_to": None}]
    o = temporal_locate("2021-01-01", overlap)
    p = fallback_plan(o)
    check("CONFLICT：版本重叠", o.status == RuleStatus.CONFLICT)
    check("CONFLICT 禁止 LLM 裁决", p["mode"] == "escalate" and not p["llm_callable"])
    check("CONFLICT 立即人工", o.review == ReviewLevel.IMMEDIATE)
    check("CONFLICT 不向 LLM 注入约束", prompt_footer(o) == "")

    # ————————————————————————————————————————
    section("规则二：位阶冲突裁决（《立法法》2023修正第100~106条）")

    # 5) 上位法优先
    o = resolve_hierarchy(
        {"law_name": "民法典", "hierarchy": "法律"},
        {"law_name": "某地方性法规", "hierarchy": "地方性法规"},
    )
    check("HIT：上位法优于下位法", o.status == RuleStatus.HIT
          and o.detail.get("precedence") == "民法典")

    # 6) 同一机关 → 新法优于旧法（非特别/一般关系时）
    o = resolve_hierarchy(
        {"law_name": "旧规定", "hierarchy": "法律", "authority": "全国人大", "promulgation_date": "2010-01-01"},
        {"law_name": "新规定", "hierarchy": "法律", "authority": "全国人大", "promulgation_date": "2020-01-01"},
    )
    check("HIT：同一机关新法优于旧法", o.status == RuleStatus.HIT
          and o.detail.get("precedence") == "新规定")

    # 7) 同一机关 → 特别法优于一般法
    o = resolve_hierarchy(
        {"law_name": "民法典", "hierarchy": "法律", "authority": "全国人大", "scope": "一般"},
        {"law_name": "消费者权益保护法", "hierarchy": "法律", "authority": "全国人大", "scope": "特别"},
    )
    check("HIT：同一机关特别法优于一般法", o.status == RuleStatus.HIT
          and o.detail.get("precedence") == "消费者权益保护法")

    # 7b) P0 关键修复：带颁布日期时，民法典(2020,一般) vs 消保法(2013,特别) 不得误判为“民法典较新优先”
    o_deference = resolve_hierarchy(
        {"law_name": "中华人民共和国民法典", "hierarchy": "法律", "authority": "全国人民代表大会",
         "promulgation_date": "2020-05-28", "scope": "一般"},
        {"law_name": "中华人民共和国消费者权益保护法", "hierarchy": "法律", "authority": "全国人民代表大会常务委员会",
         "promulgation_date": "2013-10-25", "scope": "特别"},
    )
    check("P0修复：民法典(2020) vs 消保法(2013) 依民法典第11条转致及立法法第103条判消保法优先",
          o_deference.status == RuleStatus.HIT
          and o_deference.detail.get("precedence") == "中华人民共和国消费者权益保护法")

    # 7c) P0 关键修复：无转致条款的「新的一般规定 vs 旧的特别规定」→ 立法法第105/106条 CONFLICT 转裁决
    o_new_gen = resolve_hierarchy(
        {"law_name": "某领域一般规定", "hierarchy": "法律", "authority": "全国人大常委会",
         "promulgation_date": "2023-01-01", "scope": "一般"},
        {"law_name": "某领域特别规定", "hierarchy": "法律", "authority": "全国人大常委会",
         "promulgation_date": "2015-01-01", "scope": "特别"},
    )
    p_new_gen = fallback_plan(o_new_gen)
    check("P0修复：新的一般规定 vs 旧的特别规定（无转致）→ CONFLICT 禁止直接套用新法优于旧法",
          o_new_gen.status == RuleStatus.CONFLICT
          and p_new_gen["mode"] == "escalate"
          and not p_new_gen["llm_callable"])

    # 8) 地方性法规 vs 部门规章 → 无权判定，LLM 也不得介入
    o = resolve_hierarchy(
        {"law_name": "某省条例", "hierarchy": "地方性法规"},
        {"law_name": "某部规章", "hierarchy": "部门规章"},
    )
    p = fallback_plan(o)
    check("地方性法规 vs 部门规章 → 转裁决", p["mode"] == "escalate" and not p["llm_callable"])
    check("转裁决为立即人工", o.review == ReviewLevel.IMMEDIATE)

    # 8b) P0 关键修复：部门规章 vs 地方政府规章 → 同等效力（《立法法》第102条、第106条），由国务院裁决
    o_dept_gov = resolve_hierarchy(
        {"law_name": "某省政府规章", "hierarchy": "地方政府规章", "authority": "某省人民政府"},
        {"law_name": "某部委规章", "hierarchy": "部门规章", "authority": "某部"},
    )
    p_dept_gov = fallback_plan(o_dept_gov)
    check("P0修复：部门规章 vs 地方政府规章具有同等效力（不得误判为上下位法），依法转国务院裁决",
          is_incomparable("部门规章", "地方政府规章")
          and p_dept_gov["mode"] == "escalate"
          and not p_dept_gov["llm_callable"]
          and o_dept_gov.review == ReviewLevel.IMMEDIATE)

    # 9) 同位阶不同机关 → 立法法第103条（原第92条）不适用
    o = resolve_hierarchy(
        {"law_name": "A", "hierarchy": "法律", "authority": "机关甲"},
        {"law_name": "B", "hierarchy": "法律", "authority": "机关乙"},
    )
    p = fallback_plan(o)
    check("同位阶不同机关 → 不得套用第103条(原第92条)", p["mode"] == "escalate" and not p["llm_callable"])

    # 10) 元数据缺失 → 允许受限兜底
    o = resolve_hierarchy(
        {"law_name": "X", "hierarchy": ""},
        {"law_name": "Y", "hierarchy": "法律"},
    )
    p = fallback_plan(o)
    check("位阶缺失 → LLM 受限兜底", o.status == RuleStatus.MISS and p["mode"] == "llm_fallback")
    check("位阶缺失 → 必审", o.review == ReviewLevel.MUST)
    check("位阶缺失 → 约束已注入", "不得生成任何未出现的条文号" in prompt_footer(o))

    # 11) 同位阶同机关但关系不明 → 部分覆盖（规则骨架 + LLM）
    o = resolve_hierarchy(
        {"law_name": "A", "hierarchy": "法律", "authority": "全国人大"},
        {"law_name": "B", "hierarchy": "法律", "authority": "全国人大"},
    )
    p = fallback_plan(o)
    check("PARTIAL：规则骨架 + LLM 补全", o.status == RuleStatus.PARTIAL and p["mode"] == "llm_assisted")
    check("PARTIAL 置信度 B", o.confidence == Confidence.B)

    # ————————————————————————————————————————
    section("多结果聚合：取最坏者 + 约束并集")
    o_partial = resolve_hierarchy(
        {"law_name": "A", "hierarchy": "法律", "authority": "全国人大"},
        {"law_name": "B", "hierarchy": "法律", "authority": "全国人大"},
    )
    o_hit = resolve_hierarchy(
        {"law_name": "民法典", "hierarchy": "法律"},
        {"law_name": "地方条例", "hierarchy": "地方性法规"},
    )
    agg = aggregate_outcomes([o_hit, o_partial])
    check("聚合：HIT + PARTIAL → PARTIAL", agg.status == RuleStatus.PARTIAL)
    check("聚合：整体必审", agg.review == ReviewLevel.MUST)
    check("聚合：HIT + PARTIAL → 仍允许 LLM 补全（HIT 不禁止 LLM）",
          agg.llm_allowed and fallback_plan(agg)["llm_callable"])

    o_miss = temporal_locate("2011-06-01", gap)
    agg4 = aggregate_outcomes([o_hit, o_miss])
    check("聚合：HIT + MISS → MISS 且允许兜底",
          agg4.status == RuleStatus.MISS and agg4.llm_allowed
          and fallback_plan(agg4)["llm_callable"])
    check("聚合：HIT + MISS → 兜底约束未丢失",
          bool(agg4.constraints) and "不得生成任何未出现的条文号" in prompt_footer(agg4))

    agg5 = aggregate_outcomes([o_hit])
    check("聚合：仅 HIT → 确定性执行，不需 LLM",
          agg5.status == RuleStatus.HIT
          and fallback_plan(agg5)["mode"] == "deterministic"
          and not fallback_plan(agg5)["llm_callable"])

    o_conflict = temporal_locate("2021-01-01", overlap)
    agg2 = aggregate_outcomes([o_hit, o_conflict])
    check("聚合：含 CONFLICT → CONFLICT", agg2.status == RuleStatus.CONFLICT)
    check("聚合：含 CONFLICT → 整体禁止 LLM",
          not agg2.llm_allowed and not fallback_plan(agg2)["llm_callable"])

    agg3 = aggregate_outcomes([])
    check("聚合：空输入 → MISS 且允许兜底",
          agg3.status == RuleStatus.MISS and agg3.llm_allowed)

    # ————————————————————————————————————————
    section("集成：版本时间线 + 时间效力定位")
    tl = build_version_timelines()
    check("时间线：已从法规库载入多部法律", len(tl) > 0)

    md = tl.get("中华人民共和国民法典", [])
    check("时间线：民法典含现行有效版本",
          any(v.get("validity") == "现行有效" for v in md))
    o = temporal_locate("2019-06-01", md)
    check("集成：2019 年行为 → 民法典不溯及既往",
          o.status == RuleStatus.HIT and o.detail.get("applicable") is False)
    o = temporal_locate("2025-06-01", md)
    check("集成：2025 年行为 → 适用民法典",
          o.status == RuleStatus.HIT and bool(o.detail.get("applicable_version")))

    htf = tl.get("中华人民共和国合同法", [])
    check("时间线：合同法已载入", len(htf) > 0)
    if htf:
        check("时间线：合同法失效日按民法典生效前一日推算",
              htf[0].get("effective_to") == "2020-12-31")
        o = temporal_locate("2010-01-01", htf)
        check("集成：2010 年行为 → 适用合同法",
              o.status == RuleStatus.HIT and bool(o.detail.get("applicable_version")))

    # ————————————————————————————————————————
    section("不变量：兜底安全性")
    samples = [
        temporal_locate("2019-06-01", versions),
        temporal_locate("2011-06-01", gap),
        temporal_locate("2021-01-01", overlap),
        resolve_hierarchy({"law_name": "民法典", "hierarchy": "法律"},
                          {"law_name": "地方条例", "hierarchy": "地方性法规"}),
        resolve_hierarchy({"law_name": "省条例", "hierarchy": "地方性法规"},
                          {"law_name": "部规章", "hierarchy": "部门规章"}),
        resolve_hierarchy({"law_name": "A", "hierarchy": ""}, {"law_name": "B", "hierarchy": "法律"}),
    ]
    check("CONFLICT 永不调用 LLM", all(
        not fallback_plan(o)["llm_callable"] for o in samples if o.status == RuleStatus.CONFLICT))
    check("非 HIT 均有核实事项或裁决路径", all(
        bool(o.verify_actions) for o in samples if o.status != RuleStatus.HIT))
    check("允许兜底者必带约束", all(
        bool(fallback_plan(o)["prompt_constraints"])
        for o in samples if fallback_plan(o)["llm_callable"]))

    # ————————————————————————————————————————
    section("位阶表：司法解释依附 + 不可比组合")
    check("司法解释依附被解释法律（法律 1 → 1.5）", rank_of("司法解释", "法律") == 1.5)
    check("司法解释位阶高于行政法规", rank_of("司法解释", "法律") < rank_of("行政法规", ""))
    check("司法解释位阶高于部门规章", rank_of("司法解释", "法律") < rank_of("部门规章", ""))
    check("司法解释无宿主时取保守位阶", rank_of("司法解释", "") == 1.5)
    check("地方性法规与部门规章不可比", is_incomparable("地方性法规", "部门规章"))
    check("部门规章与地方政府规章不可比", is_incomparable("部门规章", "地方政府规章"))
    check("法律与行政法规可机械比较", not is_incomparable("法律", "行政法规"))

    o = resolve_hierarchy(
        {"law_name": "民法典合同编通则解释", "hierarchy": "司法解释",
         "host_law": "中华人民共和国民法典", "host_hierarchy": "法律"},
        {"law_name": "中华人民共和国民法典", "hierarchy": "法律"},
    )
    check("司法解释 vs 被解释法律 → 解释关系而非冲突",
          o.status == RuleStatus.HIT and o.detail.get("relation") == "interpretation")
    check("解释关系以被解释法律为准", o.detail.get("precedence") == "中华人民共和国民法典")

    # ————————————————————————————————————————
    section("同一立法机关：《立法法》第103条（原第92条）的适用边界")
    check("全国人大与其常委会属同一立法机关",
          _same_legislative_body("全国人民代表大会", "全国人民代表大会常务委员会"))
    check("部委与省级人大常委会非同一机关",
          not _same_legislative_body("劳动部", "上海市人民代表大会常务委员会"))
    o = resolve_hierarchy(
        {"law_name": "中华人民共和国民法典", "hierarchy": "法律",
         "authority": "全国人民代表大会", "scope": "一般"},
        {"law_name": "中华人民共和国消费者权益保护法", "hierarchy": "法律",
         "authority": "全国人民代表大会常务委员会", "scope": "特别"},
    )
    check("人大与常委会之间可适用特别法优于一般法",
          o.status == RuleStatus.HIT
          and o.detail.get("precedence") == "中华人民共和国消费者权益保护法")

    # ————————————————————————————————————————
    section("条文级时间适用性：旧法保留 + P0 多维时间规则族")
    versions = [
        {"version": "2018修正", "effective_from": "2018-10-26", "effective_to": "2024-06-30"},
        {"version": "2023修订", "effective_from": "2024-07-01", "effective_to": None},
    ]
    o = locate_article("2024-03-01", "2018-10-26", versions, effective_to="2024-06-30")
    check("行为日落在旧法区间 → 适用旧法并标记历史法",
          bool(o.detail.get("applicable")) and bool(o.detail.get("historical")))
    check("历史法必须提示现已失效", "失效" in (o.detail.get("note") or ""))
    o = locate_article("2024-03-01", "2024-07-01", versions)
    check("新法尚未施行 → 不溯及既往，不适用", o.detail.get("applicable") is False)
    o = locate_article("2025-01-01", "2024-07-01", versions)
    check("新法施行后 → 适用新法且非历史法",
          bool(o.detail.get("applicable")) and not o.detail.get("historical"))
    o = locate_article("2025-01-01", "2018-10-26", versions, effective_to="2024-06-30")
    check("行为日在旧法失效之后 → 不再适用旧法", o.detail.get("applicable") is False)
    o = locate_article("bad-date", "2024-07-01", versions)
    check("行为日缺失 → MISS 且允许受限兜底",
          o.status == RuleStatus.MISS and fallback_plan(o)["llm_callable"])

    # P0 新增测试：条文原始生效日 vs 修正案生效日
    o_art_eff = locate_article(
        "2010-05-01",
        "2013-07-01",
        [],
        "",
        "《中华人民共和国劳动合同法》第38条",
        article_effective_from="2008-01-01",
    )
    check("P0修复：2010年事实适用自2008-01-01起施行的《劳动合同法》第38条（不受2012修正施行日误导）",
          o_art_eff.status == RuleStatus.HIT and o_art_eff.detail.get("applicable") is True)

    # P0 新增测试：司法解释按「审理时点已施行 + 行为发生于被解释法生效后」适用
    o_jiti_ok = locate_article(
        "2022-06-01",
        "2023-12-05",
        [],
        "",
        "《民法典合同编通则解释》第65条",
        norm_type="judicial_interpretation",
        host_effective_from="2021-01-01",
        ref_date="2025-06-01",
    )
    check("P0修复：2022年违约行为（民法典施行后）在2025年审理/咨询时适用2023年施行的《合同编通则解释》",
          o_jiti_ok.status == RuleStatus.HIT and o_jiti_ok.detail.get("applicable") is True)

    o_jiti_pre = locate_article(
        "2019-06-01",
        "2023-12-05",
        [],
        "",
        "《民法典合同编通则解释》第65条",
        norm_type="judicial_interpretation",
        host_effective_from="2021-01-01",
        ref_date="2025-06-01",
    )
    check("P0修复：2019年行为（早于被解释法民法典施行日）原则上不适用民法典司法解释",
          o_jiti_pre.status == RuleStatus.HIT and o_jiti_pre.detail.get("applicable") is False)

    # P0 新增测试：程序法遵循「程序从新」
    o_proc = locate_article(
        "2020-06-01",
        "2022-01-01",
        [],
        "",
        "《中华人民共和国民事诉讼法》第67条",
        norm_type="procedural",
        ref_date="2025-06-01",
    )
    check("P0修复：程序法遵循「程序从新」，2020年纠纷在2025年审理适用现行《民事诉讼法》",
          o_proc.status == RuleStatus.HIT and o_proc.detail.get("applicable") is True)

    # P0 新增测试：行政处罚「从旧兼从轻」
    o_pen = locate_article(
        "2021-03-01",
        "2021-07-15",
        [],
        "",
        "《中华人民共和国行政处罚法》第33条",
        norm_type="administrative_penalty",
        ref_date="2025-06-01",
    )
    check("P0修复：行政处罚新规（首违不罚）对修法前未决行为触发「从旧兼从轻」PARTIAL 核实",
          o_pen.status == RuleStatus.PARTIAL and o_pen.detail.get("applicable") is True)

    # P0 新增测试：持续性法律事实跨新旧法
    o_cont = locate_article(
        "2020-03-01",
        "2021-01-01",
        [],
        "",
        "《中华人民共和国民法典》第577条",
        act_end_date="2021-08-01",
    )
    check("P0修复：持续性法律事实（2020-03至2021-08）持续至民法典施行后，依法适用《民法典》",
          o_cont.status == RuleStatus.HIT and o_cont.detail.get("applicable") is True)

    # ————————————————————————————————————————
    section("集成：真实库上的新旧法版本链与事实防错层")
    tl = build_version_timelines()
    gs_old = tl.get("中华人民共和国公司法（2018年修正）", [])
    if gs_old:
        check("旧公司法失效日已标注为 2024-06-30", gs_old[0].get("effective_to") == "2024-06-30")
        o = locate_article("2024-03-01", "2018-10-26", gs_old, effective_to="2024-06-30")
        check("集成：2024-03-01 适用 2018 年修正文本",
              bool(o.detail.get("applicable")) and bool(o.detail.get("historical")))
    gs_new = tl.get("中华人民共和国公司法", [])
    check("新公司法已载入时间线", bool(gs_new))
    o = locate_article("2024-03-01", "2024-07-01", gs_new)
    check("集成：2024-03-01 不适用 2023 年修订文本（尚未施行）",
          o.detail.get("applicable") is False)
    o = locate_article("2025-01-01", "2024-07-01", gs_new)
    check("集成：2025-01-01 适用 2023 年修订文本", bool(o.detail.get("applicable")))

    # 事实层测试
    fs_no_date = analyze_facts("公司拖欠工资且未签劳动合同，我能否解除并要求赔偿？")
    check("事实抽取：未传日期且文本无日期 → act_date_source=assumed_today",
          fs_no_date.act_date_source == "assumed_today" and len(fs_no_date.missing_facts) > 0)
    check("事实抽取：劳动争议自动附带仲裁前置与1年仲裁时效提示",
          any("仲裁" in p for p in fs_no_date.procedural_notes))

    fs_with_date = analyze_facts("我在上海工作，2023年5月10日入职持续到2024年8月1日，公司一直没签劳动合同")
    check("事实抽取：从文本提取起止日期与上海地域",
          fs_with_date.act_date == "2023-05-10"
          and fs_with_date.act_end_date == "2024-08-01"
          and fs_with_date.region == "上海")

    reg_bj = check_regional_applicability("上海市劳动合同条例", "地方性法规", "上海市人大常委会", "北京")
    check("地域核验：北京案件排除《上海市劳动合同条例》", reg_bj["applicable"] is False)

    print(f"\n=== 通过 {PASS} 项，失败 {FAIL} 项 ===")
    assert FAIL == 0, "存在失败断言"
    print("✅ 规则引擎、时间/位阶P0修复与事实防错层验证全部通过")


if __name__ == "__main__":
    main()
