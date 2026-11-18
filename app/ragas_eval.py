"""基于 Ragas (vibrantlabsai/ragas) 独立微服务容器的法律顾问 RAG 多维自动评测调度引擎。
核心特性：
1. 独立容器优先 + 零依赖公式级无缝回退：优先请求 docker/ragas 微服务 (:8003/evaluate)，
   当容器未启动或处于离线测试环境时，自动无缝回退至遵循 Ragas 数学定义的本地确定性评测引擎；
2. 三大评测数据源模式：
   - golden: 内置 10 条权威黄金基准集（G01~G10），全面回归时间效力、位阶裁决、地域管辖与防幻觉；
   - production: 抽取 MySQL/SQLite 真实在线会话流水（session_snapshots）进行事后质量体检；
   - custom: 单条自定义案情沙盒，现场跑通 RAG 检索推演并立即出具 6 维 Ragas 评分；
3. 零侵入：完全不改变现有问答、审核、审计与统计主流程。
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.models import Citation
from app.nodes import (
    decompose,
    extract_facts_node,
    rank_hierarchy,
    rerank_node,
    retrieve,
    rewrite,
    rules_node,
    synthesize,
    validate,
    verify_citation_closure,
)
from app.rules import rank_of

logger = logging.getLogger(__name__)

_HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "ragas_eval_history.json"
)
_HISTORY_LOCK = threading.Lock()

_STOPWORDS = frozenset({
    "的", "了", "和", "与", "或", "及", "在", "对", "为", "是", "有", "可以", "应当", "需要",
    "根据", "依据", "按照", "规定", "如何", "怎么", "什么", "哪些", "是否", "能否", "我", "你",
    "他", "她", "公司", "一方", "对方", "当事人", "情况", "问题", "如下", "进行", "予以",
})

GOLDEN_BENCHMARK_CASES: List[Dict[str, Any]] = [
    {
        "id": "G01_现行合同违约与引致扩展",
        "title": "现行合同违约继续履行与损害赔偿",
        "category": "合同违约 · 现行法",
        "payload": {
            "query": "对方不履行合同义务造成我方损失，我可以要求继续履行并赔偿损失吗？",
            "act_date": "2025-05-01",
        },
        "ground_truth": "依据《中华人民共和国民法典》第577条与第584条，当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任；损失赔偿额应当相当于因违约所造成的损失。",
        "must_include": [("中华人民共和国民法典", "第577条"), ("中华人民共和国民法典", "第584条")],
        "must_exclude": [("中华人民共和国合同法", "第107条")],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G02_民法典施行前历史合同纠纷",
        "title": "2020 年历史违约行为不溯及既往",
        "category": "时间效力 · 不溯及既往",
        "payload": {
            "query": "对方不履行合同义务，我可以要求继续履行并赔偿损失吗？",
            "act_date": "2020-06-01",
        },
        "ground_truth": "案涉违约行为发生在2020年6月1日（《民法典》施行前），依法不溯及既往，应当适用行为时施行的《中华人民共和国合同法》第107条追究继续履行、赔偿损失等违约责任，不得适用《民法典》第577条。",
        "must_include": [("中华人民共和国合同法", "第107条")],
        "must_exclude": [("中华人民共和国民法典", "第577条")],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G03_新法词问旧案_公司五年实缴回填2018旧法",
        "title": "2023 年公司出资期限回填 2018 旧公司法",
        "category": "新旧法衔接 · 语义回填",
        "payload": {
            "query": "有限责任公司全体股东认缴的出资额必须在五年内缴足吗？",
            "act_date": "2023-05-01",
        },
        "ground_truth": "行为发生日2023年5月1日处于2023年修订《公司法》（2024年7月1日施行）施行前，应当适用当时有效的《中华人民共和国公司法（2018年修正）》第26条，适用注册资本认缴登记制（无五年缴足强制限制），不适用新《公司法》第47条。",
        "must_include": [("中华人民共和国公司法（2018年修正）", "第26条")],
        "must_exclude": [("中华人民共和国公司法", "第47条")],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G04_口语多争点_炒鱿鱼赔偿与买假货退一赔三",
        "title": "多争点拆解：违法解雇 2N 与消保退一赔三",
        "category": "多争点拆解 · 语义改写",
        "payload": {
            "query": "我被老板无故炒鱿鱼了，能拿多少赔偿？另外在商场买到假货能退一赔三吗？",
            "act_date": "2025-05-01",
        },
        "ground_truth": "用人单位违法解除劳动合同的，依据《劳动合同法》第87条与第47条应按经济补偿标准的二倍（2N）支付赔偿金；经营者提供商品有欺诈行为的，依据《消费者权益保护法》第55条应按消费者购买商品价款的三倍增加赔偿（退一赔三）。",
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
        "title": "职务送货交通事故侵权与特别法优先",
        "category": "特别法优先 · 侵权竞合",
        "payload": {
            "query": "员工开车执行工作任务发生机动车交通事故撞伤行人，侵权赔偿责任如何承担？",
            "act_date": "2025-05-01",
        },
        "ground_truth": "依据《民法典》第1191条，用人单位的工作人员因执行工作任务造成他人损害的，由用人单位承担侵权责任；同时在机动车交通事故责任分担方面，优先适用特别法《中华人民共和国道路交通安全法》第76条关于交强险及过错比例赔偿的规定。",
        "must_include": [
            ("中华人民共和国道路交通安全法", "第76条"),
            ("中华人民共和国民法典", "第1191条"),
        ],
        "must_exclude": [],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G06_地域管辖核验_北京劳动纠纷排除上海地方性法规",
        "title": "北京劳动争议自动排除异地上海法规",
        "category": "地域效力 · 跨省隔离",
        "payload": {
            "query": "我在北京一家公司上班，用人单位违法解除劳动合同，劳动合同条例或劳动合同法如何规定赔偿金？",
            "act_date": "2025-05-01",
            "region": "北京",
        },
        "ground_truth": "本案发生在北京，应依据全国性法律《中华人民共和国劳动合同法》第87条主张二倍赔偿金，应当自动排除仅在上海市域内施行的《上海市劳动合同条例》。",
        "must_include": [("中华人民共和国劳动合同法", "第87条")],
        "must_exclude": [("上海市劳动合同条例", "第1条")],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G07_跨民法典施行日持续性违约事实_按结束日适用新法",
        "title": "跨越 2021-01-01 持续性违约行为适用新法",
        "category": "时间效力 · 持续行为",
        "payload": {
            "query": "对方从2020年5月1日开始持续不履行合同义务，一直持续到2021年3月1日，我能否要求违约损害赔偿？",
        },
        "ground_truth": "违约行为自2020年5月1日持续至2021年3月1日（跨越《民法典》施行日），依据《最高人民法院关于适用〈中华人民共和国民法典〉时间效力的若干规定》，持续至施行后的民事法律事实适用《中华人民共和国民法典》第577条。",
        "must_include": [("中华人民共和国民法典", "第577条")],
        "must_exclude": [("中华人民共和国合同法", "第107条")],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G08_诉讼时效期间与劳动争议仲裁时效区分",
        "title": "三年民事诉讼时效与一年劳动仲裁时效辨析",
        "category": "程序红线 · 时效区分",
        "payload": {
            "query": "向人民法院请求保护民事权利的三年诉讼时效期间怎么起算？劳动争议申请仲裁的时效期间是多久？",
            "act_date": "2025-05-01",
        },
        "ground_truth": "依据《民法典》第188条，向人民法院请求保护民事权利的诉讼时效期间为三年，自权利人知道或者应当知道权利受到损害以及义务人之日起计算；依据《劳动争议调解仲裁法》第27条，劳动争议申请仲裁的时效期间为一年。",
        "must_include": [
            ("中华人民共和国民法典", "第188条"),
            ("中华人民共和国劳动争议调解仲裁法", "第27条"),
        ],
        "must_exclude": [],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G09_新公司法出资加速到期与引致五年认缴",
        "title": "新《公司法》出资加速到期与图谱引致扩展",
        "category": "引致图谱 · 新公司法",
        "payload": {
            "query": "公司不能清偿到期债务，债权人能否要求未届出资期限的股东提前缴纳出资？",
            "act_date": "2025-05-01",
        },
        "ground_truth": "依据现行《中华人民共和国公司法》第54条，公司不能清偿到期债务的，公司或者已到期债权的债权人有权要求已认缴出资但未届出资期限的股东提前缴纳出资；并通过引致关联第47条明确五年内缴足认缴期限要求。",
        "must_include": [
            ("中华人民共和国公司法", "第54条"),
            ("中华人民共和国公司法", "第47条"),
        ],
        "must_exclude": [("中华人民共和国公司法（2018年修正）", "第26条")],
        "expect_high_risk_calibration": False,
    },
    {
        "id": "G10_未传日期高危场景_强制降级置信度与人工必审",
        "title": "未提供事实日期高危案情强制降级必审",
        "category": "合规防错 · 高危校准",
        "payload": {
            "query": "对方违约不履行合同义务，我能否要求继续履行并赔偿损失？",
        },
        "ground_truth": "未提供法律事实发生日期时暂按当前基准日定位《民法典》第577条，但系统必须主动降级回答置信度（不得标为最高等级 A）并强制触发执业律师人工必审及待核实日期提醒。",
        "must_include": [("中华人民共和国民法典", "第577条")],
        "must_exclude": [("中华人民共和国合同法", "第107条")],
        "expect_high_risk_calibration": True,
    },
]


def _now_beijing_str() -> str:
    bj_tz = timezone(timedelta(hours=8))
    return datetime.now(bj_tz).strftime("%Y-%m-%d %H:%M:%S")


def _tokenize_zh(text: str) -> List[str]:
    if not text:
        return []
    words = re.findall(r"[\u4e00-\u9fff]{2,}|[a-zA-Z0-9]+", text)
    tokens: List[str] = []
    for w in words:
        if w not in _STOPWORDS:
            tokens.append(w.lower())
        if len(w) > 3 and re.match(r"^[\u4e00-\u9fff]+$", w):
            for i in range(len(w) - 1):
                bg = w[i : i + 2]
                if bg not in _STOPWORDS:
                    tokens.append(bg)
    return list(dict.fromkeys(tokens))


def _normalize_law_key(law_name: str, article: str) -> Tuple[str, str]:
    lname = re.sub(r"[《》\s]+", "", law_name or "").replace("中华人民共和国", "")
    art = re.sub(r"\s+", "", article or "")
    return lname, art


def _jaccard_overlap(text_a: str, text_b: str) -> float:
    toks_a = _tokenize_zh(text_a)
    toks_b = set(_tokenize_zh(text_b))
    if not toks_a:
        return 0.0
    hits = sum(1 for t in toks_a if t in toks_b)
    return hits / len(toks_a)


def _check_hierarchy_monotonic(citations: List[Dict[str, Any]]) -> bool:
    """校验检索引用列表是否满足法理位阶单调性（上位法不低于下位法）。"""
    if len(citations) <= 1:
        return True
    levels = [
        rank_of(c.get("hierarchy", "法律"), c.get("host_hierarchy", ""))
        for c in citations
    ]
    # 允许同位阶重排微调，仅检测明确逆位阶（下位法排到上位法前且层级差 >= 1.0）
    for i in range(len(levels) - 1):
        if levels[i] - levels[i + 1] > 1.0:
            return False
    return True


def compute_ragas_metrics_for_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """基于 Ragas 数学公式的本地确定性六维评测计算器（容器不可达时自动接管兜底）。"""
    question = str(sample.get("question") or "")
    answer = str(sample.get("answer") or "")
    ground_truth = str(sample.get("ground_truth") or "")
    contexts = [str(c) for c in (sample.get("contexts") or [])]
    citations = sample.get("citations") or []
    sentence_traces = sample.get("sentence_traces") or []
    must_include = sample.get("must_include") or []
    must_exclude = sample.get("must_exclude") or []
    rule_plan = sample.get("rule_plan") or {}
    act_date_source = str(sample.get("act_date_source") or "")
    expect_cal = bool(sample.get("expect_high_risk_calibration"))

    notes: List[str] = []
    combined_ctx = "\n".join(contexts)
    cited_set = {_normalize_law_key(c.get("law_name", ""), c.get("article", "")) for c in citations}

    # 1. Faithfulness (忠实度)
    if not answer.strip():
        faithfulness = 0.0
        notes.append("未生成有效回答文本")
    else:
        main_body = answer.split("【五、诉讼/仲裁时效与程序路径提示】")[0]
        main_body = main_body.split("【六、建议法务/律师核实引证的补充条文】")[0]

        closure_warns = sample.get("closure_warnings")
        if closure_warns is None:
            mentioned = re.findall(r"《([^》]{2,40})》\s*(第[一二三四五六七八九十百千0-9]+条)", main_body)
            hallucinated = [
                f"《{ln}》{ar}"
                for ln, ar in mentioned
                if _normalize_law_key(ln, ar) not in cited_set and ar not in combined_ctx
            ]
        else:
            hallucinated = list(closure_warns)

        if hallucinated:
            notes.append(f"存在未检索支撑的法条引用: {', '.join(hallucinated[:2])}")

        sents = [
            st.strip()
            for st in re.split(r"[。；;\n！？!?]", main_body)
            if len(st.strip()) >= 8 and not st.strip().startswith("【一、") and not st.strip().startswith("【二、") and not st.strip().startswith("【三、") and not st.strip().startswith("【四、")
        ]
        trace_map = {
            (t.get("sentence_text") or "").strip(): float(t.get("confidence_score") or 0.0)
            for t in sentence_traces
            if isinstance(t, dict)
        }
        rule_markers = ("系统提示", "规则判定", "时间效力基准", "争议焦点", "待核实", "建议", "补证", "原因：", "人工复核", "行为发生", "仲裁", "诉讼时效", "适用规则")
        if not sents:
            faithfulness = 0.9 if cited_set else 0.5
        else:
            sup = 0.0
            for st in sents:
                if st in trace_map and trace_map[st] >= 0.15:
                    sup += min(1.0, 0.8 + 0.2 * trace_map[st])
                elif any(k in st for k in rule_markers):
                    sup += 0.85
                else:
                    ov = _jaccard_overlap(st, combined_ctx)
                    if ov >= 0.18:
                        sup += 1.0
                    elif ov >= 0.08:
                        sup += 0.75
                    elif not hallucinated and cited_set:
                        sup += 0.5
            faithfulness = max(0.0, min(1.0, sup / len(sents)) - 0.3 * len(hallucinated))
        faithfulness = round(faithfulness, 4)

    # 2. Answer Relevancy (回答相关性)
    if not answer.strip() or not question.strip():
        answer_relevancy = 0.0
    else:
        q_ov = _jaccard_overlap(question, answer)
        gt_ov = _jaccard_overlap(ground_truth, answer) if ground_truth else q_ov
        struct_hits = sum(1 for m in ("争议焦点", "适用法规", "涵摄", "结论", "时效") if m in answer)
        len_factor = min(1.0, len(answer.strip()) / 120.0)
        raw_ar = (0.45 * q_ov + 0.35 * gt_ov + min(0.25, struct_hits * 0.05)) * (0.6 + 0.4 * len_factor)
        if q_ov >= 0.4:
            raw_ar += 0.15
        answer_relevancy = round(min(1.0, max(0.0, raw_ar)), 4)

    # 3. Context Precision@K (上下文精确率 AP@K)
    items = citations if citations else [{"content": c} for c in contexts]
    must_set = {_normalize_law_key(m[0], m[1]) for m in must_include if len(m) >= 2}
    ref_text = f"{question} {ground_truth}".strip()
    rel_flags: List[int] = []
    for idx, item in enumerate(items):
        ln, ar = item.get("law_name", ""), item.get("article", "")
        content = item.get("content", "") or (contexts[idx] if idx < len(contexts) else "")
        nk = _normalize_law_key(ln, ar)
        if must_set and nk in must_set:
            rel_flags.append(1)
        elif not must_set and (float(item.get("rerank_score") or item.get("score") or 0.0) >= 0.08 or _jaccard_overlap(ref_text, f"{ln} {ar} {content}") >= 0.15):
            rel_flags.append(1)
        elif _jaccard_overlap(ref_text, f"{ln} {ar} {content}") >= 0.25:
            rel_flags.append(1)
        else:
            rel_flags.append(0)
    if sum(rel_flags) == 0:
        context_precision = 0.0
    else:
        ap, cum = 0.0, 0
        for k, flg in enumerate(rel_flags, start=1):
            if flg == 1:
                cum += 1
                ap += cum / k
        context_precision = round(ap / sum(rel_flags), 4)

    # 4. Context Recall (上下文召回率)
    if not contexts and not citations:
        context_recall = 0.0
    else:
        if must_include:
            inc_pairs = [_normalize_law_key(m[0], m[1]) for m in must_include if len(m) >= 2]
            hits = sum(
                1 for ln, ar in inc_pairs
                if (ln, ar) in cited_set or (ln and ln in re.sub(r"[《》\s]+", "", combined_ctx) and ar in combined_ctx)
            )
            law_rec = hits / max(len(inc_pairs), 1)
        else:
            law_rec = 1.0 if citations else 0.5
        if ground_truth:
            gt_ctx_ov = _jaccard_overlap(ground_truth, combined_ctx)
            context_recall = round(min(1.0, 0.7 * law_rec + 0.3 * min(1.0, gt_ctx_ov * 1.4)), 4)
        else:
            context_recall = round(min(1.0, law_rec), 4)

    # 5. Citation Accuracy (法条引用准确率)
    if must_include or must_exclude:
        inc_p = [_normalize_law_key(m[0], m[1]) for m in must_include if len(m) >= 2]
        exc_p = [_normalize_law_key(m[0], m[1]) for m in must_exclude if len(m) >= 2]
        inc_hits = sum(1 for p in inc_p if p in cited_set)
        exc_hits = sum(1 for p in exc_p if p in cited_set)
        if exc_hits > 0:
            notes.append(f"误用失效或异地禁用法条 ({exc_hits} 条)")
        if inc_hits < len(inc_p):
            notes.append(f"核心必备法条漏召回 ({len(inc_p) - inc_hits} 条)")
        cit_acc = round(max(0.0, (inc_hits / len(inc_p) if inc_p else 1.0) - 0.5 * exc_hits), 4)
    else:
        app_cnt = sum(1 for c in citations if c.get("applicable_at", True))
        cit_acc = round((app_cnt / len(citations)) if citations else 0.6, 4)

    # 6. Temporal & Hierarchy Compliance (时空效力与位阶合规度)
    comp = 1.0
    for c in citations:
        if c.get("applicable_at") is False:
            comp -= 0.4
            notes.append(f"不适用当前时空条件条文混入最终引用: 《{c.get('law_name','')}》{c.get('article','')}")
            break
    if sample.get("temporal_compliant") is False:
        comp -= 0.35
        notes.append("时空效力规则校验未通过")
    if sample.get("hierarchy_monotonic") is False:
        comp -= 0.25
        notes.append("法律位阶单调排序未通过")
    if expect_cal or act_date_source == "assumed_today":
        conf = str(rule_plan.get("confidence", "B")).upper()
        rev = str(rule_plan.get("review", "must")).lower()
        if conf == "A" or rev not in ("must", "expert", "immediate"):
            comp -= 0.4
            notes.append("未提供行为日期高危案情未降级置信度或未触发必审")
    comp = round(min(1.0, max(0.0, comp)), 4)

    overall = round(
        0.22 * faithfulness
        + 0.18 * answer_relevancy
        + 0.15 * context_precision
        + 0.15 * context_recall
        + 0.15 * cit_acc
        + 0.15 * comp,
        4,
    )
    grade = "S" if overall >= 0.90 else ("A" if overall >= 0.78 else ("B" if overall >= 0.62 else "C"))
    return {
        "case_id": sample.get("case_id") or "sample",
        "question": question,
        "metrics": {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "citation_accuracy": cit_acc,
            "temporal_hierarchy_compliance": comp,
        },
        "overall_score": overall,
        "grade": grade,
        "passed": overall >= 0.75 and cit_acc >= 0.75 and comp >= 0.75,
        "critique": "；".join(notes) if notes else "各项指标优良，符合 Ragas 忠实度与法理时空位阶约束",
    }


def _evaluate_locally(samples: List[Dict[str, Any]], start_t: float) -> Dict[str, Any]:
    results = [compute_ragas_metrics_for_sample(s) for s in samples]
    n = max(len(results), 1)
    metric_keys = (
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "citation_accuracy",
        "temporal_hierarchy_compliance",
    )
    summary_metrics = {
        k: round(sum(r["metrics"][k] for r in results) / n, 4) for k in metric_keys
    }
    avg_overall = round(sum(r["overall_score"] for r in results) / n, 4)
    pass_count = sum(1 for r in results if r["passed"])
    summary_grade = "S" if avg_overall >= 0.90 else ("A" if avg_overall >= 0.78 else ("B" if avg_overall >= 0.62 else "C"))
    return {
        "status": "ok",
        "engine": "local_ragas_formula_fallback",
        "elapse_ms": round((time.time() - start_t) * 1000, 2),
        "sample_count": len(results),
        "pass_count": pass_count,
        "pass_rate": round(pass_count / n, 4),
        "summary": {
            **summary_metrics,
            "overall_score": avg_overall,
            "grade": summary_grade,
        },
        "results": results,
    }


class RagasEvaluationManager:
    """Ragas 自动评测编排管理器。"""

    def _service_base(self) -> str:
        url = (getattr(settings, "ragas_service_url", "http://localhost:8003/evaluate") or "").strip()
        if "/evaluate" in url:
            return url.split("/evaluate")[0].rstrip("/")
        return url.rstrip("/")

    def _run_rag_pipeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """无侵入执行核心 RAG 推理链（不写入生产 session_snapshots 污染运营流水）。"""
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

    def _load_history(self) -> List[Dict[str, Any]]:
        with _HISTORY_LOCK:
            if not os.path.exists(_HISTORY_PATH):
                return []
            try:
                with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else []
            except Exception:
                return []

    def _save_history_entry(self, entry: Dict[str, Any]) -> None:
        with _HISTORY_LOCK:
            os.makedirs(os.path.dirname(_HISTORY_PATH), exist_ok=True)
            hist: List[Dict[str, Any]] = []
            if os.path.exists(_HISTORY_PATH):
                try:
                    with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, list):
                            hist = loaded
                except Exception:
                    hist = []
            hist.insert(0, entry)
            hist = hist[:20]
            try:
                with open(_HISTORY_PATH, "w", encoding="utf-8") as f:
                    json.dump(hist, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"[RagasEval] 保存评测历史失败: {e}")

    def get_status(self) -> Dict[str, Any]:
        """探测独立 Ragas Docker 容器运行状态并返回黄金基准集与近期记录。"""
        base = self._service_base()
        health_url = f"{base}/health"
        docker_connected = False
        sdk_ready = False
        sdk_version = "unavailable"
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data.get("status") == "ok":
                    docker_connected = True
                    sdk_ready = bool(data.get("ragas_sdk_ready"))
                    sdk_version = str(data.get("ragas_version") or "unknown")
        except Exception:
            docker_connected = False

        history = self._load_history()
        golden_meta = [
            {
                "id": c["id"],
                "title": c["title"],
                "category": c["category"],
                "query": c["payload"]["query"],
                "act_date": c["payload"].get("act_date", ""),
                "region": c["payload"].get("region", ""),
            }
            for c in GOLDEN_BENCHMARK_CASES
        ]
        return {
            "docker_connected": docker_connected,
            "service_url": getattr(settings, "ragas_service_url", "http://localhost:8003/evaluate"),
            "ragas_sdk_ready": sdk_ready,
            "ragas_version": sdk_version,
            "fallback_available": True,
            "golden_cases": golden_meta,
            "history_count": len(history),
            "latest_run": history[0] if history else None,
        }

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._load_history()[:limit]

    def _pack_sample_payload(
        self,
        case_id: str,
        question: str,
        ans_text: str,
        raw_citations: List[Any],
        raw_traces: List[Any],
        rule_plan: Dict[str, Any],
        act_date_source: str,
        ground_truth: str = "",
        must_include: Optional[List[Any]] = None,
        must_exclude: Optional[List[Any]] = None,
        expect_high_risk_calibration: bool = False,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """统一组装 Ragas 评测样本结构（复用系统权威引用闭包核验与受信任元数据上下文）。"""
        cit_objs: List[Citation] = []
        cit_dicts: List[Dict[str, Any]] = []
        for c in raw_citations or []:
            if isinstance(c, Citation):
                cit_objs.append(c)
                cit_dicts.append(c.model_dump())
            elif isinstance(c, dict):
                cit_dicts.append(dict(c))
                try:
                    cit_objs.append(Citation(**c))
                except Exception:
                    pass

        trace_dicts = [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in (raw_traces or [])]
        contexts = [
            f"《{cd.get('law_name','')}》{cd.get('article','')}：{cd.get('content','')}"
            + (f"（关联备注：{cd['note']}）" if cd.get("note") else "")
            for cd in cit_dicts
        ]
        closure_warns = verify_citation_closure(ans_text, cit_objs)
        sample = {
            "case_id": case_id,
            "question": question,
            "answer": ans_text,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "must_include": [list(x) for x in (must_include or [])],
            "must_exclude": [list(x) for x in (must_exclude or [])],
            "citations": cit_dicts,
            "sentence_traces": trace_dicts,
            "closure_warnings": closure_warns,
            "rule_plan": rule_plan or {},
            "act_date_source": act_date_source,
            "expect_high_risk_calibration": expect_high_risk_calibration,
            "temporal_compliant": all(cd.get("applicable_at", True) for cd in cit_dicts),
            "hierarchy_monotonic": _check_hierarchy_monotonic(cit_dicts),
        }
        return sample, cit_dicts

    def _build_golden_samples(self, case_ids: Optional[List[str]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        cases = GOLDEN_BENCHMARK_CASES
        if case_ids:
            wanted = set(case_ids)
            filtered = [c for c in GOLDEN_BENCHMARK_CASES if c["id"] in wanted]
            if filtered:
                cases = filtered

        samples: List[Dict[str, Any]] = []
        extra_meta: Dict[str, Any] = {}
        for case in cases:
            cid = case["id"]
            st = self._run_rag_pipeline(case["payload"])
            fo = st.get("final_output")
            ans_text = getattr(fo, "answer", "") if fo else ""
            cits = getattr(fo, "citations", []) if fo else []
            traces = getattr(fo, "sentence_traces", []) if fo else st.get("sentence_traces", [])
            sample, cit_dicts = self._pack_sample_payload(
                case_id=cid,
                question=case["payload"]["query"],
                ans_text=ans_text,
                raw_citations=cits,
                raw_traces=traces,
                rule_plan=st.get("rule_plan") or {},
                act_date_source=(st.get("fact_summary") or {}).get("act_date_source", ""),
                ground_truth=case["ground_truth"],
                must_include=case.get("must_include"),
                must_exclude=case.get("must_exclude"),
                expect_high_risk_calibration=bool(case.get("expect_high_risk_calibration")),
            )
            samples.append(sample)
            extra_meta[cid] = {
                "title": case["title"],
                "category": case["category"],
                "citations_preview": [f"《{cd.get('law_name','')}》{cd.get('article','')}" for cd in cit_dicts[:4]],
                "answer_preview": (ans_text or "")[:140].replace("\n", " "),
            }
        return samples, extra_meta

    def _build_production_samples(self, limit: int = 5) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        from app.mysql_store import mysql_store

        rows = mysql_store.query(
            "SELECT thread_id, updated_at, status, query, state_json FROM session_snapshots ORDER BY updated_at DESC LIMIT ?",
            (max(1, min(limit, 20)),),
        )
        samples: List[Dict[str, Any]] = []
        extra_meta: Dict[str, Any] = {}
        for row in rows:
            tid = row.get("thread_id") or ""
            try:
                st_data = json.loads(row.get("state_json") or "{}")
            except Exception:
                st_data = {}
            fo = st_data.get("final_output") or {}
            ans_text = fo.get("answer", "") if isinstance(fo, dict) else ""
            raw_cits = fo.get("citations", []) if isinstance(fo, dict) else []
            raw_traces = fo.get("sentence_traces", []) if isinstance(fo, dict) else []
            query_text = row.get("query") or st_data.get("query") or ""
            if not query_text:
                continue

            cid = f"PROD_{tid[:8]}"
            sample, cit_dicts = self._pack_sample_payload(
                case_id=cid,
                question=query_text,
                ans_text=ans_text,
                raw_citations=raw_cits,
                raw_traces=raw_traces,
                rule_plan=st_data.get("rule_plan") or {},
                act_date_source=(st_data.get("fact_summary") or {}).get("act_date_source", ""),
            )
            samples.append(sample)
            extra_meta[cid] = {
                "title": query_text[:28] + ("..." if len(query_text) > 28 else ""),
                "category": "线上真实流水抽样",
                "thread_id": tid,
                "trace_id": st_data.get("trace_id") or "",
                "citations_preview": [f"《{c.get('law_name','')}》{c.get('article','')}" for c in cit_dicts[:4]],
                "answer_preview": (ans_text or "")[:140].replace("\n", " "),
            }
        return samples, extra_meta

    def _build_custom_samples(self, custom_case: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        query_text = str(custom_case.get("query") or "").strip()
        if not query_text:
            raise ValueError("自定义评测问题不能为空")
        payload: Dict[str, Any] = {"query": query_text}
        if custom_case.get("act_date"):
            payload["act_date"] = str(custom_case["act_date"]).strip()
        if custom_case.get("region"):
            payload["region"] = str(custom_case["region"]).strip()

        st = self._run_rag_pipeline(payload)
        fo = st.get("final_output")
        ans_text = getattr(fo, "answer", "") if fo else ""
        cits = getattr(fo, "citations", []) if fo else []
        traces = getattr(fo, "sentence_traces", []) if fo else st.get("sentence_traces", [])

        cid = "CUSTOM_SANDBOX"
        sample, cit_dicts = self._pack_sample_payload(
            case_id=cid,
            question=query_text,
            ans_text=ans_text,
            raw_citations=cits,
            raw_traces=traces,
            rule_plan=st.get("rule_plan") or {},
            act_date_source=(st.get("fact_summary") or {}).get("act_date_source", ""),
            ground_truth=str(custom_case.get("ground_truth") or ""),
        )
        extra_meta = {
            cid: {
                "title": "自定义案情沙盒",
                "category": f"现场实时推理 ({payload.get('act_date') or '默认基准日'})",
                "citations_preview": [f"《{cd.get('law_name','')}》{cd.get('article','')}" for cd in cit_dicts[:4]],
                "answer_preview": (ans_text or "")[:140].replace("\n", " "),
            }
        }
        return [sample], extra_meta

    def run_evaluation(
        self,
        mode: str = "golden",
        case_ids: Optional[List[str]] = None,
        limit: int = 5,
        custom_case: Optional[Dict[str, Any]] = None,
        use_llm: bool = False,
    ) -> Dict[str, Any]:
        """执行 Ragas 多维评测：优先调度独立 Docker 微服务，不可达时自动降级本地公式引擎。"""
        start_t = time.time()
        clean_mode = (mode or "golden").strip().lower()
        if clean_mode == "production":
            samples, extra_meta = self._build_production_samples(limit=limit)
        elif clean_mode == "custom":
            samples, extra_meta = self._build_custom_samples(custom_case or {})
        else:
            clean_mode = "golden"
            samples, extra_meta = self._build_golden_samples(case_ids=case_ids)

        if not samples:
            raise ValueError("当前评测模式下无有效问答样本可供评测（如生产流水为空可先执行智能问答或切换黄金基准集）")

        llm_cfg = {
            "use_llm_judge": bool(use_llm and not getattr(settings, "mock", True) and getattr(settings, "llm_api_key", "")),
            "api_base": getattr(settings, "llm_api_base", ""),
            "api_key": getattr(settings, "llm_api_key", ""),
            "model": getattr(settings, "llm_model", "qwen-plus"),
            "embed_api_base": getattr(settings, "embed_api_base", "") or getattr(settings, "llm_api_base", ""),
            "embed_api_key": getattr(settings, "embed_api_key", "") or getattr(settings, "llm_api_key", ""),
            "embed_model": getattr(settings, "embed_model", "BAAI/bge-m3"),
        }

        eval_resp: Optional[Dict[str, Any]] = None
        container_used = False
        eval_url = getattr(settings, "ragas_service_url", "http://localhost:8003/evaluate")
        try:
            body = json.dumps({"samples": samples, "llm_config": llm_cfg}, ensure_ascii=False, default=str).encode("utf-8")
            req = urllib.request.Request(
                eval_url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            timeout_s = float(getattr(settings, "ragas_timeout", 60.0) or 60.0)
            with urllib.request.urlopen(req, timeout=min(timeout_s, 10.0) if not llm_cfg["use_llm_judge"] else timeout_s) as resp:
                eval_resp = json.loads(resp.read().decode("utf-8"))
                container_used = True
        except Exception as e:
            logger.info(f"[RagasEval] 独立 Ragas 容器暂不可达 ({e})，自动使用内置 Ragas 公式计算引擎")
            eval_resp = _evaluate_locally(samples, start_t)
            container_used = False

        # 合并元数据与耗时，组装标准报告结构
        results = eval_resp.get("results") or []
        for r in results:
            cid = r.get("case_id", "")
            meta = extra_meta.get(cid, {})
            r["title"] = meta.get("title", cid)
            r["category"] = meta.get("category", clean_mode)
            r["citations_preview"] = meta.get("citations_preview", [])
            r["answer_preview"] = meta.get("answer_preview", "")
            if meta.get("trace_id"):
                r["trace_id"] = meta["trace_id"]

        run_record = {
            "run_id": f"ragas_{uuid.uuid4().hex[:10]}",
            "timestamp": _now_beijing_str(),
            "mode": clean_mode,
            "engine": eval_resp.get("engine", "ragas_docker_formula"),
            "container_used": container_used,
            "llm_judge_used": bool(llm_cfg["use_llm_judge"] and container_used),
            "elapse_ms": round((time.time() - start_t) * 1000, 2),
            "sample_count": eval_resp.get("sample_count", len(results)),
            "pass_count": eval_resp.get("pass_count", 0),
            "pass_rate": eval_resp.get("pass_rate", 0.0),
            "summary": eval_resp.get("summary", {}),
            "results": results,
        }
        self._save_history_entry(run_record)
        return run_record


ragas_eval_mgr = RagasEvaluationManager()
