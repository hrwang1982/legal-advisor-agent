"""独立 Ragas 自动化评测微服务 (Legal Advisor Ragas Evaluation Service)
服务端口: 8003
核心职责:
1. 隔离部署 vibrantlabsai/ragas 及其依赖（datasets、langchain-openai），避免污染主业务容器；
2. 提供 /health 健康探测及支持的评测指标元信息；
3. 提供 /evaluate 评测接口，支持 Ragas 原生 LLM/Embedding 评测与基于 Ragas 数学公式的确定性六维评测融合：
   - faithfulness（忠实度 / 防幻觉）
   - answer_relevancy（回答相关性）
   - context_precision（上下文精确率 AP@K）
   - context_recall（上下文召回率）
   - citation_accuracy（法条引用准确率）
   - temporal_hierarchy_compliance（法律时空效力与位阶合规度）
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
    _HAS_FASTAPI = True
except ImportError:
    FastAPI = None  # type: ignore
    CORSMiddleware = None  # type: ignore
    BaseModel = object  # type: ignore
    def Field(*args, **kwargs):  # type: ignore
        return None
    _HAS_FASTAPI = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("legal_ragas_service")

if _HAS_FASTAPI:
    app = FastAPI(title="Legal Advisor Ragas Evaluation Microservice", version="1.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app = None

SUPPORTED_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "citation_accuracy",
    "temporal_hierarchy_compliance",
]

_STOPWORDS = frozenset({
    "的", "了", "和", "与", "或", "及", "在", "对", "为", "是", "有", "可以", "应当", "需要",
    "根据", "依据", "按照", "规定", "如何", "怎么", "什么", "哪些", "是否", "能否", "我", "你",
    "他", "她", "公司", "一方", "对方", "当事人", "情况", "问题", "如下", "进行", "予以",
})


def _probe_ragas_sdk() -> Tuple[bool, str]:
    """探测 vibrantlabsai/ragas SDK 是否已成功安装于当前容器环境。"""
    try:
        import ragas  # type: ignore

        ver = getattr(ragas, "__version__", "unknown")
        return True, str(ver)
    except Exception:
        return False, "unavailable"


def _tokenize_zh(text: str) -> List[str]:
    """轻量中英文混合分词（兼顾法律实词与二元字符切分）。"""
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
    """归一化法规名称与条号，便于精准比对。"""
    lname = re.sub(r"[《》\s]+", "", law_name or "").replace("中华人民共和国", "")
    art = re.sub(r"\s+", "", article or "")
    return lname, art


def _extract_article_mentions(text: str) -> List[str]:
    """从文本中提取形如 《XX法》第X条 或 第X条 的条号。"""
    if not text:
        return []
    matches = re.findall(r"《([^》]{2,40})》\s*(第[一二三四五六七八九十百千0-9]+条)", text)
    return [f"{_normalize_law_key(lw, ar)[0]}:{_normalize_law_key(lw, ar)[1]}" for lw, ar in matches]


def _jaccard_overlap(text_a: str, text_b: str) -> float:
    """计算两段文本间的词元覆盖率（针对 text_a 中关键词被 text_b 覆盖的比例）。"""
    toks_a = _tokenize_zh(text_a)
    toks_b = set(_tokenize_zh(text_b))
    if not toks_a:
        return 0.0
    hits = sum(1 for t in toks_a if t in toks_b)
    return hits / len(toks_a)


def _compute_faithfulness(
    answer: str,
    contexts: List[str],
    citations: List[Dict[str, Any]],
    sentence_traces: List[Dict[str, Any]],
    closure_warnings: Optional[List[str]] = None,
) -> Tuple[float, List[str]]:
    """计算 Ragas Faithfulness（忠实度 = 受检索上下文支撑的回答陈述句比例，叠加法条闭包约束）。"""
    if not answer or not answer.strip():
        return 0.0, ["未生成有效回答文本"]

    warnings: List[str] = []
    combined_ctx = "\n".join(contexts)
    cited_keys = {
        f"{_normalize_law_key(c.get('law_name', ''), c.get('article', ''))[0]}:"
        f"{_normalize_law_key(c.get('law_name', ''), c.get('article', ''))[1]}"
        for c in citations
    }

    # 与权威闭包的边界一致：排除【五、诉讼/仲裁时效与程序路径提示】与【六、建议法务/律师核实引证的补充条文】
    main_body = answer.split("【五、诉讼/仲裁时效与程序路径提示】")[0]
    main_body = main_body.split("【六、建议法务/律师核实引证的补充条文】")[0]
    if closure_warnings is None:
        mentioned_keys = _extract_article_mentions(main_body)
        hallucinated = [k for k in mentioned_keys if k not in cited_keys and k not in combined_ctx]
    else:
        hallucinated = list(closure_warnings)
    if hallucinated:
        warnings.append(f"存在未检索支撑的法条引用: {', '.join(hallucinated[:3])}")

    # 分句评估支撑比例
    sentences = [
        s.strip()
        for s in re.split(r"[。；;\n！？!?]", main_body)
        if len(s.strip()) >= 8
        and not s.strip().startswith("【一、")
        and not s.strip().startswith("【二、")
        and not s.strip().startswith("【三、")
        and not s.strip().startswith("【四、")
    ]
    if not sentences:
        return (0.9 if cited_keys else 0.5), warnings

    trace_scores = {
        (t.get("sentence_text") or "").strip(): float(t.get("confidence_score") or 0.0)
        for t in sentence_traces
        if isinstance(t, dict)
    }
    rule_markers = (
        "系统提示", "规则判定", "时间效力基准", "争议焦点", "待核实", "建议",
        "补证", "原因：", "人工复核", "行为发生", "仲裁", "诉讼时效", "适用规则",
    )

    supported_weight = 0.0
    for sent in sentences:
        if sent in trace_scores and trace_scores[sent] >= 0.15:
            supported_weight += min(1.0, 0.8 + 0.2 * trace_scores[sent])
        elif any(k in sent for k in rule_markers):
            supported_weight += 0.85
        else:
            overlap = _jaccard_overlap(sent, combined_ctx)
            if overlap >= 0.18:
                supported_weight += 1.0
            elif overlap >= 0.08:
                supported_weight += 0.75
            elif not hallucinated and cited_keys:
                supported_weight += 0.5

    raw_score = min(1.0, supported_weight / len(sentences))
    if hallucinated:
        raw_score = max(0.0, raw_score - 0.3 * len(hallucinated))
    return round(raw_score, 4), warnings


def _compute_answer_relevancy(question: str, answer: str, ground_truth: str) -> float:
    """计算 Ragas Answer Relevancy（回答对提问核心诉求及法律结构要件的相关度）。"""
    if not answer or not question:
        return 0.0

    q_overlap = _jaccard_overlap(question, answer)
    gt_overlap = _jaccard_overlap(ground_truth, answer) if ground_truth else q_overlap

    # 检查结构化法律意见核心要件是否完备
    structure_markers = ["争议焦点", "适用法规", "涵摄", "结论", "时效"]
    struct_hits = sum(1 for m in structure_markers if m in answer)
    struct_bonus = min(0.25, struct_hits * 0.05)

    # 防止过短或纯报错文本虚高
    length_factor = min(1.0, len(answer.strip()) / 120.0)
    score = (0.45 * q_overlap + 0.35 * gt_overlap + struct_bonus) * (0.6 + 0.4 * length_factor)
    return round(min(1.0, max(0.0, score + (0.15 if q_overlap >= 0.4 else 0.0))), 4)


def _compute_context_precision(
    question: str,
    contexts: List[str],
    citations: List[Dict[str, Any]],
    must_include: List[List[str]],
    ground_truth: str,
) -> float:
    """计算 Ragas Context Precision@K（AP@K：有效检索片段排位越靠前得分越高）。"""
    items = citations if citations else [{"content": c} for c in contexts]
    if not items:
        return 0.0

    must_set = {_normalize_law_key(m[0], m[1]) for m in must_include if len(m) >= 2}
    ref_text = f"{question} {ground_truth}".strip()

    relevant_flags: List[int] = []
    for idx, item in enumerate(items):
        lname = item.get("law_name", "")
        art = item.get("article", "")
        content = item.get("content", "") or (contexts[idx] if idx < len(contexts) else "")
        norm_key = _normalize_law_key(lname, art)

        is_rel = False
        if must_set and norm_key in must_set:
            is_rel = True
        elif not must_set:
            score_val = float(item.get("rerank_score") or item.get("score") or 0.0)
            if score_val >= 0.08 or _jaccard_overlap(ref_text, f"{lname} {art} {content}") >= 0.15:
                is_rel = True
        else:
            if _jaccard_overlap(ref_text, f"{lname} {art} {content}") >= 0.25:
                is_rel = True
        relevant_flags.append(1 if is_rel else 0)

    total_rel = sum(relevant_flags)
    if total_rel == 0:
        return 0.0

    ap_sum = 0.0
    cum_rel = 0
    for k, flag in enumerate(relevant_flags, start=1):
        if flag == 1:
            cum_rel += 1
            ap_sum += cum_rel / k
    return round(ap_sum / total_rel, 4)


def _compute_context_recall(
    contexts: List[str],
    citations: List[Dict[str, Any]],
    must_include: List[List[str]],
    ground_truth: str,
) -> float:
    """计算 Ragas Context Recall（召回上下文对标准依据与 Ground Truth 要件的覆盖率）。"""
    if not contexts and not citations:
        return 0.0

    cited_set = {_normalize_law_key(c.get("law_name", ""), c.get("article", "")) for c in citations}
    combined_ctx = "\n".join(contexts)

    if must_include:
        must_pairs = [_normalize_law_key(m[0], m[1]) for m in must_include if len(m) >= 2]
        hit_count = 0
        for ln, ar in must_pairs:
            if (ln, ar) in cited_set or (ln and ln in re.sub(r"[《》\s]+", "", combined_ctx) and ar in combined_ctx):
                hit_count += 1
        law_recall = hit_count / max(len(must_pairs), 1)
    else:
        law_recall = 1.0 if citations else 0.5

    if ground_truth:
        gt_overlap = _jaccard_overlap(ground_truth, combined_ctx)
        score = 0.7 * law_recall + 0.3 * min(1.0, gt_overlap * 1.4)
    else:
        score = law_recall
    return round(min(1.0, score), 4)


def _compute_citation_and_compliance(
    citations: List[Dict[str, Any]],
    must_include: List[List[str]],
    must_exclude: List[List[str]],
    rule_plan: Dict[str, Any],
    act_date_source: str,
    expect_high_risk_calibration: bool,
    temporal_compliant: Optional[bool],
    hierarchy_monotonic: Optional[bool],
) -> Tuple[float, float, List[str]]:
    """计算法律垂直维度核心指标：法条引用准确率 & 时空效力/位阶合规度。"""
    notes: List[str] = []
    cited_set = {_normalize_law_key(c.get("law_name", ""), c.get("article", "")) for c in citations}

    # 1. 法条引用准确率
    if must_include or must_exclude:
        inc_pairs = [_normalize_law_key(m[0], m[1]) for m in must_include if len(m) >= 2]
        exc_pairs = [_normalize_law_key(m[0], m[1]) for m in must_exclude if len(m) >= 2]
        inc_hits = sum(1 for p in inc_pairs if p in cited_set)
        exc_violations = sum(1 for p in exc_pairs if p in cited_set)

        inc_ratio = (inc_hits / len(inc_pairs)) if inc_pairs else 1.0
        if exc_violations > 0:
            notes.append(f"误用失效/禁用条文 ({exc_violations} 条)")
        if inc_hits < len(inc_pairs):
            notes.append(f"必备核心法条漏召回 ({len(inc_pairs) - inc_hits} 条)")
        cit_acc = max(0.0, inc_ratio - 0.5 * exc_violations)
    else:
        verified_cnt = sum(1 for c in citations if c.get("applicable_at", True))
        cit_acc = (verified_cnt / len(citations)) if citations else 0.6

    # 2. 时空效力与位阶合规度
    comp_score = 1.0
    for c in citations:
        if c.get("applicable_at") is False:
            comp_score -= 0.4
            notes.append(f"不适用当前时间/地域条文混入引用: 《{c.get('law_name','')}》{c.get('article','')}")
            break
    if temporal_compliant is False:
        comp_score -= 0.35
        notes.append("时空效力规则校验不通过")
    if hierarchy_monotonic is False:
        comp_score -= 0.25
        notes.append("法律位阶单调排序不通过")

    if expect_high_risk_calibration or act_date_source == "assumed_today":
        conf = str(rule_plan.get("confidence", "B")).upper()
        rev = str(rule_plan.get("review", "must")).lower()
        if conf == "A" or rev not in ("must", "expert", "immediate"):
            comp_score -= 0.4
            notes.append("未传行为日期的高危案情未降级置信度并触发必审")

    return round(min(1.0, max(0.0, cit_acc)), 4), round(min(1.0, max(0.0, comp_score)), 4), notes


def _run_ragas_llm_judge(
    samples: List[Dict[str, Any]],
    llm_config: Dict[str, Any],
) -> Optional[List[Dict[str, float]]]:
    """若开启 LLM Judge 且容器内 Ragas SDK 与 API Key 就绪，调用 Ragas 原生 evaluate() 执行深度评估。"""
    if not llm_config.get("use_llm_judge"):
        return None
    api_key = str(llm_config.get("api_key") or "").strip()
    if not api_key:
        return None

    ok, _ = _probe_ragas_sdk()
    if not ok:
        return None

    try:
        from datasets import Dataset  # type: ignore
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # type: ignore
        from ragas import evaluate as ragas_evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        api_base = str(llm_config.get("api_base") or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
        model = str(llm_config.get("model") or "qwen-plus").strip()
        embed_model = str(llm_config.get("embed_model") or "text-embedding-v3").strip()
        embed_base = str(llm_config.get("embed_api_base") or api_base).strip()
        embed_key = str(llm_config.get("embed_api_key") or api_key).strip()

        llm = ChatOpenAI(model=model, api_key=api_key, base_url=api_base, temperature=0.0, timeout=45.0)
        embeddings = OpenAIEmbeddings(model=embed_model, api_key=embed_key, base_url=embed_base)

        ds_rows = {
            "question": [s.get("question", "") for s in samples],
            "answer": [s.get("answer", "") for s in samples],
            "contexts": [s.get("contexts") or ["无召回上下文"] for s in samples],
            "ground_truth": [s.get("ground_truth") or s.get("question", "") for s in samples],
        }
        dataset = Dataset.from_dict(ds_rows)
        result = ragas_evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False,
        )
        df = result.to_pandas()
        out_scores: List[Dict[str, float]] = []
        for _, row in df.iterrows():
            item: Dict[str, float] = {}
            for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
                val = row.get(m)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    item[m] = round(float(val), 4)
            out_scores.append(item)
        return out_scores
    except Exception as e:
        logger.warning(f"Ragas 原生 LLM Judge 执行异常，已无缝保留公式级混合得分: {e}")
        return None


def evaluate_samples_payload(
    samples: List[Dict[str, Any]],
    llm_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """统一评测入口（既供 FastAPI 路由调用，也便于单测直接导入校验）。"""
    start_t = time.time()
    llm_cfg = llm_config or {}
    llm_scores_list = _run_ragas_llm_judge(samples, llm_cfg)
    engine_mode = "ragas_sdk_llm_hybrid" if llm_scores_list is not None else "ragas_docker_formula"

    results: List[Dict[str, Any]] = []
    for idx, s in enumerate(samples):
        question = str(s.get("question") or "")
        answer = str(s.get("answer") or "")
        ground_truth = str(s.get("ground_truth") or "")
        contexts = [str(c) for c in (s.get("contexts") or [])]
        citations = s.get("citations") or []
        sentence_traces = s.get("sentence_traces") or []
        must_include = s.get("must_include") or []
        must_exclude = s.get("must_exclude") or []
        rule_plan = s.get("rule_plan") or {}

        f_score, f_warns = _compute_faithfulness(
            answer,
            contexts,
            citations,
            sentence_traces,
            closure_warnings=s.get("closure_warnings"),
        )
        ar_score = _compute_answer_relevancy(question, answer, ground_truth)
        cp_score = _compute_context_precision(question, contexts, citations, must_include, ground_truth)
        cr_score = _compute_context_recall(contexts, citations, must_include, ground_truth)
        cit_acc, comp_score, comp_notes = _compute_citation_and_compliance(
            citations=citations,
            must_include=must_include,
            must_exclude=must_exclude,
            rule_plan=rule_plan,
            act_date_source=str(s.get("act_date_source") or ""),
            expect_high_risk_calibration=bool(s.get("expect_high_risk_calibration")),
            temporal_compliant=s.get("temporal_compliant"),
            hierarchy_monotonic=s.get("hierarchy_monotonic"),
        )

        # 若启用了 Ragas 原生 LLM Judge，加权融合 LLM 语义评分与确定性法理校验评分
        if llm_scores_list and idx < len(llm_scores_list):
            ls = llm_scores_list[idx]
            if "faithfulness" in ls:
                f_score = round(0.65 * ls["faithfulness"] + 0.35 * f_score, 4)
            if "answer_relevancy" in ls:
                ar_score = round(0.65 * ls["answer_relevancy"] + 0.35 * ar_score, 4)
            if "context_precision" in ls:
                cp_score = round(0.65 * ls["context_precision"] + 0.35 * cp_score, 4)
            if "context_recall" in ls:
                cr_score = round(0.65 * ls["context_recall"] + 0.35 * cr_score, 4)

        overall = round(
            0.22 * f_score
            + 0.18 * ar_score
            + 0.15 * cp_score
            + 0.15 * cr_score
            + 0.15 * cit_acc
            + 0.15 * comp_score,
            4,
        )
        grade = "S" if overall >= 0.90 else ("A" if overall >= 0.78 else ("B" if overall >= 0.62 else "C"))
        all_notes = f_warns + comp_notes
        critique = "；".join(all_notes) if all_notes else "各项指标优良，符合 Ragas 忠实度与法理时空位阶约束"

        results.append({
            "case_id": s.get("case_id") or f"case_{idx + 1}",
            "question": question,
            "metrics": {
                "faithfulness": f_score,
                "answer_relevancy": ar_score,
                "context_precision": cp_score,
                "context_recall": cr_score,
                "citation_accuracy": cit_acc,
                "temporal_hierarchy_compliance": comp_score,
            },
            "overall_score": overall,
            "grade": grade,
            "passed": overall >= 0.75 and cit_acc >= 0.75 and comp_score >= 0.75,
            "critique": critique,
        })

    n = max(len(results), 1)
    summary_metrics = {
        m: round(sum(r["metrics"][m] for r in results) / n, 4)
        for m in SUPPORTED_METRICS
    }
    avg_overall = round(sum(r["overall_score"] for r in results) / n, 4)
    pass_count = sum(1 for r in results if r["passed"])
    summary_grade = "S" if avg_overall >= 0.90 else ("A" if avg_overall >= 0.78 else ("B" if avg_overall >= 0.62 else "C"))

    return {
        "status": "ok",
        "engine": engine_mode,
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


class EvalSampleReq(BaseModel):
    case_id: Optional[str] = None
    question: str
    answer: str = ""
    contexts: List[str] = Field(default_factory=list)
    ground_truth: str = ""
    must_include: List[List[str]] = Field(default_factory=list)
    must_exclude: List[List[str]] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    sentence_traces: List[Dict[str, Any]] = Field(default_factory=list)
    closure_warnings: Optional[List[str]] = None
    rule_plan: Dict[str, Any] = Field(default_factory=dict)
    act_date_source: str = ""
    expect_high_risk_calibration: bool = False
    temporal_compliant: Optional[bool] = None
    hierarchy_monotonic: Optional[bool] = None


class EvaluateReq(BaseModel):
    samples: List[EvalSampleReq]
    llm_config: Optional[Dict[str, Any]] = None


if _HAS_FASTAPI:
    @app.get("/health")
    def health_check():
        """Ragas 独立微服务健康检查接口。"""
        sdk_ready, sdk_ver = _probe_ragas_sdk()
        return {
            "status": "ok",
            "service": "legal-advisor-ragas",
            "ragas_sdk_ready": sdk_ready,
            "ragas_version": sdk_ver,
            "metrics": SUPPORTED_METRICS,
        }

    @app.post("/evaluate")
    def evaluate_endpoint(req: EvaluateReq):
        """执行 Ragas 多维指标评估。"""
        raw_samples = [s.model_dump() for s in req.samples]
        return evaluate_samples_payload(raw_samples, req.llm_config)
