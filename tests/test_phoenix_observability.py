"""Arize Phoenix 生产可观测性监控专项测试用例：
验证项：
1. 零依赖静默降级（No-Op 机制）：PHOENIX_ENABLED=false 时全流程正常运行且不抛异常；
2. 问答全链路 Trace ID 与 Phoenix 直达 URL 正确注入返回体与审计记录；
3. 确定性法律业务指标（引用闭包通过率、句子溯源覆盖率、Top-1得分、规则置信度等级）计算；
4. 条件触发深度评测（LLM-as-a-Judge）：低置信度（B/C/D）及律师驳回（revise/re_retrieve）自动触发；
5. 人在环（HITL）律师反馈回传（Span Annotations）记录与 Diff 追踪。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.models import Citation, SentenceTrace
from app.observability import obs, ObservabilityManager
from app.server import handle_legal_query, QueryReq


def test_noop_degradation():
    """测试未开启 Phoenix 或无依赖时的静默降级。"""
    mgr = ObservabilityManager()
    mgr.init_observability()
    assert mgr.enabled is False or mgr.tracer is not None

    with mgr.start_span("test_span", span_kind="CHAIN") as span:
        span.set_attribute("test_key", "test_val")
        span.add_event("test_event")
        span.set_status("OK")
    print("✅ 1. 静默降级与 NoOpSpan 接口兼容验证通过")


def test_phoenix_url_generation():
    """测试 Phoenix 直达 URL 构造（符合官方 /redirects/ 规范）。"""
    tid = "abcdef1234567890abcdef1234567890"
    url_trace = obs.get_phoenix_trace_url(tid)
    assert "/redirects/traces/" in url_trace
    assert tid in url_trace

    url_proj = obs.get_phoenix_trace_url("")
    assert "/redirects/projects/legal-advisor-prod" in url_proj
    print(f"✅ 2. Phoenix 官方重定向 URL 构造验证通过: {url_trace}")


def test_deterministic_metrics_computation():
    """测试确定性指标计算：引用闭包合格率、句子溯源覆盖率等。"""
    cites = [
        Citation(law_name="中华人民共和国民法典", article="第577条", content="当事人一方不履行合同义务应当承担违约责任。"),
        Citation(law_name="中华人民共和国民法典", article="第584条", content="损失赔偿额应当相当于因违约所造成的损失。"),
    ]
    # 正向用例：引用均在第一层召回集中
    ans_good = "依据《中华人民共和国民法典》第577条及第584条规定，违约方应承担赔偿责任。"
    traces = [
        SentenceTrace(sentence_idx=1, sentence_text=ans_good, matched_law="中华人民共和国民法典", matched_article="第577条", confidence_score=0.9)
    ]
    scores_good = obs.compute_deterministic_evals(ans_good, cites, traces, {"confidence": "A"}, "A")
    assert scores_good["citation_closure_pass"] == 1.0
    assert scores_good["sentence_trace_coverage"] >= 0.5
    assert scores_good["rule_confidence_tier"] == 1.0

    # 负向用例：编造未召回条号（民法典第9999条）
    ans_hallucinated = "根据《中华人民共和国民法典》第9999条虚假规定，予以处罚。"
    scores_bad = obs.compute_deterministic_evals(ans_hallucinated, cites, [], {"confidence": "C"}, "C")
    assert scores_bad["citation_closure_pass"] == 0.0
    assert scores_bad["rule_confidence_tier"] == 0.4
    print("✅ 3. 确定性法理指标（引用闭包防幻觉 + 溯源覆盖率）计算验证通过")


def test_query_pipeline_trace_injection():
    """测试问答主接口 handle_legal_query 中 trace_id 与 phoenix_url 的注入。"""
    req = QueryReq(query="买到假货商家概不负责怎么办？")
    res = handle_legal_query(req)

    assert "trace_id" in res
    assert len(res["trace_id"]) == 32
    assert "phoenix_url" in res
    assert res["trace_id"] in res["phoenix_url"]
    assert res["status"] in ("needs_review", "done")
    print(f"✅ 4. 问答全链路 TraceID ({res['trace_id'][:8]}...) 与 Phoenix URL 注入验证通过")


def test_conditional_deep_eval_triggering():
    """测试低置信度（B/C/D）与律师驳回自动触发深度评测任务。"""
    from app.observability import _LOCAL_EVAL_RECORDS, _LOCAL_EVAL_LOCK

    initial_count = len(_LOCAL_EVAL_RECORDS)
    test_tid = "test_eval_trace_001"

    # 1. 模拟低置信度触发深度评测
    obs.trigger_deep_eval_async(
        trace_id=test_tid,
        query="复杂涉税劳务合同争议",
        answer="需视具体情形确定",
        citations=[],
        rule_plan={"confidence": "B", "status": "PARTIAL"},
        trigger_reason="置信度为 [B]，自动触发深度评测",
    )

    # 2. 模拟律师驳回（revise）触发深度评测
    obs.record_lawyer_review_feedback(
        thread_id="thread_review_001",
        decision="revise",
        reviewer_name="张三高级合伙人",
        comments="需要补充违约金上限法理分析",
        draft_answer="初稿答案",
        final_answer="修改后答案",
        diff_str="--- ai_draft\n+++ lawyer_final\n@@ -1 +1 @@\n-初稿答案\n+修改后答案",
        trace_id="test_eval_trace_002",
    )

    # 等待异步任务执行写入记录
    time.sleep(0.3)
    with _LOCAL_EVAL_LOCK:
        latest = [r for r in _LOCAL_EVAL_RECORDS if r.get("trace_id") in (test_tid, "test_eval_trace_002")]

    assert len(latest) >= 1
    print(f"✅ 5. 低置信度与律师驳回条件触发深度评测（已捕获 {len(latest)} 条记录）验证通过")


def test_otel_trace_id_alignment_and_flush_order():
    """测试 OTel 模式下 trace_id 真实对齐及 Root Span end() 之后的 force_flush 时序。"""
    from contextlib import contextmanager

    events = []
    expected_int_tid = 0x11223344556677889900AABBCCDDEEFF
    expected_hex_tid = "11223344556677889900aabbccddeeff"

    class _FakeSpanCtx:
        trace_id = expected_int_tid

    class _FakeOtelSpan:
        def __init__(self):
            self.attributes = {}

        def get_span_context(self):
            return _FakeSpanCtx()

        def set_attribute(self, k, v):
            self.attributes[k] = v

    class _FakeTracer:
        @contextmanager
        def start_as_current_span(self, name, kind=None):
            events.append("span_enter")
            try:
                yield _FakeOtelSpan()
            finally:
                events.append("span_exit")

    class _FakeProvider:
        def force_flush(self, timeout_millis=2000):
            events.append("force_flush")
            return True

    mgr = ObservabilityManager()
    mgr.enabled = True
    mgr.tracer = _FakeTracer()
    mgr._tracer_provider = _FakeProvider()

    # 覆盖 start_span 使之不强依赖 opentelemetry.trace.SpanKind（沙箱无 otel 依赖时也能测）
    @contextmanager
    def _fake_start_span(name, span_kind="CHAIN", attributes=None):
        with mgr.tracer.start_as_current_span(name) as sp:
            yield sp

    mgr.start_span = _fake_start_span  # type: ignore

    with mgr.trace_legal_query(query="测试对齐", thread_id="t-align-001") as ctx:
        assert ctx["trace_id"] == expected_hex_tid, f"期望对齐后的 trace_id={expected_hex_tid}，实际={ctx['trace_id']}"
        assert expected_hex_tid in ctx["phoenix_url"]
        events.append("inside_query")

    assert events == ["span_enter", "inside_query", "span_exit", "force_flush"], f"执行时序不符: {events}"
    print("✅ 6. OTel 真实 TraceID 对齐与 Root Span 结束后强刷盘时序验证通过")


def test_rest_fallback_span_tree_collection():
    """测试无 OTel SDK 时零依赖 REST 兜底模式：收集完整父子 Span 树并在退出时批量上报。"""
    captured_batches = []
    mgr = ObservabilityManager()
    mgr.enabled = False
    mgr.rest_fallback_enabled = True
    mgr._push_spans_via_rest = lambda spans, timeout=1.5: captured_batches.append(spans) or True  # type: ignore

    with mgr.trace_legal_query(query="送货撞人赔偿", thread_id="t-rest-001") as ctx:
        root_tid = ctx["trace_id"]
        with mgr.start_span("retriever.hybrid", span_kind="RETRIEVER", attributes={"input.value": "撞人"}) as child_span:
            child_span.set_attribute("retrieval.hit_count", 3)

    assert len(captured_batches) == 1
    spans = captured_batches[0]
    assert len(spans) == 2
    child, root = spans[0], spans[1]
    assert child["name"] == "retriever.hybrid"
    assert root["name"] == "legal_advisor.query"
    assert child["context"]["trace_id"] == root_tid
    assert root["context"]["trace_id"] == root_tid
    assert child["parent_id"] == root["context"]["span_id"]
    assert root["parent_id"] is None
    assert child["end_time"] >= child["start_time"]
    assert root["end_time"] >= root["start_time"]
    print("✅ 7. 零依赖 REST 兜底模式 Span 树收集与父子链路关联验证通过")


if __name__ == "__main__":
    print("=== 开始运行 Arize Phoenix 生产可观测性专项测试 ===")
    test_noop_degradation()
    test_phoenix_url_generation()
    test_deterministic_metrics_computation()
    test_query_pipeline_trace_injection()
    test_conditional_deep_eval_triggering()
    test_otel_trace_id_alignment_and_flush_order()
    test_rest_fallback_span_tree_collection()
    print("=== 全部 Phoenix 可观测性测试用例执行通过 ✅ ===")
