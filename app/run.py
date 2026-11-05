"""命令行演示：跑通完整链路（问题拆解→改写→检索→校验→重排→位阶→两层合成→人在环）。

用法：
    python -m app.run
"""
import json

from langgraph.types import Command

from app.config import settings
from app.graph import build_graph


def _pending_interrupt(g, cfg) -> bool:
    """新版 langgraph 中 interrupt 不再抛异常，需从 state snapshot 检测。"""
    st = g.get_state(cfg)
    for task in getattr(st, "tasks", ()) or ():
        if getattr(task, "interrupts", ()):
            return True
    return False


def _print_state(st):
    fo = st.values.get("final_output")
    print("\n==================== 两层结构回答 ====================")
    print(json.dumps(fo.model_dump(), ensure_ascii=False, indent=2, default=str))
    notes = st.values.get("validation_notes") or []
    if notes:
        print("\n[有效性校验提示]")
        for n in notes:
            print(" -", n)
    rr = st.values.get("rerank_scores") or []
    if rr:
        print("\n[重排评分]")
        for n in rr:
            print(" -", n)

    plan = st.values.get("rule_plan") or {}
    if plan:
        conf = st.values.get("answer_confidence") or plan.get("confidence")
        print(f"\n[规则判定｜置信度 {conf}｜策略 {plan.get('mode')}｜人审 {plan.get('review')}]")
        for n in st.values.get("rule_notes") or []:
            print(" -", n)


def main():
    print(f"法律顾问 Agent 演示（模式：{'MOCK(免密钥)' if settings.mock else 'LLM'}）")
    g = build_graph()
    cfg = {"configurable": {"thread_id": "cli"}}
    q = input("\n请输入您的法律问题：\n> ").strip()
    if not q:
        q = "公司拖欠工资且未签劳动合同，我能否解除劳动合同并要求赔偿？"

    act = input("法律事实发生日（YYYY-MM-DD，直接回车表示按当日）：").strip()
    payload = {"query": q}
    if act:
        payload["act_date"] = act

    g.invoke(payload, cfg)

    printed = False
    while _pending_interrupt(g, cfg):
        st = g.get_state(cfg)
        _print_state(st)
        printed = True

        d = input("\n执业律师审核 [approve 通过 / revise 修改]：").strip().lower()
        comments = ""
        if d == "revise":
            comments = input("请输入修改意见：").strip()
        g.invoke(Command(resume={"decision": d, "comments": comments}), cfg)

    if not printed:  # 极少数未走到人审节点的情况，兜底展示一次
        _print_state(g.get_state(cfg))
    print("\n✅ 审核通过，流程结束。")


if __name__ == "__main__":
    main()
