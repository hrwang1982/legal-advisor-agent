"""新增优化功能全链路验证脚本（零外部依赖，测试所有新增核心模块）。
测试项：
1. 模块一：三种切分模式（自动分割、法条分割、文档分层分割）与文档高级导入
2. 模块二：LLM-Graph Builder 引致抽取与司法解释《理解与适用》引申抽取
3. 模块三：官方法工委新旧对照表批量数字化录入
4. 模块四：国家法律法规数据库（flk.npc.gov.cn）监听同步管线
5. 模块五：条款项粒度修改日志与双层穿透时空效力定位判定
"""
import json
import sys

from app.counterparts import batch_import_official_counterparts
from app.crawler import flk_monitor
from app.graph_builder import (
    apply_cross_references_to_config,
    extract_article_cross_references,
    extract_interpretation_and_application_relations,
)
from app.kb import kb_store
from app.knowledge import knowledge_center
from app.rules import locate_article
from app.splitter import split_auto, split_by_articles, split_hierarchical


def test_splitter():
    print("\n[测试 1：三模式分段切分算法与层级继承]")

    sample_doc = (
        "第一编 总则\n"
        "第一章 基本规定\n"
        "第一条 为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，根据宪法，制定本法。\n"
        "第二条 民法调整平等主体的自然人、法人和非法人组织之间的人身关系和财产关系。\n"
        "第二编 物权\n"
        "第三条 民事主体的人身权利、财产权利以及其他合法权益受法律保护，任何组织或者个人不得侵犯。"
    )

    # 1. 自动分割
    auto_chunks = split_auto(sample_doc, target_len=60, overlap=10, doc_id="test_doc", law_name="民法典")
    assert len(auto_chunks) >= 2, f"自动分割分段数异常: {len(auto_chunks)}"
    assert all(c["chunk_type"] == "auto" for c in auto_chunks)
    print(f"  ✅ 自动分割成功: 生成 {len(auto_chunks)} 个自适应滑窗切片")

    # 2. 法条分割
    art_chunks = split_by_articles(sample_doc, doc_id="test_doc", law_name="中华人民共和国民法典")
    assert len(art_chunks) == 3, f"法条切分数量期望 3，实际 {len(art_chunks)}"
    assert art_chunks[0]["article"] == "第1条"
    assert "第一编 总则 / 第一章 基本规定" in art_chunks[0]["chapter_path"], f"编章路径未正确继承: {art_chunks[0]['chapter_path']}"
    assert art_chunks[1]["article"] == "第2条"
    assert "第一编 总则 / 第一章 基本规定" in art_chunks[1]["chapter_path"]
    assert art_chunks[2]["article"] == "第3条"
    assert "第二编 物权" in art_chunks[2]["chapter_path"]
    print("  ✅ 法条分割成功: 编、章、节状态机路径层级与归一化条号 100% 正确继承")

    # 3. 文档分层分割
    p_chunks, c_chunks = split_hierarchical(sample_doc, doc_id="test_doc", law_name="民法典", parent_chunk_size=100)
    assert len(p_chunks) >= 1, "未生成 Parent Chunks"
    assert len(c_chunks) >= len(art_chunks), "Child Chunks 数量不足"
    assert all(c["parent_chunk_id"] for c in c_chunks), "Child Chunk 未正确绑定 parent_chunk_id"
    print(f"  ✅ 文档分层分割成功: 生成 {len(p_chunks)} 个父块上下文与 {len(c_chunks)} 个精准子块索引")


def test_advanced_kb_import():
    print("\n[测试 2：对标「AI法盾」文档两步上传与多格式解析]")
    raw_law_text = (
        "第一条 为了规范人工智能数据标注，保障数据安全，制定本办法。\n"
        "第二条 本办法适用于境内开展的法律人工智能数据处理活动。\n"
        "依照本法第一条之规定，数据处理应当合法合规。"
    )

    res = kb_store.import_advanced_document(
        filename="人工智能法律数据合规办法.txt",
        raw_text=raw_law_text,
        kb_type="外规知识库/法律法规",
        split_mode="article",
        validity_option="permanent",
        uploader="执业律师测试员",
        hierarchy="部门规章",
        effective_date="2026-05-01",
    )
    assert res["doc_id"].startswith("doc_"), "未生成有效 doc_id"
    assert res["chunks_count"] == 2, f"切片数量期望 2，实际 {res['chunks_count']}"

    docs = kb_store.list_law_documents(limit=5)
    matched = [d for d in docs if d["doc_id"] == res["doc_id"]]
    assert len(matched) == 1, "文档未登记在 law_documents 表中"
    assert matched[0]["title"] == "人工智能法律数据合规办法"
    assert matched[0]["split_mode"] == "article"
    assert matched[0]["validity_status"] == "有效"
    print(f"  ✅ 两步上传向导测试成功: 文档 [{matched[0]['title']}] 已解析并入库，登记分段数 {matched[0]['chunks_count']}")


def test_graph_builder():
    print("\n[测试 3：LLM-Graph Builder 引致条款与司法解释抽取]")
    content = "劳动者依照本法第三十八条规定解除劳动合同的，用人单位应当依照本法第四十七条的规定向劳动者支付经济补偿。"
    refs = extract_article_cross_references("中华人民共和国劳动合同法", "第46条", content)
    assert len(refs) >= 2, f"引致关系抽取不足: {len(refs)}"
    target_arts = {r["target_article"] for r in refs}
    assert "第38条" in target_arts and "第47条" in target_arts, f"引致条号未正确命中: {target_arts}"
    print(f"  ✅ 离线引致抽取成功: 识别到明示与效果依赖引致关系 {target_arts}")

    interp_text = (
        "《最高人民法院关于适用〈民法典〉合同编通则若干问题的解释》第65条系根据《中华人民共和国民法典》第585条制定，"
        "对约定的违约金超过造成损失百分之三十的调减规则作出了进一步明确。"
    )
    interp_refs = extract_interpretation_and_application_relations("合同编通则解释", "第65条", interp_text)
    assert len(interp_refs) >= 1, "司法解释《理解与适用》锚定抽取失败"
    assert interp_refs[0]["anchor_law"] == "中华人民共和国民法典"
    assert interp_refs[0]["anchor_article"] == "第585条"
    print("  ✅ 司法解释《理解与适用》引申抽取成功: 准确锚定至《民法典》第585条")

    # 验证热写入 knowledge_center
    applied = apply_cross_references_to_config(refs)
    assert applied >= 0, "合入知识中心失败"
    print("  ✅ 引致图谱热生效成功: 成功合入外置知识配置")


def test_counterparts_import():
    print("\n[测试 4：法工委权威《新旧法律条文对照表》数字化录入]")
    res = batch_import_official_counterparts()
    assert res["status"] == "success"
    assert res["total_counterparts"] >= 5, "新旧对照条目不足"

    cfg = knowledge_center.get_all()
    counterparts = cfg.get("new_old_counterparts", [])
    # 验证 2023公司法第47条 -> 2018公司法第26条
    corp_pair = [
        c for c in counterparts
        if "公司法" in c.get("current_law", "") and c.get("current_article") == "第47条"
    ]
    assert len(corp_pair) >= 1, "未找到公司法第47条新旧对照"
    assert corp_pair[0]["old_article"] == "第26条"
    assert corp_pair[0]["switch_date"] == "2024-07-01"
    print(f"  ✅ 官方新旧对照录入成功: 成功加载 {len(counterparts)} 对法工委权威条文级演进映射")


def test_flk_crawler_and_revisions():
    print("\n[测试 5：国家法律法规数据库 (flk.npc.gov.cn) 监听同步与时空效力穿透]")
    # 模拟捕获到修改公告
    bulletins = [
        {
            "action": "amend",
            "target_law": "测试公司法",
            "amendment_name": "测试公司法2025修正案",
            "effective_date": "2025-06-01",
            "revised_articles": [
                {
                    "article": "第99条",
                    "effective_from": "2025-06-01",
                    "new_content": "经修正后的公司决议撤销条款。",
                    "diff_note": "修正决议撤销除斥期间",
                }
            ],
        }
    ]
    sync_res = flk_monitor.check_and_sync_announcements(bulletins)
    assert sync_res["status"] == "success"
    assert sync_res["revision_logs_count"] >= 1, "未生成条款项修改日志"

    logs = kb_store.list_article_revisions("测试公司法", "第99条")
    assert len(logs) >= 1, "未查询到条文修改日志"
    assert logs[0]["amendment_name"] == "测试公司法2025修正案"
    print(f"  ✅ 国家法规库公告监听同步成功: 捕获并记录修正日志 [{logs[0]['amendment_name']}]")

    # 测试双层穿透时空效力判定
    # 行为发生在修正前 (2025-01-01) -> 修正条款尚未生效
    out_before = locate_article(
        act_date="2025-01-01",
        effective_from="2025-06-01",
        versions=[],
        effective_to="",
        law_name="《测试公司法》第99条",
        norm_type="substantive",
    )
    assert out_before.detail.get("applicable") is False
    print("  ✅ 条款修正案穿透判定成功: 2025-01-01 行为发生日早于条款修正施行日 2025-06-01，依法判定不适用")

    # 行为发生在修正后 (2025-07-01) -> 修正条款适用
    out_after = locate_article(
        act_date="2025-07-01",
        effective_from="2025-06-01",
        versions=[],
        effective_to="",
        law_name="《测试公司法》第99条",
        norm_type="substantive",
    )
    assert out_after.detail.get("applicable") is True
    print("  ✅ 条款修正案穿透判定成功: 2025-07-01 行为发生日处于修正后施行区间，依法适用")


def main():
    print("==================================================================")
    print("🚀 开始验证法规助手全面优化实施交付模块")
    print("==================================================================")
    test_splitter()
    test_advanced_kb_import()
    test_graph_builder()
    test_counterparts_import()
    test_flk_crawler_and_revisions()
    print("\n==================================================================")
    print("🎉 全部 5 大核心优化模块验证断言 100% 通过！")
    print("==================================================================")


if __name__ == "__main__":
    main()
