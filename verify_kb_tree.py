#!/usr/bin/env python3
"""
验证知识库创建、删除、改名与层级目录管理、文档上传归属及按目录筛选
"""
import sys
from app.kb import kb_store

def test_kb_tree_operations():
    print("==================================================================")
    print("🚀 开始验证知识库与层级目录完整管理闭环 (CRUD + 文档归属 + 路径穿透)")
    print("==================================================================")

    # 1. 验证基础树结构
    tree = kb_store.get_kb_tree()
    assert len(tree) >= 2, "应至少存在默认的内规与外规知识库"
    kb_names = [n["name"] for n in tree]
    assert "内规知识库" in kb_names, "应包含内规知识库"
    assert "外规知识库" in kb_names, "应包含外规知识库"
    print("  ✅ 默认知识库树结构获取成功: 包含", kb_names)

    # 验证外规知识库下的子目录
    external_kb = next(n for n in tree if n["name"] == "外规知识库")
    sub_names = [c["name"] for c in external_kb.get("children", [])]
    for expected in ["指导性文件", "法律法规", "其他文件", "hl_test"]:
        assert expected in sub_names, f"外规知识库应包含子目录 [{expected}]"
    print("  ✅ 外规知识库子目录结构验证通过: 包含", sub_names)

    # 2. 创建新知识库 (node_type='kb')
    new_kb = kb_store.create_kb_tree_node(
        name="合规专项知识库",
        node_type="kb",
        description="用于存放金融与数据合规专属文件"
    )
    new_kb_id = new_kb["node_id"]
    assert new_kb_id.startswith("kb_")
    print(f"  ✅ 成功创建顶层知识库: [{new_kb['name']}] (ID: {new_kb_id})")

    # 3. 在新知识库下创建一级子目录 (node_type='folder')
    sub_folder_1 = kb_store.create_kb_tree_node(
        name="数据安全规章",
        parent_id=new_kb_id,
        node_type="folder",
        description="数据安全相关规章"
    )
    sub_folder_1_id = sub_folder_1["node_id"]
    assert sub_folder_1["parent_id"] == new_kb_id
    print(f"  ✅ 成功创建一级子目录: [{sub_folder_1['name']}] (ID: {sub_folder_1_id})")

    # 4. 在一级子目录下创建多级深层子目录
    sub_folder_deep = kb_store.create_kb_tree_node(
        name="出境评估指南",
        parent_id=sub_folder_1_id,
        node_type="folder",
        description="数据出境安全评估标准"
    )
    sub_folder_deep_id = sub_folder_deep["node_id"]
    deep_path = kb_store.get_node_path(sub_folder_deep_id)
    assert deep_path == "合规专项知识库/数据安全规章/出境评估指南"
    print(f"  ✅ 成功创建多级深层目录并解析路径: {deep_path}")

    # 5. 重命名子目录与知识库
    renamed = kb_store.rename_kb_tree_node(sub_folder_deep_id, "数据出境监管评估指南")
    assert renamed is True
    updated_path = kb_store.get_node_path(sub_folder_deep_id)
    assert updated_path == "合规专项知识库/数据安全规章/数据出境监管评估指南"
    print(f"  ✅ 成功重命名子目录，路径自动同步为: {updated_path}")

    # 6. 上传法规文档并归属于该层级目录
    doc_res = kb_store.import_advanced_document(
        filename="数据出境安全评估申报指引(第六版).txt",
        raw_text="第一条 为了规范数据出境安全评估申报工作，根据《数据出境安全评估办法》，制定本指引。\n第二条 数据处理者向境外提供重要数据，应当申报数据出境安全评估。",
        kb_type=updated_path,
        split_mode="article",
        validity_option="permanent",
        uploader="合规审计员"
    )
    doc_id = doc_res["doc_id"]
    print(f"  ✅ 成功上传文档并归属到目录 [{updated_path}]，DocID: {doc_id}")

    # 7. 验证树节点 doc_count 统计与路径过滤
    tree_after_upload = kb_store.get_kb_tree()
    target_kb = next((n for n in tree_after_upload if n["node_id"] == new_kb_id), None)
    assert target_kb is not None
    assert target_kb["doc_count"] >= 1, f"知识库总计数应累加子目录文档，当前为: {target_kb['doc_count']}"
    print(f"  ✅ 知识库及其层级目录文档计数验证通过: {target_kb['doc_count']} 篇")

    # 验证按目录路径筛选文档
    matched_docs = kb_store.list_law_documents(kb_type=updated_path)
    assert len(matched_docs) >= 1
    assert any(d["doc_id"] == doc_id for d in matched_docs)
    print(f"  ✅ 按目录路径 [{updated_path}] 精准检索出对应法规文档！")

    # 8. 级联删除知识库及其下所有层级目录与文档
    deleted = kb_store.delete_kb_tree_node(new_kb_id)
    assert deleted is True
    print(f"  ✅ 成功执行级联删除知识库 [{new_kb_id}]")

    # 验证删除彻底
    tree_final = kb_store.get_kb_tree()
    assert not any(n["node_id"] == new_kb_id for n in tree_final), "被删除的知识库不应再出现在树中"
    docs_remaining = kb_store.list_law_documents(kb_type=updated_path)
    assert not any(d["doc_id"] == doc_id for d in docs_remaining), "被删除目录下的文档应级联移除"
    print("  ✅ 级联清理验证通过: 子目录与关联文档已完全清除！")

    print("\n==================================================================")
    print("🎉 知识库与层级目录全部功能测试断言 100% 通过！")
    print("==================================================================")

if __name__ == "__main__":
    test_kb_tree_operations()
