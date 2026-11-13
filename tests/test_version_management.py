import unittest
from app.kb import kb_store


class TestVersionManagement(unittest.TestCase):
    def test_sniff_document_version(self):
        text_2018 = (
            "中华人民共和国劳动法\n"
            "（1994年7月5日第八届全国人民代表大会常务委员会第八次会议通过 "
            "根据2009年8月27日《全国人民代表大会常务委员会关于修改部分法律的决定》第一次修正 "
            "根据2018年12月29日第十三届全国人民代表大会常务委员会第七次会议《关于修改〈中华人民共和国劳动法〉等七部法律的决定》第二次修正）\n"
            "第一条 为了保护劳动者的合法权益，制定本法。"
        )
        res = kb_store.sniff_document_version("中华人民共和国劳动法.docx", text_2018)
        self.assertIn("2018", res["version_code"])
        self.assertIn("2018", res["version_name"])
        self.assertEqual(res["effective_date"], "2018-12-29")
        self.assertTrue(res["sniffed"])

    def test_multi_version_lifecycle_and_diff(self):
        law_title = "测试多版本管理条例_单元测试专用"
        # Clean up any leftover test docs
        for d in kb_store.list_law_documents():
            if d.get("title") in (law_title, f"{law_title}_独立重命名版"):
                kb_store.delete_law_document(d["doc_id"])

        try:
            # 1. Upload v1 (2018版)
            r1 = kb_store.import_advanced_document(
                filename=f"{law_title}.txt",
                kb_type="内规知识库",
                raw_text=(
                    f"《{law_title}》\n"
                    "（2018年6月1日通过）\n"
                    "第一条 为了规范测试流程，制定本条例。\n"
                    "第二条 罚款金额最高不超过五万元。\n"
                    "第三条 旧版独有条款，后续将被删除。"
                ),
                split_mode="article",
                version_name="2018年初始版",
                effective_date="2018-06-01",
                conflict_action="append_new",
            )
            doc_id_v1 = r1["doc_id"]
            self.assertEqual(r1["is_latest"], 1)

            # 2. Check conflict before uploading v2
            conflict = kb_store.check_version_conflict(
                filename=f"{law_title}.txt",
                kb_type="内规知识库",
                raw_text="（2024年3月15日修订）\n第一条 为了规范测试流程，制定本条例。",
            )
            self.assertTrue(conflict["has_conflict"])
            self.assertEqual(len(conflict["existing_versions"]), 1)

            # 3. Upload v2 (2024修订版) via append_new
            r2 = kb_store.import_advanced_document(
                filename=f"{law_title}.txt",
                kb_type="内规知识库",
                raw_text=(
                    f"《{law_title}》\n"
                    "（2024年3月15日修订）\n"
                    "第一条 为了规范测试流程，制定本条例。\n"
                    "第二条 罚款金额最高不超过二十万元。\n"
                    "第四条 新增合规审计要求条款。"
                ),
                split_mode="article",
                version_name="2024年修订版",
                effective_date="2024-03-15",
                conflict_action="append_new",
            )
            doc_id_v2 = r2["doc_id"]
            self.assertNotEqual(doc_id_v1, doc_id_v2)

            # Verify v2 is latest and v1 is archived (is_latest=0)
            docs = [d for d in kb_store.list_law_documents() if d.get("title") == law_title]
            self.assertEqual(len(docs), 2)
            v_map = {d["doc_id"]: d for d in docs}
            self.assertEqual(int(v_map[doc_id_v2]["is_latest"]), 1)
            self.assertEqual(int(v_map[doc_id_v1]["is_latest"]), 0)

            # 4. Compare version diff between v2 and v1
            diff = kb_store.compare_document_versions(doc_id_v2, doc_id_v1)
            self.assertEqual(diff["status"], "ok")
            summary = diff["summary"]
            self.assertGreaterEqual(summary["modified_count"], 1)  # 第二条 modified
            self.assertGreaterEqual(summary["added_count"], 1)     # 第四条 added
            self.assertGreaterEqual(summary["removed_count"], 1)   # 第三条 removed

            # 5. Switch v1 back to latest
            set_res = kb_store.set_document_as_latest(doc_id_v1)
            self.assertEqual(set_res["status"], "ok")
            docs_after = {d["doc_id"]: d for d in kb_store.list_law_documents() if d.get("title") == law_title}
            self.assertEqual(int(docs_after[doc_id_v1]["is_latest"]), 1)
            self.assertEqual(int(docs_after[doc_id_v2]["is_latest"]), 0)
        finally:
            for d in kb_store.list_law_documents():
                if d.get("title") in (law_title, f"{law_title}_独立重命名版"):
                    kb_store.delete_law_document(d["doc_id"])

    def test_out_of_order_upload_preserves_newer_effective_date_as_latest(self):
        """模拟国家法规库从上往下拉取：先入库 2025年新版 (sxx=3)，后入库 2021年旧版 (sxx=2)，
        断言系统自动按施行日期/时效状态裁决 2025版保持 is_latest=1，2021版归入历史版本 is_latest=0。
        """
        law_title = "测试逆序入库监察实施条例_单元测试专用"
        for d in kb_store.list_law_documents():
            if d.get("title") == law_title:
                kb_store.delete_law_document(d["doc_id"])

        try:
            # 1. 先导入 2025年新版 (2025-06-01)
            r_2025 = kb_store.import_advanced_document(
                filename=f"{law_title}.docx",
                kb_type="外规知识库/监察法规",
                raw_text=(
                    f"《{law_title}》\n"
                    "（2021年7月20日决定 根据2025年4月28日决定修正 2025年6月1日公布）\n"
                    "第一条 为了深化国家监察体制改革，制定本条例（2025修正版）。"
                ),
                split_mode="article",
                effective_date="2025-06-01",
                conflict_action="append_new",
                official_sxx=3,
            )
            self.assertEqual(r_2025["is_latest"], 1)

            # 2. 后导入 2021年旧版 (2021-09-20, 已修改 sxx=2)
            r_2021 = kb_store.import_advanced_document(
                filename=f"{law_title}.docx",
                kb_type="外规知识库/监察法规",
                raw_text=(
                    f"《{law_title}》\n"
                    "（2021年7月20日决定 2021年9月20日公布 自2021年9月20日起施行）\n"
                    "第一条 为了深化国家监察体制改革，制定本条例（2021初始施行版）。"
                ),
                split_mode="article",
                effective_date="2021-09-20",
                conflict_action="append_new",
                official_sxx=2,
            )
            # 后入库的 2021 旧版绝不能夺取 2025 新版的 is_latest=1 地位
            self.assertEqual(r_2021["is_latest"], 0)

            docs = {d["doc_id"]: d for d in kb_store.list_law_documents() if d.get("title") == law_title}
            self.assertEqual(len(docs), 2)
            self.assertEqual(int(docs[r_2025["doc_id"]]["is_latest"]), 1)
            self.assertEqual(docs[r_2025["doc_id"]]["validity_status"], "有效")
            self.assertEqual(int(docs[r_2021["doc_id"]]["is_latest"]), 0)
            self.assertEqual(docs[r_2021["doc_id"]]["validity_status"], "已修正")
            self.assertEqual(docs[r_2021["doc_id"]]["valid_until"], "2025-05-31")
        finally:
            for d in kb_store.list_law_documents():
                if d.get("title") == law_title:
                    kb_store.delete_law_document(d["doc_id"])


if __name__ == "__main__":
    unittest.main()
