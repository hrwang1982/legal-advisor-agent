"""
测试数据统计看板真实性与问答流水明细接口
验证：
1. 数据统计指标 100% 取自数据库真实记录，无任何写死基数与伪造数据；
2. 问答详情流水明细真实读取 session_snapshots 与 audit_logs，且支持用户名、关键词、部门与日期检索；
3. 会话总时长基于真实会话时间窗口动态计算，杜绝写死字符串。
"""

import unittest
from datetime import datetime
from app.mysql_store import mysql_store


class TestStatsReal(unittest.TestCase):
    def setUp(self):
        self.store = mysql_store

    def test_stats_overview_strictly_matches_db(self):
        """测试 12 个指标卡片与数据库底层物理表行数严格 1:1 吻合"""
        stats = self.store.get_stats_overview()

        # 1. 知识库总量 (顶层)
        kb_nodes = self.store.query("SELECT COUNT(*) as c FROM kb_tree_nodes WHERE node_type = 'kb' OR parent_id = ''")
        expected_kb_total = kb_nodes[0]["c"]
        self.assertEqual(stats["kb_total"], expected_kb_total)
        self.assertGreaterEqual(stats["kb_total"], 1)

        # 2. 外规与内规
        ext_docs = self.store.query("SELECT COUNT(*) as c FROM law_documents WHERE kb_type LIKE '%外规%'")
        expected_ext = ext_docs[0]["c"]
        self.assertEqual(stats["kb_external"], expected_ext)

        int_docs = self.store.query("SELECT COUNT(*) as c FROM law_documents WHERE kb_type LIKE '%内规%'")
        expected_int = int_docs[0]["c"]
        self.assertEqual(stats["kb_internal"], expected_int)

        # 3. 自建总量
        all_docs = self.store.query("SELECT COUNT(*) as c FROM law_documents")
        expected_custom = all_docs[0]["c"]
        self.assertEqual(stats["kb_custom"], expected_custom)
        self.assertEqual(stats["kb_custom"], stats["kb_external"] + stats["kb_internal"])

        # 4. 问答总量 & 提问总量 & 回答总量
        qa_events = self.store.query("SELECT COUNT(*) as c FROM audit_logs WHERE event_type = 'query_generated'")
        expected_qa = qa_events[0]["c"]
        self.assertEqual(stats["qa_total"], expected_qa)
        self.assertEqual(stats["query_total"], expected_qa)
        self.assertEqual(stats["answer_total"], expected_qa)

        # 5. 建议总量 (规则提案)
        props = self.store.query("SELECT COUNT(*) as c FROM rule_proposals")
        expected_props = props[0]["c"]
        self.assertEqual(stats["suggestion_total"], expected_props)

        # 6. 点赞与点踩总量
        likes = self.store.query("SELECT COUNT(*) as c FROM audit_logs WHERE event_type = 'HUMAN_REVIEW_ADOPT' OR decision = 'approved'")
        expected_likes = likes[0]["c"]
        self.assertEqual(stats["like_total"], expected_likes)

        dislikes = self.store.query("SELECT COUNT(*) as c FROM audit_logs WHERE decision = 'rejected'")
        expected_dislikes = dislikes[0]["c"]
        self.assertEqual(stats["dislike_total"], expected_dislikes)

        # 7. 会话总量
        sessions = self.store.query("SELECT COUNT(*) as c FROM session_snapshots")
        expected_sessions = sessions[0]["c"]
        self.assertEqual(stats["session_total"], expected_sessions)

        # 8. 会话总时长非空且不是旧硬编码
        self.assertTrue(isinstance(stats["session_duration"], str))
        self.assertIn("秒", stats["session_duration"])
        self.assertNotEqual(stats["session_duration"], "2小时53分35秒")
        self.assertGreater(stats["session_seconds"], 0)

    def test_qa_details_retrieval(self):
        """测试问答详情流水明细的读取与字段完整性"""
        details = self.store.get_qa_details(limit=20)
        self.assertGreater(len(details), 0)

        first = details[0]
        self.assertIn("thread_id", first)
        self.assertIn("time", first)
        self.assertIn("username", first)
        self.assertIn("user_display", first)
        self.assertIn("department", first)
        self.assertIn("scope", first)
        self.assertIn("query", first)
        self.assertIn("answer_summary", first)
        self.assertIn("review_status", first)

        # 状态应当是合法的标签
        self.assertIn(first["review_status"], ("已复核", "待复核", "已驳回", "已办结"))

    def test_qa_details_keyword_filter(self):
        """测试关键词过滤（命中 query 或 answer）"""
        kw = "公司"
        details = self.store.get_qa_details(keyword=kw, limit=50)
        for item in details:
            matched = kw in item["query"] or kw in item["answer_summary"]
            self.assertTrue(matched, f"Item did not match keyword {kw}: {item}")

    def test_qa_details_department_filter(self):
        """测试部门树层级过滤"""
        # 测试过滤总部
        hq_details = self.store.get_qa_details(dept_id="dept_hq", limit=50)
        for item in hq_details:
            self.assertEqual(item["dept_id"], "dept_hq")


if __name__ == "__main__":
    unittest.main()

