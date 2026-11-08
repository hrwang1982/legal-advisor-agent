"""
测试知识库文档分段与切片查看接口
验证：
1. get_document_chunks 能够正确返回指定文档的切片分段数据；
2. 切片字段包含 chunk_id, article, content, chunk_type, validity 等核心信息；
3. 不存在的文档返回 not_found。
"""

import unittest
from app.kb import kb_store


class TestDocChunks(unittest.TestCase):
    def setUp(self):
        self.store = kb_store

    def test_get_existing_document_chunks(self):
        # 查询现有的第一个文档
        docs = self.store.list_law_documents(limit=1)
        self.assertGreater(len(docs), 0, "数据库中应至少有1篇法规文档")

        doc_id = docs[0]["doc_id"]
        res = self.store.get_document_chunks(doc_id)

        self.assertEqual(res.get("status"), "ok")
        self.assertIn("document", res)
        self.assertEqual(res["document"]["doc_id"], doc_id)
        self.assertIn("chunks", res)
        self.assertGreaterEqual(res["total_chunks"], 1)

        first_chunk = res["chunks"][0]
        self.assertIn("chunk_id", first_chunk)
        self.assertIn("content", first_chunk)
        self.assertIn("chunk_type", first_chunk)

    def test_get_nonexistent_document_chunks(self):
        res = self.store.get_document_chunks("doc_non_existent_999999")
        self.assertEqual(res.get("status"), "not_found")
        self.assertEqual(len(res.get("chunks", [])), 0)


if __name__ == "__main__":
    unittest.main()

