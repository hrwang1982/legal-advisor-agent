"""Docling 微服务接入与平滑降级集成测试 (Docling Integration & Fallback Tests)"""
import io
import unittest
from unittest.mock import patch
import zipfile

from app.docling_engine import is_docling_available, parse_document_via_docling
from app.splitter import extract_text_from_file_bytes, split_by_articles


class TestDoclingIntegration(unittest.TestCase):
    def test_docling_offline_safety(self):
        """当 Docling 服务不可达或超时时，探测函数安全返回 False，不抛出异常"""
        with patch.dict("os.environ", {"DOCLING_SERVICE_URL": "http://127.0.0.1:59999/parse"}):
            self.assertFalse(is_docling_available(timeout=0.1))
            res = parse_document_via_docling("test.pdf", b"fake-pdf-content", timeout=0.1)
            self.assertIsNone(res)

    def test_docling_online_priority_and_markdown_retention(self):
        """当 Docling 服务在线时，优先使用 Docling 返回的高保真 Markdown（包含标题与表格）"""
        mock_md = (
            "# 中华人民共和国民法典\n\n"
            "## 第一编 总则\n\n"
            "第一条 为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，制定本法。\n\n"
            "| 条款 | 适用情形 |\n"
            "| --- | --- |\n"
            "| 第一条 | 基本原则与立法目的 |\n"
        )

        with patch("app.docling_engine.is_docling_available", return_value=True), \
             patch("app.docling_engine.parse_document_via_docling", return_value=mock_md):

            extracted = extract_text_from_file_bytes("民法典.pdf", b"fake-pdf-bytes")
            self.assertEqual(extracted, mock_md.strip())

            # 验证切分器能够无缝接纳 Docling 产出的 Markdown 并完成法条切分
            chunks = split_by_articles(extracted, doc_id="doc_mock_1", law_name="民法典")
            self.assertTrue(len(chunks) >= 1)
            self.assertEqual(chunks[0]["article"], "第1条")
            self.assertIn("保护民事主体的合法权益", chunks[0]["content"])

    def test_docling_fallback_to_legacy_parser(self):
        """当 Docling 离线时，系统 100% 自动无缝降级回原生的 docx 解析引擎"""
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zf.writestr(
                "word/document.xml",
                "<w:document><w:body><w:p><w:t>第二条 本法适用于民商事纠纷仲裁</w:t></w:p></w:body></w:document>",
            )
        docx_bytes = bio.getvalue()

        # 模拟 Docling 容器离线
        with patch("app.docling_engine.is_docling_available", return_value=False):
            extracted = extract_text_from_file_bytes("仲裁法.docx", docx_bytes)
            self.assertIn("第二条 本法适用于民商事纠纷仲裁", extracted)


if __name__ == "__main__":
    unittest.main()
