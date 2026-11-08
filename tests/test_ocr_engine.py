import io
import unittest
import zipfile
from app.ocr_engine import is_ocr_available, ocr_image_bytes, ocr_pdf_pages, ocr_docx_embedded_images
from app.splitter import extract_text_from_file_bytes, split_by_articles, split_auto


class TestOCREngine(unittest.TestCase):
    def test_ocr_availability_and_empty_handling(self):
        # 即使无 OCR 模型或空字节，也绝不报错并安全返回
        self.assertEqual(ocr_image_bytes(b""), "")
        self.assertIsInstance(ocr_image_bytes(b"not-an-image"), str)

    def test_docx_embedded_image_extraction(self):
        # 构造包含 word/media/image1.png 的最小 docx 内存压缩包
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w") as zf:
            zf.writestr("word/document.xml", "<w:document><w:body><w:p><w:t>第一条 劳动合同订立</w:t></w:p></w:body></w:document>")
            # 存入一个测试假图
            zf.writestr("word/media/image1.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR...")
        docx_bytes = bio.getvalue()

        # 验证提取能力
        results = ocr_docx_embedded_images(docx_bytes)
        self.assertIsInstance(results, list)

        # 验证通过 extract_text_from_file_bytes 解析含有附图的 docx
        extracted_text = extract_text_from_file_bytes("contract.docx", docx_bytes)
        self.assertIn("第一条 劳动合同订立", extracted_text)

    def test_image_direct_upload_handling(self):
        # 模拟上传图片
        fake_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4"
        res = extract_text_from_file_bytes("stamp.png", fake_png)
        self.assertIsInstance(res, str)
        self.assertTrue(len(res) > 0)

    def test_pdf_scan_handling(self):
        # 验证非法/空 PDF 传入 ocr_pdf_pages 安全兜底
        res = ocr_pdf_pages(b"not-a-pdf")
        self.assertEqual(res, "")


if __name__ == "__main__":
    unittest.main()

