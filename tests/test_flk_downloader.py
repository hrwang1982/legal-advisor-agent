import json
import unittest
import urllib.error
from unittest.mock import patch, MagicMock

from app.flk_downloader import FLKDownloader, FLK_TYPE_MAPPING


class TestFLKDownloader(unittest.TestCase):
    def test_no_static_offline_registry_exists(self):
        """确保完全移除了静态本地硬编码兜底方法与 OFFLINE_REGISTRY"""
        import app.flk_downloader as mod
        self.assertFalse(hasattr(mod, "OFFLINE_REGISTRY"))
        dl = FLKDownloader()
        self.assertFalse(hasattr(dl, "_fallback_offline_search"))
        self.assertFalse(hasattr(dl, "_build_official_word_document_bytes"))

    def test_type_code_mappings_complete(self):
        """确保 flfg, xzfg, sfjs, dfxfg 均具备完整的新版 flfgCodeId 映射"""
        for code in ("flfg", "xzfg", "sfjs", "dfxfg"):
            self.assertIn(code, FLK_TYPE_MAPPING)
            self.assertTrue(len(FLK_TYPE_MAPPING[code]["flfgCodeId"]) > 0)

    @patch("app.flk_downloader.time.sleep", return_value=None)
    @patch("app.flk_downloader.urllib.request.urlopen")
    def test_network_timeout_retries_3_times_5s_and_returns_unreachable(self, mock_urlopen, _mock_sleep):
        """当网络探测/请求超时时，严格重试 3 次（每次超时 5 秒），并返回 network_unreachable 提示"""
        mock_urlopen.side_effect = urllib.error.URLError("timed out")
        dl = FLKDownloader(max_retries=3, timeout=5.0)
        res = dl.search_laws(keyword="", type_code="flfg", page=1, size=20)

        self.assertEqual(mock_urlopen.call_count, 3)
        for call_args in mock_urlopen.call_args_list:
            self.assertEqual(call_args.kwargs.get("timeout"), 5.0)

        self.assertEqual(res["status"], "network_unreachable")
        self.assertEqual(res["retry_count"], 3)
        self.assertEqual(res["timeout_seconds"], 5)
        self.assertIn("网络不可达", res["message"])
        self.assertEqual(res["laws"], [])

    @patch("app.flk_downloader.urllib.request.urlopen")
    def test_live_search_parses_official_new_api_response(self, mock_urlopen):
        """测试对接新版 /law-search/search/list 接口的真实响应结构解析"""
        fake_payload = {
            "code": 200,
            "msg": "查询成功",
            "total": 305,
            "rows": [
                {
                    "bbbs": "c554e0d172b24782b66125d58760a128",
                    "title": "<font color='red'>中华人民共和国</font>公司法",
                    "flxz": "法律",
                    "zdjgName": "全国人民代表大会常务委员会",
                    "gbrq": "2023-12-29",
                    "sxrq": "2024-07-01",
                    "sxx": 3,
                }
            ],
        }
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(fake_payload, ensure_ascii=False).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        dl = FLKDownloader(max_retries=3, timeout=5.0)
        res = dl.search_laws(keyword="公司法", type_code="flfg", page=1, size=20, sxx=[3])

        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total"], 305)
        self.assertEqual(len(res["laws"]), 1)
        law = res["laws"][0]
        self.assertEqual(law["id"], "c554e0d172b24782b66125d58760a128")
        self.assertEqual(law["title"], "中华人民共和国公司法")
        self.assertEqual(law["validity_label"], "现行有效")
        self.assertEqual(law["effective_date"], "2024-07-01")


if __name__ == "__main__":
    unittest.main()

