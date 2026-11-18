"""Ragas 独立 Docker 微服务与自动化评测模块专项测试。
验证项：
1. 独立微服务 docker/ragas/server.py 核心六维评测算法（正向合规通过 + 负向幻觉/误用旧法惩罚）；
2. 主应用 app/ragas_eval.py 全量黄金基准集（G01~G10）评测与容器不可达时的公式引擎无缝回退；
3. 生产真实流水抽样评测模式（production）与自定义沙盒评测模式（custom）；
4. 独立容器 HTTP 调度优先级与状态探测验证；
5. FastAPI 路由（/eval/ragas/status, /eval/ragas/run, /eval/ragas/history）注册及可用性验证。
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MOCK", "true")

import app.ragas_eval as ragas_eval_mod
from app.config import settings
from app.ragas_eval import GOLDEN_BENCHMARK_CASES, RagasEvaluationManager, ragas_eval_mgr


def _load_docker_ragas_server():
    """动态加载 docker/ragas/server.py（验证其可独立运行无主项目耦合）。"""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "docker",
        "ragas",
        "server.py",
    )
    spec = importlib.util.spec_from_file_location("docker_ragas_server", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


class TestRagasEvaluation(unittest.TestCase):
    def setUp(self):
        self._orig_mock = settings.mock
        settings.mock = True
        self._orig_history_path = ragas_eval_mod._HISTORY_PATH
        self._tmp_dir = tempfile.TemporaryDirectory()
        ragas_eval_mod._HISTORY_PATH = os.path.join(self._tmp_dir.name, "ragas_eval_history.json")

    def tearDown(self):
        settings.mock = self._orig_mock
        ragas_eval_mod._HISTORY_PATH = self._orig_history_path
        self._tmp_dir.cleanup()

    def test_01_docker_microservice_algorithm(self):
        """测试独立 Ragas 微服务 docker/ragas/server.py 的六维评测与防幻觉/防旧法惩罚逻辑"""
        srv = _load_docker_ragas_server()
        good_sample = {
            "case_id": "CASE_GOOD",
            "question": "对方不履行合同义务造成我方损失，我可以要求继续履行并赔偿损失吗？",
            "answer": "【一、争议焦点与事实前提】违约赔偿。\n【二、适用法规与效力要旨（大前提）】依据《中华人民共和国民法典》第577条规定，当事人一方不履行合同义务应当承担继续履行、赔偿损失等违约责任；依据《中华人民共和国民法典》第584条，损失赔偿额应当相当于因违约所造成的损失。\n【三、涵摄分析与初步结论】可要求继续履行并赔偿损失。\n【五、诉讼/仲裁时效与程序路径提示】三年诉讼时效。",
            "contexts": [
                "《中华人民共和国民法典》第577条：当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。",
                "《中华人民共和国民法典》第584条：当事人一方不履行合同义务或者履行合同义务不符合约定，造成对方损失的，损失赔偿额应当相当于因违约所造成的损失。",
            ],
            "ground_truth": "依据《中华人民共和国民法典》第577条和第584条要求继续履行并赔偿损失。",
            "must_include": [["中华人民共和国民法典", "第577条"], ["中华人民共和国民法典", "第584条"]],
            "must_exclude": [["中华人民共和国合同法", "第107条"]],
            "citations": [
                {"law_name": "中华人民共和国民法典", "article": "第577条", "applicable_at": True},
                {"law_name": "中华人民共和国民法典", "article": "第584条", "applicable_at": True},
            ],
            "temporal_compliant": True,
            "hierarchy_monotonic": True,
        }
        bad_sample = {
            "case_id": "CASE_HALLUCINATED",
            "question": "2025年合同违约怎么赔偿？",
            "answer": "根据《中华人民共和国民法典》第9999条虚构规定以及已废止《中华人民共和国合同法》第107条予以双倍罚款。",
            "contexts": ["《中华人民共和国民法典》第577条：违约责任。"],
            "ground_truth": "依据《中华人民共和国民法典》第577条主张违约责任。",
            "must_include": [["中华人民共和国民法典", "第577条"]],
            "must_exclude": [["中华人民共和国合同法", "第107条"]],
            "citations": [
                {"law_name": "中华人民共和国合同法", "article": "第107条", "applicable_at": False},
            ],
            "temporal_compliant": False,
            "hierarchy_monotonic": True,
        }

        resp = srv.evaluate_samples_payload([good_sample, bad_sample])
        self.assertEqual(resp["status"], "ok")
        self.assertEqual(resp["sample_count"], 2)
        res_good, res_bad = resp["results"][0], resp["results"][1]

        self.assertTrue(res_good["passed"], f"合规正向样本应通过: {res_good}")
        self.assertGreaterEqual(res_good["metrics"]["citation_accuracy"], 0.99)
        self.assertGreaterEqual(res_good["metrics"]["context_precision"], 0.99)
        self.assertGreaterEqual(res_good["metrics"]["temporal_hierarchy_compliance"], 0.99)

        self.assertFalse(res_bad["passed"], f"幻觉与误用旧法样本不得通过: {res_bad}")
        self.assertLess(res_bad["metrics"]["citation_accuracy"], 0.5)
        self.assertLess(res_bad["metrics"]["temporal_hierarchy_compliance"], 0.5)
        self.assertIn("未检索支撑", res_bad["critique"])
        print(f"✅ 1. Ragas 独立容器算法验证通过 (正向={res_good['overall_score']}, 负向={res_bad['overall_score']})")

    def test_02_golden_benchmark_all_10_cases(self):
        """测试主应用 G01~G10 黄金基准集全量 Ragas 自动评测（含容器不可达无缝公式回退）"""
        status = ragas_eval_mgr.get_status()
        self.assertEqual(len(status["golden_cases"]), 10)
        self.assertTrue(status["fallback_available"])

        report = ragas_eval_mgr.run_evaluation(mode="golden")
        self.assertEqual(report["mode"], "golden")
        self.assertEqual(report["sample_count"], len(GOLDEN_BENCHMARK_CASES))
        self.assertEqual(report["pass_count"], 10, f"期望黄金集 10/10 全通过，实际结果: {report['results']}")
        self.assertEqual(report["pass_rate"], 1.0)
        self.assertGreaterEqual(report["summary"]["citation_accuracy"], 0.99)
        self.assertGreaterEqual(report["summary"]["temporal_hierarchy_compliance"], 0.99)
        self.assertGreaterEqual(report["summary"]["overall_score"], 0.78)

        # 验证历史文件持久化写入成功
        history = ragas_eval_mgr.get_history(limit=5)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["run_id"], report["run_id"])
        print(
            f"✅ 2. 黄金基准集 G01~G10 全量 Ragas 评测通过 (综合分={report['summary']['overall_score']}, 等级={report['summary']['grade']}, 通过率=100%)"
        )

    def test_03_production_and_custom_modes(self):
        """测试生产真实问答流水抽样评测模式与自定义案情沙盒模式"""
        # 1. 自定义沙盒模式
        custom_rep = ragas_eval_mgr.run_evaluation(
            mode="custom",
            custom_case={
                "query": "员工送货途中开车撞伤行人怎么赔偿？",
                "act_date": "2025-05-01",
                "ground_truth": "依据《民法典》第1191条由用人单位承担侵权责任，机动车事故优先适用《道路交通安全法》第76条。",
            },
        )
        self.assertEqual(custom_rep["mode"], "custom")
        self.assertEqual(custom_rep["sample_count"], 1)
        self.assertGreater(custom_rep["summary"]["faithfulness"], 0.5)
        self.assertTrue(len(custom_rep["results"][0]["citations_preview"]) >= 1)

        # 2. 生产真实流水抽样模式（如果数据库中有流水快照则必须成功完成评估）
        from app.mysql_store import mysql_store
        existing_rows = mysql_store.query("SELECT COUNT(*) as c FROM session_snapshots")
        if existing_rows and existing_rows[0]["c"] > 0:
            prod_rep = ragas_eval_mgr.run_evaluation(mode="production", limit=3)
            self.assertEqual(prod_rep["mode"], "production")
            self.assertGreaterEqual(prod_rep["sample_count"], 1)
            self.assertLessEqual(prod_rep["sample_count"], 3)
            print(f"✅ 3. 生产真实流水 (N={prod_rep['sample_count']}) 与自定义沙盒评测模式验证通过")
        else:
            print("✅ 3. 自定义沙盒评测模式验证通过（当前库无历史生产流水，跳过 production 断言）")

    def test_04_container_priority_dispatch_and_routes(self):
        """测试独立 Ragas 容器优先调度机制以及 FastAPI 路由注册完整性"""
        import urllib.request
        from io import BytesIO

        fake_container_resp = {
            "status": "ok",
            "engine": "ragas_docker_formula",
            "elapse_ms": 12.5,
            "sample_count": 1,
            "pass_count": 1,
            "pass_rate": 1.0,
            "summary": {
                "faithfulness": 0.95,
                "answer_relevancy": 0.92,
                "context_precision": 1.0,
                "context_recall": 1.0,
                "citation_accuracy": 1.0,
                "temporal_hierarchy_compliance": 1.0,
                "overall_score": 0.97,
                "grade": "S",
            },
            "results": [
                {
                    "case_id": "G01_现行合同违约与引致扩展",
                    "question": "测试容器调度",
                    "metrics": {
                        "faithfulness": 0.95,
                        "answer_relevancy": 0.92,
                        "context_precision": 1.0,
                        "context_recall": 1.0,
                        "citation_accuracy": 1.0,
                        "temporal_hierarchy_compliance": 1.0,
                    },
                    "overall_score": 0.97,
                    "grade": "S",
                    "passed": True,
                    "critique": "来自 Ragas 独立容器响应",
                }
            ],
        }

        class _FakeHTTPResp(BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        orig_urlopen = urllib.request.urlopen
        try:
            urllib.request.urlopen = lambda req, timeout=10: _FakeHTTPResp(
                json.dumps(fake_container_resp, ensure_ascii=False).encode("utf-8")
            )
            mgr = RagasEvaluationManager()
            rep = mgr.run_evaluation(mode="golden", case_ids=["G01_现行合同违约与引致扩展"])
            self.assertTrue(rep["container_used"])
            self.assertEqual(rep["summary"]["overall_score"], 0.97)
            self.assertEqual(rep["results"][0]["critique"], "来自 Ragas 独立容器响应")
        finally:
            urllib.request.urlopen = orig_urlopen

        # 若当前环境安装了 FastAPI，检查三个 Ragas 路由是否已正确挂载
        from app.server import app
        if app is not None:
            paths = {r.path for r in app.routes}
            self.assertIn("/eval/ragas/status", paths)
            self.assertIn("/eval/ragas/run", paths)
            self.assertIn("/eval/ragas/history", paths)
        print("✅ 4. 独立 Ragas 容器优先 HTTP 调度及路由注册完整性验证通过")


if __name__ == "__main__":
    print("=== 开始运行 Ragas 独立容器自动评测专项测试 ===")
    unittest.main(verbosity=2)
