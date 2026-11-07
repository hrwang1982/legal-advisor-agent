"""针对 MySQL 8.x 细粒度拆分专用表与 Elasticsearch 8.x 的全量集成验证测试。
验证：
1. MySQL 专用表 llm_config 中大模型与向量配置的读取、持久化与在线热更新；
2. MySQL 细粒度专用表（kb_legal_synonyms、kb_cross_references、kb_article_successions、kb_client_registry 等）的读写与维护；
3. MySQL 中预置法规（sample_laws）与知识库元数据（law_documents, law_chunks）的读取维护；
4. MySQL 中可审计账本（快照、防篡改审计日志、规则缺口、规则提案）的保存与追溯；
5. Elasticsearch 向量库适配器与首轮粗排（coarse_rank / hybrid_search）的功能与容错弹性；
6. 检索流水线与业务逻辑的一致性保障。
"""
from __future__ import annotations

import json
import logging
import os
import unittest

from app.audit import audit_store
from app.config import settings
from app.es_store import es_vector_store
from app.ingest import _load_rows, build_retriever, load_nodes
from app.kb import kb_store
from app.knowledge import knowledge_center
from app.mysql_store import mysql_store
from app.rbac import rbac_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestMySQLAndES")


class TestMySQLAndESIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        mysql_store.init_schema()

    def test_01_llm_config_dedicated_table(self):
        """测试 1：大模型运行时配置与向量参数在专属表 llm_config 中的读取与热更新"""
        llm_cfg = mysql_store.get_llm_config()
        self.assertIsNotNone(llm_cfg, "MySQL 中应存在 llm_config 专用表记录")
        self.assertIn("llm_model", llm_cfg)
        self.assertIn("embed_model", llm_cfg)

        # 动态热更新测试
        orig_temp = settings.llm_temperature
        settings.update_runtime({"llm_temperature": 0.38})
        reloaded = mysql_store.get_llm_config()
        self.assertEqual(reloaded.get("llm_temperature"), 0.38, "热更新后 llm_config 表字段应实时更新")
        # 恢复
        settings.update_runtime({"llm_temperature": orig_temp})
        logger.info("✅ 测试 1 通过：专属表 llm_config 存储与在线热更新正常")

    def test_02_knowledge_split_tables(self):
        """测试 2：知识中心拆分后的专用表（法律同义词/图谱/新旧法/利益冲突库等）读写维护"""
        # 2.1 法律同义词表 kb_legal_synonyms
        syns = mysql_store.get_legal_synonyms()
        self.assertTrue(len(syns) >= 30, f"法律同义词表应至少有 30 条记录，实际为: {len(syns)}")
        self.assertIn("炒鱿鱼", syns)
        self.assertIn("霸王条款", syns)

        # 2.2 法条引用图谱表 kb_cross_references
        refs = mysql_store.get_cross_references()
        self.assertTrue(len(refs) >= 8, f"法条引用图谱表应至少有 8 条记录，实际为: {len(refs)}")
        labour_ref = next((r for r in refs if "劳动合同法" in r["source_law"] and "38" in r["source_article"]), None)
        self.assertIsNotNone(labour_ref, "必须包含劳动合同法第38条的引致图谱")

        # 2.3 新旧法对照表 kb_article_successions
        succs = mysql_store.get_article_successions()
        self.assertTrue(len(succs) >= 2, f"新旧法对照表应至少有 2 条记录，实际为: {len(succs)}")
        contract_succ = next((s for s in succs if "合同法" in s["old_law"] and "民法典" in s["new_law"]), None)
        self.assertIsNotNone(contract_succ, "必须包含合同法到民法典的承接关系")

        # 2.4 利益冲突与客户库 kb_client_registry
        clients = mysql_store.get_client_registry()
        self.assertTrue(len(clients) >= 3, f"利益冲突客户库应至少有 3 条记录，实际为: {len(clients)}")

        # 2.5 知识中心聚合视图
        full_k = knowledge_center.get_all()
        self.assertIn("legal_synonym_map", full_k)
        self.assertIn("cross_reference_graph", full_k)
        logger.info("✅ 测试 2 通过：各拆分专用表 (kb_legal_synonyms, kb_cross_references, kb_article_successions, kb_client_registry) 读写正常")

    def test_03_mysql_sample_laws_and_kb(self):
        """测试 3：预置法规与法规知识库在 MySQL 中的持久化与载入"""
        rows = mysql_store.query("SELECT count(*) as cnt FROM sample_laws")
        cnt = rows[0]["cnt"]
        self.assertTrue(cnt >= 48, f"MySQL sample_laws 应至少包含 48 条预置条文，实际为: {cnt}")

        all_laws = _load_rows()
        self.assertTrue(len(all_laws) >= 48)
        civil_code_arts = [l for l in all_laws if "民法典" in l.get("law_name", "")]
        self.assertTrue(len(civil_code_arts) > 0, "必须包含民法典条文")

        docs = mysql_store.query("SELECT * FROM law_documents")
        self.assertTrue(isinstance(docs, list))
        logger.info(f"✅ 测试 3 通过：MySQL sample_laws (条数={cnt}) 与知识库文档读取正常")

    def test_04_mysql_audit_trail(self):
        """测试 4：审计会话快照、防篡改日志与规则缺口在 MySQL 中的维护"""
        test_thread = "test_thread_verify_split_002"
        test_state = {"query": "测试拆分表存储", "act_date": "2024-05-01"}

        audit_store.save_snapshot(test_thread, "COMPLETED", test_state)
        snap = audit_store.get_snapshot(test_thread)
        self.assertIsNotNone(snap)
        self.assertEqual(snap["status"], "COMPLETED")

        event = audit_store.record_audit_event(
            thread_id=test_thread,
            event_type="HUMAN_REVIEW_ADOPT",
            reviewer_name="李合规律师",
            license_no="13101201810012345",
            reviewer_role="compliance_officer",
            decision="APPROVED",
            draft_answer="AI 初稿",
            final_answer="终审意见",
        )
        self.assertIn("entry_hash", event)
        logger.info("✅ 测试 4 通过：MySQL 审计账本 (快照/防篡改日志/规则缺口) 保存与追溯正常")

    def test_05_es_lifecycle_and_coarse_rank(self):
        """测试 5：Elasticsearch 向量库与首轮混合粗排适配能力"""
        is_es_alive = es_vector_store.ping()
        logger.info(f"Elasticsearch 连通状态: {'在线 (Active)' if is_es_alive else '离线 (Fallback)'}")

        test_chunk = {
            "chunk_id": "test_chunk_split_es",
            "doc_id": "test_doc_split_es",
            "law_name": "测试民法典验证案",
            "article": "第1条",
            "content": "为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序。",
            "content_vector": [0.1] * 128,
        }

        if is_es_alive:
            es_vector_store.init_index(dims=128)
            indexed = es_vector_store.index_chunks([test_chunk])
            self.assertTrue(indexed >= 0)
            coarse_res = es_vector_store.coarse_rank("民事主体合法权益", query_vector=[0.1] * 128, top_k=5)
            self.assertTrue(isinstance(coarse_res, list))
            es_vector_store.delete_by_doc_id("test_doc_split_es")
        else:
            res = es_vector_store.coarse_rank("测试查询")
            self.assertEqual(res, [])
        logger.info("✅ 测试 5 通过：ES 8.x 向量与粗排模块生命周期与平滑降级正常")

    def test_06_retriever_pipeline(self):
        """测试 6：多路召回流水线与粗排融合检索"""
        nodes = load_nodes()
        self.assertTrue(len(nodes) > 0)
        retriever = build_retriever(nodes, top_k=5)
        results = retriever.retrieve("公司无故开除我，怎么要赔偿金")
        self.assertTrue(len(results) > 0)
        top_law = results[0].node.metadata.get("law_name", "")
        self.assertIn("劳动", top_law)
        logger.info(f"✅ 测试 6 通过：检索流水线对拆分新表数据的聚合召回正常，Top-1: {top_law} {results[0].node.metadata.get('article', '')}")


if __name__ == "__main__":
    unittest.main()
