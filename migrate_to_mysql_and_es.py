"""数据细粒度解耦迁移引擎（JSON & SQLite -> MySQL 8.x 专用表 + Elasticsearch 8.x）：
将原先统合在 sys_configs 中的粗粒度配置：
1. 细粒度拆分出 llm_config 专用表（大模型参数、向量配置、独立凭据）；
2. 细粒度拆分出知识中心系列专用表：
   - kb_legal_synonyms（法律同义词）
   - kb_cross_references（法条引用图谱）
   - kb_article_successions（新旧法条文级对照）
   - kb_client_registry（利益冲突客户库）
   - kb_special_general_relations（特别法一般法冲突规则）
   - kb_law_aliases（法规别名简称映射）
   - kb_case_type_rules（案由规则库）
   - kb_certainty_patterns（确定性表述约束）
   - kb_new_old_counterparts（立法对照表数据）
   - kb_allowed_procedural_meta_citations（程序性条文许可）
   - kb_reviewers_registry（审核专家注册表）
3. RBAC、预置法规 sample_laws、文档与分块以及可追责审计层分别映射到各自专用表；
4. 重新导出全套规范的 docker/mysql/init.sql 引导脚本。
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List

from app.config import settings
from app.es_store import es_vector_store
from app.kb import BGEM3Embedder
from app.mysql_store import mysql_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataMigrator")


def escape_sql_str(val: Any) -> str:
    """转义 SQL 单引号与反斜杠，生成安全的 SQL 插入值。"""
    if val is None:
        return "NULL"
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, bool):
        return "1" if val else "0"
    s = str(val)
    s = s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")
    return f"'{s}'"


def migrate_all():
    logger.info("====== 开始执行 MySQL 细粒度拆分与全量数据迁移流程 ======")

    # 1. 初始化 MySQL 表结构（包含 llm_config 与 kb_* 系列拆分新表）
    mysql_store.init_schema()
    logger.info("✅ 关系表结构（包含拆分后的专用表）初始化完成")

    now_ts = datetime.now(timezone.utc).isoformat()

    sql_statements: List[str] = [
        "-- AI 法盾与智能法律顾问系统 MySQL 8.x 初始化脚本（细粒度专用表拆分规范化架构）",
        "SET NAMES utf8mb4;",
        "SET FOREIGN_KEY_CHECKS = 0;\n",
    ]

    # 添加全套 DDL
    ddl_sqls = [
        "CREATE TABLE IF NOT EXISTS llm_config (id INT PRIMARY KEY DEFAULT 1, mock TINYINT(1) DEFAULT 1, llm_provider VARCHAR(64) NOT NULL DEFAULT 'qwen', llm_api_base VARCHAR(255) NOT NULL, llm_model VARCHAR(128) NOT NULL, llm_api_key VARCHAR(255) DEFAULT '', llm_temperature FLOAT DEFAULT 0.0, llm_timeout FLOAT DEFAULT 30.0, llm_max_retries INT DEFAULT 2, embed_enabled TINYINT(1) DEFAULT 1, embed_api_base VARCHAR(255) DEFAULT '', embed_model VARCHAR(128) DEFAULT 'BAAI/bge-m3', embed_api_key VARCHAR(255) DEFAULT '', embed_dim INT DEFAULT 1024, hybrid_dense_weight FLOAT DEFAULT 0.45, strict_review_auth TINYINT(1) DEFAULT 0, top_k INT DEFAULT 5, platform_credentials_json LONGTEXT, updated_at VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS kb_legal_synonyms (id INT AUTO_INCREMENT PRIMARY KEY, colloquial_term VARCHAR(128) NOT NULL UNIQUE, legal_terms TEXT NOT NULL, updated_at VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS kb_cross_references (id INT AUTO_INCREMENT PRIMARY KEY, source_law VARCHAR(255) NOT NULL, source_article VARCHAR(64) NOT NULL, targets_json LONGTEXT NOT NULL, updated_at VARCHAR(64) NOT NULL, UNIQUE KEY uk_cross_source (source_law, source_article)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS kb_article_successions (id INT AUTO_INCREMENT PRIMARY KEY, old_law VARCHAR(255) NOT NULL, old_article VARCHAR(64) NOT NULL, new_law VARCHAR(255) NOT NULL, new_article VARCHAR(64) NOT NULL, switch_date VARCHAR(32) NOT NULL, diff_note TEXT, updated_at VARCHAR(64) NOT NULL, INDEX idx_succ_old (old_law, old_article), INDEX idx_succ_new (new_law, new_article)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS kb_client_registry (id INT AUTO_INCREMENT PRIMARY KEY, client_name VARCHAR(255) NOT NULL UNIQUE, client_type VARCHAR(128) NOT NULL DEFAULT '本所常年法律顾问客户', conflict_rules TEXT, updated_at VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS kb_special_general_relations (id INT AUTO_INCREMENT PRIMARY KEY, special_law VARCHAR(255) NOT NULL, general_law VARCHAR(255) NOT NULL, description TEXT, has_deference TINYINT(1) DEFAULT 1, updated_at VARCHAR(64) NOT NULL, UNIQUE KEY uk_spec_gen (special_law, general_law)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS kb_law_aliases (id INT AUTO_INCREMENT PRIMARY KEY, alias_name VARCHAR(128) NOT NULL UNIQUE, canonical_name VARCHAR(255) NOT NULL, updated_at VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS kb_case_type_rules (id INT AUTO_INCREMENT PRIMARY KEY, case_type VARCHAR(128) NOT NULL UNIQUE, rules_json LONGTEXT NOT NULL, updated_at VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS kb_certainty_patterns (id INT AUTO_INCREMENT PRIMARY KEY, pattern VARCHAR(255) NOT NULL, replacement VARCHAR(255) NOT NULL, updated_at VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS kb_new_old_counterparts (id INT AUTO_INCREMENT PRIMARY KEY, current_law VARCHAR(255) NOT NULL, current_article VARCHAR(64) NOT NULL, old_law VARCHAR(255) NOT NULL, old_article VARCHAR(64) NOT NULL, relation VARCHAR(64) DEFAULT 'one_to_one', switch_date VARCHAR(32) NOT NULL, diff_note TEXT, authority VARCHAR(255) DEFAULT '全国人大常委会法制工作委员会', updated_at VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS kb_allowed_procedural_meta_citations (id INT AUTO_INCREMENT PRIMARY KEY, law_name VARCHAR(255) NOT NULL, article VARCHAR(64) NOT NULL, updated_at VARCHAR(64) NOT NULL, UNIQUE KEY uk_meta_cite (law_name, article)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS kb_reviewers_registry (reviewer_id VARCHAR(64) PRIMARY KEY, name VARCHAR(128) NOT NULL, license_no VARCHAR(128) NOT NULL, role VARCHAR(64) NOT NULL, department VARCHAR(128) DEFAULT '', info_json TEXT, updated_at VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS sample_laws (id INT AUTO_INCREMENT PRIMARY KEY, law_name VARCHAR(255) NOT NULL, document_number VARCHAR(128) DEFAULT '', hierarchy VARCHAR(64) NOT NULL, host_law VARCHAR(255) DEFAULT '', host_hierarchy VARCHAR(64) DEFAULT '', promulgation_date VARCHAR(32) DEFAULT '', effective_date VARCHAR(32) DEFAULT '', effective_to VARCHAR(32) DEFAULT '', validity VARCHAR(32) NOT NULL DEFAULT '现行有效', superseded_by VARCHAR(255) DEFAULT '', article VARCHAR(64) NOT NULL, paragraph VARCHAR(64) DEFAULT '', content LONGTEXT NOT NULL, source VARCHAR(128) DEFAULT '', category VARCHAR(64) DEFAULT '', scope VARCHAR(64) DEFAULT '', partially_invalid TINYINT(1) DEFAULT 0, invalid_parts VARCHAR(255) DEFAULT '', trial TINYINT(1) DEFAULT 0, verified TINYINT(1) DEFAULT 1, note VARCHAR(255) DEFAULT '', article_effective_date VARCHAR(32) DEFAULT '', norm_type VARCHAR(64) DEFAULT 'substantive', metadata_json LONGTEXT, created_at VARCHAR(64) DEFAULT '', INDEX idx_sample_law_art (law_name, article)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS sys_rbac_roles (role_id VARCHAR(64) PRIMARY KEY, role_name VARCHAR(128) NOT NULL, description TEXT, is_system TINYINT(1) DEFAULT 0, menus_json TEXT NOT NULL, kb_permissions_json TEXT NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS sys_rbac_users (user_id VARCHAR(64) PRIMARY KEY, username VARCHAR(64) NOT NULL UNIQUE, name VARCHAR(128) NOT NULL, department VARCHAR(128) DEFAULT '', license_no VARCHAR(128) DEFAULT '', roles_json TEXT NOT NULL, status VARCHAR(32) DEFAULT 'active', created_at VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS kb_collections (collection_id VARCHAR(128) PRIMARY KEY, name VARCHAR(255) NOT NULL, description TEXT, embed_model VARCHAR(128) NOT NULL, created_at VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS law_documents (doc_id VARCHAR(128) PRIMARY KEY, title VARCHAR(255) NOT NULL, kb_type VARCHAR(128) NOT NULL DEFAULT '外规知识库/法律法规', department VARCHAR(128) DEFAULT '法律合规部', uploader VARCHAR(128) DEFAULT '系统管理员', doc_size VARCHAR(64) DEFAULT '0 KB', doc_format VARCHAR(32) DEFAULT 'TXT', split_mode VARCHAR(64) NOT NULL DEFAULT 'article', chunks_count INT DEFAULT 0, validity_status VARCHAR(64) DEFAULT '有效', doc_status VARCHAR(64) DEFAULT '已解析', valid_until VARCHAR(64) DEFAULT '', created_at VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS law_chunks (chunk_id VARCHAR(128) PRIMARY KEY, doc_id VARCHAR(128) NOT NULL, law_name VARCHAR(255) NOT NULL, chapter_path TEXT, article VARCHAR(64) NOT NULL, paragraph VARCHAR(64) DEFAULT '', content LONGTEXT NOT NULL, chunk_type VARCHAR(64) NOT NULL, parent_chunk_id VARCHAR(128) DEFAULT '', hierarchy VARCHAR(64) NOT NULL, effective_date VARCHAR(32) DEFAULT '', effective_to VARCHAR(32) DEFAULT '', validity VARCHAR(32) NOT NULL, metadata_json LONGTEXT NOT NULL, embedding_json LONGTEXT, created_at VARCHAR(64) NOT NULL, INDEX idx_chunk_doc (doc_id), INDEX idx_chunk_law (law_name)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS law_article_revision_log (id INT AUTO_INCREMENT PRIMARY KEY, law_name VARCHAR(255) NOT NULL, article VARCHAR(64) NOT NULL, paragraph VARCHAR(64) DEFAULT '', revision_type VARCHAR(64) NOT NULL, amendment_name VARCHAR(255) NOT NULL, effective_from VARCHAR(32) NOT NULL, effective_to VARCHAR(32) DEFAULT '', old_content LONGTEXT, new_content LONGTEXT NOT NULL, diff_summary TEXT, created_at VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS kb_tree_nodes (node_id VARCHAR(128) PRIMARY KEY, parent_id VARCHAR(128) NOT NULL DEFAULT '', name VARCHAR(255) NOT NULL, node_type VARCHAR(64) NOT NULL DEFAULT 'folder', description TEXT, sort_order INT DEFAULT 0, created_at VARCHAR(64) NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS session_snapshots (thread_id VARCHAR(128) PRIMARY KEY, updated_at VARCHAR(64) NOT NULL, status VARCHAR(64) NOT NULL, query TEXT, act_date VARCHAR(32), region VARCHAR(64), confidence VARCHAR(32), review_level VARCHAR(64), state_json LONGTEXT NOT NULL) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS audit_logs (id INT AUTO_INCREMENT PRIMARY KEY, thread_id VARCHAR(128) NOT NULL, timestamp VARCHAR(64) NOT NULL, event_type VARCHAR(64) NOT NULL, reviewer_name VARCHAR(128), license_no VARCHAR(128), reviewer_role VARCHAR(64), decision VARCHAR(64), comments TEXT, confidence VARCHAR(32), review_level VARCHAR(64), citations_json LONGTEXT, draft_answer LONGTEXT, final_answer LONGTEXT, answer_diff LONGTEXT, prev_hash VARCHAR(128) NOT NULL, entry_hash VARCHAR(128) NOT NULL, INDEX idx_audit_thread (thread_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS rule_gaps (id INT AUTO_INCREMENT PRIMARY KEY, thread_id VARCHAR(128) NOT NULL, timestamp VARCHAR(64) NOT NULL, mode VARCHAR(64), confidence VARCHAR(32), review_level VARCHAR(64), reason TEXT, verify_actions_json TEXT, query TEXT, INDEX idx_gap_thread (thread_id)) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
        "CREATE TABLE IF NOT EXISTS rule_proposals (id INT AUTO_INCREMENT PRIMARY KEY, thread_id VARCHAR(128) NOT NULL, timestamp VARCHAR(64) NOT NULL, proposal_type VARCHAR(64) NOT NULL, proposal_key VARCHAR(128), proposal_data_json LONGTEXT NOT NULL, status VARCHAR(64) DEFAULT 'PENDING', reviewer_name VARCHAR(128), review_notes TEXT, adopted_at VARCHAR(64), query TEXT) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;",
    ]
    sql_statements.extend(ddl_sqls)

    # ————————————————————————————————————————————————
    # 2. 迁移大模型配置至专用表 llm_config
    # ————————————————————————————————————————————————
    llm_path = settings.llm_config_path
    if os.path.exists(llm_path):
        with open(llm_path, "r", encoding="utf-8") as f:
            llm_cfg = json.load(f)
        mysql_store.save_llm_config(llm_cfg)
        plat_json = json.dumps(llm_cfg.get("platform_credentials", {}), ensure_ascii=False)
        sql_statements.append(
            f"REPLACE INTO llm_config (id, mock, llm_provider, llm_api_base, llm_model, llm_api_key, "
            f"llm_temperature, llm_timeout, llm_max_retries, embed_enabled, embed_api_base, embed_model, "
            f"embed_api_key, embed_dim, hybrid_dense_weight, strict_review_auth, top_k, platform_credentials_json, updated_at) "
            f"VALUES (1, {1 if llm_cfg.get('mock') else 0}, {escape_sql_str(llm_cfg.get('llm_provider', 'qwen'))}, "
            f"{escape_sql_str(llm_cfg.get('llm_api_base', ''))}, {escape_sql_str(llm_cfg.get('llm_model', 'qwen-plus'))}, "
            f"{escape_sql_str(llm_cfg.get('llm_api_key', ''))}, {llm_cfg.get('llm_temperature', 0.0)}, "
            f"{llm_cfg.get('llm_timeout', 30.0)}, {llm_cfg.get('llm_max_retries', 2)}, "
            f"{1 if llm_cfg.get('embed_enabled', True) else 0}, {escape_sql_str(llm_cfg.get('embed_api_base', ''))}, "
            f"{escape_sql_str(llm_cfg.get('embed_model', 'BAAI/bge-m3'))}, {escape_sql_str(llm_cfg.get('embed_api_key', ''))}, "
            f"{llm_cfg.get('embed_dim', 1024)}, {llm_cfg.get('hybrid_dense_weight', 0.45)}, "
            f"{1 if llm_cfg.get('strict_review_auth') else 0}, {llm_cfg.get('top_k', 5)}, "
            f"{escape_sql_str(plat_json)}, {escape_sql_str(now_ts)});"
        )
        logger.info("✅ 成功迁移大模型运行时与向量配置 -> MySQL llm_config 专用表")

    # ————————————————————————————————————————————————
    # 3. 拆分迁移知识中心数据至各个专用表
    # ————————————————————————————————————————————————
    k_path = settings.knowledge_config_path
    if os.path.exists(k_path):
        with open(k_path, "r", encoding="utf-8") as f:
            k_data = json.load(f)

        # 3.1 法律同义词表 (kb_legal_synonyms)
        syns = k_data.get("legal_synonym_map", {})
        mysql_store.save_legal_synonyms(syns)
        for term, expansion in syns.items():
            sql_statements.append(
                f"REPLACE INTO kb_legal_synonyms (colloquial_term, legal_terms, updated_at) "
                f"VALUES ({escape_sql_str(term)}, {escape_sql_str(expansion)}, {escape_sql_str(now_ts)});"
            )

        # 3.2 法条引用图谱表 (kb_cross_references)
        refs = k_data.get("cross_reference_graph", [])
        mysql_store.save_cross_references(refs)
        for r in refs:
            sql_statements.append(
                f"REPLACE INTO kb_cross_references (source_law, source_article, targets_json, updated_at) "
                f"VALUES ({escape_sql_str(r.get('source_law'))}, {escape_sql_str(r.get('source_article'))}, "
                f"{escape_sql_str(json.dumps(r.get('targets', []), ensure_ascii=False))}, {escape_sql_str(now_ts)});"
            )

        # 3.3 新旧法对照表 (kb_article_successions)
        succs = k_data.get("article_succession_list", [])
        mysql_store.save_article_successions(succs)
        for s in succs:
            sql_statements.append(
                f"INSERT INTO kb_article_successions (old_law, old_article, new_law, new_article, switch_date, diff_note, updated_at) "
                f"VALUES ({escape_sql_str(s.get('old_law'))}, {escape_sql_str(s.get('old_article'))}, "
                f"{escape_sql_str(s.get('new_law'))}, {escape_sql_str(s.get('new_article'))}, "
                f"{escape_sql_str(s.get('switch_date'))}, {escape_sql_str(s.get('diff_note', ''))}, {escape_sql_str(now_ts)});"
            )

        # 3.4 利益冲突与客户登记库 (kb_client_registry)
        clients = k_data.get("client_registry", {})
        mysql_store.save_client_registry(clients)
        for cname, ctype in clients.items():
            sql_statements.append(
                f"REPLACE INTO kb_client_registry (client_name, client_type, updated_at) "
                f"VALUES ({escape_sql_str(cname)}, {escape_sql_str(ctype)}, {escape_sql_str(now_ts)});"
            )

        # 3.5 特别法与一般法冲突优选规则表 (kb_special_general_relations)
        spec_gen = k_data.get("special_general_relations", [])
        mysql_store.save_special_general_relations(spec_gen)
        for sg in spec_gen:
            sql_statements.append(
                f"REPLACE INTO kb_special_general_relations (special_law, general_law, description, has_deference, updated_at) "
                f"VALUES ({escape_sql_str(sg.get('special_law'))}, {escape_sql_str(sg.get('general_law'))}, "
                f"{escape_sql_str(sg.get('description', ''))}, {1 if sg.get('has_deference', True) else 0}, {escape_sql_str(now_ts)});"
            )

        # 3.6 法规别名与简称表 (kb_law_aliases)
        aliases = k_data.get("law_aliases", {})
        mysql_store.save_law_aliases(aliases)
        for aname, cname in aliases.items():
            sql_statements.append(
                f"REPLACE INTO kb_law_aliases (alias_name, canonical_name, updated_at) "
                f"VALUES ({escape_sql_str(aname)}, {escape_sql_str(cname)}, {escape_sql_str(now_ts)});"
            )

        # 3.7 案由分类规则库 (kb_case_type_rules)
        case_rules = k_data.get("case_type_rules", {})
        for ct, cr in case_rules.items():
            mysql_store.execute(
                "REPLACE INTO kb_case_type_rules (case_type, rules_json, updated_at) VALUES (?, ?, ?)",
                (ct, json.dumps(cr, ensure_ascii=False), now_ts),
            )
            sql_statements.append(
                f"REPLACE INTO kb_case_type_rules (case_type, rules_json, updated_at) "
                f"VALUES ({escape_sql_str(ct)}, {escape_sql_str(json.dumps(cr, ensure_ascii=False))}, {escape_sql_str(now_ts)});"
            )

        # 3.8 确定性表述模式表 (kb_certainty_patterns)
        for pat, rep in k_data.get("certainty_patterns", []):
            mysql_store.execute(
                "INSERT INTO kb_certainty_patterns (pattern, replacement, updated_at) VALUES (?, ?, ?)",
                (pat, rep, now_ts),
            )
            sql_statements.append(
                f"INSERT INTO kb_certainty_patterns (pattern, replacement, updated_at) "
                f"VALUES ({escape_sql_str(pat)}, {escape_sql_str(rep)}, {escape_sql_str(now_ts)});"
            )

        # 3.9 官方新旧法对照集表 (kb_new_old_counterparts)
        for cp in k_data.get("new_old_counterparts", []):
            mysql_store.execute(
                """
                INSERT INTO kb_new_old_counterparts (current_law, current_article, old_law, old_article, relation, switch_date, diff_note, authority, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (cp.get("current_law", ""), cp.get("current_article", ""), cp.get("old_law", ""), cp.get("old_article", ""), cp.get("relation", "one_to_one"), cp.get("switch_date", ""), cp.get("diff_note", ""), cp.get("authority", ""), now_ts),
            )
            sql_statements.append(
                f"INSERT INTO kb_new_old_counterparts (current_law, current_article, old_law, old_article, relation, switch_date, diff_note, authority, updated_at) "
                f"VALUES ({escape_sql_str(cp.get('current_law'))}, {escape_sql_str(cp.get('current_article'))}, "
                f"{escape_sql_str(cp.get('old_law'))}, {escape_sql_str(cp.get('old_article'))}, "
                f"{escape_sql_str(cp.get('relation', 'one_to_one'))}, {escape_sql_str(cp.get('switch_date'))}, "
                f"{escape_sql_str(cp.get('diff_note', ''))}, {escape_sql_str(cp.get('authority', ''))}, {escape_sql_str(now_ts)});"
            )

        # 3.10 程序性条文表 (kb_allowed_procedural_meta_citations)
        for it in k_data.get("allowed_procedural_meta_citations", []):
            if isinstance(it, (list, tuple)) and len(it) >= 2:
                mysql_store.execute(
                    "REPLACE INTO kb_allowed_procedural_meta_citations (law_name, article, updated_at) VALUES (?, ?, ?)",
                    (it[0], it[1], now_ts),
                )
                sql_statements.append(
                    f"REPLACE INTO kb_allowed_procedural_meta_citations (law_name, article, updated_at) "
                    f"VALUES ({escape_sql_str(it[0])}, {escape_sql_str(it[1])}, {escape_sql_str(now_ts)});"
                )

        # 3.11 审核员注册表 (kb_reviewers_registry)
        for rid, rinfo in k_data.get("reviewers_registry", {}).items():
            mysql_store.execute(
                """
                REPLACE INTO kb_reviewers_registry (reviewer_id, name, license_no, role, department, info_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (rid, rinfo.get("name", ""), rinfo.get("license_no", ""), rinfo.get("role", ""), rinfo.get("department", ""), json.dumps(rinfo, ensure_ascii=False), now_ts),
            )
            sql_statements.append(
                f"REPLACE INTO kb_reviewers_registry (reviewer_id, name, license_no, role, department, info_json, updated_at) "
                f"VALUES ({escape_sql_str(rid)}, {escape_sql_str(rinfo.get('name'))}, {escape_sql_str(rinfo.get('license_no'))}, "
                f"{escape_sql_str(rinfo.get('role'))}, {escape_sql_str(rinfo.get('department', ''))}, "
                f"{escape_sql_str(json.dumps(rinfo, ensure_ascii=False))}, {escape_sql_str(now_ts)});"
            )

        logger.info("✅ 成功细粒度拆分迁移知识中心数据 -> MySQL 各专属新表 (同义词/图谱/新旧法/客户库/别名等)")

    # ————————————————————————————————————————————————
    # 4. 迁移 RBAC 角色与用户表
    # ————————————————————————————————————————————————
    rbac_path = "app/data/rbac_config.json"
    if os.path.exists(rbac_path):
        with open(rbac_path, "r", encoding="utf-8") as f:
            rbac_data = json.load(f)
        for r in rbac_data.get("roles", []):
            mysql_store.execute(
                """
                REPLACE INTO sys_rbac_roles (role_id, role_name, description, is_system, menus_json, kb_permissions_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    r.get("role_id"),
                    r.get("role_name"),
                    r.get("description", ""),
                    1 if r.get("is_system") else 0,
                    json.dumps(r.get("menus", []), ensure_ascii=False),
                    json.dumps(r.get("kb_permissions", []), ensure_ascii=False),
                ),
            )
            sql_statements.append(
                f"REPLACE INTO sys_rbac_roles (role_id, role_name, description, is_system, menus_json, kb_permissions_json) "
                f"VALUES ({escape_sql_str(r.get('role_id'))}, {escape_sql_str(r.get('role_name'))}, "
                f"{escape_sql_str(r.get('description', ''))}, {1 if r.get('is_system') else 0}, "
                f"{escape_sql_str(json.dumps(r.get('menus', []), ensure_ascii=False))}, "
                f"{escape_sql_str(json.dumps(r.get('kb_permissions', []), ensure_ascii=False))});"
            )

        for u in rbac_data.get("users", []):
            mysql_store.execute(
                """
                REPLACE INTO sys_rbac_users (user_id, username, name, department, license_no, roles_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    u.get("user_id"),
                    u.get("username"),
                    u.get("name"),
                    u.get("department", ""),
                    u.get("license_no", ""),
                    json.dumps(u.get("roles", []), ensure_ascii=False),
                    u.get("status", "active"),
                    u.get("created_at", now_ts),
                ),
            )
            sql_statements.append(
                f"REPLACE INTO sys_rbac_users (user_id, username, name, department, license_no, roles_json, status, created_at) "
                f"VALUES ({escape_sql_str(u.get('user_id'))}, {escape_sql_str(u.get('username'))}, {escape_sql_str(u.get('name'))}, "
                f"{escape_sql_str(u.get('department', ''))}, {escape_sql_str(u.get('license_no', ''))}, "
                f"{escape_sql_str(json.dumps(u.get('roles', []), ensure_ascii=False))}, "
                f"{escape_sql_str(u.get('status', 'active'))}, {escape_sql_str(u.get('created_at', now_ts))});"
            )
        logger.info("✅ 成功迁移 RBAC 角色与用户数据 -> MySQL sys_rbac_roles / sys_rbac_users")

    # ————————————————————————————————————————————————
    # 5. 迁移预置法规表 sample_laws
    # ————————————————————————————————————————————————
    laws_path = settings.data_path
    laws_data: List[Dict[str, Any]] = []
    if os.path.exists(laws_path):
        with open(laws_path, "r", encoding="utf-8") as f:
            laws_data = json.load(f)
        for l in laws_data:
            mysql_store.execute(
                """
                INSERT INTO sample_laws (
                    law_name, document_number, hierarchy, host_law, host_hierarchy,
                    promulgation_date, effective_date, effective_to, validity, superseded_by,
                    article, paragraph, content, source, category, scope, partially_invalid,
                    invalid_parts, trial, verified, note, article_effective_date, norm_type,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    l.get("law_name", ""),
                    l.get("document_number", ""),
                    l.get("hierarchy", "法律"),
                    l.get("host_law", ""),
                    l.get("host_hierarchy", ""),
                    l.get("promulgation_date", ""),
                    l.get("effective_date", ""),
                    l.get("effective_to", ""),
                    l.get("validity", "现行有效"),
                    l.get("superseded_by", ""),
                    l.get("article", ""),
                    l.get("paragraph", ""),
                    l.get("content", ""),
                    l.get("source", ""),
                    l.get("category", ""),
                    l.get("scope", ""),
                    1 if l.get("partially_invalid") else 0,
                    l.get("invalid_parts", ""),
                    1 if l.get("trial") else 0,
                    1 if l.get("verified", True) else 0,
                    l.get("note", ""),
                    l.get("article_effective_date", ""),
                    l.get("norm_type", "substantive"),
                    json.dumps(l, ensure_ascii=False),
                    now_ts,
                ),
            )
            sql_statements.append(
                f"INSERT INTO sample_laws (law_name, document_number, hierarchy, host_law, host_hierarchy, "
                f"promulgation_date, effective_date, effective_to, validity, superseded_by, article, paragraph, "
                f"content, source, category, scope, partially_invalid, invalid_parts, trial, verified, note, "
                f"article_effective_date, norm_type, metadata_json, created_at) VALUES ("
                f"{escape_sql_str(l.get('law_name', ''))}, {escape_sql_str(l.get('document_number', ''))}, "
                f"{escape_sql_str(l.get('hierarchy', '法律'))}, {escape_sql_str(l.get('host_law', ''))}, "
                f"{escape_sql_str(l.get('host_hierarchy', ''))}, {escape_sql_str(l.get('promulgation_date', ''))}, "
                f"{escape_sql_str(l.get('effective_date', ''))}, {escape_sql_str(l.get('effective_to', ''))}, "
                f"{escape_sql_str(l.get('validity', '现行有效'))}, {escape_sql_str(l.get('superseded_by', ''))}, "
                f"{escape_sql_str(l.get('article', ''))}, {escape_sql_str(l.get('paragraph', ''))}, "
                f"{escape_sql_str(l.get('content', ''))}, {escape_sql_str(l.get('source', ''))}, "
                f"{escape_sql_str(l.get('category', ''))}, {escape_sql_str(l.get('scope', ''))}, "
                f"{1 if l.get('partially_invalid') else 0}, {escape_sql_str(l.get('invalid_parts', ''))}, "
                f"{1 if l.get('trial') else 0}, {1 if l.get('verified', True) else 0}, {escape_sql_str(l.get('note', ''))}, "
                f"{escape_sql_str(l.get('article_effective_date', ''))}, {escape_sql_str(l.get('norm_type', 'substantive'))}, "
                f"{escape_sql_str(json.dumps(l, ensure_ascii=False))}, {escape_sql_str(now_ts)});"
            )
        logger.info(f"✅ 成功迁移预置法规数据: {len(laws_data)} 条 -> MySQL sample_laws")

    # ————————————————————————————————————————————————
    # 6. 迁移 SQLite kb_vectors.db 数据
    # ————————————————————————————————————————————————
    kb_chunks_for_es: List[Dict[str, Any]] = []
    kb_db_path = settings.kb_db_path
    if os.path.exists(kb_db_path):
        kb_conn = sqlite3.connect(kb_db_path)
        kb_conn.row_factory = sqlite3.Row
        cur = kb_conn.cursor()

        cur.execute("SELECT * FROM kb_collections")
        for row in cur.fetchall():
            d = dict(row)
            mysql_store.execute(
                "REPLACE INTO kb_collections (collection_id, name, description, embed_model, created_at) VALUES (?, ?, ?, ?, ?)",
                (d["collection_id"], d["name"], d.get("description", ""), d["embed_model"], d["created_at"]),
            )
            sql_statements.append(
                f"REPLACE INTO kb_collections (collection_id, name, description, embed_model, created_at) "
                f"VALUES ({escape_sql_str(d['collection_id'])}, {escape_sql_str(d['name'])}, "
                f"{escape_sql_str(d.get('description', ''))}, {escape_sql_str(d['embed_model'])}, {escape_sql_str(d['created_at'])});"
            )

        cur.execute("SELECT * FROM law_documents")
        for row in cur.fetchall():
            d = dict(row)
            mysql_store.execute(
                """
                REPLACE INTO law_documents (
                    doc_id, title, kb_type, department, uploader, doc_size, doc_format,
                    split_mode, chunks_count, validity_status, doc_status, valid_until, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["doc_id"], d["title"], d["kb_type"], d.get("department", ""),
                    d.get("uploader", ""), d.get("doc_size", "0 KB"), d.get("doc_format", "TXT"),
                    d.get("split_mode", "article"), d.get("chunks_count", 0),
                    d.get("validity_status", "有效"), d.get("doc_status", "已解析"),
                    d.get("valid_until", ""), d["created_at"],
                ),
            )
            sql_statements.append(
                f"REPLACE INTO law_documents (doc_id, title, kb_type, department, uploader, doc_size, doc_format, "
                f"split_mode, chunks_count, validity_status, doc_status, valid_until, created_at) VALUES ("
                f"{escape_sql_str(d['doc_id'])}, {escape_sql_str(d['title'])}, {escape_sql_str(d['kb_type'])}, "
                f"{escape_sql_str(d.get('department', ''))}, {escape_sql_str(d.get('uploader', ''))}, "
                f"{escape_sql_str(d.get('doc_size', '0 KB'))}, {escape_sql_str(d.get('doc_format', 'TXT'))}, "
                f"{escape_sql_str(d.get('split_mode', 'article'))}, {d.get('chunks_count', 0)}, "
                f"{escape_sql_str(d.get('validity_status', '有效'))}, {escape_sql_str(d.get('doc_status', '已解析'))}, "
                f"{escape_sql_str(d.get('valid_until', ''))}, {escape_sql_str(d['created_at'])});"
            )

        cur.execute("SELECT * FROM law_chunks")
        for row in cur.fetchall():
            d = dict(row)
            mysql_store.execute(
                """
                REPLACE INTO law_chunks (
                    chunk_id, doc_id, law_name, chapter_path, article, paragraph, content,
                    chunk_type, parent_chunk_id, hierarchy, effective_date, effective_to,
                    validity, metadata_json, embedding_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["chunk_id"], d["doc_id"], d["law_name"], d.get("chapter_path", ""),
                    d["article"], d.get("paragraph", ""), d["content"], d["chunk_type"],
                    d.get("parent_chunk_id", ""), d.get("hierarchy", "法律"),
                    d.get("effective_date", ""), d.get("effective_to", ""),
                    d.get("validity", "现行有效"), d["metadata_json"], d.get("embedding_json", ""),
                    d["created_at"],
                ),
            )
            sql_statements.append(
                f"REPLACE INTO law_chunks (chunk_id, doc_id, law_name, chapter_path, article, paragraph, content, "
                f"chunk_type, parent_chunk_id, hierarchy, effective_date, effective_to, validity, metadata_json, "
                f"embedding_json, created_at) VALUES ("
                f"{escape_sql_str(d['chunk_id'])}, {escape_sql_str(d['doc_id'])}, {escape_sql_str(d['law_name'])}, "
                f"{escape_sql_str(d.get('chapter_path', ''))}, {escape_sql_str(d['article'])}, {escape_sql_str(d.get('paragraph', ''))}, "
                f"{escape_sql_str(d['content'])}, {escape_sql_str(d['chunk_type'])}, {escape_sql_str(d.get('parent_chunk_id', ''))}, "
                f"{escape_sql_str(d.get('hierarchy', '法律'))}, {escape_sql_str(d.get('effective_date', ''))}, "
                f"{escape_sql_str(d.get('effective_to', ''))}, {escape_sql_str(d.get('validity', '现行有效'))}, "
                f"{escape_sql_str(d['metadata_json'])}, {escape_sql_str(d.get('embedding_json', ''))}, {escape_sql_str(d['created_at'])});"
            )
            vec = []
            if d.get("embedding_json"):
                try:
                    vec = json.loads(d["embedding_json"])
                except Exception:
                    pass
            kb_chunks_for_es.append(
                {
                    "chunk_id": d["chunk_id"],
                    "doc_id": d["doc_id"],
                    "law_name": d["law_name"],
                    "chapter_path": d.get("chapter_path", ""),
                    "article": d["article"],
                    "paragraph": d.get("paragraph", ""),
                    "content": d["content"],
                    "chunk_type": d["chunk_type"],
                    "parent_chunk_id": d.get("parent_chunk_id", ""),
                    "hierarchy": d.get("hierarchy", "法律"),
                    "effective_date": d.get("effective_date", ""),
                    "effective_to": d.get("effective_to", ""),
                    "validity": d.get("validity", "现行有效"),
                    "content_vector": vec,
                }
            )

        cur.execute("SELECT * FROM law_article_revision_log")
        for row in cur.fetchall():
            d = dict(row)
            mysql_store.execute(
                """
                INSERT INTO law_article_revision_log (
                    law_name, article, paragraph, revision_type, amendment_name,
                    effective_from, effective_to, old_content, new_content, diff_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["law_name"], d["article"], d.get("paragraph", ""), d["revision_type"],
                    d["amendment_name"], d["effective_from"], d.get("effective_to", ""),
                    d.get("old_content", ""), d["new_content"], d.get("diff_summary", ""), d["created_at"],
                ),
            )
            sql_statements.append(
                f"INSERT INTO law_article_revision_log (law_name, article, paragraph, revision_type, amendment_name, "
                f"effective_from, effective_to, old_content, new_content, diff_summary, created_at) VALUES ("
                f"{escape_sql_str(d['law_name'])}, {escape_sql_str(d['article'])}, {escape_sql_str(d.get('paragraph', ''))}, "
                f"{escape_sql_str(d['revision_type'])}, {escape_sql_str(d['amendment_name'])}, {escape_sql_str(d['effective_from'])}, "
                f"{escape_sql_str(d.get('effective_to', ''))}, {escape_sql_str(d.get('old_content', ''))}, "
                f"{escape_sql_str(d['new_content'])}, {escape_sql_str(d.get('diff_summary', ''))}, {escape_sql_str(d['created_at'])});"
            )

        cur.execute("SELECT * FROM kb_tree_nodes")
        for row in cur.fetchall():
            d = dict(row)
            mysql_store.execute(
                """
                REPLACE INTO kb_tree_nodes (
                    node_id, parent_id, name, node_type, description, sort_order, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["node_id"], d.get("parent_id", ""), d["name"], d.get("node_type", "folder"),
                    d.get("description", ""), d.get("sort_order", 0), d["created_at"],
                ),
            )
            sql_statements.append(
                f"REPLACE INTO kb_tree_nodes (node_id, parent_id, name, node_type, description, sort_order, created_at) "
                f"VALUES ({escape_sql_str(d['node_id'])}, {escape_sql_str(d.get('parent_id', ''))}, "
                f"{escape_sql_str(d['name'])}, {escape_sql_str(d.get('node_type', 'folder'))}, "
                f"{escape_sql_str(d.get('description', ''))}, {d.get('sort_order', 0)}, {escape_sql_str(d['created_at'])});"
            )
        kb_conn.close()
        logger.info("✅ 成功迁移 SQLite kb_vectors.db 全量表数据 -> MySQL")

    # ————————————————————————————————————————————————
    # 7. 迁移 SQLite audit_trail.db 数据
    # ————————————————————————————————————————————————
    audit_db_path = "app/data/audit_trail.db"
    if os.path.exists(audit_db_path):
        aud_conn = sqlite3.connect(audit_db_path)
        aud_conn.row_factory = sqlite3.Row
        cur = aud_conn.cursor()

        cur.execute("SELECT * FROM session_snapshots")
        for row in cur.fetchall():
            d = dict(row)
            mysql_store.execute(
                """
                REPLACE INTO session_snapshots (
                    thread_id, updated_at, status, query, act_date, region, confidence, review_level, state_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["thread_id"], d["updated_at"], d["status"], d.get("query", ""),
                    d.get("act_date", ""), d.get("region", ""), d.get("confidence", ""),
                    d.get("review_level", ""), d["state_json"],
                ),
            )
            sql_statements.append(
                f"REPLACE INTO session_snapshots (thread_id, updated_at, status, query, act_date, region, "
                f"confidence, review_level, state_json) VALUES ("
                f"{escape_sql_str(d['thread_id'])}, {escape_sql_str(d['updated_at'])}, {escape_sql_str(d['status'])}, "
                f"{escape_sql_str(d.get('query', ''))}, {escape_sql_str(d.get('act_date', ''))}, "
                f"{escape_sql_str(d.get('region', ''))}, {escape_sql_str(d.get('confidence', ''))}, "
                f"{escape_sql_str(d.get('review_level', ''))}, {escape_sql_str(d['state_json'])});"
            )

        cur.execute("SELECT * FROM audit_logs")
        for row in cur.fetchall():
            d = dict(row)
            mysql_store.execute(
                """
                INSERT INTO audit_logs (
                    thread_id, timestamp, event_type, reviewer_name, license_no, reviewer_role,
                    decision, comments, confidence, review_level, citations_json, draft_answer,
                    final_answer, answer_diff, prev_hash, entry_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["thread_id"], d["timestamp"], d["event_type"], d.get("reviewer_name", ""),
                    d.get("license_no", ""), d.get("reviewer_role", ""), d.get("decision", ""),
                    d.get("comments", ""), d.get("confidence", ""), d.get("review_level", ""),
                    d.get("citations_json", ""), d.get("draft_answer", ""), d.get("final_answer", ""),
                    d.get("answer_diff", ""), d["prev_hash"], d["entry_hash"],
                ),
            )
            sql_statements.append(
                f"INSERT INTO audit_logs (thread_id, timestamp, event_type, reviewer_name, license_no, reviewer_role, "
                f"decision, comments, confidence, review_level, citations_json, draft_answer, final_answer, "
                f"answer_diff, prev_hash, entry_hash) VALUES ("
                f"{escape_sql_str(d['thread_id'])}, {escape_sql_str(d['timestamp'])}, {escape_sql_str(d['event_type'])}, "
                f"{escape_sql_str(d.get('reviewer_name', ''))}, {escape_sql_str(d.get('license_no', ''))}, "
                f"{escape_sql_str(d.get('reviewer_role', ''))}, {escape_sql_str(d.get('decision', ''))}, "
                f"{escape_sql_str(d.get('comments', ''))}, {escape_sql_str(d.get('confidence', ''))}, "
                f"{escape_sql_str(d.get('review_level', ''))}, {escape_sql_str(d.get('citations_json', ''))}, "
                f"{escape_sql_str(d.get('draft_answer', ''))}, {escape_sql_str(d.get('final_answer', ''))}, "
                f"{escape_sql_str(d.get('answer_diff', ''))}, {escape_sql_str(d['prev_hash'])}, {escape_sql_str(d['entry_hash'])});"
            )

        cur.execute("SELECT * FROM rule_gaps")
        for row in cur.fetchall():
            d = dict(row)
            mysql_store.execute(
                """
                INSERT INTO rule_gaps (
                    thread_id, timestamp, mode, confidence, review_level, reason, verify_actions_json, query
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["thread_id"], d["timestamp"], d.get("mode", ""), d.get("confidence", ""),
                    d.get("review_level", ""), d.get("reason", ""), d.get("verify_actions_json", ""),
                    d.get("query", ""),
                ),
            )
            sql_statements.append(
                f"INSERT INTO rule_gaps (thread_id, timestamp, mode, confidence, review_level, reason, "
                f"verify_actions_json, query) VALUES ("
                f"{escape_sql_str(d['thread_id'])}, {escape_sql_str(d['timestamp'])}, {escape_sql_str(d.get('mode', ''))}, "
                f"{escape_sql_str(d.get('confidence', ''))}, {escape_sql_str(d.get('review_level', ''))}, "
                f"{escape_sql_str(d.get('reason', ''))}, {escape_sql_str(d.get('verify_actions_json', ''))}, "
                f"{escape_sql_str(d.get('query', ''))});"
            )

        cur.execute("SELECT * FROM rule_proposals")
        for row in cur.fetchall():
            d = dict(row)
            mysql_store.execute(
                """
                INSERT INTO rule_proposals (
                    thread_id, timestamp, proposal_type, proposal_key, proposal_data_json,
                    status, reviewer_name, review_notes, adopted_at, query
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["thread_id"], d["timestamp"], d["proposal_type"], d.get("proposal_key", ""),
                    d["proposal_data_json"], d.get("status", "PENDING"), d.get("reviewer_name", ""),
                    d.get("review_notes", ""), d.get("adopted_at", ""), d.get("query", ""),
                ),
            )
            sql_statements.append(
                f"INSERT INTO rule_proposals (thread_id, timestamp, proposal_type, proposal_key, proposal_data_json, "
                f"status, reviewer_name, review_notes, adopted_at, query) VALUES ("
                f"{escape_sql_str(d['thread_id'])}, {escape_sql_str(d['timestamp'])}, {escape_sql_str(d['proposal_type'])}, "
                f"{escape_sql_str(d.get('proposal_key', ''))}, {escape_sql_str(d['proposal_data_json'])}, "
                f"{escape_sql_str(d.get('status', 'PENDING'))}, {escape_sql_str(d.get('reviewer_name', ''))}, "
                f"{escape_sql_str(d.get('review_notes', ''))}, {escape_sql_str(d.get('adopted_at', ''))}, "
                f"{escape_sql_str(d.get('query', ''))});"
            )
        aud_conn.close()
        logger.info("✅ 成功迁移 SQLite audit_trail.db 全量表数据 -> MySQL")

    # ————————————————————————————————————————————————
    # 8. 重新导出标准 docker/mysql/init.sql 引导脚本
    # ————————————————————————————————————————————————
    sql_statements.append("\nSET FOREIGN_KEY_CHECKS = 1;\n")
    init_sql_path = "docker/mysql/init.sql"
    os.makedirs(os.path.dirname(init_sql_path), exist_ok=True)
    with open(init_sql_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sql_statements))
    logger.info(f"✅ 成功重新生成规范化 MySQL 初始化脚本: {init_sql_path} (语句总数: {len(sql_statements)})")

    # ————————————————————————————————————————————————
    # 9. 检查并同步 Elasticsearch 8.x 向量与粗排索引
    # ————————————————————————————————————————————————
    if es_vector_store.ping():
        embedder = BGEM3Embedder()
        all_es_chunks = list(kb_chunks_for_es)
        law_texts = [f"{l.get('law_name', '')} {l.get('article', '')} {l.get('content', '')}" for l in laws_data]
        vectors, _ = embedder.embed_texts(law_texts)
        for l, vec in zip(laws_data, vectors):
            all_es_chunks.append(
                {
                    "chunk_id": f"sample_{l.get('law_name', '')}_{l.get('article', '')}",
                    "doc_id": "sample_laws",
                    "law_name": l.get("law_name", ""),
                    "chapter_path": "",
                    "article": l.get("article", ""),
                    "paragraph": l.get("paragraph", ""),
                    "content": l.get("content", ""),
                    "chunk_type": "article",
                    "parent_chunk_id": "",
                    "hierarchy": l.get("hierarchy", "法律"),
                    "effective_date": l.get("effective_date", ""),
                    "effective_to": l.get("effective_to", ""),
                    "validity": l.get("validity", "现行有效"),
                    "content_vector": vec,
                }
            )

        dims = len(all_es_chunks[0]["content_vector"]) if all_es_chunks and all_es_chunks[0]["content_vector"] else 128
        es_vector_store.init_index(dims=dims)
        indexed = es_vector_store.index_chunks(all_es_chunks)
        logger.info(f"✅ 成功将 {indexed} 条法规文档切片及向量数据写入 Elasticsearch 8.x")
    else:
        logger.info("ℹ️ Elasticsearch 当前未运行或不可达，数据已在 MySQL 各专用表中规范化入库完毕。")

    logger.info("====== 细粒度拆分迁移全部圆满完成！ ======")


if __name__ == "__main__":
    migrate_all()
