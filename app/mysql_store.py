"""MySQL 8.x 关系型数据存储与交互管理引擎：
支持系统细粒度解耦存储（llm_config、kb_legal_synonyms、kb_cross_references、
kb_article_successions、kb_client_registry 等专用表）、法规知识库及审计账本。
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

try:
    import pymysql
    import pymysql.cursors
    HAS_PYMYSQL = True
except ImportError:
    pymysql = None
    HAS_PYMYSQL = False


class MySQLStore:
    """企业级关系存储引擎，支持 MySQL 8.x 细粒度表结构交互与本地兼容代理。"""

    _instance: Optional["MySQLStore"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        if getattr(self, "_initialized", False):
            return

        self.host = host or settings.mysql_host
        self.port = port or settings.mysql_port
        self.user = user or settings.mysql_user
        self.password = password or settings.mysql_password
        self.database = database or settings.mysql_database

        self._fallback_db_path = os.getenv("MYSQL_FALLBACK_SQLITE", "app/data/mysql_mirror.db")
        self._is_mysql_active: Optional[bool] = None
        self._thread_local = threading.local()
        self._initialized = True

    def ping(self) -> bool:
        """探测 MySQL 服务是否真实可用且能成功认证。"""
        if not HAS_PYMYSQL:
            self._is_mysql_active = False
            return False

        try:
            conn = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset="utf8mb4",
                connect_timeout=2,
            )
            conn.close()
            self._is_mysql_active = True
            return True
        except Exception as e:
            logger.debug(f"[MySQLStore] MySQL 连接探测失败 ({self.host}:{self.port}/{self.database}): {e}")
            self._is_mysql_active = False
            return False

    @property
    def is_mysql(self) -> bool:
        if self._is_mysql_active is None:
            self.ping()
        return bool(self._is_mysql_active)

    def get_connection(self):
        """获取底层数据库物理连接。"""
        if self.is_mysql and HAS_PYMYSQL:
            return pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                autocommit=True,
            )
        else:
            parent = os.path.dirname(os.path.abspath(self._fallback_db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(self._fallback_db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn

    def init_schema(self) -> bool:
        """初始化全量数据表结构（兼容 MySQL 8.x 与 SQLite 语法）。"""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            if self.is_mysql:
                # 1. 大模型与向量参数专门配置表 (llm_config)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS llm_config (
                        id INT PRIMARY KEY DEFAULT 1,
                        mock TINYINT(1) DEFAULT 1,
                        llm_provider VARCHAR(64) NOT NULL DEFAULT 'qwen',
                        llm_api_base VARCHAR(255) NOT NULL,
                        llm_model VARCHAR(128) NOT NULL,
                        llm_api_key VARCHAR(255) DEFAULT '',
                        llm_temperature FLOAT DEFAULT 0.0,
                        llm_timeout FLOAT DEFAULT 30.0,
                        llm_max_retries INT DEFAULT 2,
                        embed_enabled TINYINT(1) DEFAULT 1,
                        embed_api_base VARCHAR(255) DEFAULT '',
                        embed_model VARCHAR(128) DEFAULT 'BAAI/bge-m3',
                        embed_api_key VARCHAR(255) DEFAULT '',
                        embed_dim INT DEFAULT 1024,
                        hybrid_dense_weight FLOAT DEFAULT 0.45,
                        strict_review_auth TINYINT(1) DEFAULT 0,
                        top_k INT DEFAULT 5,
                        platform_credentials_json LONGTEXT,
                        updated_at VARCHAR(64) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 2. 法律同义词表 (kb_legal_synonyms)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_legal_synonyms (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        colloquial_term VARCHAR(128) NOT NULL UNIQUE,
                        legal_terms TEXT NOT NULL,
                        updated_at VARCHAR(64) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 3. 法条引用图谱表 (kb_cross_references)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_cross_references (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        source_law VARCHAR(255) NOT NULL,
                        source_article VARCHAR(64) NOT NULL,
                        targets_json LONGTEXT NOT NULL,
                        updated_at VARCHAR(64) NOT NULL,
                        UNIQUE KEY uk_cross_source (source_law, source_article)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 4. 新旧法条文级演进与对照表 (kb_article_successions)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_article_successions (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        old_law VARCHAR(255) NOT NULL,
                        old_article VARCHAR(64) NOT NULL,
                        new_law VARCHAR(255) NOT NULL,
                        new_article VARCHAR(64) NOT NULL,
                        switch_date VARCHAR(32) NOT NULL,
                        diff_note TEXT,
                        updated_at VARCHAR(64) NOT NULL,
                        INDEX idx_succ_old (old_law, old_article),
                        INDEX idx_succ_new (new_law, new_article)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 5. 利益冲突与客户登记表 (kb_client_registry)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_client_registry (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        client_name VARCHAR(255) NOT NULL UNIQUE,
                        client_type VARCHAR(128) NOT NULL DEFAULT '本所常年法律顾问客户',
                        conflict_rules TEXT,
                        updated_at VARCHAR(64) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 6. 特别法与一般法冲突优选规则表 (kb_special_general_relations)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_special_general_relations (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        special_law VARCHAR(255) NOT NULL,
                        general_law VARCHAR(255) NOT NULL,
                        description TEXT,
                        has_deference TINYINT(1) DEFAULT 1,
                        updated_at VARCHAR(64) NOT NULL,
                        UNIQUE KEY uk_spec_gen (special_law, general_law)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 7. 法规别名与简称映射表 (kb_law_aliases)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_law_aliases (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        alias_name VARCHAR(128) NOT NULL UNIQUE,
                        canonical_name VARCHAR(255) NOT NULL,
                        updated_at VARCHAR(64) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 8. 案由分类规则与事实要件库 (kb_case_type_rules)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_case_type_rules (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        case_type VARCHAR(128) NOT NULL UNIQUE,
                        rules_json LONGTEXT NOT NULL,
                        updated_at VARCHAR(64) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 9. 确定性表述模式表 (kb_certainty_patterns)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_certainty_patterns (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        pattern VARCHAR(255) NOT NULL,
                        replacement VARCHAR(255) NOT NULL,
                        updated_at VARCHAR(64) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 10. 官方立法新旧法对照集表 (kb_new_old_counterparts)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_new_old_counterparts (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        current_law VARCHAR(255) NOT NULL,
                        current_article VARCHAR(64) NOT NULL,
                        old_law VARCHAR(255) NOT NULL,
                        old_article VARCHAR(64) NOT NULL,
                        relation VARCHAR(64) DEFAULT 'one_to_one',
                        switch_date VARCHAR(32) NOT NULL,
                        diff_note TEXT,
                        authority VARCHAR(255) DEFAULT '全国人大常委会法制工作委员会',
                        updated_at VARCHAR(64) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 11. 程序性元引用表 (kb_allowed_procedural_meta_citations)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_allowed_procedural_meta_citations (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        law_name VARCHAR(255) NOT NULL,
                        article VARCHAR(64) NOT NULL,
                        updated_at VARCHAR(64) NOT NULL,
                        UNIQUE KEY uk_meta_cite (law_name, article)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 12. 审核专家与律师注册表 (kb_reviewers_registry)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_reviewers_registry (
                        reviewer_id VARCHAR(64) PRIMARY KEY,
                        name VARCHAR(128) NOT NULL,
                        license_no VARCHAR(128) NOT NULL,
                        role VARCHAR(64) NOT NULL,
                        department VARCHAR(128) DEFAULT '',
                        info_json TEXT,
                        updated_at VARCHAR(64) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 13. 预置基础法规表 (sample_laws)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sample_laws (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        law_name VARCHAR(255) NOT NULL,
                        document_number VARCHAR(128) DEFAULT '',
                        hierarchy VARCHAR(64) NOT NULL,
                        host_law VARCHAR(255) DEFAULT '',
                        host_hierarchy VARCHAR(64) DEFAULT '',
                        promulgation_date VARCHAR(32) DEFAULT '',
                        effective_date VARCHAR(32) DEFAULT '',
                        effective_to VARCHAR(32) DEFAULT '',
                        validity VARCHAR(32) NOT NULL DEFAULT '现行有效',
                        superseded_by VARCHAR(255) DEFAULT '',
                        article VARCHAR(64) NOT NULL,
                        paragraph VARCHAR(64) DEFAULT '',
                        content LONGTEXT NOT NULL,
                        source VARCHAR(128) DEFAULT '',
                        category VARCHAR(64) DEFAULT '',
                        scope VARCHAR(64) DEFAULT '',
                        partially_invalid TINYINT(1) DEFAULT 0,
                        invalid_parts VARCHAR(255) DEFAULT '',
                        trial TINYINT(1) DEFAULT 0,
                        verified TINYINT(1) DEFAULT 1,
                        note VARCHAR(255) DEFAULT '',
                        article_effective_date VARCHAR(32) DEFAULT '',
                        norm_type VARCHAR(64) DEFAULT 'substantive',
                        metadata_json LONGTEXT,
                        created_at VARCHAR(64) DEFAULT '',
                        INDEX idx_sample_law_art (law_name, article)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 14. RBAC 角色表
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sys_rbac_roles (
                        role_id VARCHAR(64) PRIMARY KEY,
                        role_name VARCHAR(128) NOT NULL,
                        description TEXT,
                        is_system TINYINT(1) DEFAULT 0,
                        menus_json TEXT NOT NULL,
                        kb_permissions_json TEXT NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 15. RBAC 用户表
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sys_rbac_users (
                        user_id VARCHAR(64) PRIMARY KEY,
                        username VARCHAR(64) NOT NULL UNIQUE,
                        name VARCHAR(128) NOT NULL,
                        dept_id VARCHAR(64) DEFAULT 'dept_hq',
                        department VARCHAR(128) DEFAULT '',
                        post_title VARCHAR(64) DEFAULT '',
                        license_no VARCHAR(128) DEFAULT '',
                        roles_json TEXT NOT NULL,
                        status VARCHAR(32) DEFAULT 'active',
                        created_at VARCHAR(64) NOT NULL,
                        INDEX idx_user_dept (dept_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                try:
                    cur.execute("ALTER TABLE sys_rbac_users ADD COLUMN dept_id VARCHAR(64) DEFAULT 'dept_hq'")
                except Exception:
                    pass
                try:
                    cur.execute("ALTER TABLE sys_rbac_users ADD COLUMN post_title VARCHAR(64) DEFAULT ''")
                except Exception:
                    pass

                # 15.1 组织机构部门表 (三级树状结构: 总部 -> 二级部室 -> 三级科室)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sys_org_departments (
                        dept_id VARCHAR(64) PRIMARY KEY,
                        dept_name VARCHAR(128) NOT NULL,
                        parent_id VARCHAR(64) DEFAULT NULL,
                        level INT NOT NULL DEFAULT 1,
                        dept_path VARCHAR(256) NOT NULL,
                        sort_order INT NOT NULL DEFAULT 0,
                        leader VARCHAR(64) DEFAULT '',
                        phone VARCHAR(32) DEFAULT '',
                        status VARCHAR(16) NOT NULL DEFAULT 'active',
                        created_at VARCHAR(64) NOT NULL,
                        updated_at VARCHAR(64) NOT NULL,
                        INDEX idx_dept_parent (parent_id),
                        INDEX idx_dept_level (level)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 16. 知识库集合表
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_collections (
                        collection_id VARCHAR(128) PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        description TEXT,
                        embed_model VARCHAR(128) NOT NULL,
                        created_at VARCHAR(64) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 16.1 法规条文旧式向量兼容表 (kb_articles)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_articles (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        collection_id VARCHAR(128) NOT NULL,
                        law_name VARCHAR(255) NOT NULL,
                        article VARCHAR(128) NOT NULL,
                        paragraph VARCHAR(128) DEFAULT '',
                        hierarchy VARCHAR(64) NOT NULL,
                        validity VARCHAR(64) NOT NULL,
                        effective_date VARCHAR(64) DEFAULT '',
                        effective_to VARCHAR(64) DEFAULT '',
                        category VARCHAR(64) DEFAULT '',
                        content LONGTEXT NOT NULL,
                        metadata_json LONGTEXT NOT NULL,
                        embedding_json LONGTEXT NOT NULL,
                        embed_backend VARCHAR(128) NOT NULL,
                        updated_at VARCHAR(64) NOT NULL,
                        UNIQUE KEY uq_kb_article (law_name, article, paragraph)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 17. 法规文档总表
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS law_documents (
                        doc_id VARCHAR(128) PRIMARY KEY,
                        title VARCHAR(255) NOT NULL,
                        kb_type VARCHAR(128) NOT NULL DEFAULT '外规知识库/法律法规',
                        department VARCHAR(128) DEFAULT '法律合规部',
                        uploader VARCHAR(128) DEFAULT '系统管理员',
                        doc_size VARCHAR(64) DEFAULT '0 KB',
                        doc_format VARCHAR(32) DEFAULT 'TXT',
                        split_mode VARCHAR(64) NOT NULL DEFAULT 'article',
                        chunks_count INT DEFAULT 0,
                        validity_status VARCHAR(64) DEFAULT '有效',
                        doc_status VARCHAR(64) DEFAULT '已解析',
                        valid_until VARCHAR(64) DEFAULT '',
                        created_at VARCHAR(64) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 18. 法规分块表
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS law_chunks (
                        chunk_id VARCHAR(128) PRIMARY KEY,
                        doc_id VARCHAR(128) NOT NULL,
                        law_name VARCHAR(255) NOT NULL,
                        chapter_path TEXT,
                        article VARCHAR(64) NOT NULL,
                        paragraph VARCHAR(64) DEFAULT '',
                        content LONGTEXT NOT NULL,
                        chunk_type VARCHAR(64) NOT NULL,
                        parent_chunk_id VARCHAR(128) DEFAULT '',
                        hierarchy VARCHAR(64) NOT NULL,
                        effective_date VARCHAR(32) DEFAULT '',
                        effective_to VARCHAR(32) DEFAULT '',
                        validity VARCHAR(32) NOT NULL,
                        metadata_json LONGTEXT NOT NULL,
                        embedding_json LONGTEXT,
                        created_at VARCHAR(64) NOT NULL,
                        INDEX idx_chunk_doc (doc_id),
                        INDEX idx_chunk_law (law_name)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 19. 修正案与生命周期追踪表
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS law_article_revision_log (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        law_name VARCHAR(255) NOT NULL,
                        article VARCHAR(64) NOT NULL,
                        paragraph VARCHAR(64) DEFAULT '',
                        revision_type VARCHAR(64) NOT NULL,
                        amendment_name VARCHAR(255) NOT NULL,
                        effective_from VARCHAR(32) NOT NULL,
                        effective_to VARCHAR(32) DEFAULT '',
                        old_content LONGTEXT,
                        new_content LONGTEXT NOT NULL,
                        diff_summary TEXT,
                        created_at VARCHAR(64) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 20. 知识库目录树节点表 (支持主责部门管控与可见性范围)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_tree_nodes (
                        node_id VARCHAR(128) PRIMARY KEY,
                        parent_id VARCHAR(128) NOT NULL DEFAULT '',
                        name VARCHAR(255) NOT NULL,
                        node_type VARCHAR(64) NOT NULL DEFAULT 'folder',
                        owner_dept_id VARCHAR(128) DEFAULT 'dept_hq',
                        visibility VARCHAR(32) DEFAULT 'dept',
                        description TEXT,
                        sort_order INT DEFAULT 0,
                        created_at VARCHAR(64) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                try:
                    cur.execute("ALTER TABLE kb_tree_nodes ADD COLUMN owner_dept_id VARCHAR(128) DEFAULT 'dept_hq'")
                except Exception:
                    pass
                try:
                    cur.execute("ALTER TABLE kb_tree_nodes ADD COLUMN visibility VARCHAR(32) DEFAULT 'dept'")
                except Exception:
                    pass
                # 21. 会话快照存储表
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_snapshots (
                        thread_id VARCHAR(128) PRIMARY KEY,
                        updated_at VARCHAR(64) NOT NULL,
                        status VARCHAR(64) NOT NULL,
                        query TEXT,
                        act_date VARCHAR(32),
                        region VARCHAR(64),
                        confidence VARCHAR(32),
                        review_level VARCHAR(64),
                        state_json LONGTEXT NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 22. 防篡改哈希链审计日志表
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        thread_id VARCHAR(128) NOT NULL,
                        timestamp VARCHAR(64) NOT NULL,
                        event_type VARCHAR(64) NOT NULL,
                        reviewer_name VARCHAR(128),
                        license_no VARCHAR(128),
                        reviewer_role VARCHAR(64),
                        decision VARCHAR(64),
                        comments TEXT,
                        confidence VARCHAR(32),
                        review_level VARCHAR(64),
                        citations_json LONGTEXT,
                        draft_answer LONGTEXT,
                        final_answer LONGTEXT,
                        answer_diff LONGTEXT,
                        prev_hash VARCHAR(128) NOT NULL,
                        entry_hash VARCHAR(128) NOT NULL,
                        INDEX idx_audit_thread (thread_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 23. 规则缺口归集表
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rule_gaps (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        thread_id VARCHAR(128) NOT NULL,
                        timestamp VARCHAR(64) NOT NULL,
                        mode VARCHAR(64),
                        confidence VARCHAR(32),
                        review_level VARCHAR(64),
                        reason TEXT,
                        verify_actions_json TEXT,
                        query TEXT,
                        INDEX idx_gap_thread (thread_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
                # 24. 规则提案表
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rule_proposals (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        thread_id VARCHAR(128) NOT NULL,
                        timestamp VARCHAR(64) NOT NULL,
                        proposal_type VARCHAR(64) NOT NULL,
                        proposal_key VARCHAR(128),
                        proposal_data_json LONGTEXT NOT NULL,
                        status VARCHAR(64) DEFAULT 'PENDING',
                        reviewer_name VARCHAR(128),
                        review_notes TEXT,
                        adopted_at VARCHAR(64),
                        query TEXT
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                )
            else:
                # SQLite 语法初始化 fallback
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS llm_config (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        mock INTEGER DEFAULT 1,
                        llm_provider TEXT NOT NULL DEFAULT 'qwen',
                        llm_api_base TEXT NOT NULL,
                        llm_model TEXT NOT NULL,
                        llm_api_key TEXT DEFAULT '',
                        llm_temperature REAL DEFAULT 0.0,
                        llm_timeout REAL DEFAULT 30.0,
                        llm_max_retries INTEGER DEFAULT 2,
                        embed_enabled INTEGER DEFAULT 1,
                        embed_api_base TEXT DEFAULT '',
                        embed_model TEXT DEFAULT 'BAAI/bge-m3',
                        embed_api_key TEXT DEFAULT '',
                        embed_dim INTEGER DEFAULT 1024,
                        hybrid_dense_weight REAL DEFAULT 0.45,
                        strict_review_auth INTEGER DEFAULT 0,
                        top_k INTEGER DEFAULT 5,
                        platform_credentials_json TEXT,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_legal_synonyms (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        colloquial_term TEXT NOT NULL UNIQUE,
                        legal_terms TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_cross_references (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        source_law TEXT NOT NULL,
                        source_article TEXT NOT NULL,
                        targets_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(source_law, source_article)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_article_successions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        old_law TEXT NOT NULL,
                        old_article TEXT NOT NULL,
                        new_law TEXT NOT NULL,
                        new_article TEXT NOT NULL,
                        switch_date TEXT NOT NULL,
                        diff_note TEXT,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_client_registry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        client_name TEXT NOT NULL UNIQUE,
                        client_type TEXT NOT NULL DEFAULT '本所常年法律顾问客户',
                        conflict_rules TEXT,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_special_general_relations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        special_law TEXT NOT NULL,
                        general_law TEXT NOT NULL,
                        description TEXT,
                        has_deference INTEGER DEFAULT 1,
                        updated_at TEXT NOT NULL,
                        UNIQUE(special_law, general_law)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_law_aliases (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alias_name TEXT NOT NULL UNIQUE,
                        canonical_name TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_case_type_rules (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        case_type TEXT NOT NULL UNIQUE,
                        rules_json TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_certainty_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern TEXT NOT NULL,
                        replacement TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_new_old_counterparts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        current_law TEXT NOT NULL,
                        current_article TEXT NOT NULL,
                        old_law TEXT NOT NULL,
                        old_article TEXT NOT NULL,
                        relation TEXT DEFAULT 'one_to_one',
                        switch_date TEXT NOT NULL,
                        diff_note TEXT,
                        authority TEXT DEFAULT '全国人大常委会法制工作委员会',
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_allowed_procedural_meta_citations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        law_name TEXT NOT NULL,
                        article TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(law_name, article)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_reviewers_registry (
                        reviewer_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        license_no TEXT NOT NULL,
                        role TEXT NOT NULL,
                        department TEXT DEFAULT '',
                        info_json TEXT,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sample_laws (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        law_name TEXT NOT NULL,
                        document_number TEXT DEFAULT '',
                        hierarchy TEXT NOT NULL,
                        host_law TEXT DEFAULT '',
                        host_hierarchy TEXT DEFAULT '',
                        promulgation_date TEXT DEFAULT '',
                        effective_date TEXT DEFAULT '',
                        effective_to TEXT DEFAULT '',
                        validity TEXT NOT NULL DEFAULT '现行有效',
                        superseded_by TEXT DEFAULT '',
                        article TEXT NOT NULL,
                        paragraph TEXT DEFAULT '',
                        content TEXT NOT NULL,
                        source TEXT DEFAULT '',
                        category TEXT DEFAULT '',
                        scope TEXT DEFAULT '',
                        partially_invalid INTEGER DEFAULT 0,
                        invalid_parts TEXT DEFAULT '',
                        trial INTEGER DEFAULT 0,
                        verified INTEGER DEFAULT 1,
                        note TEXT DEFAULT '',
                        article_effective_date TEXT DEFAULT '',
                        norm_type TEXT DEFAULT 'substantive',
                        metadata_json TEXT,
                        created_at TEXT DEFAULT ''
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sys_rbac_roles (
                        role_id TEXT PRIMARY KEY,
                        role_name TEXT NOT NULL,
                        description TEXT,
                        is_system INTEGER DEFAULT 0,
                        menus_json TEXT NOT NULL,
                        kb_permissions_json TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sys_rbac_users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT NOT NULL UNIQUE,
                        name TEXT NOT NULL,
                        dept_id TEXT DEFAULT 'dept_hq',
                        department TEXT DEFAULT '',
                        post_title TEXT DEFAULT '',
                        license_no TEXT DEFAULT '',
                        roles_json TEXT NOT NULL,
                        status TEXT DEFAULT 'active',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                try:
                    cur.execute("ALTER TABLE sys_rbac_users ADD COLUMN dept_id TEXT DEFAULT 'dept_hq'")
                except Exception:
                    pass
                try:
                    cur.execute("ALTER TABLE sys_rbac_users ADD COLUMN post_title TEXT DEFAULT ''")
                except Exception:
                    pass

                # 15.1 组织机构部门表 (SQLite 镜像)
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sys_org_departments (
                        dept_id TEXT PRIMARY KEY,
                        dept_name TEXT NOT NULL,
                        parent_id TEXT DEFAULT NULL,
                        level INTEGER NOT NULL DEFAULT 1,
                        dept_path TEXT NOT NULL,
                        sort_order INTEGER NOT NULL DEFAULT 0,
                        leader TEXT DEFAULT '',
                        phone TEXT DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'active',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_collections (
                        collection_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        description TEXT,
                        embed_model TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_articles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        collection_id TEXT NOT NULL,
                        law_name TEXT NOT NULL,
                        article TEXT NOT NULL,
                        paragraph TEXT DEFAULT '',
                        hierarchy TEXT NOT NULL,
                        validity TEXT NOT NULL,
                        effective_date TEXT DEFAULT '',
                        effective_to TEXT DEFAULT '',
                        category TEXT DEFAULT '',
                        content TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        embed_backend TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(law_name, article, paragraph)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS law_documents (
                        doc_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        kb_type TEXT NOT NULL DEFAULT '外规知识库/法律法规',
                        department TEXT DEFAULT '法律合规部',
                        uploader TEXT DEFAULT '系统管理员',
                        doc_size TEXT DEFAULT '0 KB',
                        doc_format TEXT DEFAULT 'TXT',
                        split_mode TEXT NOT NULL DEFAULT 'article',
                        chunks_count INTEGER DEFAULT 0,
                        validity_status TEXT DEFAULT '有效',
                        doc_status TEXT DEFAULT '已解析',
                        valid_until TEXT DEFAULT '',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS law_chunks (
                        chunk_id TEXT PRIMARY KEY,
                        doc_id TEXT NOT NULL,
                        law_name TEXT NOT NULL,
                        chapter_path TEXT DEFAULT '',
                        article TEXT NOT NULL,
                        paragraph TEXT DEFAULT '',
                        content TEXT NOT NULL,
                        chunk_type TEXT NOT NULL,
                        parent_chunk_id TEXT DEFAULT '',
                        hierarchy TEXT NOT NULL,
                        effective_date TEXT DEFAULT '',
                        effective_to TEXT DEFAULT '',
                        validity TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        embedding_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS law_article_revision_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        law_name TEXT NOT NULL,
                        article TEXT NOT NULL,
                        paragraph TEXT DEFAULT '',
                        revision_type TEXT NOT NULL,
                        amendment_name TEXT NOT NULL,
                        effective_from TEXT NOT NULL,
                        effective_to TEXT DEFAULT '',
                        old_content TEXT DEFAULT '',
                        new_content TEXT NOT NULL,
                        diff_summary TEXT DEFAULT '',
                        created_at TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_tree_nodes (
                        node_id TEXT PRIMARY KEY,
                        parent_id TEXT NOT NULL DEFAULT '',
                        name TEXT NOT NULL,
                        node_type TEXT NOT NULL DEFAULT 'folder',
                        owner_dept_id TEXT DEFAULT 'dept_hq',
                        visibility TEXT DEFAULT 'dept',
                        description TEXT DEFAULT '',
                        sort_order INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                try:
                    cur.execute("ALTER TABLE kb_tree_nodes ADD COLUMN owner_dept_id TEXT DEFAULT 'dept_hq'")
                except Exception:
                    pass
                try:
                    cur.execute("ALTER TABLE kb_tree_nodes ADD COLUMN visibility TEXT DEFAULT 'dept'")
                except Exception:
                    pass
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS session_snapshots (
                        thread_id TEXT PRIMARY KEY,
                        updated_at TEXT NOT NULL,
                        status TEXT NOT NULL,
                        query TEXT,
                        act_date TEXT,
                        region TEXT,
                        confidence TEXT,
                        review_level TEXT,
                        state_json TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        reviewer_name TEXT,
                        license_no TEXT,
                        reviewer_role TEXT,
                        decision TEXT,
                        comments TEXT,
                        confidence TEXT,
                        review_level TEXT,
                        citations_json TEXT,
                        draft_answer TEXT,
                        final_answer TEXT,
                        answer_diff TEXT,
                        prev_hash TEXT NOT NULL,
                        entry_hash TEXT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rule_gaps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        mode TEXT,
                        confidence TEXT,
                        review_level TEXT,
                        reason TEXT,
                        verify_actions_json TEXT,
                        query TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS rule_proposals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        thread_id TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        proposal_type TEXT NOT NULL,
                        proposal_key TEXT,
                        proposal_data_json TEXT NOT NULL,
                        status TEXT DEFAULT 'PENDING',
                        reviewer_name TEXT,
                        review_notes TEXT,
                        adopted_at TEXT,
                        query TEXT
                    )
                    """
                )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"[MySQLStore] 初始化数据库结构失败: {e}")
            return False
        finally:
            cur.close()
            conn.close()

    def execute(self, sql: str, params: Optional[Sequence[Any] | Dict[str, Any]] = None) -> int:
        """执行 INSERT / UPDATE / DELETE 语句。"""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            adjusted_sql = sql
            if self.is_mysql:
                adjusted_sql = adjusted_sql.replace("?", "%s")
            else:
                adjusted_sql = adjusted_sql.replace("%s", "?")

            cur.execute(adjusted_sql, params or ())
            if not self.is_mysql:
                conn.commit()
            return cur.rowcount
        finally:
            cur.close()
            conn.close()

    def query(self, sql: str, params: Optional[Sequence[Any] | Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """执行 SELECT 语句，返回字典列表。"""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            adjusted_sql = sql
            if self.is_mysql:
                adjusted_sql = adjusted_sql.replace("?", "%s")
            else:
                adjusted_sql = adjusted_sql.replace("%s", "?")

            cur.execute(adjusted_sql, params or ())
            rows = cur.fetchall()
            if self.is_mysql:
                return list(rows)
            else:
                return [dict(r) for r in rows]
        finally:
            cur.close()
            conn.close()

    def query_one(self, sql: str, params: Optional[Sequence[Any] | Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """查询单行记录。"""
        res = self.query(sql, params)
        return res[0] if res else None

    # ————————————————————————————————————————————————
    # 1. 大模型运行时与向量配置表 (llm_config) 专属维护接口
    # ————————————————————————————————————————————————
    def get_llm_config(self) -> Optional[Dict[str, Any]]:
        """从专用表 llm_config 中载入大模型与向量参数。"""
        row = self.query_one("SELECT * FROM llm_config WHERE id = 1")
        if not row:
            return None
        cfg = dict(row)
        if "platform_credentials_json" in cfg and cfg["platform_credentials_json"]:
            try:
                cfg["platform_credentials"] = json.loads(cfg["platform_credentials_json"])
            except Exception:
                cfg["platform_credentials"] = {}
        cfg["mock"] = bool(cfg.get("mock", True))
        cfg["embed_enabled"] = bool(cfg.get("embed_enabled", True))
        cfg["strict_review_auth"] = bool(cfg.get("strict_review_auth", False))
        return cfg

    def save_llm_config(self, cfg: Dict[str, Any]) -> bool:
        """持久化保存大模型与向量配置至 llm_config 专用表。"""
        now_ts = datetime.now(timezone.utc).isoformat()
        plat_json = json.dumps(cfg.get("platform_credentials", {}), ensure_ascii=False)
        sql = """
            REPLACE INTO llm_config (
                id, mock, llm_provider, llm_api_base, llm_model, llm_api_key,
                llm_temperature, llm_timeout, llm_max_retries, embed_enabled,
                embed_api_base, embed_model, embed_api_key, embed_dim,
                hybrid_dense_weight, strict_review_auth, top_k, platform_credentials_json, updated_at
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            1 if cfg.get("mock") else 0,
            str(cfg.get("llm_provider", "qwen")),
            str(cfg.get("llm_api_base", "")),
            str(cfg.get("llm_model", "qwen-plus")),
            str(cfg.get("llm_api_key", "")),
            float(cfg.get("llm_temperature", 0.0)),
            float(cfg.get("llm_timeout", 30.0)),
            int(cfg.get("llm_max_retries", 2)),
            1 if cfg.get("embed_enabled", True) else 0,
            str(cfg.get("embed_api_base", "")),
            str(cfg.get("embed_model", "BAAI/bge-m3")),
            str(cfg.get("embed_api_key", "")),
            int(cfg.get("embed_dim", 1024)),
            float(cfg.get("hybrid_dense_weight", 0.45)),
            1 if cfg.get("strict_review_auth") else 0,
            int(cfg.get("top_k", 5)),
            plat_json,
            now_ts,
        )
        try:
            self.execute(sql, params)
            return True
        except Exception as e:
            logger.error(f"[MySQLStore] 保存 llm_config 失败: {e}")
            return False

    # ————————————————————————————————————————————————
    # 2. 法律同义词表 (kb_legal_synonyms) 维护接口
    # ————————————————————————————————————————————————
    def get_legal_synonyms(self) -> Dict[str, str]:
        rows = self.query("SELECT colloquial_term, legal_terms FROM kb_legal_synonyms")
        return {r["colloquial_term"]: r["legal_terms"] for r in rows}

    def save_legal_synonyms(self, syn_map: Dict[str, str]) -> bool:
        now_ts = datetime.now(timezone.utc).isoformat()
        try:
            self.execute("DELETE FROM kb_legal_synonyms")
            for term, expansion in syn_map.items():
                self.execute(
                    "REPLACE INTO kb_legal_synonyms (colloquial_term, legal_terms, updated_at) VALUES (?, ?, ?)",
                    (term, expansion, now_ts),
                )
            return True
        except Exception as e:
            logger.error(f"[MySQLStore] 保存法律同义词表失败: {e}")
            return False

    # ————————————————————————————————————————————————
    # 3. 法条引用图谱表 (kb_cross_references) 维护接口
    # ————————————————————————————————————————————————
    def get_cross_references(self) -> List[Dict[str, Any]]:
        rows = self.query("SELECT source_law, source_article, targets_json FROM kb_cross_references")
        out = []
        for r in rows:
            targets = []
            if r.get("targets_json"):
                try:
                    targets = json.loads(r["targets_json"])
                except Exception:
                    pass
            out.append({
                "source_law": r["source_law"],
                "source_article": r["source_article"],
                "targets": targets,
            })
        return out

    def save_cross_references(self, refs: List[Dict[str, Any]]) -> bool:
        now_ts = datetime.now(timezone.utc).isoformat()
        try:
            self.execute("DELETE FROM kb_cross_references")
            for item in refs:
                self.execute(
                    "REPLACE INTO kb_cross_references (source_law, source_article, targets_json, updated_at) VALUES (?, ?, ?, ?)",
                    (item.get("source_law", ""), item.get("source_article", ""), json.dumps(item.get("targets", []), ensure_ascii=False), now_ts),
                )
            return True
        except Exception as e:
            logger.error(f"[MySQLStore] 保存法条引用图谱失败: {e}")
            return False

    # ————————————————————————————————————————————————
    # 4. 新旧法对照表 (kb_article_successions) 维护接口
    # ————————————————————————————————————————————————
    def get_article_successions(self) -> List[Dict[str, Any]]:
        return self.query("SELECT old_law, old_article, new_law, new_article, switch_date, diff_note FROM kb_article_successions")

    def save_article_successions(self, succ_list: List[Dict[str, Any]]) -> bool:
        now_ts = datetime.now(timezone.utc).isoformat()
        try:
            self.execute("DELETE FROM kb_article_successions")
            for s in succ_list:
                self.execute(
                    """
                    INSERT INTO kb_article_successions (old_law, old_article, new_law, new_article, switch_date, diff_note, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (s.get("old_law", ""), s.get("old_article", ""), s.get("new_law", ""), s.get("new_article", ""), s.get("switch_date", ""), s.get("diff_note", ""), now_ts),
                )
            return True
        except Exception as e:
            logger.error(f"[MySQLStore] 保存新旧法对照表失败: {e}")
            return False

    # ————————————————————————————————————————————————
    # 5. 利益冲突与客户库 (kb_client_registry) 维护接口
    # ————————————————————————————————————————————————
    def get_client_registry(self) -> Dict[str, str]:
        rows = self.query("SELECT client_name, client_type FROM kb_client_registry")
        return {r["client_name"]: r["client_type"] for r in rows}

    def save_client_registry(self, clients: Dict[str, str]) -> bool:
        now_ts = datetime.now(timezone.utc).isoformat()
        try:
            self.execute("DELETE FROM kb_client_registry")
            for name, desc in clients.items():
                self.execute(
                    "REPLACE INTO kb_client_registry (client_name, client_type, updated_at) VALUES (?, ?, ?)",
                    (name, desc, now_ts),
                )
            return True
        except Exception as e:
            logger.error(f"[MySQLStore] 保存利益冲突客户库失败: {e}")
            return False

    # ————————————————————————————————————————————————
    # 6. 特别法与一般法冲突优选表 (kb_special_general_relations)
    # ————————————————————————————————————————————————
    def get_special_general_relations(self) -> List[Dict[str, Any]]:
        return self.query("SELECT special_law, general_law, description, has_deference FROM kb_special_general_relations")

    def save_special_general_relations(self, relations: List[Dict[str, Any]]) -> bool:
        now_ts = datetime.now(timezone.utc).isoformat()
        try:
            self.execute("DELETE FROM kb_special_general_relations")
            for r in relations:
                self.execute(
                    """
                    REPLACE INTO kb_special_general_relations (special_law, general_law, description, has_deference, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (r.get("special_law", ""), r.get("general_law", ""), r.get("description", ""), 1 if r.get("has_deference", True) else 0, now_ts),
                )
            return True
        except Exception as e:
            logger.error(f"[MySQLStore] 保存特别法一般法关系失败: {e}")
            return False

    # ————————————————————————————————————————————————
    # 7. 法律别名表 (kb_law_aliases)
    # ————————————————————————————————————————————————
    def get_law_aliases(self) -> Dict[str, str]:
        rows = self.query("SELECT alias_name, canonical_name FROM kb_law_aliases")
        return {r["alias_name"]: r["canonical_name"] for r in rows}

    def save_law_aliases(self, aliases: Dict[str, str]) -> bool:
        now_ts = datetime.now(timezone.utc).isoformat()
        try:
            self.execute("DELETE FROM kb_law_aliases")
            for alias, canon in aliases.items():
                self.execute(
                    "REPLACE INTO kb_law_aliases (alias_name, canonical_name, updated_at) VALUES (?, ?, ?)",
                    (alias, canon, now_ts),
                )
            return True
        except Exception as e:
            logger.error(f"[MySQLStore] 保存法规别名表失败: {e}")
            return False

    # ————————————————————————————————————————————————
    # 8. 知识中心聚合视图获取与写回（全面兼容 KnowledgeCenter）
    # ————————————————————————————————————————————————
    def get_full_knowledge_config(self) -> Optional[Dict[str, Any]]:
        """从拆分后的各专用表聚合成完整的知识中心字典对象。"""
        syns = self.get_legal_synonyms()
        if not syns:
            return None

        refs = self.get_cross_references()
        succs = self.get_article_successions()
        spec_gen = self.get_special_general_relations()
        aliases = self.get_law_aliases()
        clients = self.get_client_registry()

        # 读取案由规则
        case_rules = {}
        for r in self.query("SELECT case_type, rules_json FROM kb_case_type_rules"):
            try:
                case_rules[r["case_type"]] = json.loads(r["rules_json"])
            except Exception:
                pass

        # 读取确定性模式
        certainty = []
        for r in self.query("SELECT pattern, replacement FROM kb_certainty_patterns"):
            certainty.append([r["pattern"], r["replacement"]])

        # 读取立法对照表
        new_old_cps = self.query("SELECT current_law, current_article, old_law, old_article, relation, switch_date, diff_note, authority FROM kb_new_old_counterparts")

        # 读取程序性条文
        allowed_procedural = []
        for r in self.query("SELECT law_name, article FROM kb_allowed_procedural_meta_citations"):
            allowed_procedural.append([r["law_name"], r["article"]])

        # 读取审核员
        reviewers = {}
        for r in self.query("SELECT reviewer_id, name, license_no, role, department, info_json FROM kb_reviewers_registry"):
            info = {}
            if r.get("info_json"):
                try:
                    info = json.loads(r["info_json"])
                except Exception:
                    pass
            reviewers[r["reviewer_id"]] = {
                "name": r["name"],
                "license_no": r["license_no"],
                "role": r["role"],
                "department": r["department"],
                **info,
            }

        return {
            "legal_synonym_map": syns,
            "cross_reference_graph": refs,
            "article_succession_list": succs,
            "special_general_relations": spec_gen,
            "law_aliases": aliases,
            "client_registry": clients,
            "case_type_rules": case_rules,
            "certainty_patterns": certainty,
            "new_old_counterparts": new_old_cps,
            "allowed_procedural_meta_citations": allowed_procedural,
            "reviewers_registry": reviewers,
        }

    def save_full_knowledge_config(self, data: Dict[str, Any]) -> bool:
        """将完整知识中心字典结构拆分持久化至各专用表中。"""
        now_ts = datetime.now(timezone.utc).isoformat()
        try:
            if "legal_synonym_map" in data:
                self.save_legal_synonyms(data["legal_synonym_map"])
            if "cross_reference_graph" in data:
                self.save_cross_references(data["cross_reference_graph"])
            if "article_succession_list" in data:
                self.save_article_successions(data["article_succession_list"])
            if "special_general_relations" in data:
                self.save_special_general_relations(data["special_general_relations"])
            if "law_aliases" in data:
                self.save_law_aliases(data["law_aliases"])
            if "client_registry" in data:
                self.save_client_registry(data["client_registry"])
            if "case_type_rules" in data:
                self.execute("DELETE FROM kb_case_type_rules")
                for ct, rules in data["case_type_rules"].items():
                    self.execute(
                        "REPLACE INTO kb_case_type_rules (case_type, rules_json, updated_at) VALUES (?, ?, ?)",
                        (ct, json.dumps(rules, ensure_ascii=False), now_ts),
                    )
            if "certainty_patterns" in data:
                for pat, rep in data["certainty_patterns"]:
                    self.execute(
                        "INSERT INTO kb_certainty_patterns (pattern, replacement, updated_at) VALUES (?, ?, ?)",
                        (pat, rep, now_ts),
                    )
            if "new_old_counterparts" in data:
                for cp in data["new_old_counterparts"]:
                    self.execute(
                        """
                        INSERT INTO kb_new_old_counterparts (current_law, current_article, old_law, old_article, relation, switch_date, diff_note, authority, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (cp.get("current_law", ""), cp.get("current_article", ""), cp.get("old_law", ""), cp.get("old_article", ""), cp.get("relation", "one_to_one"), cp.get("switch_date", ""), cp.get("diff_note", ""), cp.get("authority", ""), now_ts),
                    )
            if "allowed_procedural_meta_citations" in data:
                for item in data["allowed_procedural_meta_citations"]:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        self.execute(
                            "REPLACE INTO kb_allowed_procedural_meta_citations (law_name, article, updated_at) VALUES (?, ?, ?)",
                            (item[0], item[1], now_ts),
                        )
            if "reviewers_registry" in data:
                self.execute("DELETE FROM kb_reviewers_registry")
                for rid, rinfo in data["reviewers_registry"].items():
                    self.execute(
                        """
                        REPLACE INTO kb_reviewers_registry (reviewer_id, name, license_no, role, department, info_json, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (rid, rinfo.get("name", ""), rinfo.get("license_no", ""), rinfo.get("role", ""), rinfo.get("department", ""), json.dumps(rinfo, ensure_ascii=False), now_ts),
                    )
            return True
        except Exception as e:
            logger.error(f"[MySQLStore] 保存知识中心拆分表失败: {e}")
            return False

    # ————————————————————————————————————————————————
    # 25. 组织机构三级部门 (sys_org_departments)
    # ————————————————————————————————————————————————
    def get_departments(self) -> List[Dict[str, Any]]:
        rows = self.query(
            """
            SELECT dept_id, dept_name, parent_id, level, dept_path, sort_order, leader, phone, status, created_at, updated_at
            FROM sys_org_departments
            ORDER BY level ASC, sort_order ASC, dept_id ASC
            """
        )
        return rows

    def get_department_by_id(self, dept_id: str) -> Optional[Dict[str, Any]]:
        rows = self.query(
            """
            SELECT dept_id, dept_name, parent_id, level, dept_path, sort_order, leader, phone, status, created_at, updated_at
            FROM sys_org_departments
            WHERE dept_id = ?
            """,
            (dept_id,),
        )
        return rows[0] if rows else None

    def save_department(self, dept: Dict[str, Any]) -> bool:
        now_ts = datetime.now(timezone.utc).isoformat()
        try:
            self.execute(
                """
                REPLACE INTO sys_org_departments (dept_id, dept_name, parent_id, level, dept_path, sort_order, leader, phone, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    dept.get("dept_id"),
                    dept.get("dept_name"),
                    dept.get("parent_id") or None,
                    int(dept.get("level") or 1),
                    dept.get("dept_path") or dept.get("dept_id"),
                    int(dept.get("sort_order") or 0),
                    dept.get("leader") or "",
                    dept.get("phone") or "",
                    dept.get("status") or "active",
                    dept.get("created_at") or now_ts,
                    now_ts,
                ),
            )
            return True
        except Exception as e:
            logger.error(f"[MySQLStore] 保存部门失败: {e}")
            return False

    def delete_department(self, dept_id: str) -> bool:
        try:
            self.execute("DELETE FROM sys_org_departments WHERE dept_id = ?", (dept_id,))
            return True
        except Exception as e:
            logger.error(f"[MySQLStore] 删除部门失败: {e}")
            return False

    def get_dynamic_kb_collections(self) -> List[Dict[str, Any]]:
        """获取所有顶层知识库集合（动态同源，用于权限控制与知识库列表）"""
        try:
            rows = self.query(
                "SELECT node_id, name, description, owner_dept_id, visibility, sort_order, created_at "
                "FROM kb_tree_nodes WHERE node_type = 'kb' OR parent_id = '' ORDER BY sort_order ASC, created_at ASC"
            )
            # 获取每个知识库的文档数统计
            doc_counts_rows = self.query("SELECT kb_type, COUNT(*) AS cnt FROM law_documents GROUP BY kb_type")
            doc_counts = {r["kb_type"]: r["cnt"] for r in doc_counts_rows}

            # 获取所有部门的名称字典
            dept_rows = self.query("SELECT dept_id, dept_name FROM sys_org_departments")
            dept_name_map = {d["dept_id"]: d["dept_name"] for d in dept_rows}

            res = []
            for r in rows:
                nid = r["node_id"]
                owner_dept = r.get("owner_dept_id") or "dept_hq"
                # 初始预置库默认归属于法务合规部 (dept_legal)
                if nid in ("kb_external", "kb_internal", "default_external_laws", "internal_rules") and (not r.get("owner_dept_id") or r.get("owner_dept_id") == "dept_hq"):
                    owner_dept = "dept_legal"

                dept_name = dept_name_map.get(owner_dept, "集团总部")
                res.append({
                    "id": nid,
                    "name": r["name"],
                    "description": r.get("description") or f"主责管理部门: {dept_name}",
                    "owner_dept_id": owner_dept,
                    "owner_dept_name": dept_name,
                    "visibility": r.get("visibility") or "dept",
                    "doc_count": doc_counts.get(nid, 0),
                })
            return res
        except Exception as e:
            logger.error(f"[MySQLStore] 获取动态知识库集合失败: {e}")
            return [
                {"id": "default_external_laws", "name": "外规知识库", "description": "外部国家法律、行政法规与司法解释集合", "owner_dept_id": "dept_legal", "owner_dept_name": "法务合规部", "visibility": "dept", "doc_count": 48},
                {"id": "internal_rules", "name": "内规知识库", "description": "行内规章制度、合规指引与内部业务规范", "owner_dept_id": "dept_legal", "owner_dept_name": "法务合规部", "visibility": "dept", "doc_count": 0},
            ]

    def update_kb_node_owner_dept(self, node_id: str, owner_dept_id: str, visibility: str = "dept") -> bool:
        """更新知识库的主责管理部门与可见性范围"""
        try:
            self.execute(
                "UPDATE kb_tree_nodes SET owner_dept_id = ?, visibility = ? WHERE node_id = ?",
                (owner_dept_id, visibility, node_id)
            )
            return True
        except Exception as e:
            logger.error(f"[MySQLStore] 更新知识库主责部门失败: {e}")
            return False

    def get_stats_overview(
        self,
        dept_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """获取系统真实运营与知识库数据概览（100% 反映数据库真实记录，无任何伪造基数）。"""
        import re

        # 1. 顶层知识库总量 (真实树根节点数)
        kb_res = self.query("SELECT COUNT(*) as c FROM kb_tree_nodes WHERE node_type = 'kb' OR parent_id = ''")
        kb_total = kb_res[0]["c"] if kb_res else 0

        # 2. 外规与内规知识量 (按真实导入法规库分类统计)
        ext_res = self.query("SELECT COUNT(*) as c FROM law_documents WHERE kb_type LIKE '%外规%'")
        kb_ext = ext_res[0]["c"] if ext_res else 0

        int_res = self.query("SELECT COUNT(*) as c FROM law_documents WHERE kb_type LIKE '%内规%'")
        kb_int = int_res[0]["c"] if int_res else 0

        # 3. 自建知识总量 (全部已导入文档数)
        custom_res = self.query("SELECT COUNT(*) as c FROM law_documents")
        kb_custom = custom_res[0]["c"] if custom_res else 0

        # 4. 问答总量 & 提问总量 & 回答总量 (真实提问事件发生轮次)
        qa_res = self.query("SELECT COUNT(*) as c FROM audit_logs WHERE event_type = 'query_generated'")
        qa_total = qa_res[0]["c"] if qa_res else 0

        # 5. 建议总量 (大模型知识飞轮沉淀待审与已采纳提案真实总数)
        prop_res = self.query("SELECT COUNT(*) as c FROM rule_proposals")
        prop_total = prop_res[0]["c"] if prop_res else 0

        # 6. 点赞总量 (人工复核采纳与审核通过真实总数)
        like_res = self.query(
            "SELECT COUNT(*) as c FROM audit_logs WHERE event_type = 'HUMAN_REVIEW_ADOPT' OR decision = 'approved'"
        )
        like_total = like_res[0]["c"] if like_res else 0

        # 7. 点踩总量 (人工复核驳回真实总数)
        dislike_res = self.query("SELECT COUNT(*) as c FROM audit_logs WHERE decision = 'rejected'")
        dislike_total = dislike_res[0]["c"] if dislike_res else 0

        # 8. 会话总量 (真实持久化咨询会话快照总数)
        sess_res = self.query("SELECT COUNT(*) as c FROM session_snapshots")
        sess_total = sess_res[0]["c"] if sess_res else 0

        # 9. 会话总时长 (基于真实各会话发生时间窗口计算连续活动时长)
        times_rows = self.query("SELECT updated_at FROM session_snapshots ORDER BY updated_at ASC")
        times = []
        for r in times_rows:
            try:
                t = datetime.fromisoformat(r["updated_at"].replace("Z", "+00:00"))
                times.append(t)
            except Exception:
                pass

        total_seconds = 0
        if times:
            curr_start = times[0]
            curr_end = times[0]
            for t in times[1:]:
                if (t - curr_end).total_seconds() < 1800:
                    curr_end = t
                else:
                    total_seconds += max(60, (curr_end - curr_start).total_seconds())
                    curr_start = t
                    curr_end = t
            total_seconds += max(60, (curr_end - curr_start).total_seconds())

        hours = int(total_seconds // 3600)
        mins = int((total_seconds % 3600) // 60)
        secs = int(total_seconds % 60)
        duration_str = f"{hours}小时{mins}分{secs}秒" if hours > 0 else f"{mins}分{secs}秒"

        return {
            "kb_total": kb_total,
            "kb_external": kb_ext,
            "kb_internal": kb_int,
            "kb_custom": kb_custom,
            "qa_total": qa_total,
            "suggestion_total": prop_total,
            "like_total": like_total,
            "dislike_total": dislike_total,
            "query_total": qa_total,
            "answer_total": qa_total,
            "session_total": sess_total,
            "session_duration": duration_str,
            "session_seconds": total_seconds,
        }

    def get_qa_details(
        self,
        username: Optional[str] = None,
        keyword: Optional[str] = None,
        dept_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """获取真实的问答详情流水明细，支持用户名、关键词、部门与日期动态过滤检索"""
        import re

        def _clean_summary(text: str, max_len: int = 140, kw: Optional[str] = None) -> str:
            if not text:
                return ""
            t = re.sub(r"#+\s*", "", text)
            t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
            t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
            t = re.sub(r"[\r\n\t]+", " ", t)
            t = t.strip()
            if not t:
                return ""
            if kw and kw.lower() in t.lower():
                idx = t.lower().find(kw.lower())
                start = max(0, idx - 25)
                end = min(len(t), idx + max_len - 25)
                sub = t[start:end]
                if start > 0:
                    sub = "..." + sub
                if end < len(t):
                    sub = sub + "..."
                return sub
            return t[:max_len] + ("..." if len(t) > max_len else "")

        # 载入部门体系与映射
        depts = self.get_departments()
        dept_map = {d["dept_id"]: d for d in depts}
        target_dept_ids = set()
        if dept_id:
            target_node = dept_map.get(dept_id)
            if target_node:
                t_path = target_node.get("dept_path") or dept_id
                for d in depts:
                    p = d.get("dept_path") or d.get("dept_id")
                    if p == t_path or p.startswith(t_path + "/") or d["dept_id"] == dept_id:
                        target_dept_ids.add(d["dept_id"])
            else:
                target_dept_ids.add(dept_id)

        # 载入用户与所属部门
        users_rows = self.query("SELECT user_id, username, name, department, dept_id FROM sys_rbac_users")
        user_by_name = {}
        for u in users_rows:
            user_by_name[u["username"]] = u
            user_by_name[u["name"]] = u

        # 真实读取会话快照与审计日志
        snapshots = self.query(
            "SELECT thread_id, updated_at, status, query, state_json FROM session_snapshots ORDER BY updated_at DESC"
        )
        audit_rows = self.query(
            "SELECT thread_id, event_type, decision, reviewer_name, reviewer_role, draft_answer, final_answer, timestamp FROM audit_logs ORDER BY id DESC"
        )
        logs_by_thread: Dict[str, List[Dict[str, Any]]] = {}
        for a in audit_rows:
            tid = a["thread_id"]
            if tid not in logs_by_thread:
                logs_by_thread[tid] = []
            logs_by_thread[tid].append(a)

        results = []
        for s in snapshots:
            tid = s["thread_id"]
            up_at = s.get("updated_at") or ""
            date_str = up_at[:10]

            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue

            st_data = {}
            if s.get("state_json"):
                try:
                    st_data = json.loads(s["state_json"])
                except Exception:
                    pass

            fo = st_data.get("final_output") or {}
            related_logs = logs_by_thread.get(tid, [])

            ans_text = ""
            if isinstance(fo, dict) and fo.get("answer"):
                ans_text = fo.get("answer")
            elif related_logs:
                ans_text = related_logs[0].get("final_answer") or related_logs[0].get("draft_answer") or ""

            u_info = st_data.get("user") or {}
            raw_u = u_info.get("username") or (related_logs[0].get("reviewer_name") if related_logs else "") or "admin"
            matched_user = user_by_name.get(raw_u, {})
            user_display = matched_user.get("name") or raw_u
            user_dept_id = matched_user.get("dept_id") or "dept_hq"
            user_dept_name = matched_user.get("department") or "集团总部"

            if target_dept_ids and user_dept_id not in target_dept_ids:
                continue

            if username and (username.lower() not in raw_u.lower() and username.lower() not in user_display.lower()):
                continue

            q_text = s.get("query") or ""
            if keyword:
                kw_low = keyword.lower()
                if kw_low not in q_text.lower() and kw_low not in ans_text.lower():
                    continue

            raw_scope = st_data.get("kb_scope") or "external"
            if raw_scope in ("external", "外规", "default_external_laws"):
                scope_display = "外规库"
            elif raw_scope in ("internal", "内规", "internal_rules"):
                scope_display = "内规库"
            else:
                scope_display = "全库检索"

            rev_status = s.get("status") or ""
            has_adopt = any(
                l.get("event_type") == "HUMAN_REVIEW_ADOPT" or l.get("decision") == "approved"
                for l in related_logs
            )
            if has_adopt or rev_status in ("approved", "COMPLETED", "done"):
                display_status = "已复核"
                status_tag = "done"
            elif rev_status == "needs_review":
                display_status = "待复核"
                status_tag = "warn"
            elif rev_status == "rejected" or any(l.get("decision") == "rejected" for l in related_logs):
                display_status = "已驳回"
                status_tag = "err"
            else:
                display_status = "已办结"
                status_tag = "done"

            clean_time = up_at[:19].replace("T", " ") if up_at else "-"

            results.append({
                "thread_id": tid,
                "time": clean_time,
                "username": raw_u,
                "user_display": f"{user_display} ({raw_u})" if user_display != raw_u else raw_u,
                "department": user_dept_name,
                "dept_id": user_dept_id,
                "scope": scope_display,
                "query": q_text,
                "answer_summary": _clean_summary(ans_text, max_len=140, kw=keyword),
                "review_status": display_status,
                "status_tag": status_tag,
            })
            if len(results) >= limit:
                break

        return results


# 全局单例
mysql_store = MySQLStore()
try:
    mysql_store.init_schema()
except Exception as e:
    logger.debug(f"[MySQLStore] 自动执行 init_schema 提示: {e}")
try:
    settings.load_from_disk()
except Exception:
    pass
