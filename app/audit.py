"""持久化审计账本、会话快照存储与规则缺口归集器（P2 可追责层）。

设计目标：
1. 默认无参调用：完全基于 MySQL 关系数据库持久化保存每个 thread_id 的完整会话快照，彻底切断本地文件依赖；
2. 支持单测显式传入 db_path 临时文件隔离运行；
3. 解决法律责任追溯问题：记录每次 AI 生成与执业律师审核（含律师姓名、执业证号、角色等级、审核决定、
   修改前后 Unified Diff），并采用 SHA-256 前后哈希链（Tamper-Evident Hash Chain）防篡改；
4. 解决规则迭代闭环问题：自动将所有 PARTIAL / MISS / CONFLICT 判定归集至 rule_gaps 表，
   形成「线上暴露缺口 → 归集清单 → 补充元数据/规则」的数据飞轮。
"""
from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.mysql_store import mysql_store

logger = logging.getLogger(__name__)


def compute_answer_diff(draft: str, final: str) -> str:
    """计算 AI 初稿与律师改稿/终稿之间的 Unified Diff。无差异则返回空字符串。"""
    a = (draft or "").strip()
    b = (final or "").strip()
    if a == b:
        return ""
    lines = list(
        difflib.unified_diff(
            a.splitlines(),
            b.splitlines(),
            fromfile="ai_draft",
            tofile="lawyer_final",
            lineterm="",
        )
    )
    return "\n".join(lines)


class AuditStore:
    """基于关系数据库（MySQL）的持久化审计账本与会话快照管理器。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path
        self._mem_conn: Optional[sqlite3.Connection] = None
        if self.db_path:
            if self.db_path == ":memory:":
                self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            else:
                parent = os.path.dirname(os.path.abspath(self.db_path))
                if parent:
                    os.makedirs(parent, exist_ok=True)
            self._init_custom_db()
        else:
            self._init_db()

    def _init_db(self) -> None:
        mysql_store.init_schema()

    def _init_custom_db(self) -> None:
        conn = self._get_custom_conn()
        cur = conn.cursor()
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
        if self._mem_conn is None:
            conn.close()

    def _get_custom_conn(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _execute(self, sql: str, params: tuple = ()) -> int:
        if self.db_path:
            conn = self._get_custom_conn()
            cur = conn.cursor()
            cur.execute(sql, params)
            last_id = cur.lastrowid
            conn.commit()
            if self._mem_conn is None:
                conn.close()
            return last_id or cur.rowcount
        else:
            return mysql_store.execute(sql, params)

    def _query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        if self.db_path:
            conn = self._get_custom_conn()
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]
            if self._mem_conn is None:
                conn.close()
            return rows
        else:
            return mysql_store.query(sql, params)

    def _query_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _calc_hash(
        prev_hash: str,
        thread_id: str,
        timestamp: str,
        event_type: str,
        reviewer_name: str,
        license_no: str,
        decision: str,
        final_answer: str,
    ) -> str:
        raw = "|".join(
            [
                prev_hash or "GENESIS",
                thread_id or "",
                timestamp or "",
                event_type or "",
                reviewer_name or "",
                license_no or "",
                decision or "",
                final_answer or "",
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def save_snapshot(
        self,
        thread_id: str,
        status: str,
        state_data: Dict[str, Any],
    ) -> None:
        """保存或更新会话状态快照。"""
        now = self._now_iso()
        plan = state_data.get("rule_plan") or {}
        payload_str = json.dumps(state_data, ensure_ascii=False, default=str)

        try:
            sql = """
                REPLACE INTO session_snapshots
                    (thread_id, updated_at, status, query, act_date, region, confidence, review_level, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            self._execute(
                sql,
                (
                    thread_id,
                    now,
                    status,
                    state_data.get("query", ""),
                    state_data.get("act_date", ""),
                    state_data.get("region", ""),
                    state_data.get("answer_confidence") or plan.get("confidence", ""),
                    plan.get("review", ""),
                    payload_str,
                ),
            )
        except Exception as e:
            logger.error(f"[AuditStore] save_snapshot 失败: {e}")

    def get_snapshot(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """按 thread_id 读取持久化的会话快照。"""
        try:
            row = self._query_one(
                "SELECT updated_at, status, state_json FROM session_snapshots WHERE thread_id = ?",
                (thread_id,),
            )
            if row and row.get("state_json"):
                return {
                    "thread_id": thread_id,
                    "updated_at": row["updated_at"],
                    "status": row["status"],
                    "state": json.loads(row["state_json"]),
                }
        except Exception as e:
            logger.error(f"[AuditStore] get_snapshot 失败: {e}")
        return None

    def record_audit_event(
        self,
        thread_id: str,
        event_type: str,
        decision: str = "",
        reviewer_name: str = "",
        license_no: str = "",
        reviewer_role: str = "",
        comments: str = "",
        confidence: str = "",
        review_level: str = "",
        citations: Optional[List[Dict]] = None,
        draft_answer: str = "",
        final_answer: str = "",
    ) -> Dict[str, Any]:
        """追加一条防篡改哈希链审计记录，并返回该记录摘要（含 entry_hash 与 answer_diff）。"""
        now = self._now_iso()
        diff_text = compute_answer_diff(draft_answer, final_answer)
        cites_str = json.dumps(
            [
                f"《{c.get('law_name', '')}》{c.get('article', '')}{c.get('paragraph', '')}"
                for c in (citations or [])
            ],
            ensure_ascii=False,
        )

        last = self._query_one("SELECT entry_hash FROM audit_logs ORDER BY id DESC LIMIT 1")
        prev_hash = last["entry_hash"] if last and last.get("entry_hash") else "GENESIS"
        entry_hash = self._calc_hash(
            prev_hash,
            thread_id,
            now,
            event_type,
            reviewer_name,
            license_no,
            decision,
            final_answer,
        )

        row_id = 0
        try:
            self._execute(
                """
                INSERT INTO audit_logs (
                    thread_id, timestamp, event_type, reviewer_name, license_no, reviewer_role,
                    decision, comments, confidence, review_level, citations_json,
                    draft_answer, final_answer, answer_diff, prev_hash, entry_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id, now, event_type, reviewer_name, license_no, reviewer_role,
                    decision, comments, confidence, review_level, cites_str,
                    draft_answer, final_answer, diff_text, prev_hash, entry_hash,
                ),
            )
            max_row = self._query_one("SELECT MAX(id) as id FROM audit_logs")
            if max_row and max_row.get("id"):
                row_id = int(max_row["id"])
        except Exception as e:
            logger.error(f"[AuditStore] record_audit_event 写入失败: {e}")

        return {
            "audit_id": row_id,
            "thread_id": thread_id,
            "timestamp": now,
            "event_type": event_type,
            "decision": decision,
            "reviewer_name": reviewer_name,
            "license_no": license_no,
            "reviewer_role": reviewer_role,
            "answer_diff": diff_text,
            "prev_hash": prev_hash,
            "entry_hash": entry_hash,
        }

    def record_rule_gap(
        self,
        thread_id: str,
        plan: Dict[str, Any],
        query: str = "",
    ) -> None:
        """当规则引擎处于非确定性命中（mode != deterministic）时，自动归集至规则缺口表。"""
        mode = (plan or {}).get("mode", "deterministic")
        if mode == "deterministic":
            return
        now = self._now_iso()

        try:
            self._execute(
                """
                INSERT INTO rule_gaps (thread_id, timestamp, mode, confidence, review_level, reason, verify_actions_json, query)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    thread_id, now, mode, plan.get("confidence", ""),
                    plan.get("review", ""), plan.get("reason", ""),
                    json.dumps(plan.get("verify_actions", []), ensure_ascii=False),
                    query,
                ),
            )
        except Exception as e:
            logger.error(f"[AuditStore] record_rule_gap 写入失败: {e}")

    def list_audit_logs(self, thread_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """查询审计日志列表。"""
        if thread_id:
            rows = self._query(
                """
                SELECT id, thread_id, timestamp, event_type, reviewer_name, license_no, reviewer_role,
                       decision, comments, confidence, review_level, citations_json, answer_diff, prev_hash, entry_hash
                FROM audit_logs WHERE thread_id = ? ORDER BY id ASC LIMIT ?
                """,
                (thread_id, limit),
            )
        else:
            rows = self._query(
                """
                SELECT id, thread_id, timestamp, event_type, reviewer_name, license_no, reviewer_role,
                       decision, comments, confidence, review_level, citations_json, answer_diff, prev_hash, entry_hash
                FROM audit_logs ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )

        out = []
        for r in rows:
            item = dict(r)
            try:
                item["citations"] = json.loads(item.pop("citations_json", None) or "[]")
            except Exception:
                item["citations"] = []
            out.append(item)
        return out

    def list_rule_gaps(self, limit: int = 50) -> List[Dict[str, Any]]:
        """查询归集的规则缺口清单（供法务与规则工程师迭代规则库）。"""
        rows = self._query(
            """
            SELECT id, thread_id, timestamp, mode, confidence, review_level, reason, verify_actions_json, query
            FROM rule_gaps ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        )
        out = []
        for r in rows:
            verify_act = []
            try:
                verify_act = json.loads(r.get("verify_actions_json") or "[]")
            except Exception:
                pass
            out.append(
                {
                    "id": r["id"],
                    "thread_id": r["thread_id"],
                    "timestamp": r["timestamp"],
                    "mode": r.get("mode"),
                    "confidence": r.get("confidence"),
                    "review_level": r.get("review_level"),
                    "reason": r.get("reason"),
                    "verify_actions": verify_act,
                    "query": r.get("query"),
                }
            )
        return out

    def record_rule_proposal(
        self,
        thread_id: str,
        proposal_type: str,
        proposal_key: str,
        proposal_data: Dict[str, Any],
        query: str = "",
    ) -> int:
        """记录大模型推导的规则提案（用于知识飞轮一键采纳）。
        如果已有同 thread_id, 同 proposal_type, 同 proposal_key 处于 PENDING 状态，则避免重复记录。
        """
        existing = self._query_one(
            """
            SELECT id FROM rule_proposals
            WHERE thread_id = ? AND proposal_type = ? AND proposal_key = ? AND status = 'PENDING'
            """,
            (thread_id, proposal_type, proposal_key),
        )
        if existing and existing.get("id"):
            return int(existing["id"])

        now = self._now_iso()
        self._execute(
            """
            INSERT INTO rule_proposals
                (thread_id, timestamp, proposal_type, proposal_key, proposal_data_json, status, query)
            VALUES (?, ?, ?, ?, ?, 'PENDING', ?)
            """,
            (
                thread_id,
                now,
                proposal_type,
                proposal_key,
                json.dumps(proposal_data, ensure_ascii=False),
                query,
            ),
        )
        max_row = self._query_one("SELECT MAX(id) as id FROM rule_proposals")
        return int(max_row["id"]) if max_row and max_row.get("id") else 0

    def list_rule_proposals(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """查询待审或已处理的规则提案列表。"""
        if status:
            rows = self._query(
                """
                SELECT id, thread_id, timestamp, proposal_type, proposal_key, proposal_data_json, status, reviewer_name, review_notes, adopted_at, query
                FROM rule_proposals WHERE status = ? ORDER BY id DESC LIMIT ?
                """,
                (status, limit),
            )
        else:
            rows = self._query(
                """
                SELECT id, thread_id, timestamp, proposal_type, proposal_key, proposal_data_json, status, reviewer_name, review_notes, adopted_at, query
                FROM rule_proposals ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
        out = []
        for r in rows:
            data = {}
            try:
                data = json.loads(r.get("proposal_data_json") or "{}")
            except Exception:
                pass
            out.append(
                {
                    "id": r["id"],
                    "thread_id": r["thread_id"],
                    "timestamp": r["timestamp"],
                    "proposal_type": r["proposal_type"],
                    "proposal_key": r.get("proposal_key"),
                    "proposal_data": data,
                    "status": r["status"],
                    "reviewer_name": r.get("reviewer_name"),
                    "review_notes": r.get("review_notes"),
                    "adopted_at": r.get("adopted_at"),
                    "query": r.get("query"),
                }
            )
        return out

    def adopt_rule_proposal(
        self,
        proposal_id: int,
        reviewer_name: str = "执业律师/规则专家",
        review_notes: str = "",
    ) -> Tuple[bool, str]:
        """知识飞轮一键采纳：将提案自动合并写入 MySQL 业务规则表并标记状态为 ADOPTED。"""
        row = self._query_one(
            """
            SELECT id, proposal_type, proposal_key, proposal_data_json, status
            FROM rule_proposals WHERE id = ?
            """,
            (proposal_id,),
        )
        if not row:
            return False, f"未找到 ID 为 {proposal_id} 的提案"

        if row.get("status") == "ADOPTED":
            return True, "该提案已采纳，无需重复操作"

        try:
            p_data = json.loads(row.get("proposal_data_json") or "{}")
        except Exception as e:
            return False, f"解析提案数据失败: {e}"

        try:
            from app.knowledge import knowledge_center
            from app.nodes import _BRACKET_CITE_RE

            p_type = row.get("proposal_type")
            p_key = row.get("proposal_key")
            cfg = knowledge_center.get_all()
            if p_type == "case_type_rules":
                rules = cfg.setdefault("case_type_rules", {})
                target_key = p_key or p_data.get("name") or "未命名案由"
                rules[target_key] = p_data
                knowledge_center.save(cfg)
            elif p_type == "suggested_citations":
                cits = p_data if isinstance(p_data, list) else p_data.get("citations", [])
                meta_list = cfg.setdefault("allowed_procedural_meta_citations", [])
                for cit in cits:
                    m = _BRACKET_CITE_RE.search(str(cit))
                    if m:
                        pair = [m.group(1).strip(), m.group(2).strip()]
                        if pair not in meta_list:
                            meta_list.append(pair)
                knowledge_center.save(cfg)
            elif p_type == "synonym":
                syn_map = cfg.setdefault("legal_synonym_map", {})
                syn_map[p_key] = p_data.get("formal", "")
                knowledge_center.save(cfg)
            else:
                knowledge_center.save(cfg)
        except Exception as e:
            return False, f"规则合入配置失败: {e}"

        now = self._now_iso()
        self._execute(
            """
            UPDATE rule_proposals
            SET status = 'ADOPTED', reviewer_name = ?, review_notes = ?, adopted_at = ?
            WHERE id = ?
            """,
            (reviewer_name, review_notes, now, proposal_id),
        )
        return True, f"提案 #{proposal_id} 已成功采纳并持久化"

    def reject_rule_proposal(
        self,
        proposal_id: int,
        reviewer_name: str = "",
        review_notes: str = "",
    ) -> Tuple[bool, str]:
        """驳回提案并标记为 REJECTED。"""
        now = self._now_iso()
        self._execute(
            """
            UPDATE rule_proposals
            SET status = 'REJECTED', reviewer_name = ?, review_notes = ?, adopted_at = ?
            WHERE id = ?
            """,
            (reviewer_name, review_notes, now, proposal_id),
        )
        return True, f"提案 #{proposal_id} 已驳回"

    def verify_hash_chain(self) -> Tuple[bool, int, str]:
        """校验全量审计账本哈希链完整性，检测是否存在篡改或中间记录删除。"""
        rows = self._query(
            """
            SELECT id, thread_id, timestamp, event_type, reviewer_name, license_no,
                   decision, final_answer, prev_hash, entry_hash
            FROM audit_logs ORDER BY id ASC
            """
        )

        expected_prev = "GENESIS"
        for r in rows:
            row_id = r["id"]
            tid = r["thread_id"]
            ts = r["timestamp"]
            ev = r["event_type"]
            r_name = r.get("reviewer_name") or ""
            lic = r.get("license_no") or ""
            dec = r.get("decision") or ""
            final_ans = r.get("final_answer") or ""
            prev_h = r["prev_hash"]
            entry_h = r["entry_hash"]

            if prev_h != expected_prev:
                return False, len(rows), f"记录 #{row_id} 的 prev_hash 断裂（期望 {expected_prev}，实际 {prev_h}）"
            calc = self._calc_hash(prev_h, tid, ts, ev, r_name, lic, dec, final_ans)
            if calc != entry_h:
                return False, len(rows), f"记录 #{row_id} 的 entry_hash 校验失败（内容疑似被篡改）"
            expected_prev = entry_h
        return True, len(rows), "审计哈希链完整无篡改"


# 默认全局持久化审计实例
audit_store = AuditStore()
