"""持久化审计账本、会话快照存储与规则缺口归集器（P2 可追责层）。

设计目标：
1. 解决 MemorySaver 重启丢失问题：基于标准库 sqlite3 持久化保存每个 thread_id 的完整会话快照；
2. 解决法律责任追溯问题：记录每次 AI 生成与执业律师审核（含律师姓名、执业证号、角色等级、审核决定、
   修改前后 Unified Diff），并采用 SHA-256 前后哈希链（Tamper-Evident Hash Chain）防篡改；
3. 解决规则迭代闭环问题：自动将所有 PARTIAL / MISS / CONFLICT 判定归集至 rule_gaps 表，
   形成「线上暴露缺口 → 归集清单 → 补充元数据/规则」的数据飞轮。

依赖：纯标准库（sqlite3 / hashlib / difflib / json / datetime），零第三方依赖。
"""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


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
    """基于 SQLite 的持久化审计账本与会话快照管理器。"""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("AUDIT_DB_PATH", "app/data/audit_trail.db")
        if self.db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(self.db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
        # 若使用 :memory:，保持长连接以便单测验证
        self._mem_conn: Optional[sqlite3.Connection] = (
            sqlite3.connect(":memory:", check_same_thread=False)
            if self.db_path == ":memory:"
            else None
        )
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if self._mem_conn is not None:
            return self._mem_conn
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self) -> None:
        conn = self._conn()
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
        """保存或更新会话状态快照（支持服务重启后恢复）。"""
        now = self._now_iso()
        plan = state_data.get("rule_plan") or {}
        payload_str = json.dumps(state_data, ensure_ascii=False, default=str)
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO session_snapshots
                (thread_id, updated_at, status, query, act_date, region, confidence, review_level, state_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                updated_at=excluded.updated_at,
                status=excluded.status,
                query=excluded.query,
                act_date=excluded.act_date,
                region=excluded.region,
                confidence=excluded.confidence,
                review_level=excluded.review_level,
                state_json=excluded.state_json
            """,
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
        conn.commit()
        if self._mem_conn is None:
            conn.close()

    def get_snapshot(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """按 thread_id 读取持久化的会话快照。"""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT updated_at, status, state_json FROM session_snapshots WHERE thread_id = ?",
            (thread_id,),
        )
        row = cur.fetchone()
        if self._mem_conn is None:
            conn.close()
        if not row:
            return None
        updated_at, status, state_json = row
        return {
            "thread_id": thread_id,
            "updated_at": updated_at,
            "status": status,
            "state": json.loads(state_json),
        }

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

        conn = self._conn()
        cur = conn.cursor()
        cur.execute("SELECT entry_hash FROM audit_logs ORDER BY id DESC LIMIT 1")
        last = cur.fetchone()
        prev_hash = last[0] if last else "GENESIS"
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

        cur.execute(
            """
            INSERT INTO audit_logs (
                thread_id, timestamp, event_type, reviewer_name, license_no, reviewer_role,
                decision, comments, confidence, review_level, citations_json,
                draft_answer, final_answer, answer_diff, prev_hash, entry_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                now,
                event_type,
                reviewer_name,
                license_no,
                reviewer_role,
                decision,
                comments,
                confidence,
                review_level,
                cites_str,
                draft_answer,
                final_answer,
                diff_text,
                prev_hash,
                entry_hash,
            ),
        )
        row_id = cur.lastrowid
        conn.commit()
        if self._mem_conn is None:
            conn.close()
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
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO rule_gaps (thread_id, timestamp, mode, confidence, review_level, reason, verify_actions_json, query)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                now,
                mode,
                plan.get("confidence", ""),
                plan.get("review", ""),
                plan.get("reason", ""),
                json.dumps(plan.get("verify_actions", []), ensure_ascii=False),
                query,
            ),
        )
        conn.commit()
        if self._mem_conn is None:
            conn.close()

    def list_audit_logs(self, thread_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """查询审计日志列表。"""
        conn = self._conn()
        cur = conn.cursor()
        if thread_id:
            cur.execute(
                """
                SELECT id, thread_id, timestamp, event_type, reviewer_name, license_no, reviewer_role,
                       decision, comments, confidence, review_level, citations_json, answer_diff, prev_hash, entry_hash
                FROM audit_logs WHERE thread_id = ? ORDER BY id ASC LIMIT ?
                """,
                (thread_id, limit),
            )
        else:
            cur.execute(
                """
                SELECT id, thread_id, timestamp, event_type, reviewer_name, license_no, reviewer_role,
                       decision, comments, confidence, review_level, citations_json, answer_diff, prev_hash, entry_hash
                FROM audit_logs ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
        rows = cur.fetchall()
        if self._mem_conn is None:
            conn.close()
        cols = [
            "id", "thread_id", "timestamp", "event_type", "reviewer_name", "license_no", "reviewer_role",
            "decision", "comments", "confidence", "review_level", "citations_json", "answer_diff", "prev_hash", "entry_hash",
        ]
        out = []
        for r in rows:
            item = dict(zip(cols, r))
            try:
                item["citations"] = json.loads(item.pop("citations_json") or "[]")
            except Exception:
                item["citations"] = []
            out.append(item)
        return out

    def list_rule_gaps(self, limit: int = 50) -> List[Dict[str, Any]]:
        """查询归集的规则缺口清单（供法务与规则工程师迭代规则库）。"""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, thread_id, timestamp, mode, confidence, review_level, reason, verify_actions_json, query
            FROM rule_gaps ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        if self._mem_conn is None:
            conn.close()
        out = []
        for r in rows:
            out.append(
                {
                    "id": r[0],
                    "thread_id": r[1],
                    "timestamp": r[2],
                    "mode": r[3],
                    "confidence": r[4],
                    "review_level": r[5],
                    "reason": r[6],
                    "verify_actions": json.loads(r[7] or "[]"),
                    "query": r[8],
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
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id FROM rule_proposals
            WHERE thread_id = ? AND proposal_type = ? AND proposal_key = ? AND status = 'PENDING'
            """,
            (thread_id, proposal_type, proposal_key),
        )
        row = cur.fetchone()
        if row:
            if self._mem_conn is None:
                conn.close()
            return row[0]

        now = self._now_iso()
        cur.execute(
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
        new_id = cur.lastrowid
        conn.commit()
        if self._mem_conn is None:
            conn.close()
        return new_id

    def list_rule_proposals(self, status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """查询待审或已处理的规则提案列表。"""
        conn = self._conn()
        cur = conn.cursor()
        if status:
            cur.execute(
                """
                SELECT id, thread_id, timestamp, proposal_type, proposal_key, proposal_data_json, status, reviewer_name, review_notes, adopted_at, query
                FROM rule_proposals WHERE status = ? ORDER BY id DESC LIMIT ?
                """,
                (status, limit),
            )
        else:
            cur.execute(
                """
                SELECT id, thread_id, timestamp, proposal_type, proposal_key, proposal_data_json, status, reviewer_name, review_notes, adopted_at, query
                FROM rule_proposals ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            )
        rows = cur.fetchall()
        if self._mem_conn is None:
            conn.close()
        out = []
        for r in rows:
            data = {}
            try:
                data = json.loads(r[5] or "{}")
            except Exception:
                pass
            out.append(
                {
                    "id": r[0],
                    "thread_id": r[1],
                    "timestamp": r[2],
                    "proposal_type": r[3],
                    "proposal_key": r[4],
                    "proposal_data": data,
                    "status": r[6],
                    "reviewer_name": r[7],
                    "review_notes": r[8],
                    "adopted_at": r[9],
                    "query": r[10],
                }
            )
        return out

    def adopt_rule_proposal(
        self,
        proposal_id: int,
        reviewer_name: str = "执业律师/规则专家",
        review_notes: str = "",
    ) -> Tuple[bool, str]:
        """知识飞轮一键采纳：将提案自动合并写入 knowledge_config.json 并标记状态为 ADOPTED。"""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, proposal_type, proposal_key, proposal_data_json, status
            FROM rule_proposals WHERE id = ?
            """,
            (proposal_id,),
        )
        row = cur.fetchone()
        if not row:
            if self._mem_conn is None:
                conn.close()
            return False, f"未找到 ID 为 {proposal_id} 的提案"

        _id, p_type, p_key, p_json, st = row
        if st == "ADOPTED":
            if self._mem_conn is None:
                conn.close()
            return True, "该提案已采纳，无需重复操作"

        try:
            p_data = json.loads(p_json or "{}")
        except Exception as e:
            if self._mem_conn is None:
                conn.close()
            return False, f"解析提案数据失败: {e}"

        try:
            import re
            from app.knowledge import knowledge_center
            from app.nodes import _BRACKET_CITE_RE

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
            if self._mem_conn is None:
                conn.close()
            return False, f"规则合入配置失败: {e}"

        now = self._now_iso()
        cur.execute(
            """
            UPDATE rule_proposals
            SET status = 'ADOPTED', reviewer_name = ?, review_notes = ?, adopted_at = ?
            WHERE id = ?
            """,
            (reviewer_name, review_notes, now, proposal_id),
        )
        conn.commit()
        if self._mem_conn is None:
            conn.close()
        return True, f"提案 #{proposal_id} 已成功采纳并热更新至 knowledge_config.json"

    def reject_rule_proposal(
        self,
        proposal_id: int,
        reviewer_name: str = "",
        review_notes: str = "",
    ) -> Tuple[bool, str]:
        """驳回提案并标记为 REJECTED。"""
        conn = self._conn()
        cur = conn.cursor()
        now = self._now_iso()
        cur.execute(
            """
            UPDATE rule_proposals
            SET status = 'REJECTED', reviewer_name = ?, review_notes = ?, adopted_at = ?
            WHERE id = ?
            """,
            (reviewer_name, review_notes, now, proposal_id),
        )
        conn.commit()
        if self._mem_conn is None:
            conn.close()
        return True, f"提案 #{proposal_id} 已驳回"

    def verify_hash_chain(self) -> Tuple[bool, int, str]:
        """校验全量审计账本哈希链完整性，检测是否存在篡改或中间记录删除。"""
        conn = self._conn()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, thread_id, timestamp, event_type, reviewer_name, license_no,
                   decision, final_answer, prev_hash, entry_hash
            FROM audit_logs ORDER BY id ASC
            """
        )
        rows = cur.fetchall()
        if self._mem_conn is None:
            conn.close()

        expected_prev = "GENESIS"
        for r in rows:
            row_id, tid, ts, ev, r_name, lic, dec, final_ans, prev_h, entry_h = r
            if prev_h != expected_prev:
                return False, len(rows), f"记录 #{row_id} 的 prev_hash 断裂（期望 {expected_prev}，实际 {prev_h}）"
            calc = self._calc_hash(prev_h, tid, ts, ev, r_name or "", lic or "", dec or "", final_ans or "")
            if calc != entry_h:
                return False, len(rows), f"记录 #{row_id} 的 entry_hash 校验失败（内容疑似被篡改）"
            expected_prev = entry_h
        return True, len(rows), "审计哈希链完整无篡改"


# 默认全局持久化审计实例
audit_store = AuditStore()

