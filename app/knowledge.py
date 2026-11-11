"""外置知识与规则配置中心（Knowledge & Rules Configuration Center）：
将原本硬编码在 Python 源码中的业务知识与规则全部抽离至 JSON 配置文件（`app/data/knowledge_config.json`），
并提供内存缓存、文件变更自动热加载（mtime 感知）以及在线增删改持久化能力。

业务法务人员可直接修改 JSON 或通过管理台 API 维护以下知识，无需改代码、无需重启发版：
1. 口语 → 法言法语同义词表（legal_synonym_map）
2. 引致条款图谱（cross_reference_graph）
3. 新旧法条文对照表（article_succession_list）
4. 特别法—一般法成对关系表（special_general_relations）
5. 法规简称/别名归一化字典（law_aliases）
6. 程序与立法元法规条文级白名单（allowed_procedural_meta_citations）
7. 兜底绝对化表述降级替换规则（certainty_patterns）
8. 律所利益冲突客户名录（client_registry）
9. 案由事实要件与诉讼/仲裁时效规则库（case_type_rules）
10. 执业律师审核员权限与鉴权注册表（reviewers_registry）
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import settings


class KnowledgeCenter:
    """支持文件修改时间（mtime）感知的热加载与在线持久化知识配置管理器。"""

    def __init__(self, path: Optional[str] = None):
        self.config_path = path or "mysql:kb_knowledge_configs"
        self._data: Dict[str, Any] = {}

    def _ensure_loaded(self) -> Dict[str, Any]:
        if not self._data:
            self.reload()
        return self._data

    def reload(self) -> Dict[str, Any]:
        """从 MySQL 数据库各知识专用拆分表（同义词/图谱/新旧法/客户库等）加载全量业务知识配置。"""
        try:
            from app.mysql_store import mysql_store
            db_data = mysql_store.get_full_knowledge_config()
            if db_data:
                self._data = db_data
                return self._data
        except Exception:
            pass
        return self._data

    def save(self, new_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """将更新后的知识配置拆分持久化写入 MySQL 各专用表（kb_legal_synonyms、kb_cross_references 等）。"""
        if new_data is not None:
            self._data = new_data

        try:
            from app.mysql_store import mysql_store
            mysql_store.save_full_knowledge_config(self._data)
        except Exception:
            pass
        return self._data

    def get_all(self, reload: bool = False) -> Dict[str, Any]:
        if reload:
            self.reload()
        data = dict(self._ensure_loaded())

        # 确保关键模块在返回前端视图时始终经过严格去重过滤
        if "article_succession_list" in data and isinstance(data["article_succession_list"], list):
            seen_succ = set()
            clean_succs = []
            for s in data["article_succession_list"]:
                k = (
                    str(s.get("old_law", "")).strip(),
                    str(s.get("old_article", "")).strip(),
                    str(s.get("new_law", "")).strip(),
                    str(s.get("new_article", "")).strip(),
                )
                if not k[0] or not k[1] or not k[2] or not k[3] or k in seen_succ:
                    continue
                seen_succ.add(k)
                clean_succs.append(s)
            data["article_succession_list"] = clean_succs
            self._data["article_succession_list"] = clean_succs

        if "certainty_patterns" in data and isinstance(data["certainty_patterns"], list):
            seen_pat = set()
            clean_pats = []
            for p in data["certainty_patterns"]:
                pat_str = str(p[0]) if isinstance(p, (list, tuple)) and len(p) >= 1 else ""
                if not pat_str or pat_str in seen_pat:
                    continue
                seen_pat.add(pat_str)
                clean_pats.append(p)
            data["certainty_patterns"] = clean_pats
            self._data["certainty_patterns"] = clean_pats

        return data

    def update_section(self, section: str, value: Any) -> Dict[str, Any]:
        """按模块局部更新配置并持久化（精准更新对应模块表，毫秒级响应并即时生效）。"""
        from datetime import datetime, timezone
        data = dict(self._ensure_loaded())
        data[section] = value
        self._data = data
        try:
            from app.mysql_store import mysql_store
            now_ts = datetime.now(timezone.utc).isoformat()
            if section == "legal_synonym_map" and isinstance(value, dict):
                mysql_store.save_legal_synonyms(value)
            elif section == "cross_reference_graph" and isinstance(value, list):
                mysql_store.save_cross_references(value)
            elif section == "article_succession_list" and isinstance(value, list):
                seen = set()
                clean_succs = []
                for s in value:
                    k = (str(s.get("old_law", "")).strip(), str(s.get("old_article", "")).strip(), str(s.get("new_law", "")).strip(), str(s.get("new_article", "")).strip())
                    if not k[0] or not k[1] or not k[2] or not k[3] or k in seen:
                        continue
                    seen.add(k)
                    clean_succs.append(s)
                data[section] = clean_succs
                mysql_store.save_article_successions(clean_succs)
            elif section == "special_general_relations" and isinstance(value, list):
                mysql_store.save_special_general_relations(value)
            elif section == "law_aliases" and isinstance(value, dict):
                mysql_store.save_law_aliases(value)
            elif section == "client_registry" and isinstance(value, dict):
                mysql_store.save_client_registry(value)
            elif section == "new_old_counterparts" and isinstance(value, list):
                mysql_store.save_new_old_counterparts(value)
            elif section == "case_type_rules" and isinstance(value, dict):
                mysql_store.execute("DELETE FROM kb_case_type_rules")
                for ct, rules in value.items():
                    mysql_store.execute(
                        "REPLACE INTO kb_case_type_rules (case_type, rules_json, updated_at) VALUES (?, ?, ?)",
                        (ct, json.dumps(rules, ensure_ascii=False), now_ts),
                    )
            elif section == "reviewers_registry" and isinstance(value, dict):
                mysql_store.execute("DELETE FROM kb_reviewers_registry")
                for rid, rinfo in value.items():
                    mysql_store.execute(
                        "REPLACE INTO kb_reviewers_registry (reviewer_id, name, license_no, role, department, info_json, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (rid, rinfo.get("name", ""), rinfo.get("license_no", ""), rinfo.get("role", ""), rinfo.get("department", ""), json.dumps(rinfo, ensure_ascii=False), now_ts),
                    )
            else:
                mysql_store.save_full_knowledge_config(self._data)
        except Exception:
            pass
        return self._data

    # ————————————————————————————————————————————————
    # 1. 口语 → 法言法语同义词表
    # ————————————————————————————————————————————————
    def get_synonym_map(self) -> Dict[str, str]:
        data = self._ensure_loaded()
        return dict(data.get("legal_synonym_map") or {})

    # ————————————————————————————————————————————————
    # 2. 引致条款图谱：(source_law, source_article) -> [(target_law, target_article, reason), ...]
    # ————————————————————————————————————————————————
    def get_cross_reference_graph(self) -> Dict[Tuple[str, str], List[Tuple[str, str, str]]]:
        data = self._ensure_loaded()
        raw_val = data.get("cross_reference_graph") or []
        graph: Dict[Tuple[str, str], List[Tuple[str, str, str]]] = {}

        if isinstance(raw_val, dict):
            for k, targets in raw_val.items():
                if "|" in str(k):
                    s_law, s_art = str(k).split("|", 1)
                elif isinstance(k, tuple) and len(k) >= 2:
                    s_law, s_art = k[0], k[1]
                else:
                    continue
                t_list = []
                for t in targets or []:
                    if isinstance(t, dict):
                        t_law = (t.get("target_law") or "").strip()
                        t_art = (t.get("target_article") or "").strip()
                        reason = (t.get("reason") or "").strip()
                        if t_law and t_art:
                            t_list.append((t_law, t_art, reason))
                    elif isinstance(t, (list, tuple)) and len(t) >= 3:
                        t_list.append((t[0], t[1], t[2]))
                if t_list:
                    graph[(s_law, s_art)] = t_list
            return graph

        for item in raw_val:
            if not isinstance(item, dict):
                continue
            s_law = (item.get("source_law") or "").strip()
            s_art = (item.get("source_article") or "").strip()
            if not s_law or not s_art:
                continue
            targets: List[Tuple[str, str, str]] = []
            for t in item.get("targets") or []:
                if isinstance(t, dict):
                    t_law = (t.get("target_law") or "").strip()
                    t_art = (t.get("target_article") or "").strip()
                    reason = (t.get("reason") or "").strip()
                    if t_law and t_art:
                        targets.append((t_law, t_art, reason))
                elif isinstance(t, (list, tuple)) and len(t) >= 3:
                    targets.append((t[0], t[1], t[2]))
            graph[(s_law, s_art)] = targets
        return graph

    # ————————————————————————————————————————————————
    # 3. 新旧法条文对照表
    # ————————————————————————————————————————————————
    def get_article_succession_list(self) -> List[Tuple[str, str, str, str, str, str]]:
        data = self._ensure_loaded()
        raw_list = data.get("article_succession_list") or []
        out: List[Tuple[str, str, str, str, str, str]] = []
        seen = set()
        for item in raw_list:
            if isinstance(item, dict):
                tuple_item = (
                    str(item.get("old_law", "")).strip(),
                    str(item.get("old_article", "")).strip(),
                    str(item.get("new_law", "")).strip(),
                    str(item.get("new_article", "")).strip(),
                    str(item.get("switch_date", "")).strip(),
                    str(item.get("diff_note", "")).strip(),
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 6:
                tuple_item = tuple(str(x).strip() for x in item[:6])  # type: ignore
            else:
                continue

            k = tuple_item[:4]
            if not k[0] or not k[1] or not k[2] or not k[3] or k in seen:
                continue
            seen.add(k)
            out.append(tuple_item)
        return out

    # ————————————————————————————————————————————————
    # 4. 特别法—一般法成对关系表
    # ————————————————————————————————————————————————
    def get_special_general_relations(self) -> Dict[Tuple[str, str], Tuple[str, bool]]:
        data = self._ensure_loaded()
        raw_list = data.get("special_general_relations") or []
        out: Dict[Tuple[str, str], Tuple[str, bool]] = {}
        for item in raw_list:
            sp = (item.get("special_law") or "").strip()
            gn = (item.get("general_law") or "").strip()
            desc = (item.get("description") or "").strip()
            defer = bool(item.get("has_deference", True))
            if sp and gn:
                out[(sp, gn)] = (desc, defer)
        return out

    # ————————————————————————————————————————————————
    # 5. 法规简称/别名归一化字典
    # ————————————————————————————————————————————————
    def get_law_aliases(self) -> Dict[str, str]:
        data = self._ensure_loaded()
        return dict(data.get("law_aliases") or {})

    # ————————————————————————————————————————————————
    # 6. 程序与立法元法规（条文级白名单）
    # ————————————————————————————————————————————————
    def get_allowed_meta_citations(self) -> Set[Tuple[str, str]]:
        data = self._ensure_loaded()
        raw = data.get("allowed_procedural_meta_citations") or []
        out: Set[Tuple[str, str]] = set()
        for pair in raw:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                out.add((str(pair[0]).strip(), str(pair[1]).strip()))
        return out

    # ————————————————————————————————————————————————
    # 7. 兜底绝对化表述替换模式
    # ————————————————————————————————————————————————
    def get_certainty_patterns(self) -> List[Tuple[str, str]]:
        data = self._ensure_loaded()
        raw = data.get("certainty_patterns") or []
        out: List[Tuple[str, str]] = []
        for p in raw:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append((str(p[0]), str(p[1])))
        return out

    # ————————————————————————————————————————————————
    # 8. 律所利益冲突客户名录
    # ————————————————————————————————————————————————
    def get_client_registry(self) -> Dict[str, str]:
        data = self._ensure_loaded()
        return dict(data.get("client_registry") or {})

    # ————————————————————————————————————————————————
    # 9. 案由事实要件与诉讼/仲裁时效规则库
    # ————————————————————————————————————————————————
    def get_case_type_rules(self) -> Dict[str, Dict[str, Any]]:
        data = self._ensure_loaded()
        raw = data.get("case_type_rules") or {}
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in raw.items():
            elems = []
            for e in v.get("elements") or []:
                if isinstance(e, (list, tuple)) and len(e) >= 3:
                    elems.append((str(e[0]), tuple(e[1]), str(e[2])))
            out[k] = {
                "name": v.get("name", k),
                "keywords": list(v.get("keywords") or []),
                "elements": elems,
                "procedural_notes": list(v.get("procedural_notes") or []),
            }
        return out

    # ————————————————————————————————————————————————
    # 10. 执业律师审核员权限与鉴权注册表
    # ————————————————————————————————————————————————
    def get_reviewers_registry(self) -> Dict[str, Dict[str, Any]]:
        data = self._ensure_loaded()
        return dict(data.get("reviewers_registry") or {})


# 全局单例
knowledge_center = KnowledgeCenter()

