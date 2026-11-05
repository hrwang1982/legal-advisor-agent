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
        self._explicit_path = path
        self._cached_mtime: float = -1.0
        self._data: Dict[str, Any] = {}

    @property
    def config_path(self) -> str:
        return self._explicit_path or settings.knowledge_config_path

    def _ensure_loaded(self) -> Dict[str, Any]:
        p = self.config_path
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            mtime = 0.0

        if not self._data or mtime != self._cached_mtime:
            self.reload()
        return self._data

    def reload(self) -> Dict[str, Any]:
        """强制从磁盘重新读取知识配置 JSON。"""
        p = self.config_path
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            try:
                self._cached_mtime = os.path.getmtime(p)
            except OSError:
                self._cached_mtime = 0.0
        else:
            self._data = {}
            self._cached_mtime = 0.0
        return self._data

    def save(self, new_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """将更新后的知识配置写回 JSON 文件并刷新缓存（立即热生效）。"""
        if new_data is not None:
            self._data = new_data
        p = self.config_path
        parent = os.path.dirname(os.path.abspath(p))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        try:
            self._cached_mtime = os.path.getmtime(p)
        except OSError:
            pass
        return self._data

    def get_all(self) -> Dict[str, Any]:
        return dict(self._ensure_loaded())

    def update_section(self, section: str, value: Any) -> Dict[str, Any]:
        """按模块局部更新配置并持久化（如更新同义词表、引致图谱、客户名录等）。"""
        data = dict(self._ensure_loaded())
        data[section] = value
        return self.save(data)

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
        for item in raw_list:
            if isinstance(item, dict):
                out.append(
                    (
                        item.get("old_law", ""),
                        item.get("old_article", ""),
                        item.get("new_law", ""),
                        item.get("new_article", ""),
                        item.get("switch_date", ""),
                        item.get("diff_note", ""),
                    )
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 6:
                out.append(tuple(item[:6]))  # type: ignore
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

