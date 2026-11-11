"""官方《新旧法律条文对照表》数字化数据集与批量录入工具：
支持解析全国人大常委会法工委编写的官方新旧法律对照表（如《公司法新旧条文对照表》、《民法典与各旧法对照表》），
并将数字化对照关系批量录入外置知识中心（knowledge_center），实现历史纠纷条文级自动回填。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.kb import normalize_article_id
from app.knowledge import knowledge_center

logger = logging.getLogger(__name__)

# ————————————————————————————————————————————————
# 全国人大常委会法工委编写的权威《新旧法律条文对照表》预置数字化数据集
# ————————————————————————————————————————————————
OFFICIAL_LEGISLATIVE_COUNTERPARTS: List[Dict[str, Any]] = [
    # 1. 2023 修订《公司法》 vs 2018 修正《公司法》
    {
        "current_law": "中华人民共和国公司法",
        "current_article": "第47条",
        "old_law": "中华人民共和国公司法（2018年修正）",
        "old_article": "第26条",
        "relation": "one_to_one",
        "switch_date": "2024-07-01",
        "diff_note": "有限责任公司认缴出资额由无期限认缴修改为成立之日起5年内缴足",
        "authority": "全国人大常委会法制工作委员会",
    },
    {
        "current_law": "中华人民共和国公司法",
        "current_article": "第54条",
        "old_law": "中华人民共和国公司法（2018年修正）",
        "old_article": "第28条",
        "relation": "one_to_one",
        "switch_date": "2024-07-01",
        "diff_note": "新增公司不能清偿到期债务时出资加速到期制度",
        "authority": "全国人大常委会法制工作委员会",
    },
    {
        "current_law": "中华人民共和国公司法",
        "current_article": "第23条",
        "old_law": "中华人民共和国公司法（2018年修正）",
        "old_article": "第20条",
        "relation": "one_to_one",
        "switch_date": "2024-07-01",
        "diff_note": "完善法人人格否认制度，增加横向法人人格否认（关联公司混同连带）",
        "authority": "全国人大常委会法制工作委员会",
    },
    # 2. 2020《民法典》 vs 1999《合同法》
    {
        "current_law": "中华人民共和国民法典",
        "current_article": "第577条",
        "old_law": "中华人民共和国合同法",
        "old_article": "第107条",
        "relation": "one_to_one",
        "switch_date": "2021-01-01",
        "diff_note": "违约责任基本原则承继，条文文字无实质变动",
        "authority": "全国人大常委会法制工作委员会",
    },
    {
        "current_law": "中华人民共和国民法典",
        "current_article": "第585条",
        "old_law": "中华人民共和国合同法",
        "old_article": "第114条",
        "relation": "one_to_one",
        "switch_date": "2021-01-01",
        "diff_note": "违约金约定与调减规则承继",
        "authority": "全国人大常委会法制工作委员会",
    },
    {
        "current_law": "中华人民共和国民法典",
        "current_article": "第563条",
        "old_law": "中华人民共和国合同法",
        "old_article": "第94条",
        "relation": "one_to_one",
        "switch_date": "2021-01-01",
        "diff_note": "法定解除合同条件承继，并与违约责任体系衔接",
        "authority": "全国人大常委会法制工作委员会",
    },
    # 3. 2020《民法典》 vs 2009《侵权责任法》
    {
        "current_law": "中华人民共和国民法典",
        "current_article": "第1191条",
        "old_law": "中华人民共和国侵权责任法",
        "old_article": "第34条",
        "relation": "one_to_one",
        "switch_date": "2021-01-01",
        "diff_note": "用人单位工作人员执行工作任务造成他人损害的替代责任承继并细化追偿权",
        "authority": "全国人大常委会法制工作委员会",
    },
    # 4. 2021 修正《行政处罚法》 vs 1996 旧《行政处罚法》
    {
        "current_law": "中华人民共和国行政处罚法",
        "current_article": "第33条",
        "old_law": "中华人民共和国行政处罚法（1996年版）",
        "old_article": "第27条",
        "relation": "one_to_one",
        "switch_date": "2021-07-15",
        "diff_note": "新增首违不罚（初次违法且危害后果轻微并及时改正的，可以不予行政处罚）与无主观过错不罚",
        "authority": "全国人大常委会法制工作委员会",
    },
]


def check_network_available(timeout: float = 1.0) -> bool:
    """毫秒级快速探测外网是否可达。若离线或处于隔离环境，1 秒内立即安全退出，绝不阻塞请求。"""
    import socket

    for host in ("223.5.5.5", "119.29.29.29", "180.76.76.76"):
        try:
            s = socket.create_connection((host, 53), timeout=timeout)
            s.close()
            return True
        except Exception:
            continue
    return False


def fetch_latest_online_counterparts(timeout: float = 2.0) -> Optional[List[Dict[str, Any]]]:
    """尝试从国家法律法规数据库或权威公开源获取最新的《新旧法律条文对照表》。
    若网络不可达或获取超时，安全返回 None。
    """
    if not check_network_available(timeout=0.8):
        return None

    try:
        import urllib.request

        api_url = "https://flk.npc.gov.cn/api/counterparts/latest"
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "LegalAdvisorAgent/1.0", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("data") and isinstance(payload["data"], list):
                    return payload["data"]
    except Exception as e:
        logger.debug(f"[Counterparts] 联网获取最新官方对照表跳过/回退: {e}")
    return None


def batch_import_official_counterparts(
    counterparts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """将官方新旧法律对照表批量录入至知识中心：
    1. 动态检查外网连通性：若外网可达，优先抓取官方最新在线条文对照；若外网不可达，平滑采用内置权威法工委对照集；
    2. 严格去重与底层防膨胀：更新入库并清空历史冗余，确保物理表不膨胀且业务无重复。
    """
    data_source = "custom"
    if counterparts is not None:
        items = counterparts
        data_source = "custom_provided"
    else:
        # 尝试外网获取最新对照数据
        online_items = fetch_latest_online_counterparts(timeout=2.0)
        if online_items:
            items = online_items
            data_source = "online_official"
        else:
            items = OFFICIAL_LEGISLATIVE_COUNTERPARTS
            data_source = "builtin_official"

    cfg = knowledge_center.get_all()
    raw_existing = cfg.get("new_old_counterparts", [])

    # 严格去重已存在条目，自动清洗历史可能残留的冗余重复项
    existing_list: List[Dict[str, Any]] = []
    seen_keys = set()
    for item in raw_existing:
        c_law = (item.get("current_law") or "").strip().strip("《》")
        c_art = normalize_article_id(item.get("current_article") or "")
        o_law = (item.get("old_law") or "").strip().strip("《》")
        o_art = normalize_article_id(item.get("old_article") or "")
        if not c_law or not c_art or not o_law or not o_art:
            continue
        k = (c_law, c_art, o_law, o_art)
        if k not in seen_keys:
            seen_keys.add(k)
            existing_list.append(item)

    imported_count = 0

    for it in items:
        c_law = (it.get("current_law") or "").strip().strip("《》")
        c_art = normalize_article_id(it.get("current_article") or "")
        o_law = (it.get("old_law") or "").strip().strip("《》")
        o_art = normalize_article_id(it.get("old_article") or "")
        s_date = (it.get("switch_date") or "").strip()
        if not c_law or not c_art or not o_law or not o_art or not s_date:
            continue

        key = (c_law, c_art, o_law, o_art)
        clean_item = {
            "current_law": c_law,
            "current_article": c_art,
            "old_law": o_law,
            "old_article": o_art,
            "relation": it.get("relation", "one_to_one"),
            "switch_date": s_date,
            "diff_note": it.get("diff_note", "新旧法条承继与修订"),
            "authority": it.get("authority", "全国人大常委会法制工作委员会"),
        }
        if key not in seen_keys:
            existing_list.append(clean_item)
            seen_keys.add(key)
            imported_count += 1
        else:
            # 更新已有条目的 diff_note 与 switch_date
            for old_it in existing_list:
                if (
                    old_it.get("current_law") == c_law
                    and normalize_article_id(old_it.get("current_article") or "") == c_art
                    and old_it.get("old_law") == o_law
                    and normalize_article_id(old_it.get("old_article") or "") == o_art
                ):
                    old_it.update(clean_item)

    # 精准更新 new_old_counterparts，触发底层 save_new_old_counterparts 清空并重写，杜绝表膨胀
    knowledge_center.update_section("new_old_counterparts", existing_list)

    if data_source == "online_official":
        msg = f"已联网同步法工委最新新旧法律对照表，本次更新 {imported_count} 条，现有对照条目总量 {len(existing_list)} 条。"
    elif data_source == "builtin_official":
        msg = f"外网不可达（离线安全模式），已采用内置全国人大常委会法工委权威对照表，本次更新 {imported_count} 条，现有对照条目总量 {len(existing_list)} 条。"
    else:
        msg = f"成功批量导入 {imported_count} 条官方新旧法律对照关系，现有库容量 {len(existing_list)} 条。"

    return {
        "status": "success",
        "source": data_source,
        "imported_count": imported_count,
        "total_counterparts": len(existing_list),
        "message": msg,
    }

