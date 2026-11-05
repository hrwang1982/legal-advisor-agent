"""合规与安全守卫层（P2）：
1. 个人信息保护（《个人信息保护法》）：身份证号、手机号、银行卡号自动识别与脱敏（PII Masking）；
2. 律所利益冲突检查（《律师法》第39条）：自动提取涉案公司/主体名称并与利益冲突当事人库比对；
3. 生成式 AI 显著标识与服务边界声明（《生成式人工智能服务管理暂行办法》）。

依赖：纯标准库（re / dataclasses），零第三方依赖，可离线单测。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from app.config import settings
from app.knowledge import knowledge_center

# ————————————————————————————————————————————————
# 1. 个人信息（PII）识别与脱敏规则
# ————————————————————————————————————————————————
# 18位身份证号（避免与普通数字混淆，校验前6位地区码+年份19/20+月份+日期+4位尾号）
_ID_CARD_RE = re.compile(
    r"(?<!\d)([1-6]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx])(?!\d)"
)
# 11位中国大陆手机号
_PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d)(\d{4})(\d{4})(?!\d)")
# 16~19位银行卡号
_BANK_CARD_RE = re.compile(r"(?<!\d)([3-6]\d{3})(\d{8,11})(\d{4})(?!\d)")


def mask_pii(text: str) -> Tuple[str, List[str]]:
    """对输入文本中的敏感个人信息（身份证、手机号、银行卡）执行自动脱敏。

    返回 (masked_text, masked_types_log)。
    """
    if not text:
        return "", []
    logs: List[str] = []
    out = text

    def _sub_id(m: re.Match) -> str:
        val = m.group(1)
        if "身份证号" not in logs:
            logs.append("身份证号")
        return f"{val[:3]}***********{val[-4:]}"

    def _sub_phone(m: re.Match) -> str:
        if "手机号码" not in logs:
            logs.append("手机号码")
        return f"{m.group(1)}****{m.group(3)}"

    def _sub_bank(m: re.Match) -> str:
        if "银行卡号" not in logs:
            logs.append("银行卡号")
        return f"{m.group(1)}********{m.group(3)}"

    out = _ID_CARD_RE.sub(_sub_id, out)
    out = _PHONE_RE.sub(_sub_phone, out)
    out = _BANK_CARD_RE.sub(_sub_bank, out)
    return out, logs


# ————————————————————————————————————————————————
# 2. 律所利益冲突检索（Conflict of Interest Check）
# ————————————————————————————————————————————————
# 示例律所常年法律顾问客户 / 在办案件当事人名录（生产环境可对接律所 OA / CRM 库）
DEFAULT_CLIENT_REGISTRY: Dict[str, str] = {
    "澜舟科技有限公司": "本所常年法律顾问客户",
    "远航物流集团": "本所在办诉讼案件委托人",
    "星辰地产有限公司": "本所常年法律顾问客户",
}

# 提取「XX公司 / XX集团 / XX事务所 / XX银行 / XX厂」等商事主体
_ENTITY_RE = re.compile(
    r"([一-龥]{2,20}(?:有限责任公司|股份有限公司|有限公司|集团|科技公司|物流公司|地产公司|贸易公司|银行|分公司))"
)


def extract_entities(text: str) -> List[str]:
    """从问题文本中提取潜在的当事人/企业主体名称。"""
    if not text:
        return []
    found: List[str] = []
    for m in _ENTITY_RE.finditer(text):
        ent = m.group(1)
        # 去除前缀动词/介词（如“起诉”“在”“与”“被”）
        ent = re.sub(r"^.*?(?:起诉|控告|投诉|状告|被|在|与|向|和|为|由|因)", "", ent)
        if len(ent) >= 4 and ent not in found:
            found.append(ent)
    return found


def check_conflict_of_interest(
    text: str,
    registry: Dict[str, str] | None = None,
) -> Dict:
    """检查咨询文本中涉及的主体是否与律所现有客户存在潜在利益冲突（《律师法》第39条）。
    支持从外置知识中心（knowledge_center）实时读取律所客户名录。
    """
    reg = registry if registry is not None else (knowledge_center.get_client_registry() or DEFAULT_CLIENT_REGISTRY)
    entities = extract_entities(text)
    conflicts: List[Dict[str, str]] = []

    for ent in entities:
        for client_name, relation in reg.items():
            short_client = re.sub(r"(?:有限责任公司|股份有限公司|有限公司)$", "", client_name)
            if client_name in ent or ent in client_name or (len(short_client) >= 2 and short_client in ent):
                conflicts.append(
                    {
                        "matched_entity": ent,
                        "registered_client": client_name,
                        "relation": relation,
                        "warning": (
                            f"利益冲突预警：检测到涉案主体「{ent}」与「{client_name}」（{relation}）高度关联，"
                            "请承办律师依《律师法》第39条完成利益冲突审查后再出具正式意见。"
                        ),
                    }
                )
    return {
        "has_conflict": bool(conflicts),
        "entities": entities,
        "conflicts": conflicts,
    }


# ————————————————————————————————————————————————
# 3. 审核权限分级校验（Role-Based Review Gate）
# ————————————————————————————————————————————————
_ROLE_RANK = {
    "lawyer": 1,         # 普通执业律师
    "senior_lawyer": 2,  # 资深律师
    "partner": 3,        # 合伙人律师
    "expert": 3,         # 专家委员
}

_REQUIRED_ROLE_FOR_LEVEL = {
    "sample": 1,
    "must": 1,
    "expert": 2,
    "immediate": 2,
    "reject": 3,
}


def verify_reviewer_permission(
    review_level: str,
    reviewer_role: str = "lawyer",
    license_no: str = "",
    reviewer_token: str = "",
) -> Tuple[bool, str]:
    """校验审核人角色、执业证号与注册表权限是否满足当前案件的 review_level 要求。

    安全升级（P0）：
    1. 若审核人的 license_no 在律所执业律师注册表（reviewers_registry）中登记，
       强制以服务端注册表核定的真实职级（registered_role）为准，防止前端伪造高职级越权审批；
    2. 若开启 STRICT_REVIEW_AUTH=true，则未登记的 license_no 或错误的 reviewer_token 一律拒绝；
    3. 职级门槛：
       - sample / must：普通执业律师（lawyer）及以上即可；
       - expert / immediate：属于重大疑难或规范冲突升级案件，要求 senior_lawyer / partner / expert 复核。
    """
    lvl = (review_level or "must").lower()
    claimed_role = (reviewer_role or "lawyer").lower()
    lic = (license_no or "").strip()

    registry = knowledge_center.get_reviewers_registry()
    effective_role = claimed_role

    if lic and lic in registry:
        reg_info = registry[lic]
        reg_role = (reg_info.get("role") or "lawyer").lower()
        # 若提供了 token 或开启了严格鉴权，校验 token
        expected_tk = reg_info.get("token", "")
        if settings.strict_review_auth and expected_tk and reviewer_token != expected_tk:
            return False, f"鉴权失败：执业证号 {lic} 的审核授权令牌（reviewer_token）校验不通过。"
        # 强制使用服务端登记的真实角色（防止普通律师执业证自报 partner 角色）
        if _ROLE_RANK.get(claimed_role, 1) > _ROLE_RANK.get(reg_role, 1):
            return (
                False,
                f"越权拦截：执业证号 {lic}（{reg_info.get('name', '')}）在律所注册表中的核定职级为「{reg_role}」，不得自报「{claimed_role}」审批。",
            )
        effective_role = reg_role
    elif settings.strict_review_auth:
        return (
            False,
            f"鉴权失败：严格审核鉴权模式已开启，执业证号「{lic or '空'}」未在律所执业律师注册表中备案。",
        )

    req_rank = _REQUIRED_ROLE_FOR_LEVEL.get(lvl, 1)
    cur_rank = _ROLE_RANK.get(effective_role, 1)

    if cur_rank < req_rank:
        return (
            False,
            f"权限不足：当前回答审核等级为「{lvl}」，需资深律师或合伙人（senior_lawyer / partner）复核，当前角色为「{effective_role}」。",
        )
    return True, "审核权限校验通过"


def build_compliance_disclaimer(
    is_human_approved: bool = False,
    reviewer_name: str = "",
    license_no: str = "",
    answer_confidence: str = "B",
) -> str:
    """生成符合《生成式人工智能服务管理暂行办法》与《律师法》的合规标识与免责声明。"""
    ai_tag = "［生成式人工智能辅助生成｜算法备案与可追溯标识］"
    if is_human_approved and reviewer_name:
        lic_str = f"（执业证号：{license_no}）" if license_no else ""
        return (
            f"{ai_tag} 本法律咨询意见初稿由 AI 辅助检索生成（系统置信度：{answer_confidence}），"
            f"已由执业律师【{reviewer_name}】{lic_str}人工复核通过。具体案件办理仍需以正式委托代理协议及庭审举证为准。"
        )
    return (
        f"{ai_tag} 本回答由 AI 基于检索法规自动推演生成（当前置信度：{answer_confidence}），"
        "尚未完成执业律师终审签名，不构成正式法律意见书，仅供法律检索参考。"
    )

