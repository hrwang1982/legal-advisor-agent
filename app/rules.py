"""确定性规则引擎（三层体系之「层 2」）：时间效力定位、位阶冲突裁决，以及无规则时的 LLM 兜底策略。

设计原则：
1. 能用规则判定的，绝不给 LLM —— 法律效力、时间定位、位阶优先都是确定性问题。
2. 判定结果分四态：HIT / PARTIAL / MISS / CONFLICT，而不是简单的「有/无规则」。
3. CONFLICT（规则互相矛盾或新一般 vs 旧特别无法确定适用）禁止 LLM 裁决：用不确定性覆盖确定性是反模式，必须升级人工或有权机关裁决。
4. MISS / PARTIAL 允许 LLM 兜底，但必须「降级不失控」：携带硬约束、标注不确定性、提升人审等级。

依赖：仅标准库（dataclasses / enum / datetime），零第三方依赖，可离线单测。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple


# ————————————————————————————————————————————————
# 枚举与数据结构
# ————————————————————————————————————————————————
class RuleStatus(str, Enum):
    HIT = "HIT"            # 规则唯一确定，可直接执行
    PARTIAL = "PARTIAL"    # 规则覆盖部分要素，余下由 LLM 在骨架内补全
    MISS = "MISS"          # 规则未覆盖 / 关键元数据缺失
    CONFLICT = "CONFLICT"  # 多条规则给出互相矛盾的结论（或新一般 vs 旧特别冲突）


class Confidence(str, Enum):
    A = "A"  # 规则确定
    B = "B"  # 规则骨架 + LLM 补全
    C = "C"  # 纯 LLM 受限兜底
    D = "D"  # 无法判定（拒答或转裁决）


class ReviewLevel(str, Enum):
    SAMPLE = "sample"        # 抽检
    MUST = "must"            # 必经人工审核
    EXPERT = "expert"        # 必审 + 资深律师/专家
    IMMEDIATE = "immediate"  # 立即人工介入（不得自动输出）
    REJECT = "reject"        # 拒答并转人工


# ————————————————————————————————————————————————
# LLM 兜底的硬约束模板（兜底时必须注入提示词）
# ————————————————————————————————————————————————
C_CITE_CLOSURE = "只能引用给定候选条文；不得生成任何未出现的条文号、法规名称或文号"
C_NO_CERTAINTY = "禁止使用“必然/一定/肯定/应当”等确定性表述；改用“在……前提下/倾向于/需进一步确认”"
C_STATE_GAP = "必须显式说明：规则未覆盖的具体原因、不确定性来源、以及结论成立所需的前提"
C_GIVE_VERIFY = "必须给出至少一项建议核实或补证事项（如行为发生日期、制定机关、事实细节）"
C_NO_AUTO = "本结论标记为兜底结果，不得进入自动对外通道，必须经人工审核后方可使用"


def _hard_constraints() -> List[str]:
    return [C_CITE_CLOSURE, C_NO_CERTAINTY, C_STATE_GAP, C_GIVE_VERIFY, C_NO_AUTO]


def hard_constraints() -> List[str]:
    """对外暴露硬约束清单（供链路在特殊场景下构造兜底结果时复用）。"""
    return _hard_constraints()


@dataclass
class RuleOutcome:
    """规则判定结果 + 兜底策略载体。"""
    status: RuleStatus
    reason: str                                   # 判定理由 / 缺口原因（用于留痕与规则建设）
    confidence: Confidence
    review: ReviewLevel
    llm_allowed: bool                             # 是否允许 LLM 兜底
    constraints: List[str] = field(default_factory=list)   # 兜底时注入的硬约束
    verify_actions: List[str] = field(default_factory=list)  # 建议核实事项
    detail: Dict = field(default_factory=dict)


# ————————————————————————————————————————————————
# 工具与位阶体系
# ————————————————————————————————————————————————
def _parse(d) -> Optional[date]:
    """宽松解析 ISO 日期（YYYY-MM-DD）。失败返回 None，绝不抛异常。"""
    if not d:
        return None
    try:
        return date.fromisoformat(str(d)[:10])
    except (ValueError, TypeError):
        return None


# 基础位阶表（不含司法解释）：数值越小位阶越高。
# 注意：
# 1. 司法解释不在此表内——它是「依附于被解释法律」的官方解释，位阶随被解释法律浮动。
# 2. 《立法法》（2023修正）第100条：地方性法规的效力高于本级和下级地方政府规章（故地方性法规=3，地方政府规章=4）。
# 3. 《立法法》（2023修正）第102条：部门规章之间、部门规章与地方政府规章之间具有同等效力，
#    因此「部门规章 vs 地方政府规章」列入 INCOMPARABLE_PAIRS，绝不按数值大小判上下位法！
BASE_RANK = {
    "宪法": 0,
    "法律": 1,
    "行政法规": 2,
    "地方性法规": 3,
    "部门规章": 3,
    "地方政府规章": 4,
    "规范性文件": 5,
}

# 司法解释相对被解释法律的偏移量：排在被解释法律之后、下一位阶之前。
JITI_OFFSET = 0.5
# 司法解释无法定位被解释法律时的保守位阶（法律之下、行政法规之上）。
JITI_FALLBACK_RANK = 1.5

_LOCAL = "地方性法规"
_DEPT = "部门规章"
_LOCAL_GOV = "地方政府规章"

# 不可机械比较位阶的组合：法律未规定何者优先，须走有权机关裁决程序。
# 1) 《立法法》（2023修正）第106条第1款第2项（原第95条）：地方性法规与部门规章之间对同一事项规定不一致，
#    由国务院提出意见；认为应适用地方性法规的决定适用地方性法规，认为应适用部门规章的提请全国人大常委会裁决。
# 2) 《立法法》（2023修正）第102条、第106条第1款第3项：部门规章与地方政府规章之间具有同等效力，
#    对同一事项的规定不一致时，由国务院裁决。
INCOMPARABLE_PAIRS = frozenset(
    {
        ("地方性法规", "部门规章"),
        ("部门规章", "地方性法规"),
        ("部门规章", "地方政府规章"),
        ("地方政府规章", "部门规章"),
    }
)


def rank_of(hierarchy: str, host_hierarchy: str = "") -> float:
    """计算位阶数值（越小越高）。

    hierarchy="司法解释" 时依附于被解释法律的位阶：host_hierarchy + 0.5。
    例如解释《民法典》（法律，1）的司法解释位阶为 1.5，
    高于行政法规（2）与部门规章（3）——这才是实务中的真实效力序列。
    """
    if hierarchy == "司法解释":
        base = BASE_RANK.get(host_hierarchy)
        if base is None:
            return JITI_FALLBACK_RANK
        return base + JITI_OFFSET
    return float(BASE_RANK.get(hierarchy, 9))


def is_incomparable(h1: str, h2: str) -> bool:
    """两个位阶是否属于「法律未定优劣、须走裁决程序」的组合。"""
    return (h1, h2) in INCOMPARABLE_PAIRS


# 《立法法》（2023修正）第103条（原第92条）限「同一机关制定」。
# 依《宪法》第58条，全国人民代表大会与其常务委员会共同行使国家立法权，常委会是全国人大的常设机关，
# 二者在法律层面应视为同一立法机关；否则人大通过的基本法律（如民法典）与常委会通过的法律（如消保法、劳动合同法）
# 之间将无法适用特别法优于一般法，与通说和裁判实务不符。
_SAME_BODY_GROUPS = (
    {"全国人民代表大会", "全国人民代表大会常务委员会", "全国人大", "全国人大常委会"},
)


def _same_legislative_body(a: str, b: str) -> bool:
    """判定两个制定机关是否属于同一立法机关（决定是否可适用《立法法》第103条）。"""
    a, b = (a or "").strip(), (b or "").strip()
    if not a or not b:
        return False
    if a == b:
        return True
    return any(a in g and b in g for g in _SAME_BODY_GROUPS)


def _clean_law_name(name: str) -> str:
    """归一化法规名称（去除书名号与「中华人民共和国」前缀），便于成对关系匹配。"""
    s = (name or "").strip().strip("《》")
    if s.startswith("中华人民共和国"):
        s = s[len("中华人民共和国") :]
    return s


# ————————————————————————————————————————————————
# 成对「特别法—一般法」关系表（解决将「特别/一般」简化为孤立字段的缺陷）
# ————————————————————————————————————————————————
# 格式：(特别法简称, 一般法简称) -> (适用领域说明, 是否存在新一般法显式转致条款)
# 注：《民法典》第11条规定：“其他法律对民事关系有特别规定的，依照其规定。”（显式立法转致保留），
# 故即便《民法典》（2020）颁布晚于《消费者权益保护法》（2013修正）、《劳动合同法》（2012修正）、
# 《道路交通安全法》（2011修正），在《民法典》第11条转致下仍确定优先适用上述特别法，未尽事宜由《民法典》补充。
from app.knowledge import knowledge_center

SPECIAL_GENERAL_RELATIONS: Dict[Tuple[str, str], Tuple[str, bool]] = {
    ("劳动合同法", "民法典"): ("劳动合同关系优先适用《劳动合同法》，未尽事宜补充适用《民法典》", True),
    ("消费者权益保护法", "民法典"): ("消费者权益保护领域依《民法典》第11条、第128条优先适用《消费者权益保护法》", True),
    ("公司法", "民法典"): ("公司组织与股东出资关系依《民法典》第11条、第71条优先适用《公司法》", True),
    ("公司法（2018年修正）", "民法典"): ("公司组织与股东关系优先适用行为时有效的《公司法》", True),
    ("道路交通安全法", "民法典"): ("机动车交通事故责任依《民法典》第1208条转致规定优先适用《道路交通安全法》", True),
    ("个人信息保护法", "民法典"): ("个人信息处理规则依《民法典》第11条、第1034条优先适用《个人信息保护法》", True),
}


def identify_special_general(a: Dict, b: Dict) -> Optional[Tuple[Dict, Dict, str, bool]]:
    """识别两条同位阶规范是否构成「特别法—一般法」关系。

    返回 (special_dict, general_dict, relation_desc, has_statutory_deference) 或 None。
    1) 优先匹配外置知识中心（knowledge_center）的成对关系表（支持在线热更新，无需改代码）；
    2) 向下兼容：若不在成对表中，当且仅当 {a.scope, b.scope} == {"特别", "一般"} 时按字段识别。
    """
    rel_map = knowledge_center.get_special_general_relations() or SPECIAL_GENERAL_RELATIONS
    na, nb = _clean_law_name(a.get("law_name", "")), _clean_law_name(b.get("law_name", ""))
    if (na, nb) in rel_map:
        desc, defer = rel_map[(na, nb)]
        return a, b, desc, defer
    if (nb, na) in rel_map:
        desc, defer = rel_map[(nb, na)]
        return b, a, desc, defer

    sa, sb = (a.get("scope") or "").strip(), (b.get("scope") or "").strip()
    if {sa, sb} == {"特别", "一般"}:
        special = a if sa == "特别" else b
        general = b if sa == "特别" else a
        return special, general, "", False
    return None


# 兼容旧调用方：RANK 仍可用，但不包含司法解释（避免被误当作静态位阶使用）
RANK = dict(BASE_RANK)

# 合法位阶取值（司法解释需配合 host_law 使用，故单独列出）
_VALID_HIERARCHIES = frozenset(set(BASE_RANK) | {"司法解释"})


# ————————————————————————————————————————————————
# 规则一：时间效力定位（法不溯及既往 + 版本区间）
# ————————————————————————————————————————————————
def temporal_locate(act_date: str, versions: Sequence[Dict]) -> RuleOutcome:
    """判断「行为发生日」应适用哪一版本。

    versions 形如：
        [{"version": "2021版", "effective_from": "2021-01-01", "effective_to": None}, ...]
    """
    ad = _parse(act_date)
    if ad is None:
        return RuleOutcome(
            status=RuleStatus.MISS,
            reason="行为发生日缺失或无法解析，无法进行时间效力定位",
            confidence=Confidence.C,
            review=ReviewLevel.MUST,
            llm_allowed=True,
            constraints=_hard_constraints(),
            verify_actions=["向用户确认法律事实发生的具体日期"],
        )

    if not versions:
        return RuleOutcome(
            status=RuleStatus.MISS,
            reason="知识库缺少该法的版本时间线，无法定位适用版本",
            confidence=Confidence.C,
            review=ReviewLevel.EXPERT,
            llm_allowed=True,
            constraints=_hard_constraints(),
            verify_actions=["补充该法的历次版本及生效/失效时间线"],
        )

    missing = [v for v in versions if _parse(v.get("effective_from")) is None]
    if missing:
        return RuleOutcome(
            status=RuleStatus.PARTIAL,
            reason="部分版本缺少生效日期，只能完成不严格的定位",
            confidence=Confidence.B,
            review=ReviewLevel.MUST,
            llm_allowed=True,
            constraints=_hard_constraints(),
            verify_actions=["补齐缺失版本的生效/失效日期"],
            detail={"missing_versions": [v.get("version") for v in missing]},
        )

    hits = []
    for v in versions:
        f = _parse(v.get("effective_from"))
        t = _parse(v.get("effective_to"))
        if f and ad >= f and (t is None or ad <= t):
            hits.append(v)

    if len(hits) == 1:
        return RuleOutcome(
            status=RuleStatus.HIT,
            reason=f"行为发生日 {act_date} 落在唯一版本的有效区间内",
            confidence=Confidence.A,
            review=ReviewLevel.SAMPLE,
            llm_allowed=False,
            detail={"applicable_version": hits[0].get("version")},
        )

    if len(hits) > 1:
        return RuleOutcome(
            status=RuleStatus.CONFLICT,
            reason="同一行为发生日命中多个版本，版本时间线互相重叠",
            confidence=Confidence.D,
            review=ReviewLevel.IMMEDIATE,
            llm_allowed=False,
            verify_actions=["修正版本时间线，消除重叠区间后再判定"],
            detail={"overlapping": [v.get("version") for v in hits]},
        )

    starts = [s for s in (_parse(v.get("effective_from")) for v in versions) if s]
    earliest = min(starts)
    if ad < earliest:
        return RuleOutcome(
            status=RuleStatus.HIT,
            reason=f"行为发生日 {act_date} 早于该法最早生效日 {earliest}，依法不溯及既往，该法不适用",
            confidence=Confidence.A,
            review=ReviewLevel.SAMPLE,
            llm_allowed=False,
            verify_actions=["确认行为发生日；如主张溯及适用，须核查是否存在特别规定（如有利溯及/空白溯及）"],
            detail={"applicable": False},
        )

    return RuleOutcome(
        status=RuleStatus.MISS,
        reason="行为发生日落在版本空档期（旧版已失效、新版未生效），适用依据不明",
        confidence=Confidence.C,
        review=ReviewLevel.EXPERT,
        llm_allowed=True,
        constraints=_hard_constraints(),
        verify_actions=["核查该空档期是否存在过渡条款或衔接规定"],
    )


# ————————————————————————————————————————————————
# 规则一·条文级：多维时间适用性判定（实体从旧 / 程序从新 / 司法解释审理时点 / 从旧兼从轻 / 持续性事实）
# ————————————————————————————————————————————————
def locate_article(
    act_date: str,
    effective_from: str,
    versions: Sequence[Dict] = (),
    effective_to: str = "",
    law_name: str = "",
    article_effective_from: str = "",
    norm_type: str = "",
    host_effective_from: str = "",
    ref_date: str = "",
    act_end_date: str = "",
) -> RuleOutcome:
    """判定「某一条文」在行为发生日（及审理/咨询时点）是否适用。

    核心规则族：
    1. 条文级原始生效日优先（`article_effective_from`）：若法律历经修正但该条文未作实质修改，
       按该条文首次施行日判定，避免将「2012年劳动合同法修正施行日 2013-07-01」误判为未修改条文（如第38条）的生效日。
    2. 持续性法律事实（`act_end_date`）：依《民法典时间效力规定》第1条第3款，行为始于旧法期间、持续至新法施行后的，适用新法。
    3. 程序法「程序从新」（`norm_type="procedural"`）：诉讼法、复议法等程序规定按审理/咨询基准日 `ref_date` 适用现行有效规定。
    4. 司法解释时间效力（`norm_type="judicial_interpretation"`）：适用于施行后尚未终审的案件（基准为 `ref_date >= 解释施行日`），
       但行为发生日不得早于被解释法律的施行日（`host_effective_from`）。
    5. 行政处罚「从旧兼从轻」（`norm_type="administrative_penalty"`）：依《行政处罚法》第37条，行为发生于旧法期间、
       但作出处罚决定时新法处罚较轻或不认为是违法的，适用新法（标记 PARTIAL 提示律师核实）。
    6. 实体法一般原则：行为发生日 < 施行日 → 不溯及既往；行为发生日 > 失效日 → 已失效排除；
       落在旧法有效区间内 → 保留为「行为时旧法」并强制标注现已失效。
    """
    name = law_name or "该条"
    ad = _parse(act_date)
    if ad is None:
        return RuleOutcome(
            status=RuleStatus.MISS,
            reason="行为发生日缺失或无法解析，无法判定条文的时间适用性",
            confidence=Confidence.C,
            review=ReviewLevel.MUST,
            llm_allowed=True,
            constraints=_hard_constraints(),
            verify_actions=["向用户确认法律事实发生的具体日期"],
        )

    # 条文级原始生效日优先于法律修正案生效日
    law_f = _parse(effective_from)
    art_f = _parse(article_effective_from)
    f = art_f or law_f

    # 7. 条文修正日志穿透检查（若存在条款项级独立修正记录，精准对齐修正生效日）
    try:
        from app.kb import kb_store, normalize_article_id
        # 从 name 中提取法规名与条号，例如 《公司法》第142条
        m_name = re.search(r"《?([^》]+)》?\s*(第[零〇一二两三四五六七八九十百千0-9]+条)", name)
        if m_name:
            t_law, t_art = m_name.group(1).strip(), normalize_article_id(m_name.group(2).strip())
            revs = kb_store.list_article_revisions(t_law, t_art)
            if revs:
                # 找到与当前候选条文最匹配的修正日志生效日期
                matched_rev = revs[0]
                rev_from = _parse(matched_rev.get("effective_from"))
                if rev_from and (art_f is None or rev_from == art_f):
                    f = rev_from
                    if not effective_to and matched_rev.get("effective_to"):
                        effective_to = matched_rev.get("effective_to")
    except Exception:
        pass

    if f is None:
        return RuleOutcome(
            status=RuleStatus.PARTIAL,
            reason=f"{name}缺少生效日期，无法完成严格的时间适用性判定",
            confidence=Confidence.B,
            review=ReviewLevel.MUST,
            llm_allowed=True,
            constraints=_hard_constraints(),
            verify_actions=["补齐该条的施行日期"],
        )

    # 失效日：显式字段优先，其次从所属法律的版本时间线推导
    t = _parse(effective_to)
    if t is None:
        for v in versions or ():
            if (str(v.get("effective_from") or "")[:10]) == (str(effective_from)[:10]):
                t = _parse(v.get("effective_to"))
                break

    rd = _parse(ref_date) or date.today()
    end_d = _parse(act_end_date)
    # 持续性法律事实：若存在合法的持续结束日且晚于起始日，以结束日判断新法是否已生效覆盖
    effective_ad = end_d if (end_d and end_d > ad) else ad
    spanned_new_law = bool(end_d and ad < f <= end_d)

    # ─── 分支 A：程序法遵循「实体从旧、程序从新」 ───
    if norm_type == "procedural":
        if rd < f:
            return RuleOutcome(
                status=RuleStatus.HIT,
                reason=f"{name}（程序法）自 {f} 起施行，当前基准日 {rd} 早于施行日，尚未施行",
                confidence=Confidence.A,
                review=ReviewLevel.SAMPLE,
                llm_allowed=False,
                detail={"applicable": False, "historical": False, "note": f"不适用：该程序条文尚未施行（施行日 {f}）"},
            )
        if t is not None and rd > t:
            return RuleOutcome(
                status=RuleStatus.HIT,
                reason=f"{name}（程序法）已于 {t} 失效，依「程序从新」原则应适用当前有效的程序规定",
                confidence=Confidence.A,
                review=ReviewLevel.SAMPLE,
                llm_allowed=False,
                detail={"applicable": False, "historical": False, "note": f"不适用：程序法遵循程序从新，该旧条文已于 {t} 失效"},
            )
        return RuleOutcome(
            status=RuleStatus.HIT,
            reason=f"{name}系程序法规范，遵循「程序从新」原则，按当前审理/咨询时点（{rd}）适用现行有效规定",
            confidence=Confidence.A,
            review=ReviewLevel.SAMPLE,
            llm_allowed=False,
            detail={
                "applicable": True,
                "historical": False,
                "note": "适用：程序法遵循「程序从新」原则，适用现行有效程序规定",
            },
        )

    # ─── 分支 B：司法解释（按「施行后尚未终审案件适用 + 被解释法律生效边界」判定） ───
    if norm_type == "judicial_interpretation":
        host_f = _parse(host_effective_from)
        if host_f is None and "民法典" in name:
            host_f = date(2021, 1, 1)
        if rd < f:
            return RuleOutcome(
                status=RuleStatus.HIT,
                reason=f"{name}自 {f} 起施行，基准日 {rd} 早于施行日，不适用",
                confidence=Confidence.A,
                review=ReviewLevel.SAMPLE,
                llm_allowed=False,
                detail={"applicable": False, "historical": False, "note": f"不适用：该司法解释尚未施行（施行日 {f}）"},
            )
        if t is not None and rd > t:
            return RuleOutcome(
                status=RuleStatus.HIT,
                reason=f"{name}已于 {t} 废止，当前不再适用",
                confidence=Confidence.A,
                review=ReviewLevel.SAMPLE,
                llm_allowed=False,
                detail={"applicable": False, "historical": False, "note": f"不适用：该司法解释已于 {t} 废止"},
            )
        if host_f is not None and effective_ad < host_f:
            return RuleOutcome(
                status=RuleStatus.HIT,
                reason=(
                    f"行为发生日 {act_date} 早于被解释法律施行日 {host_f}，"
                    f"依法不适用被解释法律及其配套司法解释{name}"
                ),
                confidence=Confidence.A,
                review=ReviewLevel.SAMPLE,
                llm_allowed=False,
                detail={
                    "applicable": False,
                    "historical": False,
                    "note": f"不适用：行为发生于被解释法律施行日（{host_f}）之前",
                },
                verify_actions=["如主张适用新法及本解释，须核查是否符合《民法典时间效力规定》有利溯及或空白溯及情形"],
            )
        if effective_ad < f:
            return RuleOutcome(
                status=RuleStatus.HIT,
                reason=(
                    f"司法解释时间效力：行为发生日 {act_date} 在被解释法律施行之后，且当前（{rd}）处于"
                    f"{name}（{f} 施行）生效后的未终审/咨询阶段，依法适用该司法解释"
                ),
                confidence=Confidence.A,
                review=ReviewLevel.SAMPLE,
                llm_allowed=False,
                detail={
                    "applicable": True,
                    "historical": False,
                    "note": f"适用：司法解释适用于施行（{f}）后尚未终审的案件（行为发生于被解释法律生效期内）",
                },
            )

    # ─── 分支 C：行政处罚「从旧兼从轻」原则（《行政处罚法》第37条） ───
    if norm_type == "administrative_penalty" and effective_ad < f and (t is None or rd <= t):
        return RuleOutcome(
            status=RuleStatus.PARTIAL,
            reason=(
                f"从旧兼从轻原则（《行政处罚法》第37条）：行为发生日 {act_date} 早于{name}施行日 {f}，"
                f"原则上适用行为时规定；但若作出行政处罚决定时新规已施行，且新规处罚较轻或不认为是违法的，适用新规"
            ),
            confidence=Confidence.B,
            review=ReviewLevel.MUST,
            llm_allowed=True,
            constraints=_hard_constraints(),
            verify_actions=[
                "核实行政处罚决定是否在 2021-07-15 之后作出",
                "比对新旧法条款，确认新规是否属于「处罚较轻或者不认为是违法」（如首违不罚/无主观过错不罚）",
            ],
            detail={
                "applicable": True,
                "historical": False,
                "note": f"条件适用（从旧兼从轻）：行为发生于 {act_date}，仅当处罚决定在 {f} 后作出且新规更轻时适用本条",
            },
        )

    # ─── 分支 D：实体法一般原则（法不溯及既往 + 行为时旧法保留） ───
    if effective_ad < f:
        return RuleOutcome(
            status=RuleStatus.HIT,
            reason=f"{name}自 {f} 起施行，行为发生日 {act_date} 早于施行日，依法不溯及既往，不适用",
            confidence=Confidence.A,
            review=ReviewLevel.SAMPLE,
            llm_allowed=False,
            detail={
                "applicable": False,
                "historical": False,
                "note": f"不适用：该条尚未施行（施行日 {f}）",
            },
            verify_actions=["如主张溯及适用，须核查是否存在特别规定（如《民法典时间效力规定》第2条有利溯及、第3条空白溯及）"],
        )

    if t is not None and effective_ad > t:
        return RuleOutcome(
            status=RuleStatus.HIT,
            reason=f"{name}已于 {t} 失效，行为发生日 {effective_ad} 在其失效之后，不适用",
            confidence=Confidence.A,
            review=ReviewLevel.SAMPLE,
            llm_allowed=False,
            detail={
                "applicable": False,
                "historical": False,
                "note": f"不适用：该条已于 {t} 失效，行为发生在失效之后",
            },
        )

    if t is not None:
        return RuleOutcome(
            status=RuleStatus.HIT,
            reason=f"行为发生日 {act_date} 落在该条有效区间（{f} ~ {t}）内，适用行为时的旧法",
            confidence=Confidence.A,
            review=ReviewLevel.MUST,
            llm_allowed=False,
            detail={
                "applicable": True,
                "historical": True,
                "effective_to": str(t),
                "note": f"适用行为时旧法（该条已于 {t} 失效，现不得以该条主张现行权利）",
            },
            verify_actions=["确认现行替代依据，并核对新旧法的衔接适用规则"],
        )

    if spanned_new_law:
        return RuleOutcome(
            status=RuleStatus.HIT,
            reason=(
                f"持续性法律事实自 {ad} 持续至 {end_d}（跨越新法施行日 {f}），"
                f"依《民法典时间效力规定》第1条第3款等持续性事实规则，适用新法{name}"
            ),
            confidence=Confidence.A,
            review=ReviewLevel.SAMPLE,
            llm_allowed=False,
            detail={
                "applicable": True,
                "historical": False,
                "note": f"适用：持续性法律事实持续至新法施行日（{f}）之后，依法适用新法",
            },
        )

    if art_f and law_f and art_f < law_f and ad < law_f:
        return RuleOutcome(
            status=RuleStatus.HIT,
            reason=(
                f"行为发生日 {act_date} 不早于{name}首次施行日 {art_f}"
                f"（后续 {law_f} 修正未改变本条实质内容），该条有效适用"
            ),
            confidence=Confidence.A,
            review=ReviewLevel.SAMPLE,
            llm_allowed=False,
            detail={
                "applicable": True,
                "historical": False,
                "note": f"适用：该条自 {art_f} 起即施行有效（{law_f} 修正未改动本条）",
            },
        )

    return RuleOutcome(
        status=RuleStatus.HIT,
        reason=f"行为发生日 {act_date} 不早于施行日 {f}，且该条现行有效",
        confidence=Confidence.A,
        review=ReviewLevel.SAMPLE,
        llm_allowed=False,
        detail={"applicable": True, "historical": False, "note": "适用：现行有效"},
    )


# ————————————————————————————————————————————————
# 规则二：位阶冲突裁决（上位法 / 特别法 / 新法，严格符合《立法法》2023修正第100~106条）
# ————————————————————————————————————————————————
def resolve_hierarchy(a: Dict, b: Dict) -> RuleOutcome:
    """两条规范的位阶与适用关系裁决。

    a / b 形如：{"law_name": ..., "hierarchy": ..., "host_law": ..., "host_hierarchy": ...,
                 "promulgation_date": ..., "authority": ..., "scope": "特别|一般",
                 "irreconcilable_conflict": bool}

    关键修正（P0）：
    1. 部门规章 vs 地方政府规章：依《立法法》第102条、第106条具有同等效力，冲突由国务院裁决，绝不误判为上下位法。
    2. 同位阶同机关时：先判「特别法 vs 一般法」，再判「新法 vs 旧法」！
       - 若属「新的一般规定 vs 旧的特别规定」：
         * 若新一般法设有明确特别法转致条款（如《民法典》第11条/第128条）且非不可调和冲突 → 适用特别法（HIT）；
         * 若无转致条款（或明确冲突无法确定如何适用）→ 依《立法法》第105条、第106条返回 CONFLICT（禁止 LLM 自行裁决，转有权机关/人工）。
    3. 全面采用 2023 年修正后的《立法法》条号（第100、102、103、105、106条，括号兼容原第92、95条）。
    """
    ha, hb = a.get("hierarchy", ""), b.get("hierarchy", "")

    if not ha or not hb or ha not in _VALID_HIERARCHIES or hb not in _VALID_HIERARCHIES:
        return RuleOutcome(
            status=RuleStatus.MISS,
            reason="位阶元数据缺失，无法执行确定性裁决",
            confidence=Confidence.C,
            review=ReviewLevel.MUST,
            llm_allowed=True,
            constraints=_hard_constraints(),
            verify_actions=["核实并补齐两条文的位阶与制定机关"],
        )

    # 1) 司法解释 vs 被解释法律：这是「解释关系」而非冲突，被解释法律为准、解释用于细化
    for x, y in ((a, b), (b, a)):
        host = (x.get("host_law") or "").strip()
        if x.get("hierarchy") == "司法解释" and host and host == (y.get("law_name") or "").strip():
            return RuleOutcome(
                status=RuleStatus.HIT,
                reason=f"司法解释依附于被解释法律：{x.get('law_name')}系对《{y.get('law_name')}》的解释，"
                       f"适用时以《{y.get('law_name')}》为准并参照该解释细化",
                confidence=Confidence.A,
                review=ReviewLevel.SAMPLE,
                llm_allowed=False,
                detail={"precedence": y.get("law_name"), "relation": "interpretation"},
            )

    # 2) 不可比位阶：法律未定优劣，规则层与 LLM 均无权判定，须走法定裁决程序
    if is_incomparable(ha, hb):
        if {ha, hb} == {_DEPT, _LOCAL_GOV}:
            reason = (
                f"{ha}与{hb}之间依《立法法》第102条具有同等效力；对同一事项规定不一致时，"
                f"依《立法法》第106条第1款第3项（原第95条）由国务院裁决，规则层与模型均无权自行判定优劣"
            )
            action = "按《立法法》第106条第1款第3项提请国务院裁决"
        else:
            reason = (
                f"{ha}与{hb}之冲突，法律未直接规定何者优先，依《立法法》第106条第1款第2项（原第95条）"
                f"应由国务院提出意见、认为应当适用部门规章的提请全国人大常委会裁决，规则层无权判定"
            )
            action = "按《立法法》第106条第1款第2项（原第95条）提请国务院/全国人大常委会裁决"
        return RuleOutcome(
            status=RuleStatus.MISS,
            reason=reason,
            confidence=Confidence.D,
            review=ReviewLevel.IMMEDIATE,
            llm_allowed=False,
            verify_actions=[action],
        )

    # 3) 位阶不同 → 上位法优于下位法（确定性）
    ra = rank_of(ha, a.get("host_hierarchy", ""))
    rb = rank_of(hb, b.get("host_hierarchy", ""))
    if ra != rb:
        upper, lower = (a, b) if ra < rb else (b, a)
        return RuleOutcome(
            status=RuleStatus.HIT,
            reason=f"上位法优于下位法：《{upper.get('law_name')}》（{upper.get('hierarchy')}）优先于《{lower.get('law_name')}》（{lower.get('hierarchy')}）",
            confidence=Confidence.A,
            review=ReviewLevel.SAMPLE,
            llm_allowed=False,
            detail={"precedence": upper.get("law_name"), "relation": "upper_lower"},
        )

    # 4) 同位阶：检查制定机关是否同一（《立法法》第103条「原第92条」适用前提）
    auth_a, auth_b = a.get("authority", ""), b.get("authority", "")
    if not auth_a or not auth_b:
        return RuleOutcome(
            status=RuleStatus.PARTIAL,
            reason="同位阶但制定机关未知，无法确认是否适用《立法法》第103条（原第92条，特别法/新法规则仅限同一机关）",
            confidence=Confidence.B,
            review=ReviewLevel.MUST,
            llm_allowed=True,
            constraints=_hard_constraints(),
            verify_actions=["核实两条文的制定机关是否同一"],
        )

    if not _same_legislative_body(auth_a, auth_b):
        return RuleOutcome(
            status=RuleStatus.MISS,
            reason="同位阶但由不同机关制定，《立法法》第103条（原第92条）不适用，需按第106条（原第95条）程序处理",
            confidence=Confidence.D,
            review=ReviewLevel.IMMEDIATE,
            llm_allowed=False,
            verify_actions=["按《立法法》第106条（原第95条）确定裁决机关与程序"],
        )

    # 5) 同一立法机关：先判「特别法 vs 一般法」，绝不先按日期盲目套用「新法优于旧法」！
    sg = identify_special_general(a, b)
    if sg is not None:
        special, general, rel_desc, has_deference = sg
        d_spec = _parse(special.get("promulgation_date"))
        d_gen = _parse(general.get("promulgation_date"))
        irreconcilable = bool(a.get("irreconcilable_conflict") or b.get("irreconcilable_conflict"))

        # 检查是否构成「新的一般规定 vs 旧的特别规定」（《立法法》第105条、第106条第1款第1项）
        if d_spec and d_gen and d_gen > d_spec:
            if has_deference and not irreconcilable:
                return RuleOutcome(
                    status=RuleStatus.HIT,
                    reason=(
                        f"同一机关制定，特别法优于一般法（《立法法》第103条；{rel_desc}）："
                        f"《{special.get('law_name')}》为特别规定，优先适用"
                    ),
                    confidence=Confidence.A,
                    review=ReviewLevel.SAMPLE,
                    llm_allowed=False,
                    detail={
                        "precedence": special.get("law_name"),
                        "secondary": general.get("law_name"),
                        "relation": "special_general",
                    },
                )
            return RuleOutcome(
                status=RuleStatus.CONFLICT,
                reason=(
                    f"同一机关制定的新的一般规定《{general.get('law_name')}》（{d_gen}）与旧的特别规定"
                    f"《{special.get('law_name')}》（{d_spec}）不一致，依《立法法》第105条、第106条不能直接套用"
                    f"「新法优于旧法」，须核查是否存在特别法保留/转致条款或依法提请制定机关裁决"
                ),
                confidence=Confidence.D,
                review=ReviewLevel.IMMEDIATE,
                llm_allowed=False,
                verify_actions=[
                    "核查新的一般规定中是否设有「其他法律有特别规定的，依照其规定」之转致条款",
                    "若确系同一事项实质冲突且无法确定如何适用，按《立法法》第105条/第106条提请全国人大常委会/国务院/制定机关裁决",
                ],
                detail={
                    "special": special.get("law_name"),
                    "general": general.get("law_name"),
                    "relation": "new_general_vs_old_special",
                },
            )

        extra = f"（{rel_desc}）" if rel_desc else ""
        return RuleOutcome(
            status=RuleStatus.HIT,
            reason=f"同一机关制定，特别法优于一般法（《立法法》第103条）{extra}：《{special.get('law_name')}》为特别规定",
            confidence=Confidence.A,
            review=ReviewLevel.SAMPLE,
            llm_allowed=False,
            detail={
                "precedence": special.get("law_name"),
                "secondary": general.get("law_name"),
                "relation": "special_general",
            },
        )

    # 6) 非特别/一般关系时，判「新法优于旧法」（《立法法》第103条）
    da, db = _parse(a.get("promulgation_date")), _parse(b.get("promulgation_date"))
    if da and db and da != db:
        newer = a if da > db else b
        return RuleOutcome(
            status=RuleStatus.HIT,
            reason=f"同一机关制定，新法优于旧法（《立法法》第103条）：《{newer.get('law_name')}》较新",
            confidence=Confidence.A,
            review=ReviewLevel.SAMPLE,
            llm_allowed=False,
            detail={"precedence": newer.get("law_name"), "relation": "new_old"},
        )

    return RuleOutcome(
        status=RuleStatus.PARTIAL,
        reason="同位阶同机关，但新旧关系与特别/一般关系均无法从元数据判定",
        confidence=Confidence.B,
        review=ReviewLevel.MUST,
        llm_allowed=True,
        constraints=_hard_constraints(),
        verify_actions=["人工确认两条文属新旧关系还是特别/一般关系"],
    )


# ————————————————————————————————————————————————
# 兜底计划：把判定结果翻译成「谁来定、带什么约束、谁来审」
# ————————————————————————————————————————————————
def fallback_plan(o: RuleOutcome) -> Dict:
    if o.status == RuleStatus.HIT:
        mode = "deterministic"
    elif o.status == RuleStatus.CONFLICT:
        mode = "escalate"
    elif o.status == RuleStatus.PARTIAL:
        mode = "llm_assisted"
    else:  # MISS
        mode = "llm_fallback" if o.llm_allowed else "escalate"

    return {
        "mode": mode,
        "confidence": o.confidence.value,
        "review": o.review.value,
        "llm_callable": o.llm_allowed and mode in ("llm_assisted", "llm_fallback"),
        "prompt_constraints": list(o.constraints),
        "verify_actions": list(o.verify_actions),
        "reason": o.reason,
    }


def prompt_footer_from_plan(plan: Dict) -> str:
    """由 fallback_plan 产出的 dict 生成提示词约束（状态中只存 dict，便于序列化）。"""
    if not plan.get("llm_callable"):
        return ""
    lines = ["【输出约束｜必须严格遵守】"]
    lines += [f"- {c}" for c in plan.get("prompt_constraints", [])]
    if plan.get("verify_actions"):
        lines.append("- 建议核实事项：" + "；".join(plan["verify_actions"]))
    lines.append(f"- 置信度等级：{plan.get('confidence')}（原因：{plan.get('reason')}）")
    return "\n".join(lines)


def prompt_footer(o: RuleOutcome) -> str:
    """把兜底约束拼成可直接追加到 LLM 提示词尾部的文本。"""
    return prompt_footer_from_plan(fallback_plan(o))


# ————————————————————————————————————————————————
# 多结果聚合：一次请求往往命中多条规则，取「最坏」者作为整体策略
# ————————————————————————————————————————————————
_SEVERITY = {
    RuleStatus.HIT: 0,
    RuleStatus.PARTIAL: 1,
    RuleStatus.MISS: 2,
    RuleStatus.CONFLICT: 3,
}
_CONF_ORDER = {Confidence.A: 0, Confidence.B: 1, Confidence.C: 2, Confidence.D: 3}
_REVIEW_ORDER = {
    ReviewLevel.SAMPLE: 0,
    ReviewLevel.MUST: 1,
    ReviewLevel.EXPERT: 2,
    ReviewLevel.IMMEDIATE: 3,
    ReviewLevel.REJECT: 4,
}


def aggregate_outcomes(
    outcomes: Sequence[RuleOutcome], fallback_reason: str = "未产生任何规则判定条目"
) -> RuleOutcome:
    """把多条规则判定合成一个整体结果：按严重度取最坏者，约束/核实项取并集。

    安全不变量：只要有一条判定禁止 LLM（如 CONFLICT 或依法需裁决），整体即禁止兜底。
    """
    items = [o for o in outcomes if o is not None]
    if not items:
        return RuleOutcome(
            status=RuleStatus.MISS,
            reason=fallback_reason,
            confidence=Confidence.C,
            review=ReviewLevel.MUST,
            llm_allowed=True,
            constraints=_hard_constraints(),
            verify_actions=["补充待判定的条文与元数据"],
        )

    worst = max(
        items,
        key=lambda o: (_SEVERITY[o.status], _REVIEW_ORDER[o.review], _CONF_ORDER[o.confidence]),
    )

    seen, reasons = set(), []
    for o in items:
        if o.reason and o.reason not in seen:
            seen.add(o.reason)
            reasons.append(o.reason)

    verify: List[str] = []
    for o in items:
        for v in o.verify_actions:
            if v not in verify:
                verify.append(v)

    constraints: List[str] = []
    for o in items:
        for c in o.constraints:
            if c not in constraints:
                constraints.append(c)

    # HIT 的 llm_allowed=False 语义是「不需要 LLM」，而非「禁止 LLM」。
    # 因此只有显式禁止的情形（CONFLICT / 依法需裁决）才让整体禁止兜底。
    forbidden = any(
        o.status == RuleStatus.CONFLICT
        or (o.status in (RuleStatus.MISS, RuleStatus.PARTIAL) and not o.llm_allowed)
        for o in items
    )
    llm_allowed = (not forbidden) and any(o.llm_allowed for o in items)

    return RuleOutcome(
        status=worst.status,
        reason="；".join(reasons),
        confidence=worst.confidence,
        review=max((o.review for o in items), key=lambda r: _REVIEW_ORDER[r]),
        llm_allowed=llm_allowed,
        constraints=constraints,
        verify_actions=verify,
        detail={
            "components": [
                {"status": o.status.value, "reason": o.reason} for o in items
            ]
        },
    )
