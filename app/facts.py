"""法律事实要素抽取、案由要件核对、地域管辖识别与程序/时效红线提示（P0「小前提」防错层）。

设计原则：
1. 法律三段论 = 大前提（法规）+ 小前提（事实）→ 结论。缺关键小前提时不得给出高置信度断语。
2. 时间、地域、主体/行为要件、程序时效是法律适用的四大事实支柱：
   - 时间不明：不得静默按今日给 A 级置信度，须标注假设前提并降级为 PARTIAL(B)/必审；
   - 地域不明或异地：地方性法规/地方政府规章不得跨行政区错用；
   - 要件缺失：按案由生成待核实事实清单（missing_facts）与追问问题（clarification_questions）；
   - 程序与时效：自动附带仲裁前置、诉讼时效、除斥期间、复议期限等程序红线提示。
3. 零第三方依赖，纯标准库实现，支持离线单测与规则引擎无缝集成。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional, Tuple

# ————————————————————————————————————————————————
# 行政区划词表（用于识别用户问题所属地域及地方性法规效力范围）
# ————————————————————————————————————————————————
PROVINCE_CITY_MAP: Dict[str, str] = {
    "北京": "北京", "北京市": "北京",
    "上海": "上海", "上海市": "上海",
    "天津": "天津", "天津市": "天津",
    "重庆": "重庆", "重庆市": "重庆",
    "广东": "广东", "广东省": "广东", "广州": "广东", "深圳": "广东", "珠海": "广东", "佛山": "广东", "东莞": "广东",
    "浙江": "浙江", "浙江省": "浙江", "杭州": "浙江", "宁波": "浙江", "温州": "浙江",
    "江苏": "江苏", "江苏省": "江苏", "南京": "江苏", "苏州": "江苏", "无锡": "江苏",
    "山东": "山东", "山东省": "山东", "济南": "山东", "青岛": "山东",
    "四川": "四川", "四川省": "四川", "成都": "四川",
    "湖北": "湖北", "湖北省": "湖北", "武汉": "湖北",
    "湖南": "湖南", "湖南省": "湖南", "长沙": "湖南",
    "河南": "河南", "河南省": "河南", "郑州": "河南",
    "河北": "河北", "河北省": "河北", "石家庄": "河北",
    "福建": "福建", "福建省": "福建", "福州": "福建", "厦门": "福建",
    "安徽": "安徽", "安徽省": "安徽", "合肥": "安徽",
    "陕西": "陕西", "陕西省": "陕西", "西安": "陕西",
    "辽宁": "辽宁", "辽宁省": "辽宁", "沈阳": "辽宁", "大连": "辽宁",
    "江西": "江西", "江西省": "江西", "南昌": "江西",
    "云南": "云南", "云南省": "云南", "昆明": "云南",
    "广西": "广西", "南宁": "广西",
    "山西": "山西", "太原": "山西",
    "贵州": "贵州", "贵阳": "贵州",
    "吉林": "吉林", "长春": "吉林",
    "黑龙江": "黑龙江", "哈尔滨": "黑龙江",
    "内蒙古": "内蒙古", "呼和浩特": "内蒙古",
    "新疆": "新疆", "乌鲁木齐": "新疆",
    "甘肃": "甘肃", "兰州": "甘肃",
    "海南": "海南", "海口": "海南", "三亚": "海南",
    "宁夏": "宁夏", "银川": "宁夏",
    "青海": "青海", "西宁": "青海",
    "西藏": "西藏", "拉萨": "西藏",
}


# ————————————————————————————————————————————————
# 六大高频案由：识别关键词、必备要件清单、法定程序与时效红线
# ————————————————————————————————————————————————
CASE_TYPE_RULES: Dict[str, Dict] = {
    "劳动": {
        "name": "劳动人事争议",
        "keywords": ["工资", "欠薪", "拖欠", "劳动合同", "辞退", "解雇", "解除劳动", "用人单位", "员工", "经济补偿", "赔偿金", "双倍工资", "二倍工资", "社保", "工伤", "加班"],
        "elements": [
            ("劳动关系与入职时间", ("入职", "用工", "工作满", "工作了", "年限", "年"), "确认建立劳动关系的具体日期（用工之日）及已工作年限"),
            ("书面劳动合同情况", ("未签", "没签", "签订", "书面劳动合同", "劳动合同"), "确认是否已签订书面劳动合同，以及未签合同持续的月数"),
            ("工资支付与违约事实", ("拖欠", "欠薪", "未足额", "未及时", "社保", "克扣", "辞退"), "确认用人单位是否存在未及时足额支付劳动报酬、未缴社保或违法解除等法定情形及证据"),
            ("当前在职/离职状态与时间", ("离职", "在职", "解除", "辞职", "开除"), "确认目前仍在职还是已解除劳动关系，以及解除通知的送达时间（关系到1年仲裁时效起算）"),
            ("用工所在地", tuple(PROVINCE_CITY_MAP.keys()), "确认实际用工所在地（不同省市最低工资、社平工资及地方性劳动法规存在差异）"),
        ],
        "procedural_notes": [
            "【程序前置】依《劳动争议调解仲裁法》第5条、第79条，劳动争议实行「仲裁前置」，当事人须先向有管辖权的劳动人事争议仲裁委员会申请仲裁，对仲裁裁决不服方可向人民法院起诉。",
            "【仲裁时效】依《劳动争议调解仲裁法》第27条，劳动争议申请仲裁的时效期间为 1 年（自知道或者应当知道权利被侵害之日起算）；拖欠劳动报酬争议在劳动关系存续期间不受 1 年限制，但劳动关系终止的，应当自终止之日起 1 年内提出。",
            "【二倍工资时效】主张未订立书面劳动合同二倍工资差额（最长期限11个月，自用工满1个月的次日至满1年的前一日）亦受 1 年仲裁时效限制。",
        ],
    },
    "合同": {
        "name": "民商事合同纠纷",
        "keywords": ["合同", "违约", "违约金", "解除合同", "继续履行", "赔偿损失", "定金", "押金", "货款", "付款", "交货", "逾期", "欺诈", "撤销"],
        "elements": [
            ("合同订立与履行时间", ("年", "月", "日", "签订", "订立", "到期"), "确认合同成立/生效时间及约定的履行期限（决定适用《民法典》还是旧《合同法》）"),
            ("违约具体情形与催告情况", ("不履行", "逾期", "迟延", "催告", "拒绝", "瑕疵", "违约"), "确认对方违约的具体表现，若属迟延履行主要债务，确认是否已发出催告函并给予合理期限"),
            ("损失金额与违约金约定", ("损失", "违约金", "赔偿", "金额", "元", "万"), "确认合同约定的违约金计算标准及守约方实际遭受的直接损失与可得利益损失"),
        ],
        "procedural_notes": [
            "【诉讼时效】依《民法典》第188条，向人民法院请求保护民事权利的普通诉讼时效期间为 3 年，自知道或者应当知道权利受到损害以及义务人之日起计算。",
            "【除斥期间】若主张受欺诈/胁迫请求撤销合同（《民法典》第148条、第152条），应当自知道或者应当知道撤销事由之日起 1 年内行使撤销权（该期间为除斥期间，不适用中止、中断）。",
            "【违约金调整】依《民法典合同编通则解释》第65条，约定违约金超过造成损失 30% 的，一般可认定为「过分高于造成的损失」，违约方（恶意违约除外）可请求法院或仲裁机构予以适当减少。",
        ],
    },
    "公司": {
        "name": "公司与股东出资纠纷",
        "keywords": ["公司", "股东", "出资", "认缴", "实缴", "注册资本", "抽逃出资", "股东会", "董事会", "分红", "利润分配", "连带责任", "法人人格否认"],
        "elements": [
            ("公司设立与出资行为时间", ("年", "月", "成立", "设立", "章程"), "确认公司设立日期、章程约定的认缴出资期限及争议行为发生日（2024-07-01 新《公司法》施行前后规则不同）"),
            ("认缴与实缴明细", ("认缴", "实缴", "未缴", "出资额"), "确认股东认缴出资额、已实缴金额及是否届满认缴期限"),
            ("债权受损与滥用权利事实", ("不能清偿", "逃避债务", "混同", "严重损害"), "若主张股东对公司债务承担连带责任或出资加速到期，需确认公司是否已不能清偿到期债务或存在财产混同"),
        ],
        "procedural_notes": [
            "【新旧法衔接】2023年修订《公司法》自 2024-07-01 起施行，规定有限责任公司认缴出资额须自成立之日起 5 年内缴足（第47条），存量公司适用国务院过渡期规定。",
            "【出资加速到期】依新《公司法》第54条，公司不能清偿到期债务的，公司或者已到期债权的债权人有权要求已认缴出资但未届出资期限的股东提前缴纳出资。",
        ],
    },
    "侵权": {
        "name": "侵权与交通事故责任纠纷",
        "keywords": ["撞伤", "交通事故", "侵权", "损害赔偿", "送货途中", "执行工作任务", "劳务派遣", "医疗费", "伤残", "过错", "行人", "机动车"],
        "elements": [
            ("侵权发生时间与地点", ("年", "月", "日", "途中"), "确认侵权行为/事故发生的具体时间与地点"),
            ("行为人身份与职务关系", ("员工", "工作人员", "执行工作", "送货", "劳务派遣"), "确认侵权行为人是否系用人单位员工或劳务派遣人员，以及行为发生时是否在执行工作任务"),
            ("事故责任认定与保险情况", ("交警", "责任认定", "全责", "主责", "交强险", "商业险"), "确认交通事故责任认定书划分的过错比例，以及车辆交强险、商业三者险投保情况"),
        ],
        "procedural_notes": [
            "【诉讼时效】依《民法典》第188条，侵权损害赔偿请求权诉讼时效为 3 年，自知道或应当知道权利受损及义务人之日起算。",
            "【赔付顺位】机动车交通事故损害先由保险公司在交强险限额内予以赔偿；不足部分按商业三者险及侵权责任比例分担；属职务行为的由用人单位承担侵权责任（单位担责后可向有故意或重大过失的员工追偿）。",
        ],
    },
    "消费": {
        "name": "消费者权益保护纠纷",
        "keywords": ["消费者", "商家", "经营者", "退一赔三", "三倍赔偿", "假货", "虚假宣传", "格式条款", "霸王条款", "店堂告示"],
        "elements": [
            ("购买时间与交易凭证", ("年", "月", "购买", "订单", "发票"), "确认购买商品或接受服务的时间、消费金额及付款/订单凭证"),
            ("消费目的与欺诈事实", ("自用", "欺诈", "虚假", "格式条款"), "确认是否属于生活消费需要，以及经营者是否存在故意告知虚假情况或隐瞒真实情况的欺诈行为"),
        ],
        "procedural_notes": [
            "【维权途径与时效】依《消费者权益保护法》第39条，可通过协商、消协调解、行政投诉（12315）、仲裁或诉讼解决；民事诉讼时效为 3 年。",
        ],
    },
    "行政": {
        "name": "行政处罚与行政救济争议",
        "keywords": ["行政处罚", "罚款", "吊销", "行政机关", "行政复议", "行政诉讼", "违法行为轻微", "初次违法", "首违不罚", "听证", "陈述", "申辩"],
        "elements": [
            ("违法行为与处罚决定时间", ("年", "月", "日", "决定书", "收到"), "确认违法行为发生时间及收到《行政处罚决定书》的具体日期"),
            ("违法情节与主观过错", ("轻微", "初次", "及时改正", "没有造成危害", "主观过错"), "确认是否属于初次违法、危害后果轻微并及时改正，或是否有证据证明无主观过错"),
            ("行政程序合法性", ("告知", "听证", "陈述", "申辩"), "确认行政机关作出处罚前是否依法告知拟处罚内容、理由、依据及陈述申辩/听证权利"),
        ],
        "procedural_notes": [
            "【行政复议期限】依《行政复议法》（2023修订）第20条，公民、法人或者其他组织认为行政行为侵犯其合法权益的，可以自知道或者应当知道该行政行为之日起 60 日内提出行政复议申请。",
            "【行政诉讼期限】依《行政诉讼法》第46条，直接向人民法院提起诉讼的，应当自知道或者应当知道作出行政行为之日起 6 个月内提出；经复议的，自收到复议决定书之日起 15 日内提起诉讼。",
        ],
    },
}


@dataclass
class FactSummary:
    """从用户输入中抽取并核对后的结构化事实摘要。"""
    act_date: str                                     # 生效计算使用的行为发生日（YYYY-MM-DD）
    act_end_date: str = ""                            # 持续性行为的结束日（若有）
    act_date_source: str = "assumed_today"            # explicit | extracted | assumed_today
    date_granularity: str = "none"                    # day | month | year | none
    is_continuous: bool = False                       # 是否持续性法律事实
    region: str = ""                                  # 识别出的省级行政区（如 上海/北京）
    case_types: List[str] = field(default_factory=list)
    extracted_facts: List[str] = field(default_factory=list)
    missing_facts: List[str] = field(default_factory=list)
    clarification_questions: List[str] = field(default_factory=list)
    procedural_notes: List[str] = field(default_factory=list)
    suggested_citations: List[str] = field(default_factory=list)
    dynamic_proposal: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict:
        return {
            "act_date": self.act_date,
            "act_end_date": self.act_end_date,
            "act_date_source": self.act_date_source,
            "date_granularity": self.date_granularity,
            "is_continuous": self.is_continuous,
            "region": self.region,
            "case_types": list(self.case_types),
            "extracted_facts": list(self.extracted_facts),
            "missing_facts": list(self.missing_facts),
            "clarification_questions": list(self.clarification_questions),
            "procedural_notes": list(self.procedural_notes),
            "suggested_citations": list(self.suggested_citations),
            "dynamic_proposal": self.dynamic_proposal,
        }


def _norm_date(y: str, m: str = "06", d: str = "01") -> Optional[str]:
    try:
        dt = date(int(y), int(m), int(d))
        return dt.isoformat()
    except (ValueError, TypeError):
        return None


def extract_dates_from_text(
    query: str,
    explicit_act_date: Optional[str] = None,
    explicit_end_date: Optional[str] = None,
) -> Tuple[str, str, str, str, bool]:
    """返回 (act_date, act_end_date, source, granularity, is_continuous)。"""
    # 1) 显式传入的日期优先
    if explicit_act_date and str(explicit_act_date).strip():
        raw = str(explicit_act_date).strip()[:10]
        try:
            date.fromisoformat(raw)
            end_raw = ""
            if explicit_end_date and str(explicit_end_date).strip():
                end_raw = str(explicit_end_date).strip()[:10]
                date.fromisoformat(end_raw)
            return raw, end_raw, "explicit", "day", bool(end_raw and end_raw > raw)
        except ValueError:
            # 显式传入了非法日期字符串，原样返回以触发 locate_article 的日期格式失败分支
            return raw, "", "explicit", "invalid", False

    text = query or ""
    found: List[Tuple[str, str]] = []  # (iso_date, granularity)

    # 匹配 YYYY-MM-DD 或 YYYY/MM/DD 或 YYYY年M月D日
    for m in re.finditer(r"(19\d{2}|20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})日?", text):
        iso = _norm_date(m.group(1), m.group(2), m.group(3))
        if iso:
            found.append((iso, "day"))

    # 若未匹配到具体日，匹配 YYYY年M月
    if not found:
        for m in re.finditer(r"(19\d{2}|20\d{2})年(\d{1,2})月", text):
            iso = _norm_date(m.group(1), m.group(2), "01")
            if iso:
                found.append((iso, "month"))

    # 若仍未匹配到，匹配单独的 YYYY年（避免把“二倍/三年/五年”误识别，限定 1990~2035）
    if not found:
        for m in re.finditer(r"(199\d|20[0-3]\d)\s*年", text):
            iso = _norm_date(m.group(1), "06", "01")
            if iso:
                found.append((iso, "year"))

    is_cont = bool(re.search(r"持续|至今|一直|到现在|期间|至\s*(?:19|20)\d{2}|到\s*(?:19|20)\d{2}", text))

    if len(found) >= 2:
        dates_sorted = sorted(found, key=lambda x: x[0])
        return dates_sorted[0][0], dates_sorted[-1][0], "extracted", dates_sorted[0][1], True
    if len(found) == 1:
        end_d = date.today().isoformat() if is_cont and found[0][0] < date.today().isoformat() else ""
        return found[0][0], end_d, "extracted", found[0][1], bool(end_d)

    # 未提供且未抽取出任何日期：暂按今日计算，但标记 source="assumed_today"
    return date.today().isoformat(), "", "assumed_today", "none", is_cont


def extract_region(query: str, explicit_region: Optional[str] = None) -> str:
    """从问题或显式参数中识别省级行政区。"""
    if explicit_region and str(explicit_region).strip():
        r = str(explicit_region).strip()
        return PROVINCE_CITY_MAP.get(r, r)
    text = query or ""
    for kw, prov in PROVINCE_CITY_MAP.items():
        if kw in text:
            return prov
    return ""


def norm_region_of(law_name: str, source: str = "") -> str:
    """识别地方性法规/地方政府规章所属的省级行政区；全国性规范返回空字符串。"""
    combined = f"{law_name or ''} {source or ''}"
    for kw, prov in PROVINCE_CITY_MAP.items():
        if kw in combined:
            return prov
    return ""


def check_regional_applicability(
    law_name: str,
    hierarchy: str,
    source: str = "",
    user_region: str = "",
) -> Dict:
    """判定地方性法规/地方政府规章在用户地域下是否适用。

    返回 dict:
      - applicable: bool (是否保留)
      - regional_warning: bool (是否因地域未确认而需降级提示)
      - law_region: str
      - note: str
    """
    if hierarchy not in ("地方性法规", "地方政府规章"):
        return {"applicable": True, "regional_warning": False, "law_region": "", "note": ""}

    law_reg = norm_region_of(law_name, source)
    if not law_reg:
        return {"applicable": True, "regional_warning": False, "law_region": "", "note": ""}

    if user_region and user_region != law_reg:
        return {
            "applicable": False,
            "regional_warning": False,
            "law_region": law_reg,
            "note": f"地域效力不符：《{law_name}》系{law_reg}{hierarchy}，仅在{law_reg}行政区域内有效，不适用于{user_region}",
        }

    if not user_region:
        return {
            "applicable": True,
            "regional_warning": True,
            "law_region": law_reg,
            "note": f"地域效力待核实：《{law_name}》系{law_reg}{hierarchy}，仅在行为地/用工地/履行地位于{law_reg}时方可适用",
        }

    return {
        "applicable": True,
        "regional_warning": False,
        "law_region": law_reg,
        "note": f"地域效力匹配：适用{law_reg}{hierarchy}",
    }


from app.knowledge import knowledge_center


def analyze_facts(
    query: str,
    explicit_act_date: Optional[str] = None,
    explicit_end_date: Optional[str] = None,
    explicit_region: Optional[str] = None,
) -> FactSummary:
    """对用户问题执行全方位事实要素抽取、缺失要件检查与程序时效提示生成（支持外置案由规则热更新）。"""
    act_date, act_end_date, src, gran, is_cont = extract_dates_from_text(
        query, explicit_act_date, explicit_end_date
    )
    region = extract_region(query, explicit_region)
    text = query or ""

    case_rules = knowledge_center.get_case_type_rules() or CASE_TYPE_RULES
    matched_types: List[str] = []
    for ctype, cfg in case_rules.items():
        if any(kw in text for kw in cfg["keywords"]):
            matched_types.append(ctype)

    extracted: List[str] = []
    missing: List[str] = []
    questions: List[str] = []
    proc_notes: List[str] = []

    if src == "explicit":
        extracted.append(f"法律事实发生日：{act_date}" + (f" 至 {act_end_date}" if act_end_date else ""))
    elif src == "extracted":
        extracted.append(
            f"从问题识别出时间：{act_date}（粒度：{gran}"
            + (f"，持续至 {act_end_date}" if act_end_date else "")
            + "）"
        )
        if gran in ("year", "month"):
            missing.append("法律事实发生的具体日期（当前仅精确到年/月，若跨新旧法施行日需精确到日）")
            questions.append("请问相关法律事实（如签约、违约、入职或事故）发生的具体年月日是哪一天？")
    else:
        missing.append("法律事实发生的具体日期（决定适用现行法还是行为时旧法，以及是否超过诉讼/仲裁时效）")
        questions.append("请问该争议事实发生的具体时间（或持续起止期间）是什么时候？")

    if region:
        extracted.append(f"涉及地域：{region}")

    for ctype in matched_types:
        cfg = case_rules[ctype]
        for elem_name, triggers, verify_desc in cfg["elements"]:
            if any(t in text for t in triggers):
                extracted.append(f"[{cfg['name']}] 已提及「{elem_name}」相关线索")
            else:
                if verify_desc not in missing:
                    missing.append(verify_desc)
                    questions.append(f"请补充说明：{verify_desc}？")
        for pn in cfg["procedural_notes"]:
            if pn not in proc_notes:
                proc_notes.append(pn)

    suggested_cits: List[str] = []
    dyn_proposal: Optional[Dict[str, Any]] = None
    if not matched_types:
        try:
            from app.llm import infer_dynamic_elements

            dyn_info = infer_dynamic_elements(text)
            if dyn_info and dyn_info.get("inferred_case_type"):
                dyn_type = dyn_info["inferred_case_type"]
                matched_types.append(f"{dyn_type}（动态要件推演）")
                extracted.append(f"[{dyn_type}] 大模型法理推演核心构成要件：" + "、".join(dyn_info.get("core_elements", [])))
                for mf in dyn_info.get("missing_facts", []):
                    if mf not in missing:
                        missing.append(mf)
                for q in dyn_info.get("clarification_questions", []):
                    if q not in questions:
                        questions.append(q)
                for pn in dyn_info.get("procedural_notes", []):
                    if pn not in proc_notes:
                        proc_notes.append(pn)
                suggested_cits = dyn_info.get("suggested_citations", [])

                # 构造符合知识配置规范的案由规则提案（用于知识飞轮一键采纳）
                kw = [w for w in [dyn_type.replace("纠纷", ""), dyn_type] if w]
                elem_tuples = [
                    (elem, (elem[:2], elem), f"核实{elem}的相关事实与凭证")
                    for elem in dyn_info.get("core_elements", [])
                ]
                dyn_proposal = {
                    "case_type": dyn_type,
                    "name": dyn_type,
                    "keywords": kw,
                    "elements": elem_tuples,
                    "procedural_notes": list(dyn_info.get("procedural_notes", [])),
                    "suggested_citations": list(dyn_info.get("suggested_citations", [])),
                }
        except Exception:
            pass

    if not proc_notes:
        proc_notes.append(
            "【通用时效提示】民事权利普通诉讼时效为 3 年（《民法典》第188条）；劳动争议申请仲裁时效为 1 年；行政复议期限一般为 60 日、行政诉讼期限一般为 6 个月。"
        )

    return FactSummary(
        act_date=act_date,
        act_end_date=act_end_date,
        act_date_source=src,
        date_granularity=gran,
        is_continuous=is_cont,
        region=region,
        case_types=matched_types,
        extracted_facts=extracted,
        missing_facts=missing,
        clarification_questions=questions,
        procedural_notes=proc_notes,
        suggested_citations=suggested_cits,
        dynamic_proposal=dyn_proposal,
    )

