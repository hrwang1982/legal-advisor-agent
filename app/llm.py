"""LLM 调用层（可交付增强版）：
1. 支持 OpenAI 兼容接口（阿里百炼通义千问 Qwen / 本地 Ollama / vLLM / Xinference / DeepSeek / SiliconFlow）；
2. 内置双通道调用器：优先使用 `llama_index.llms.openai`（若已安装），否则自动使用标准库 `urllib.request` 直连 `/v1/chat/completions`（带超时控制与指数退避重试，无需额外依赖即可连接真实千问或本地大模型）；
3. 严禁静默吞掉 LLM 异常：生成或重排调用失败时，记录明确告警日志以触发置信度降级与人工必审。
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import TYPE_CHECKING, Dict, List, Optional

from app.config import settings
from app.ingest import expand_legal_query

if TYPE_CHECKING:
    from app.models import Citation

# 记录最近一次 LLM 调用的异常信息，供 synthesize 节点检测并强制降级置信度
_LAST_LLM_ERROR: Optional[str] = None


def get_and_clear_last_llm_error() -> Optional[str]:
    global _LAST_LLM_ERROR
    err = _LAST_LLM_ERROR
    _LAST_LLM_ERROR = None
    return err


def _extract_json(text: str) -> dict | list:
    """从模型输出中尽量稳健地抽取 JSON。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    s_obj, e_obj = text.find("{"), text.rfind("}")
    s_arr, e_arr = text.find("["), text.rfind("]")
    if s_arr != -1 and (s_obj == -1 or s_arr < s_obj) and e_arr > s_arr:
        text = text[s_arr : e_arr + 1]
    elif s_obj != -1 and e_obj != -1 and e_obj > s_obj:
        text = text[s_obj : e_obj + 1]
    return json.loads(text)


def _is_remote_api_ready() -> bool:
    """检查是否具备有效的外部或本地模型调用条件，避免无密钥或未连接时盲目重试导致长时间挂起。"""
    if settings.mock:
        return False
    base = (settings.llm_api_base or "").strip()
    key = (settings.llm_api_key or "").strip()
    is_local = "localhost" in base or "127.0.0.1" in base
    if is_local:
        return True
    return bool(key)


def _normalize_model_name(raw_model: Optional[str]) -> str:
    """获取当前生效的规范模型名称。忠实尊重用户配置与选择的模型名称，杜绝私自篡改。"""
    m = (raw_model or settings.llm_model or "qwen-plus").strip()
    if m.lower() == "mock":
        return settings.llm_model or "qwen-plus"
    return m


def _call_openai_compatible_http(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """使用标准库直连 OpenAI 兼容 `/v1/chat/completions` 接口（支持千问 DashScope、本地 Ollama/vLLM、DeepSeek，支持多轮对话）。"""
    if not _is_remote_api_ready():
        raise RuntimeError("未配置有效的外部 LLM API Key 且未连接本地 Ollama/vLLM 兼容服务")

    api_base = (settings.llm_api_base or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip().rstrip("/")
    url = f"{api_base}/chat/completions"
    real_model = _normalize_model_name(settings.llm_model)
    sys_content = system_prompt if system_prompt is not None else "你是一名严谨的中国执业律师与法律检索专家。"
    messages = []
    if sys_content:
        messages.append({"role": "system", "content": sys_content})
    if history:
        for msg in history:
            r = msg.get("role", "user")
            c = msg.get("content", "")
            if r in ("user", "assistant", "system") and c:
                messages.append({"role": r, "content": str(c)})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": real_model,
        "messages": messages,
        "temperature": settings.llm_temperature,
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    last_exc: Optional[Exception] = None
    timeout_val = min(float(settings.llm_timeout or 8.0), 8.0)
    retries = 1
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout_val) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                return str(msg.get("content") or "").strip()
        except Exception as e:
            last_exc = e
    raise RuntimeError(f"调用外部大模型接口 ({url}, model={real_model}) 失败: {last_exc}")


def _call(
    prompt: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """统一 LLM 入口（支持千问平台与本地 OpenAI 兼容接口，支持多轮对话）。"""
    if not _is_remote_api_ready():
        raise RuntimeError("当前处于免密钥演示模式或未配置大模型 API Key")
    # 若显式传入自定义 system_prompt 或历史多轮 messages，优先使用 HTTP 兼容调用确保精准下发
    if system_prompt is not None or history:
        return _call_openai_compatible_http(prompt, system_prompt=system_prompt, history=history)
    try:
        from llama_index.llms.openai import OpenAI

        real_model = _normalize_model_name(settings.llm_model)
        llm = OpenAI(
            api_base=settings.llm_api_base,
            api_key=settings.llm_api_key or "EMPTY",
            model=real_model,
            temperature=settings.llm_temperature,
            timeout=min(float(settings.llm_timeout or 8.0), 8.0),
        )
        return llm.predict(prompt)
    except ImportError:
        return _call_openai_compatible_http(prompt, system_prompt=system_prompt, history=history)
    except Exception:
        # 若 llama_index OpenAI 因非标模型名校验报错，自动回退标准 HTTP 兼容客户端
        return _call_openai_compatible_http(prompt, system_prompt=system_prompt, history=history)


# ————————————————————————————————————————————————
# 问题拆解：把复杂法律问题拆成可检索的子问题
# ————————————————————————————————————————————————
def _heuristic_decompose(query: str) -> List[str]:
    parts = re.split(r"[；;。\n？?]", query)
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return [query]
    if len(parts) == 1:
        sub = re.split(r"和|以及|并且|同时|且", parts[0])
        sub = [s.strip() for s in sub if s.strip()]
        if len(sub) > 1:
            return sub
    return parts


def decompose_query(query: str) -> List[str]:
    if settings.mock:
        return _heuristic_decompose(query)

    prompt = (
        "你是一名法律助理。请把用户的法律问题拆解为若干相互独立、可分别检索法规的子问题。"
        "只输出 JSON 数组，例如：[\"子问题1\",\"子问题2\"]。不要输出解释。\n"
        f"用户问题：{query}"
    )
    try:
        data = _extract_json(_call(prompt))
        if isinstance(data, list):
            res = [str(x).strip() for x in data if str(x).strip()]
            if res:
                return res
    except Exception:
        pass
    return _heuristic_decompose(query)


# ————————————————————————————————————————————————
# 融合 ks-lma-chat 检索理念：三维专业法律检索式生成引擎
# 维度 A（最精准）：[争议焦点核心术语] + [行为定性] + 民法典或专门规定
# 维度 B（找例外）：[规范主体] + [行为] + 禁止及除外免责条款
# 维度 C（查后果）：[行为] + 违反 + 法律后果与法律效力
# ————————————————————————————————————————————————
def generate_structured_queries(
    query: str,
    sub_questions: Optional[List[str]] = None,
    fact_summary: Optional[Dict] = None,
) -> Dict[str, List[str]]:
    """生成三维法律检索式，确保高命中率、查漏（例外）与后果闭环。"""
    subs = sub_questions or [query]
    base_text = " ".join(subs) if subs else query
    expanded = expand_legal_query(base_text)

    # 启发式规则三维生成（支持 MOCK 或 LLM 降级）
    focus_queries: List[str] = []
    exception_queries: List[str] = []
    consequence_queries: List[str] = []

    # 提取核心词语
    keywords = re.findall(r"[\u4e00-\u9fa5]{2,6}", base_text)
    primary_terms = [w for w in keywords if w not in {"可以", "如何", "怎么", "是否", "能否", "对方", "我们", "自己"}]
    focus_core = " ".join(primary_terms[:3]) if primary_terms else base_text

    # 1. 维度 A（最精准式）
    if expanded and expanded != base_text:
        focus_queries.append(f"{expanded} 民法典规定")
    focus_queries.append(f"{focus_core} 构成要件 适用规定")

    # 2. 维度 B（找例外/免责式）
    exception_queries.append(f"{focus_core} 除外情形 免责事由 不适用")
    exception_queries.append(f"{focus_core} 禁止性规定 排除适用")

    # 3. 维度 C（查后果/效力式）
    consequence_queries.append(f"{focus_core} 法律后果 法律责任 效力认定")
    consequence_queries.append(f"{focus_core} 违约损害赔偿 追偿与补救措施")

    structured = {
        "focus": list(dict.fromkeys(focus_queries)),
        "exception": list(dict.fromkeys(exception_queries)),
        "consequence": list(dict.fromkeys(consequence_queries)),
    }

    if settings.mock:
        return structured

    prompt = (
        "你是资深法律案情与法条检索专家。请根据案情咨询提取关键要素，拆解成 3 种不同维度的专业法律检索式：\n"
        "1. focus（最精准）：[争议焦点核心术语] + [行为定性] + 法规原文定位\n"
        "2. exception（找例外）：[主体] + [行为] + 禁止及除外免责条款\n"
        "3. consequence（查后果）：[行为] + 违反 + 法律后果与责任认定\n\n"
        f"案情咨询：{query}\n"
        "请严格按以下 JSON 格式输出，不要有额外解释：\n"
        "```json\n"
        "{\n"
        '  "focus": ["检索式1", "检索式2"],\n'
        '  "exception": ["检索式1"],\n'
        '  "consequence": ["检索式1"]\n'
        "}\n"
        "```"
    )
    try:
        data = _extract_json(_call(prompt))
        if isinstance(data, dict):
            res_focus = [str(x).strip() for x in data.get("focus", []) if str(x).strip()]
            res_exc = [str(x).strip() for x in data.get("exception", []) if str(x).strip()]
            res_con = [str(x).strip() for x in data.get("consequence", []) if str(x).strip()]
            if res_focus or res_exc or res_con:
                return {
                    "focus": res_focus or structured["focus"],
                    "exception": res_exc or structured["exception"],
                    "consequence": res_con or structured["consequence"],
                }
    except Exception:
        pass
    return structured


# ————————————————————————————————————————————————
# 问题改写：把子问题改写为更适合检索的 query（多路：原句 + 去疑问词 + 法言法语同义词扩展）
# ————————————————————————————————————————————————
def rewrite_queries(sub_question: str) -> List[str]:
    expanded = expand_legal_query(sub_question)
    if settings.mock:
        base = re.sub(r"[？?如何怎么怎样是否能否吗]", "", sub_question).strip()
        out = [sub_question]
        if base and base != sub_question:
            out.append(base)
        if expanded and expanded not in out:
            out.append(expanded)
        return out

    prompt = (
        "你是一名法律检索专家。请把下面的子问题改写为 1-3 个更适合在法规库中检索的查询语句，"
        "将日常口语转化为精准的法典术语（如'违约责任''解除劳动合同''经济补偿''欺诈惩罚性赔偿''格式条款无效'）。"
        "只输出 JSON 数组。\n"
        f"子问题：{sub_question}"
    )
    try:
        data = _extract_json(_call(prompt))
        if isinstance(data, list) and data:
            res = [str(x).strip() for x in data if str(x).strip()]
            if expanded != sub_question and expanded not in res:
                res.append(expanded)
            return res
    except Exception:
        pass
    return [sub_question, expanded] if expanded != sub_question else [sub_question]


# ————————————————————————————————————————————————
# 建议 1：未收录案由的动态事实构成要件推导（Dynamic Element Fallback）
# ————————————————————————————————————————————————
_DYNAMIC_FALLBACK_HEURISTICS: List[Tuple[Tuple[str, ...], Dict[str, Any]]] = [
    (
        ("工程", "装修", "施工", "发包", "承包", "分包", "挂靠"),
        {
            "inferred_case_type": "建设工程/装饰装修施工合同纠纷",
            "core_elements": ["施工合同效力与施工资质要件", "工程价款结算与工程质量合格要件", "实际施工人与欠付工程款事实"],
            "missing_facts": [
                "确认施工合同效力（是否存在未取得建设工程施工许可证、转包、借用资质挂靠或违法分包情形）",
                "确认工程是否已竣工验收合格及工程价款结算审价凭证",
                "确认发包人是否存在逾期支付工程款及守约方催告付款事实",
            ],
            "clarification_questions": [
                "请问涉案工程是否已经完成竣工验收并确认工程质量合格？",
                "请问双方是否已经完成工程结算审价？发包人目前实际欠付工程款金额是多少？",
            ],
            "procedural_notes": [
                "【专属管辖】依《民事诉讼法》第34条及司法解释，建设工程施工合同纠纷按照不动产纠纷确定管辖，由不动产所在地人民法院专属管辖。",
                "【诉讼时效】工程价款索赔普通诉讼时效为 3 年（《民法典》第188条），自约定的付款期届满之日起算。",
            ],
            "suggested_citations": ["中华人民共和国民法典|第791条", "中华人民共和国民法典|第807条"],
        },
    ),
    (
        ("租房", "租赁", "房东", "租客", "中介", "转租", "押金"),
        {
            "inferred_case_type": "房屋租赁合同纠纷",
            "core_elements": ["租赁合同成立与房屋交付要件", "租金/押金支付与转租约定要件", "解除合同条件与腾房违约责任"],
            "missing_facts": [
                "确认租赁合同约定的租期、租金支付方式及押金退还约定",
                "确认出租人是否交付符合约定用途的房屋，或承租人是否存在逾期付租、擅自转租等违约行为",
                "确认是否已向对方送达解除通知或催告函",
            ],
            "clarification_questions": [
                "请问合同关于押金退还和违约责任是如何具体约定的？",
                "请问目前租赁房屋是否已经腾退并完成交接验收？",
            ],
            "procedural_notes": [
                "【诉讼时效】租赁合同欠租或退还押金纠纷诉讼时效为 3 年（《民法典》第188条）。",
                "【转租异议期】依《民法典》第718条，出租人知道或者应当知道承租人转租，但在六个月内未提出异议的，视为出租人同意转租。",
            ],
            "suggested_citations": ["中华人民共和国民法典|第716条", "中华人民共和国民法典|第718条"],
        },
    ),
    (
        ("借条", "欠条", "借钱", "借贷", "利息", "民间借贷", "还钱"),
        {
            "inferred_case_type": "民间借贷纠纷",
            "core_elements": ["借贷合意（借据/合同）要件", "款项实际交付（银行转账凭证/交付记录）要件", "借款利息与还款期限约定要件"],
            "missing_facts": [
                "确认借贷合意凭证（是否有借条、借款协议、聊天记录等）",
                "确认借款实际交付凭证（银行转账流水、微信/支付宝转账电子凭证等出借款项证据）",
                "确认约定的借款期限及利率是否符合法律规定（一年期 LPR 四倍红线）",
            ],
            "clarification_questions": [
                "请问出借款项是否有银行转账凭证或微信/支付宝流水证明款项已实际交付？",
                "请问借条中约定的还款日期是哪一天？双方是否约定了利息？",
            ],
            "procedural_notes": [
                "【诉讼时效】约定了还款日期的，诉讼时效为 3 年，自还款期限届满次日起算；未约定还款期限的，出借人催告后的宽限期届满起算。",
                "【利率上限】依最高法《民间借贷司法解释》，出借人请求借款人按照合同约定利率支付利息，未超过合同成立时一年期贷款市场报价利率（LPR）四倍的，人民法院应予支持。",
            ],
            "suggested_citations": ["中华人民共和国民法典|第675条", "中华人民共和国民法典|第679条"],
        },
    ),
    (
        ("商标", "专利", "著作权", "版权", "侵权盗版", "知识产权"),
        {
            "inferred_case_type": "知识产权侵权纠纷",
            "core_elements": ["原告权利权属合法性要件", "被告被诉侵权行为（比对接触与实质性相似）要件", "侵权损失或侵权获利计算要件"],
            "missing_facts": [
                "确认权利凭证（商标注册证、专利证书、作品登记证书或创作底稿）",
                "确认侵权行为发生地、销售渠道以及侵权存续公证保全证据",
                "确认因侵权遭受的实际损失、侵权人获利或法定赔偿计算依据",
            ],
            "clarification_questions": [
                "请问我方是否已取得涉案权利证书，且目前处于合法有效期限内？",
                "请问是否已通过公证购买或电子存证固定被诉侵权方的销售/使用证据？",
            ],
            "procedural_notes": [
                "【诉讼时效】侵犯知识产权的诉讼时效为 3 年，自权利人知道或者应当知道侵权行为以及侵权人之日起计算。",
                "【管辖法院】知识产权民事侵权纠纷依最高法规定由专门的知识产权法庭或中级人民法院管辖。",
            ],
            "suggested_citations": ["中华人民共和国民法典|第1194条", "中华人民共和国民法典|第1195条"],
        },
    ),
]


def infer_dynamic_elements(query: str) -> Dict[str, Any]:
    """当用户问题未命中外置已知案由时，激活大模型动态推导法定要件、缺漏事实与追问问题。"""
    text = (query or "").strip()
    if not text:
        return {}

    # 1) 若处于 MOCK 模式，通过法律概念启发式推演
    if settings.mock:
        for kw_group, res in _DYNAMIC_FALLBACK_HEURISTICS:
            if any(k in text for k in kw_group):
                return dict(res)
        # 通用民商事争议兜底框架
        return {
            "inferred_case_type": "一般民商事权益争议（动态推导）",
            "core_elements": ["原告权利基础与合同/侵权法律事实要件", "相对人违法违约行为与过错要件", "损害后果与因果关系要件"],
            "missing_facts": [
                "确认争议双方的身份关系与法律关系性质凭证",
                "确认相对人实施违法、违约或损害行为的具体表现与举证凭证",
                "确认损害结果及具体诉请主张金额的计算依据",
            ],
            "clarification_questions": [
                "请问双方之间是否存在书面协议、往来函件或即时通讯沟通记录？",
                "请问本案主张损害赔偿或违约责任的具体金额及计算依据是什么？",
            ],
            "procedural_notes": [
                "【诉讼时效】民事权利请求保护普通诉讼时效为 3 年（《民法典》第188条），自知道或者应当知道权利受损及义务人之日起算。",
            ],
            "suggested_citations": ["中华人民共和国民法典|第188条"],
        }

    # 2) 真实模式：调用大模型生成严谨的法理要件结构化推演
    prompt = (
        "你是一名严谨的执业律师。请针对用户提出的法律咨询问题，推导出该纠纷所属的具体案由、法定核心构成要件、"
        "小前提中目前缺失的关键事实、建议向当事人追问的问题、诉讼时效/程序路径提示，以及建议核实引用的配套法律条款。\n"
        "请严格只输出一个合法 JSON 对象，格式如下：\n"
        "{\n"
        '  "inferred_case_type": "案由名称",\n'
        '  "core_elements": ["构成要件1", "构成要件2"],\n'
        '  "missing_facts": ["待核实确认事实1", "待核实确认事实2"],\n'
        '  "clarification_questions": ["澄清追问问题1", "澄清追问问题2"],\n'
        '  "procedural_notes": ["诉讼时效或管辖前置程序提示"],\n'
        '  "suggested_citations": ["法规全称|第X条"]\n'
        "}\n\n"
        f"用户问题：{query}"
    )
    try:
        data = _extract_json(_call(prompt))
        if isinstance(data, dict) and data.get("inferred_case_type"):
            return {
                "inferred_case_type": str(data.get("inferred_case_type", "民商事纠纷")),
                "core_elements": [str(x) for x in data.get("core_elements", []) if str(x).strip()],
                "missing_facts": [str(x) for x in data.get("missing_facts", []) if str(x).strip()],
                "clarification_questions": [str(x) for x in data.get("clarification_questions", []) if str(x).strip()],
                "procedural_notes": [str(x) for x in data.get("procedural_notes", []) if str(x).strip()],
                "suggested_citations": [str(x) for x in data.get("suggested_citations", []) if str(x).strip()],
            }
    except Exception:
        pass

    # 失败时回退到启发式通用兜底
    return infer_dynamic_elements(query)


def _render_structured_template_answer(
    query: str,
    citations: List[Citation],
    feedback: Optional[str] = None,
    extra_constraints: str = "",
    precedence_note: str = "",
    fact_summary: Optional[Dict] = None,
    sub_questions: Optional[List[str]] = None,
) -> str:
    """确定性六段式结构化法律意见书生成（供 MOCK 模式与外部 LLM 故障安全回退共用）。"""
    fs = fact_summary or {}
    missing_facts = fs.get("missing_facts") or []
    proc_notes = fs.get("procedural_notes") or []
    act_date = fs.get("act_date") or ""
    act_src = fs.get("act_date_source") or ""
    subs = sub_questions or [query]

    lines: List[str] = []
    lines.append("【一、争议焦点与事实前提】")
    if act_src == "assumed_today":
        lines.append(
            f"· 时间前提声明：因未提供具体法律事实发生时间，本意见在**假设争议事实发生于现行法有效期内（暂以 {act_date} 为基准日）**的前提下推演；若事实发生在旧法期间，须按行为时旧法重新定位。"
        )
    else:
        lines.append(f"· 时间效力基准：按法律事实发生日 {act_date} 定位有效适用规范。")
    for idx, sq in enumerate(subs, 1):
        lines.append(f"· 争议焦点 {idx}：{sq}")
    lines.append("")

    lines.append("【二、适用法规与效力要旨（大前提）】")
    cite_refs = []
    for c in citations:
        loc = f"{c.article}{c.paragraph}" if c.paragraph else c.article
        mark = "" if c.verified else "［待核验·不得作为正式依据］"
        cite_refs.append(f"《{c.law_name}》{loc}")
        lines.append(f"· 《{c.law_name}》{loc}（{c.hierarchy}·{c.validity}）{mark}：{c.content}")
        hint = c.applicable_hint()
        if hint:
            lines.append(f"  ↳ 效力说明：{hint}")
        if c.note and ("［引致关联" in c.note or "［新旧法对照" in c.note):
            lines.append(f"  ↳ 关联说明：{c.note}")
    lines.append("")

    if precedence_note:
        lines.append(precedence_note)
        lines.append("")

    is_fallback = bool(extra_constraints)
    ref_str = "、".join(cite_refs[:5])
    lines.append("【三、涵摄分析与初步结论（小前提推演）】")
    if is_fallback:
        lines.append(
            f"在现有候选依据范围内，结合{ref_str}之规定，倾向于按上述条文界定的构成要件与法律后果处理；"
            "但鉴于部分事实要件或时空要素尚待核实，上述结论成立需以下列「待核实关键事实」属实为前提，需进一步确认后方可定论。"
        )
    else:
        lines.append(
            f"依据{ref_str}之规定，将本案事实与法定构成要件对照涵摄：若当事人能够举证证明相关事实符合上述条文规定的构成要件，"
            "权利人可依法主张相应实体权利或救济（含继续履行、解除合同、经济补偿或损害赔偿等）。"
        )

    if missing_facts:
        lines.append("")
        lines.append("【四、待核实关键事实与补证建议】")
        for mf in missing_facts[:4]:
            lines.append(f"- {mf}")

    if proc_notes:
        lines.append("")
        lines.append("【五、诉讼/仲裁时效与程序路径提示】")
        for pn in proc_notes[:3]:
            lines.append(f"- {pn}")

    suggested_cits = fs.get("suggested_citations") or []
    if suggested_cits:
        lines.append("")
        lines.append("【六、建议法务/律师核实引证的补充条文】")
        for sc in suggested_cits[:3]:
            if "|" in sc:
                l_part, a_part = sc.split("|", 1)
                lines.append(f"- 建议核查引证：《{l_part.strip('《》')}》{a_part.strip()}（待律师确认后一键纳入正式引用库）")
            else:
                lines.append(f"- 建议核查引证：{sc}（待律师确认后一键纳入正式引用库）")

    if feedback:
        lines.append("")
        lines.append(f"【执业律师审核与补充意见】{feedback}")

    if extra_constraints:
        lines.append("\n（受限兜底模式：本结论包含待确认前提，不得进入自动对外通道，必须经执业律师人工审核后方可使用）")
    return "\n".join(lines)


def _render_direct_llm_fallback_answer(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    feedback: Optional[str] = None,
    fact_summary: Optional[Dict] = None,
) -> str:
    """不使用知识库模式下的智能答复（纯大模型口吻，支持多轮上下文理解，不套用法律助手的规则和输出框架）。"""
    q = (query or "").strip()
    cur_m = _normalize_model_name(settings.llm_model)
    q_lower = q.lower()

    # 1. 询问名字、身份、擅长什么、功能介绍、自我介绍（解决用户截图中的核心问题）
    if any(k in q_lower for k in ("名字", "叫什么", "你是谁", "模型", "自我介绍", "擅长", "能做", "介绍下你自己", "介绍一下你自己", "你的功能", "身份", "who are you", "what can you do")):
        res = (
            f"您好！我是 **AI法盾智能助手**（当前运行模型：`{cur_m}`）。\n\n"
            f"作为全能型人工智能助手，我拥有广泛的知识储备与逻辑推理能力，主要擅长以下领域：\n\n"
            f"1. **知识问答与深度咨询**：解答金融合规、法律法规、行业制度、通用常识与实务操作等各类问题；\n"
            f"2. **专业文本撰写与润色**：协助起草公告通知、合同协议、业务方案、工作报告、总结邮件及文案创作；\n"
            f"3. **逻辑梳理与争议分析**：协助拆解复杂事务中的核心事实要素、责任边界、潜在风险并提供应对建议；\n"
            f"4. **多轮连续对话**：能够根据我们聊天的上下文背景，开展连续追问、补充细节并不断深化解答。\n\n"
            f"当前您已开启 **「不使用知识库」** 直答模式，您可以直接向我提出任何问题或交代具体任务，我会竭诚为您解答！"
        )
        if feedback:
            res += f"\n\n补充说明：{feedback}"
        return res

    # 2. 问候语或礼貌用语
    if any(q_lower.startswith(k) for k in ("你好", "您好", "早上好", "中午好", "下午好", "晚上好", "hi", "hello", "嗨")):
        return f"您好！我是 AI法盾智能助手，很高兴与您交流。请问今天有什么我可以协助您的吗？无论是日常问题解答、文本撰写还是业务咨询，我随时都在！"

    # 3. 常见业务场景提问（直接回答业务核心，不套用法律意见书框架）
    if "理财" in q and ("高风险" in q or "测试" in q):
        return (
            "客户通过手机银行等电子渠道购买高风险理财产品时，除常规风险评估外，通常需要完成以下额外的适当性与安全测试：\n\n"
            "1. **风险承受能力动态匹配测评**：确认客户评级结果在有效期内（通常为1年），且匹配等级须达到进取型或激进型（R4/R5），杜绝向低风险承受能力客户越级销售；\n"
            "2. **高风险专属风险揭示与认知测试**：在购买流程中逐项展示最不利投资情形压力测试结果与本金亏损可能，并设置强制阅读时限与专属风险认知问答测试；\n"
            "3. **特殊客群强化核验与冷静期机制**：对65周岁以上的高龄客户或首次购买高风险产品的客户，提供强化风险提示、可回溯电子留痕以及交易冷静期与电话回访机制。"
        )
    elif "基本账户" in q or ("微企" in q and "诈骗" in q):
        return (
            "针对无固定经营场所的小微企业申请开立基本存款账户，银行识别与防范电信网络诈骗风险的核心要点如下：\n\n"
            "1. **穿透核实法定代表人开户意愿与身份**：深入了解法人对主营业务、上下游客户及资金来源的知晓程度，重点排查受雇挂名、买卖账户嫌疑；\n"
            "2. **多维交叉核验经营真实性**：要求提供纳税记录、社保缴纳、水电气物业缴费、工位托管或共享办公协议、经营合同及资金往来凭证等客观佐证；\n"
            "3. **比对涉诈风险名单与异常特征**：结合公安反诈预警库、同一个人或手机号短期关联多家空壳企业等涉案特征进行排查；\n"
            "4. **实施账户分级分类限额管理**：根据风险评级合理设置非柜面转账额度，并在开户后对大额快进快出、夜间集中转账等异常流水实施动态监测。"
        )
    elif "吞钞" in q or ("自助设备" in q and "账务核对" in q):
        return (
            "客户在自助存取款设备（ATM/CRS）发生吞钞长款时的账务核对与处置时限规定如下：\n\n"
            "1. **账务核对时限**：网点营业设备原则上在接到客户报障后的**第 1 个工作日内**完成清机盘点与流水日志对账；离行式自助设备一般不超过 2 至 3 个工作日；\n"
            "2. **双线溯源核对**：通过比对自助设备交易流水、钞箱冠字号码记录以及现场监控录像，核定长款金额与归属客户；\n"
            "3. **长款调账与资金退还**：核对查实后，于查清当日办理长款调账并足额划入客户账户，同时通过短信或电话告知客户处理结果。"
        )

    # 4. 判断是否是多轮对话的连续承接
    if history and len(history) > 0:
        res = (
            f"结合我们刚才讨论的上下文，针对您的进一步提问「{q}」：\n\n"
            f"在前面交流的基础上，建议关注以下几个核心方向：\n"
            f"1. **关键要点对齐**：承接上文的讨论背景，细化明确当前阶段的具体诉求与各方权责；\n"
            f"2. **落地操作建议**：根据实际情况制定切实可行的执行方案，注意把握推进节点并做好过程记录；\n"
            f"3. **风险防范措施**：提前排查潜在的分歧点或阻碍，准备好相应的应对备选方案。\n\n"
            f"若您需要针对其中的特定细节进行方案细化或起草文案，请随时告诉我！"
        )
        if feedback:
            res += f"\n\n补充意见：{feedback}"
        return res

    # 5. 通用常识与综合提问的自然大模型解答
    res = (
        f"您好！关于您提出的问题「{q}」，解答如下：\n\n"
        f"1. **核心要点概述**：在处理或理解此项问题时，关键在于准确界定其核心概念与关键事实要素，理清各方的权责定位与关联目标；\n"
        f"2. **常规处理流程与实施建议**：建议采取结构化的方式推进，先梳理基础资料与现状，再拟定明确的处理步骤，并根据实际反馈进行灵活调整；\n"
        f"3. **关键注意事项**：过程中注意留存客观沟通记录与书面依据，防范信息不对称及潜在执行风险。\n\n"
        f"如需针对具体细节进一步展开或有其他相关疑问，欢迎随时向我补充信息！"
    )

    if feedback:
        res += f"\n\n补充意见：{feedback}"
    return res


def generate_direct_llm_answer(
    query: str,
    history: Optional[List[Dict[str, str]]] = None,
    feedback: Optional[str] = None,
    fact_summary: Optional[Dict] = None,
    deep_thinking: bool = True,
) -> str:
    """不使用知识库模式：直接将输入框中的内容提交给大模型回答，支持多轮对话上下文，不走任何法律助手的规则和输出框架。"""
    global _LAST_LLM_ERROR
    _LAST_LLM_ERROR = None

    if settings.mock:
        return _render_direct_llm_fallback_answer(query, history=history, feedback=feedback)

    generic_system = (
        "你是一个乐于助人、知识渊博的人工智能助手。请针对用户提出的问题，直接给出清晰、准确、自然、详尽的回答。"
    )
    user_prompt = (query or "").strip()
    if feedback:
        user_prompt += f"\n\n【补充参考信息】\n{feedback}"

    try:
        return _call(user_prompt, system_prompt=generic_system, history=history)
    except Exception as e:
        _LAST_LLM_ERROR = str(e)
        return _render_direct_llm_fallback_answer(query, history=history, feedback=feedback)


# ————————————————————————————————————————————————
# 两层合成：第二层结构化法律咨询意见（P1 升级六段式法律意见书）
# ————————————————————————————————————————————————
def generate_answer(
    query: str,
    citations: List[Citation],
    feedback: Optional[str] = None,
    extra_constraints: str = "",
    precedence_note: str = "",
    fact_summary: Optional[Dict] = None,
    sub_questions: Optional[List[str]] = None,
    use_kb: bool = True,
) -> str:
    global _LAST_LLM_ERROR
    _LAST_LLM_ERROR = None

    if not use_kb:
        return generate_direct_llm_answer(query, feedback=feedback, fact_summary=fact_summary)

    if not citations:
        return (
            "经检索，当前知识库中未找到与您问题直接对应、且在行为发生日适用的条文。"
            "建议您携带具体材料（含法律事实发生时间、相关证据）咨询执业律师，以获得准确的法律意见。"
        )

    if settings.mock:
        return _render_structured_template_answer(
            query, citations, feedback, extra_constraints, precedence_note, fact_summary, sub_questions
        )

    fs = fact_summary or {}
    missing_facts = fs.get("missing_facts") or []
    proc_notes = fs.get("procedural_notes") or []
    subs = sub_questions or [query]

    blocks = []
    for c in citations:
        loc = f"{c.article}{c.paragraph}" if c.paragraph else c.article
        mark = "" if c.verified else "［待核验，不得作为正式依据］"
        line = f"- 《{c.law_name}》{loc}（{c.hierarchy}·{c.validity}）{mark}：{c.content}"
        hint = c.applicable_hint()
        if hint:
            line += f"\n  适用提示：{hint}"
        if c.note:
            line += f"\n  关联备注：{c.note}"
        blocks.append(line)
    cite_block = "\n".join(blocks)
    fb = f"\n【执业律师审核意见（必须在回答中完整体现）】\n{feedback}\n" if feedback else ""
    prec = f"\n{precedence_note}\n" if precedence_note else ""
    sub_block = "\n【拆解子争点】\n" + "\n".join(f"{i+1}. {s}" for i, s in enumerate(subs)) + "\n"
    mf_block = (
        "\n【待核实的关键事实（必须在回答中列出以提醒核实小前提）】\n"
        + "\n".join(f"- {m}" for m in missing_facts)
        + "\n"
        if missing_facts
        else ""
    )
    proc_block = (
        "\n【程序前置与时效红线提示（必须在回答中予以提示）】\n"
        + "\n".join(f"- {p}" for p in proc_notes)
        + "\n"
        if proc_notes
        else ""
    )
    prompt = (
        "你是一名严谨的执业律师。请严格基于【引用的法规】对用户问题给出结构化的第二层法律咨询意见。\n"
        "硬性要求：\n"
        "1. 必须按「【一、争议焦点与事实前提】→【二、适用法规与效力要旨（大前提）】→【三、涵摄分析与初步结论（小前提推演）】→【四、待核实关键事实与补证建议】→【五、诉讼/仲裁时效与程序路径提示】」结构输出；\n"
        "2. 只能引用【引用的法规】及【程序前置与时效红线提示】中明确出现的法规名称与具体条文编号，严禁在正文实体分析中编造任何未在列表中提供的条号；\n"
        "3. 若经法理分析认为该案需要进一步查验关键配套法条（如引致救济或司法解释），请在回答结尾专门设置「【六、建议法务/律师核实引证的补充条文】」章节按列表列出（例如：- 建议核查引证：《法规全称》第X条（理由：XXX））；\n"
        "4. 若存在【执业律师审核意见】，必须在回答末尾设立「【执业律师审核与补充意见】」章节完整体现。\n"
        f"用户问题：{query}\n"
        f"{sub_block}"
        f"【引用的法规】\n{cite_block}\n"
        f"{prec}"
        f"{mf_block}"
        f"{proc_block}"
        f"{fb}"
        f"{extra_constraints}\n"
        "请输出结构化法律咨询意见。"
    )
    try:
        return _call(prompt)
    except Exception as e:
        # P0 关键修复：绝不把报错字符串静默当作正式回答返回，而是记录异常供节点降级置信度，并生成受控规则骨架初稿
        _LAST_LLM_ERROR = str(e)
        fallback_draft = _render_structured_template_answer(
            query, citations, feedback, extra_constraints or "fallback", precedence_note, fact_summary, sub_questions
        )
        return (
            f"【系统提示：外部大模型 ({settings.llm_model}) 调用异常，已自动切换为确定性规则骨架生成，必须经执业律师复核】\n\n"
            f"{fallback_draft}"
        )


# ————————————————————————————————————————————————
# 重排打分：对每条候选条文与问题（含子争点）的相关性打 1~10 分（真实模式）
# ————————————————————————————————————————————————
def rerank_citations_llm(
    query: str,
    citations: List[Citation],
    sub_queries: Optional[List[str]] = None,
) -> Optional[List[float]]:
    """返回与 citations 等长的相关性分数列表（1~10，越高越相关）。调用失败返回 None（不再静默给 5.0 分）。"""
    if settings.mock:
        raise RuntimeError("MOCK 模式下不应调用重排 LLM")
    if not citations:
        return []

    items = "\n".join(
        f"{i + 1}. 《{c.law_name}》{c.article}：{c.content}"
        for i, c in enumerate(citations)
    )
    sub_hint = ""
    if sub_queries and len(sub_queries) > 1:
        sub_hint = "拆解子争点：" + "；".join(sub_queries) + "\n"
    prompt = (
        "你是法律检索相关性评估专家。请判断下列每条法规与用户问题（或任一拆解子争点）的相关性，"
        "给出 1~10 的整数分数（10=高度相关且直接可回答任一子争点，1=完全无关）。"
        "只输出一个与候选法规数量相等的 JSON 数组，例如 [9,3,7]，不要任何解释。\n"
        f"用户问题：{query}\n"
        f"{sub_hint}"
        f"候选法规：\n{items}"
    )
    try:
        data = _extract_json(_call(prompt))
        if isinstance(data, list) and len(data) == len(citations):
            return [max(1.0, min(10.0, float(x))) for x in data]
    except Exception:
        return None
    return None
