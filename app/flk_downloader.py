"""国家法律法规数据库（flk.npc.gov.cn）官方 Word 文档自动拉取、分类与知识库挂载引擎：
1. 支持按 type 分类代码（flfg: 法律, xzfg: 行政法规, sfjs: 司法解释, dfxfg: 地方性法规, jcfg: 监察法规）直连官方新版检索接口 `/law-search/search/list`；
2. 彻底摒弃静态本地硬编码兜底：所有列表检索与 Word 文档均实时从国家法律法规数据库拉取；
3. 严格网络探测与重试机制：每次请求最多重试 3 次（max_retries=3），单次超时时长 5 秒（timeout=5.0s）；若 3 次均失败或超时，明确返回 `network_unreachable` 状态并在前端弹出网络不可达提示；
4. 反爬风控与安全对抗：完整浏览器指纹模拟、随机时延抖动（Jitter）；
5. 自动下载官方 Word (.docx) 格式文档（直连 `/law-search/download/pc` 与官方 OBS 预签名链接），并自动按分类在「外规知识库」下创建对应知识库目录完成法条切分与向量入库。
"""
from __future__ import annotations

import json
import logging
import random
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.kb import kb_store

logger = logging.getLogger(__name__)

# ————————————————————————————————————————————————
# 1. 国家法规库新版分类代码 (flfgCodeId) 与知识库目录映射配置
# 数据同源于 https://flk.npc.gov.cn/law-search/search/enumData
# ————————————————————————————————————————————————
FLK_TYPE_MAPPING: Dict[str, Dict[str, Any]] = {
    "flfg": {
        "name": "法律",
        "description": "全国人民代表大会及其常务委员会制定的国家法律",
        "hierarchy": "法律",
        # 对应官网“法律”大类全部子项 codeIdList（若仅筛选纯法律正典不含修改决定则为 [102,110,120,130,140,150,155,160,170]）
        "flfgCodeId": [101, 102, 110, 120, 130, 140, 150, 155, 160, 170, 180, 190, 195, 200],
        "statuteOnlyCodeId": [102, 110, 120, 130, 140, 150, 155, 160, 170],
    },
    "xzfg": {
        "name": "行政法规",
        "description": "中华人民共和国国务院制定的行政法规",
        "hierarchy": "行政法规",
        "flfgCodeId": [201, 210, 215],
        "statuteOnlyCodeId": [210],
    },
    "sfjs": {
        "name": "司法解释",
        "description": "最高人民法院、最高人民检察院发布的司法解释",
        "hierarchy": "司法解释",
        "flfgCodeId": [311, 320, 330, 340, 350],
        "statuteOnlyCodeId": [320, 330, 340],
    },
    "dfxfg": {
        "name": "地方性法规",
        "description": "省、自治区、直辖市人民代表大会及其常委会制定的地方性法规",
        "hierarchy": "地方性法规",
        "flfgCodeId": [221, 222, 230, 260, 270, 290, 295, 300, 305, 310],
        "statuteOnlyCodeId": [222, 230, 260, 270, 290, 295, 300],
    },
    "jcfg": {
        "name": "监察法规",
        "description": "国家监察委员会制定的监察法规",
        "hierarchy": "监察法规",
        "flfgCodeId": [220],
        "statuteOnlyCodeId": [220],
    },
    "xf": {
        "name": "宪法",
        "description": "中华人民共和国宪法及修正案",
        "hierarchy": "宪法",
        "flfgCodeId": [100],
        "statuteOnlyCodeId": [100],
    },
}

# 时效性状态码映射（对应新版官网 sxx 字段）
SXX_STATUS_MAP: Dict[int, str] = {
    3: "现行有效",
    4: "尚未生效",
    2: "已修改",
    1: "已废止",
}

# 伪装现代真实浏览器报头池
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]

FLK_HOST = "https://flk.npc.gov.cn"
FLK_SEARCH_LIST_URL = f"{FLK_HOST}/law-search/search/list"
FLK_DOWNLOAD_PC_URL = f"{FLK_HOST}/law-search/download/pc"
FLK_DETAILS_URL = f"{FLK_HOST}/law-search/search/flfgDetails"

# 固定重试与超时参数：重试 3 次，单次超时 5 秒
MAX_RETRIES = 3
TIMEOUT_SECONDS = 5.0


def get_random_headers(is_json_post: bool = False) -> Dict[str, str]:
    """生成具备反爬对抗能力的动态真实浏览器请求头。"""
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Origin": FLK_HOST,
        "Referer": f"{FLK_HOST}/search",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    if is_json_post:
        headers["Content-Type"] = "application/json;charset=utf-8"
    return headers


def ensure_kb_category_folder(type_code: str) -> Tuple[str, str]:
    """检查并自动创建外规知识库下的分类子目录（如 外规知识库/法律、外规知识库/行政法规）。
    返回: (kb_type_path, hierarchy)
    """
    conf = FLK_TYPE_MAPPING.get(type_code) or {
        "name": "法律法规",
        "description": "国家法律法规",
        "hierarchy": "法律",
    }
    target_name = conf["name"]
    hierarchy = conf["hierarchy"]

    tree = kb_store.get_kb_tree()
    ext_node = next((n for n in tree if n.get("node_id") == "kb_external" or n.get("name") == "外规知识库"), None)
    if not ext_node:
        ext_node = kb_store.create_kb_tree_node(
            name="外规知识库",
            node_type="kb",
            description="国家法律法规、司法解释与部委规章",
        )

    ext_node_id = ext_node.get("node_id", "kb_external")

    children = ext_node.get("children", [])
    child_node = next((c for c in children if c.get("name") == target_name), None)

    if not child_node:
        try:
            child_node = kb_store.create_kb_tree_node(
                parent_id=ext_node_id,
                name=target_name,
                node_type="folder",
                description=conf["description"],
            )
            logger.info(f"[FLKDownloader] 自动为类型 [{type_code}] 创建知识库目录: 外规知识库/{target_name}")
        except Exception as e:
            logger.warning(f"[FLKDownloader] 创建目录外规知识库/{target_name}轻微异常: {e}")

    kb_type_path = f"外规知识库/{target_name}"
    return kb_type_path, hierarchy


class FLKNetworkUnreachableError(Exception):
    """当连续重试 MAX_RETRIES (3) 次、每次超时 TIMEOUT_SECONDS (5s) 均无法连通国家法规库时抛出。"""

    def __init__(self, message: str, last_error: str = "", retries: int = MAX_RETRIES, timeout: float = TIMEOUT_SECONDS):
        super().__init__(message)
        self.message = message
        self.last_error = last_error
        self.retries = retries
        self.timeout = timeout


class FLKDownloader:
    """国家法律法规数据库实时检索、Word下载与知识库归类入库器（无任何本地硬编码兜底）。"""

    def __init__(
        self,
        request_delay_range: Tuple[float, float] = (0.35, 0.95),
        max_retries: int = MAX_RETRIES,
        timeout: float = TIMEOUT_SECONDS,
    ):
        self.delay_range = request_delay_range
        self.max_retries = max_retries
        self.timeout = timeout
        self._ssl_ctx = ssl._create_unverified_context()

    def _sleep_jitter(self) -> None:
        """随机休眠抖动，避免高频并发触发防火墙 WAF 频控。"""
        delay = random.uniform(*self.delay_range)
        time.sleep(delay)

    def _request_with_retry(
        self,
        url: str,
        method: str = "GET",
        json_body: Optional[Dict[str, Any]] = None,
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> bytes:
        """执行带有严格「重试 3 次、单次超时 5 秒」机制的 HTTP 请求。
        若 3 次尝试全部超时或网络异常，抛出 FLKNetworkUnreachableError。
        """
        is_post = method.upper() == "POST"
        data_bytes = json.dumps(json_body, ensure_ascii=False).encode("utf-8") if (is_post and json_body is not None) else None
        last_err_str = ""

        for attempt in range(1, self.max_retries + 1):
            headers = custom_headers or get_random_headers(is_json_post=is_post)
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method.upper())
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=self._ssl_ctx) as resp:
                    body = resp.read()
                    if not body:
                        raise urllib.error.URLError("响应体为空")
                    return body
            except Exception as e:
                last_err_str = f"{type(e).__name__}: {e}"
                logger.warning(
                    f"[FLKDownloader] 请求国家法规库超时/异常 (第 {attempt}/{self.max_retries} 次尝试, 超时阈值 {self.timeout}s): {url} -> {last_err_str}"
                )
                if attempt < self.max_retries:
                    time.sleep(0.4 * attempt)

        raise FLKNetworkUnreachableError(
            message=f"网络不可达：连接国家法律法规数据库 (flk.npc.gov.cn) 失败（已重试 {self.max_retries} 次，单次超时 {int(self.timeout)} 秒），请检查网络连接或防火墙策略后重试。",
            last_error=last_err_str,
            retries=self.max_retries,
            timeout=self.timeout,
        )

    def search_laws(
        self,
        keyword: str = "",
        type_code: str = "flfg",
        page: int = 1,
        size: int = 20,
        sxx: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """实时向国家法律法规数据库新版接口 `/law-search/search/list` 检索法规列表。
        绝不使用任何本地静态硬编码；若重试 3 次（每次 5 秒）后仍不可达，返回 `status='network_unreachable'`。
        """
        clean_kw = (keyword or "").strip()
        conf = FLK_TYPE_MAPPING.get(type_code) or FLK_TYPE_MAPPING["flfg"]
        code_id_list = conf["flfgCodeId"]

        payload: Dict[str, Any] = {
            "searchRange": 1,
            "sxrq": [],
            "gbrq": [],
            "searchType": 2,  # 2: 模糊检索, 1: 精确检索
            "sxx": sxx if sxx is not None else [],
            "gbrqYear": [],
            "flfgCodeId": code_id_list,
            "zdjgCodeId": [],
            "searchContent": clean_kw,
            "orderByParam": {"order": "-1", "sort": ""},
            "pageNum": max(1, int(page)),
            "pageSize": max(1, min(int(size), 100)),
        }

        try:
            raw_bytes = self._request_with_retry(
                url=FLK_SEARCH_LIST_URL,
                method="POST",
                json_body=payload,
            )
            resp_text = raw_bytes.decode("utf-8", errors="replace")
            if resp_text.lstrip().startswith("<!doctype html") or "<html" in resp_text[:120].lower():
                raise FLKNetworkUnreachableError(
                    message=f"网络不可达：国家法律法规数据库 (flk.npc.gov.cn) 返回了网关拦截页而非有效数据（已校验重试策略 {self.max_retries} 次 / {int(self.timeout)} 秒超时）。",
                    last_error="WAF HTML Challenge Response",
                )

            data = json.loads(resp_text)
            if data.get("code") != 200:
                raise FLKNetworkUnreachableError(
                    message=f"国家法律法规数据库接口返回异常状态 (code={data.get('code')}, msg={data.get('msg')})",
                    last_error=str(data.get("msg") or ""),
                )

            rows = data.get("rows") or []
            total = int(data.get("total") or len(rows))
            items: List[Dict[str, Any]] = []
            for it in rows:
                raw_title = it.get("title") or ""
                clean_title = re.sub(r"<[^>]+>", "", raw_title).strip()
                sxx_val = it.get("sxx")
                sxx_label = SXX_STATUS_MAP.get(sxx_val, "有效" if sxx_val is None else str(sxx_val))
                items.append(
                    {
                        "id": it.get("bbbs") or it.get("id") or "",
                        "bbbs": it.get("bbbs") or it.get("id") or "",
                        "title": clean_title,
                        "type": it.get("flxz") or conf.get("name", "法律"),
                        "type_code": type_code,
                        "office": it.get("zdjgName") or "全国人民代表大会常务委员会",
                        "publish": str(it.get("gbrq") or "")[:10],
                        "effective_date": str(it.get("sxrq") or "")[:10],
                        "sxx": sxx_val,
                        "validity_label": sxx_label,
                        "has_word": True,
                        "source": "official_live",
                    }
                )

            return {
                "status": "success",
                "total": total,
                "page": int(page),
                "size": int(size),
                "type_code": type_code,
                "type_name": conf.get("name", "法律"),
                "source": "official_live",
                "laws": items,
            }

        except FLKNetworkUnreachableError as net_err:
            return {
                "status": "network_unreachable",
                "message": net_err.message,
                "error": net_err.message,
                "detail": net_err.last_error,
                "retry_count": self.max_retries,
                "timeout_seconds": int(self.timeout),
                "total": 0,
                "page": int(page),
                "size": int(size),
                "source": "network_unreachable",
                "laws": [],
            }
        except Exception as e:
            err_msg = f"网络不可达：连接国家法律法规数据库 (flk.npc.gov.cn) 异常（重试 {self.max_retries} 次，超时 {int(self.timeout)} 秒）：{e}"
            return {
                "status": "network_unreachable",
                "message": err_msg,
                "error": err_msg,
                "detail": str(e),
                "retry_count": self.max_retries,
                "timeout_seconds": int(self.timeout),
                "total": 0,
                "page": int(page),
                "size": int(size),
                "source": "network_unreachable",
                "laws": [],
            }

    def download_and_import(
        self,
        flk_id: Optional[str] = None,
        title: Optional[str] = None,
        type_code: str = "flfg",
        uploader: str = "国家法规库同步器",
        publish_date: str = "",
        effective_date: str = "",
    ) -> Dict[str, Any]:
        """从国家法律法规数据库实时下载指定法规的官方 Word (.docx) 文件，自动按分类创建知识库并完成法条切分与向量入库。
        绝不使用本地静态模板造假；若重试 3 次（每次超时 5 秒）后网络不可达，返回 `status='network_unreachable'`。
        """
        clean_title = re.sub(r"<[^>]+>", "", (title or "")).strip().strip("《》")
        clean_id = (flk_id or "").strip()
        if not clean_title and not clean_id:
            raise ValueError("必须提供法规标题或国家法规库唯一标识 (bbbs)")

        # 1. 自动确保分类知识库存在 (如 外规知识库/法律、外规知识库/行政法规)
        kb_path, hierarchy = ensure_kb_category_folder(type_code)

        # 2. 若未传入 flk_id (bbbs)，先通过实时检索接口按标题定位 bbbs
        if not clean_id and clean_title:
            search_res = self.search_laws(keyword=clean_title, type_code=type_code, page=1, size=10)
            if search_res.get("status") == "network_unreachable":
                return {
                    "status": "network_unreachable",
                    "message": search_res.get("message") or "网络不可达：无法连接国家法律法规数据库",
                    "retry_count": self.max_retries,
                    "timeout_seconds": int(self.timeout),
                    "title": clean_title,
                }
            laws = search_res.get("laws") or []
            matched = next((x for x in laws if x.get("title") == clean_title), laws[0] if laws else None)
            if not matched or not matched.get("id"):
                return {
                    "status": "error",
                    "message": f"未在国家法律法规数据库中检索到《{clean_title}》的对应条目。",
                    "title": clean_title,
                }
            clean_id = matched["id"]
            publish_date = publish_date or matched.get("publish", "")
            effective_date = effective_date or matched.get("effective_date", "")

        # 3. 实时从国家法规库拉取官方 Word (.docx) 附件文件流（带 3 次重试、5 秒超时）
        try:
            word_bytes, fetched_pub_date, fetched_eff_date, raw_text_fallback = self._fetch_live_word_bytes(clean_id)
        except FLKNetworkUnreachableError as net_err:
            return {
                "status": "network_unreachable",
                "message": net_err.message,
                "error": net_err.message,
                "detail": net_err.last_error,
                "retry_count": self.max_retries,
                "timeout_seconds": int(self.timeout),
                "title": clean_title,
            }

        pub_date = publish_date or fetched_pub_date or datetime.now().strftime("%Y-%m-%d")
        eff_date = effective_date or fetched_eff_date or pub_date

        if not word_bytes and not raw_text_fallback:
            return {
                "status": "error",
                "message": f"国家法律法规数据库暂未提供《{clean_title}》(bbbs={clean_id}) 的可下载 Word 文档或正文内容。",
                "title": clean_title,
            }

        # 4. 调用企业级切分与多版本入库管线（支持同名法规自动作为新版本演进或去重）
        filename = f"{clean_title}.docx" if word_bytes else f"{clean_title}.txt"
        ingest_res = kb_store.import_advanced_document(
            filename=filename,
            content_bytes=word_bytes,
            raw_text=raw_text_fallback if not word_bytes else None,
            law_name=clean_title,
            kb_type=kb_path,
            department="法律合规部",
            uploader=uploader,
            split_mode="article",
            validity_option="permanent",
            hierarchy=hierarchy,
            promulgation_date=pub_date,
            effective_date=eff_date,
            conflict_action="append_new",
        )

        return {
            "status": "success",
            "message": f"成功从国家法律法规数据库下载《{clean_title}》(官方Word版)，并自动归类导入至「{kb_path}」！",
            "title": clean_title,
            "kb_type": kb_path,
            "hierarchy": hierarchy,
            "doc_id": ingest_res.get("doc_id"),
            "version_name": ingest_res.get("version_name", ""),
            "chunks_count": ingest_res.get("chunks_count", 0),
            "doc_size": f"{len(word_bytes) / 1024:.1f} KB" if word_bytes else f"{len((raw_text_fallback or '').encode('utf-8')) / 1024:.1f} KB",
            "format": "DOCX" if word_bytes else "TXT",
        }

    def _fetch_live_word_bytes(self, bbbs: str) -> Tuple[Optional[bytes], str, str, str]:
        """向国家法规库请求 `/law-search/download/pc` 获取官方预签名 Word (.docx) 下载链接并拉取二进制流。
        若单次超时（5秒），自动按 3 次重试策略执行；若 3 次均超时则向上抛出 FLKNetworkUnreachableError。
        返回: (word_bytes, publish_date, effective_date, raw_text_from_details_if_no_docx)
        """
        self._sleep_jitter()
        query = urllib.parse.urlencode({"format": "docx", "bbbs": bbbs, "fileId": ""})
        dl_meta_url = f"{FLK_DOWNLOAD_PC_URL}?{query}"

        raw_meta = self._request_with_retry(url=dl_meta_url, method="GET")
        meta_json = json.loads(raw_meta.decode("utf-8", errors="replace"))

        dl_url = ""
        if meta_json.get("code") == 200 and isinstance(meta_json.get("data"), dict):
            dl_url = (meta_json["data"].get("url") or "").strip()

        if dl_url:
            self._sleep_jitter()
            # 下载 OBS 预签名 Word 文件（同样遵循重试 3 次，单次超时 5 秒）
            dl_headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "*/*",
                "Referer": f"{FLK_HOST}/",
            }
            word_bytes = self._request_with_retry(url=dl_url, method="GET", custom_headers=dl_headers)
            if word_bytes and len(word_bytes) > 100:
                return word_bytes, "", "", ""

        # 若该法规在官网仅提供在线正文（如早期决定无独立 docx 附件），实时查询官网详情接口 `/law-search/search/flfgDetails`
        det_query = urllib.parse.urlencode({"bbbs": bbbs})
        det_url = f"{FLK_DETAILS_URL}?{det_query}"
        raw_det = self._request_with_retry(url=det_url, method="GET")
        det_json = json.loads(raw_det.decode("utf-8", errors="replace"))
        if det_json.get("code") == 200 and isinstance(det_json.get("data"), dict):
            d_obj = det_json["data"]
            pub_date = str(d_obj.get("gbrq") or "")[:10]
            eff_date = str(d_obj.get("sxrq") or "")[:10]
            content_html = d_obj.get("content") or ""
            clean_text = re.sub(r"<[^>]+>", "\n", content_html).strip()
            return None, pub_date, eff_date, clean_text

        return None, "", "", ""


flk_downloader = FLKDownloader()
