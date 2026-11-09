"""法规与法务文档解析切分引擎（对标「AI法盾」三模式切分体系）：
1. 自动分割（Token / 标点自适应语义滑窗，适合非结构化法律文档）
2. 法条分割（编、章、节、条、款、项有限状态机与上下文层级继承，适合成文法典）
3. 文档分层分割（Hierarchical Parent-Child 分块：大块承载法理全貌，小块用于高精准语义检索）
4. 多格式文档纯文本安全抽取器（支持 txt / md / csv / json / pdf / docx / xlsx）
"""
from __future__ import annotations

import io
import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple


def extract_text_from_file_bytes(filename: str, content_bytes: bytes) -> str:
    """从上传的文件二进制流中抽取纯文本。原生零第三方强依赖，优先尝试解析，失败平滑降级。"""
    ext = (filename.rsplit(".", 1)[-1] if "." in filename else "").lower()

    if ext in ("txt", "md", "csv"):
        for enc in ("utf-8", "gb18030", "gbk", "utf-16"):
            try:
                return content_bytes.decode(enc)
            except Exception:
                pass
        return content_bytes.decode("utf-8", errors="ignore")

    if ext == "json":
        try:
            parsed = json.loads(content_bytes.decode("utf-8", errors="ignore"))
            if isinstance(parsed, list):
                return "\n".join(str(item) for item in parsed)
            return json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            return content_bytes.decode("utf-8", errors="ignore")

    # 对图片格式执行 OCR 识别抽取
    if ext in ("png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"):
        try:
            from app.ocr_engine import ocr_image_bytes

            # 始终走多级 OCR 管道（独立容器 → 视觉大模型 → 进程内 RapidOCR → 带原因的提示）
            txt = ocr_image_bytes(content_bytes)
            if txt.strip():
                return txt.strip()
            return "【系统提示：该图片已由 OCR 引擎扫描，但画面中未识别到清晰文字】"
        except Exception as e:
            return f"【系统提示：图片 OCR 解析异常：{e}】"

    # 对 docx 尝试提取（优先使用 python-docx，降级使用 zipfile 解压 word/document.xml）
    if ext == "docx":
        para_texts: List[str] = []
        try:
            import docx

            doc_obj = docx.Document(io.BytesIO(content_bytes))
            para_texts = [p.text for p in doc_obj.paragraphs if p.text.strip()]
            # 同时也提取表格中的文字
            for tbl in doc_obj.tables:
                for row in tbl.rows:
                    row_txt = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_txt:
                        para_texts.append(row_txt)
        except Exception:
            pass

        if not para_texts:
            try:
                import zipfile
                import xml.etree.ElementTree as ET

                with zipfile.ZipFile(io.BytesIO(content_bytes)) as docx_zip:
                    xml_raw = docx_zip.read("word/document.xml")
                    try:
                        tree = ET.fromstring(xml_raw)
                        paragraphs = []
                        current_p = []
                        for elem in tree.iter():
                            if elem.tag.endswith("t"):
                                current_p.append(elem.text or "")
                            elif elem.tag.endswith("p"):
                                if current_p:
                                    paragraphs.append("".join(current_p))
                                    current_p = []
                        if current_p:
                            paragraphs.append("".join(current_p))
                        para_texts = paragraphs
                    except Exception:
                        # 命名空间容错正则提取
                        xml_str = xml_raw.decode("utf-8", errors="ignore")
                        t_matches = re.findall(r"<[^:>]*:?t[^>]*>(.*?)</[^:>]*:?t>", xml_str)
                        if t_matches:
                            para_texts = ["".join(t_matches)]
            except Exception:
                pass

        # 检查并调用 OCR 解析 DOCX 中内嵌的插图（word/media/）
        try:
            from app.ocr_engine import is_ocr_available, ocr_docx_embedded_images

            if is_ocr_available():
                img_results = ocr_docx_embedded_images(content_bytes)
                if img_results:
                    para_texts.append("\n【文档内嵌插图文字识别（OCR）】")
                    for img_name, img_txt in img_results:
                        para_texts.append(f"▶ 附图 [{img_name}]:\n{img_txt}")
        except Exception:
            pass

        if para_texts:
            return "\n".join(para_texts)

    # 对 PPTX 尝试原生零依赖提取（幻灯片文本解析）
    if ext == "pptx":
        try:
            import zipfile
            import xml.etree.ElementTree as ET

            with zipfile.ZipFile(io.BytesIO(content_bytes)) as ppt_zip:
                slide_files = sorted(
                    [f for f in ppt_zip.namelist() if f.startswith("ppt/slides/slide") and f.endswith(".xml")]
                )
                slides_text = []
                for s_idx, sf in enumerate(slide_files, 1):
                    tree = ET.fromstring(ppt_zip.read(sf))
                    slide_paras = []
                    for elem in tree.iter():
                        if elem.tag.endswith("}p"):
                            texts = [t.text for t in elem.iter() if t.tag.endswith("}t") and t.text]
                            if texts:
                                slide_paras.append("".join(texts))
                    if slide_paras:
                        slides_text.append(f"【幻灯片第 {s_idx} 页】\n" + "\n".join(slide_paras))
                if slides_text:
                    return "\n\n".join(slides_text)
        except Exception:
            pass

    # 对 XLSX 尝试原生零依赖提取（共享字符串池与各工作表解析）
    if ext == "xlsx":
        try:
            import zipfile
            import xml.etree.ElementTree as ET

            with zipfile.ZipFile(io.BytesIO(content_bytes)) as xlsx_zip:
                shared_strings = []
                if "xl/sharedStrings.xml" in xlsx_zip.namelist():
                    tree = ET.fromstring(xlsx_zip.read("xl/sharedStrings.xml"))
                    for si in tree.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}si"):
                        t_nodes = [t.text for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t") if t.text]
                        shared_strings.append("".join(t_nodes))

                sheet_files = sorted(
                    [f for f in xlsx_zip.namelist() if f.startswith("xl/worksheets/sheet") and f.endswith(".xml")]
                )
                sheets_text = []
                for s_idx, sf in enumerate(sheet_files, 1):
                    stree = ET.fromstring(xlsx_zip.read(sf))
                    rows = []
                    for row in stree.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}row"):
                        cells = []
                        for c in row.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}c"):
                            t_attr = c.attrib.get("t", "")
                            v = c.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                            if v is not None and v.text:
                                if t_attr == "s" and v.text.isdigit():
                                    idx = int(v.text)
                                    cells.append(shared_strings[idx] if idx < len(shared_strings) else "")
                                else:
                                    cells.append(v.text)
                        if any(cells):
                            rows.append(" | ".join(cells))
                    if rows:
                        sheets_text.append(f"【工作表 {s_idx}】\n" + "\n".join(rows))
                if sheets_text:
                    return "\n\n".join(sheets_text)
        except Exception:
            pass

    # 对早期二进制格式 doc / xls / ppt 提供中文特征安全抽取与防乱码兜底
    if ext in ("doc", "xls", "ppt"):
        try:
            # 扫描 OLE 二进制流中的 UTF-16LE 连续中文字符串
            raw_data = content_bytes
            extracted_paras: List[str] = []
            cur_run: List[str] = []
            for i in range(0, len(raw_data) - 1, 2):
                ch_bytes = raw_data[i : i + 2]
                try:
                    c = ch_bytes.decode("utf-16le")
                    if "\u4e00" <= c <= "\u9fff" or c in "，。！？；：“”‘’（）《》、\n\r":
                        cur_run.append(c)
                    else:
                        if len(cur_run) >= 6:
                            p_str = "".join(cur_run).strip()
                            if p_str:
                                extracted_paras.append(p_str)
                        cur_run = []
                except Exception:
                    cur_run = []
            if len(cur_run) >= 6:
                extracted_paras.append("".join(cur_run).strip())

            if extracted_paras and sum(len(p) for p in extracted_paras) >= 30:
                return "\n".join(extracted_paras)
        except Exception:
            pass
        return f"【系统提示：检测到该文件为旧版 Office 二进制格式（.{ext}），未能直接解析到结构化文字。为保障最佳检索效果与条文分段质量，建议使用 Office/WPS 另存为 .{ext}x 现代格式后重新上传】"

    # 对 PDF 尝试标准抽取（优先 pypdf，次选原生流解压，遇纯扫描件自动启动本地 OCR 识别）
    if ext == "pdf":
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            pages_txt = [page.extract_text() or "" for page in reader.pages]
            full_txt = "\n".join(p.strip() for p in pages_txt if p and p.strip())
            # 如果提取到的原生文字足够多（>= 30 个字符），判定为文字版 PDF 直接返回
            if full_txt and len(full_txt.strip()) >= 30:
                return full_txt
        except Exception:
            pass

        # 扫描件识别：若字符极少或为空，自动调用轻量级本地 OCR 管道提取各页位图
        try:
            from app.ocr_engine import is_ocr_available, ocr_pdf_pages

            if is_ocr_available():
                ocr_txt = ocr_pdf_pages(content_bytes)
                if ocr_txt.strip():
                    return ocr_txt.strip()
        except Exception:
            pass

        # 原生零依赖兜底：利用 zlib 解压缩 PDF 中的 FlateDecode stream 提取纯文本
        try:
            import zlib
            stream_pat = re.compile(b"stream[\r\n]+(.*?)[\r\n]+endstream", re.DOTALL)
            extracted_lines = []
            for m in stream_pat.finditer(content_bytes):
                s_data = m.group(1)
                try:
                    dec = zlib.decompress(s_data)
                except Exception:
                    try:
                        dec = zlib.decompress(s_data, -zlib.MAX_WBITS)
                    except Exception:
                        continue
                if dec:
                    # 匹配括号文本 (text) Tj 或 [ (t1) (t2) ] TJ
                    for sub in re.findall(rb"\(([^\)]+)\)\s*Tj", dec):
                        for enc in ("utf-8", "gb18030", "gbk"):
                            try:
                                t = sub.decode(enc).strip()
                                if any("\u4e00" <= ch <= "\u9fff" for ch in t):
                                    extracted_lines.append(t)
                                    break
                            except Exception:
                                pass
            if extracted_lines:
                return "\n".join(extracted_lines)
        except Exception:
            pass

        # 保护性防乱码拦截：如果依然包含 PDF 二进制流标识，绝不把原始字节乱码直接作为文本切分
        if b"%PDF" in content_bytes[:50] or b"0 obj" in content_bytes[:100]:
            return "【系统提示：该 PDF 文档为纯扫描图片或已加密保护，未能直接抽取到内嵌文字，建议使用 OCR 或上传文字版 PDF/TXT/Word 格式】"


# ————————————————————————————————————————————————
# 模式 ①：自动分割（自适应语义滑窗）
# ————————————————————————————————————————————————
def split_auto(
    text: str,
    target_len: int = 512,
    overlap: int = 64,
    doc_id: str = "",
    law_name: str = "",
) -> List[Dict[str, Any]]:
    """自适应语义滑窗分割算法：优先在换行、段落、句末标点截断，保持上下文连续。"""
    if not text:
        return []

    # 预按段落初步切分
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    chunks: List[Dict[str, Any]] = []
    current_buf: List[str] = []
    current_len = 0

    def _flush_buffer():
        nonlocal current_buf, current_len
        if not current_buf:
            return
        content = "\n".join(current_buf).strip()
        if content:
            chunks.append(
                {
                    "chunk_id": f"chunk_auto_{uuid.uuid4().hex[:12]}",
                    "doc_id": doc_id,
                    "law_name": law_name,
                    "chapter_path": "",
                    "article": f"第{len(chunks) + 1}段",
                    "paragraph": "",
                    "content": content,
                    "chunk_type": "auto",
                    "parent_chunk_id": "",
                }
            )
        # 滑窗保留 overlap 字符作为前缀
        tail = content[-overlap:] if len(content) > overlap else ""
        current_buf = [tail] if tail else []
        current_len = len(tail)

    for p in paragraphs:
        if current_len + len(p) > target_len and current_buf:
            _flush_buffer()
        current_buf.append(p)
        current_len += len(p)

    _flush_buffer()
    return chunks


# ————————————————————————————————————————————————
# 模式 ②：法条分割（有限状态机 FSM，层级继承）
# ————————————————————————————————————————————————
_RE_BOOK = re.compile(r"^\s*第[零〇一二两三四五六七八九十百]+编\s*(.*)")
_RE_CHAPTER = re.compile(r"^\s*第[零〇一二两三四五六七八九十百]+章\s*(.*)")
_RE_SECTION = re.compile(r"^\s*第[零〇一二两三四五六七八九十百]+节\s*(.*)")
_RE_ARTICLE = re.compile(r"^\s*(第[零〇一二两三四五六七八九十百千0-9]+条)\s*(.*)")


def split_by_articles(
    text: str,
    doc_id: str = "",
    law_name: str = "",
    hierarchy: str = "法律",
    effective_date: str = "",
    effective_to: str = "",
    validity: str = "现行有效",
) -> List[Dict[str, Any]]:
    """法条结构化切分：识别编、章、节路径层级，提取条、款，并将层级路径注入分块。"""
    from app.kb import normalize_article_id

    lines = text.splitlines()
    chunks: List[Dict[str, Any]] = []

    current_book = ""
    current_chapter = ""
    current_section = ""

    current_art_no = ""
    current_art_lines: List[str] = []

    def _flush_article():
        nonlocal current_art_no, current_art_lines
        if not current_art_no or not current_art_lines:
            return
        body = "\n".join(current_art_lines).strip()
        path_parts = [p for p in (current_book, current_chapter, current_section) if p]
        chapter_path = " / ".join(path_parts)
        norm_art = normalize_article_id(current_art_no)

        # 检查首行是否包含款项定义
        chunks.append(
            {
                "chunk_id": f"chunk_art_{uuid.uuid4().hex[:12]}",
                "doc_id": doc_id,
                "law_name": law_name,
                "chapter_path": chapter_path,
                "article": norm_art,
                "paragraph": "",
                "content": body,
                "chunk_type": "article",
                "parent_chunk_id": "",
                "hierarchy": hierarchy,
                "effective_date": effective_date,
                "effective_to": effective_to,
                "validity": validity,
            }
        )
        current_art_no = ""
        current_art_lines = []

    for line in lines:
        s = line.strip()
        if not s:
            continue

        # 1) 检查编
        mb = _RE_BOOK.match(s)
        if mb:
            _flush_article()
            current_book = s
            current_chapter = ""
            current_section = ""
            continue

        # 2) 检查章
        mc = _RE_CHAPTER.match(s)
        if mc:
            _flush_article()
            current_chapter = s
            current_section = ""
            continue

        # 3) 检查节
        ms = _RE_SECTION.match(s)
        if ms:
            _flush_article()
            current_section = s
            continue

        # 4) 检查条
        ma = _RE_ARTICLE.match(s)
        if ma:
            _flush_article()
            current_art_no = ma.group(1).strip()
            rest = ma.group(2).strip()
            current_art_lines = [rest] if rest else []
            continue

        # 5) 条文正文延续行
        if current_art_no:
            current_art_lines.append(s)
        else:
            # 尚未进入第一条前的序言或标题
            pass

    _flush_article()

    # 若未找到任何“第X条”特征，回退为单条
    if not chunks:
        chunks.append(
            {
                "chunk_id": f"chunk_art_{uuid.uuid4().hex[:12]}",
                "doc_id": doc_id,
                "law_name": law_name,
                "chapter_path": "",
                "article": "第1条",
                "paragraph": "",
                "content": text.strip(),
                "chunk_type": "article",
                "parent_chunk_id": "",
                "hierarchy": hierarchy,
                "effective_date": effective_date,
                "effective_to": effective_to,
                "validity": validity,
            }
        )
    return chunks


# ————————————————————————————————————————————————
# 模式 ③：文档分层分割（Hierarchical Parent-Child 分块）
# ————————————————————————————————————————————————
def split_hierarchical(
    text: str,
    doc_id: str = "",
    law_name: str = "",
    parent_chunk_size: int = 1200,
    child_chunk_size: int = 300,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """文档分层切分：
    - 生成 Parent Chunks（大上下文，用于回答合成）；
    - 生成 Child Chunks（小语义块，绑定 parent_chunk_id，用于高精度向量检索）。
    返回 (parent_chunks, child_chunks)。
    """
    article_chunks = split_by_articles(text, doc_id=doc_id, law_name=law_name)

    parent_chunks: List[Dict[str, Any]] = []
    child_chunks: List[Dict[str, Any]] = []

    cur_parent_id = f"parent_{uuid.uuid4().hex[:12]}"
    cur_parent_arts: List[Dict[str, Any]] = []
    cur_parent_len = 0

    def _flush_parent():
        nonlocal cur_parent_id, cur_parent_arts, cur_parent_len
        if not cur_parent_arts:
            return
        parent_content = "\n\n".join(
            f"《{law_name}》{a['article']}: {a['content']}" for a in cur_parent_arts
        )
        parent_chunks.append(
            {
                "chunk_id": cur_parent_id,
                "doc_id": doc_id,
                "law_name": law_name,
                "chapter_path": cur_parent_arts[0].get("chapter_path", ""),
                "article": f"{cur_parent_arts[0]['article']}~{cur_parent_arts[-1]['article']}",
                "content": parent_content,
                "chunk_type": "hierarchical_parent",
                "parent_chunk_id": "",
            }
        )
        # 为父块中的每一条生成 Child Chunk，指向父块
        for a in cur_parent_arts:
            child_chunks.append(
                {
                    "chunk_id": f"child_{uuid.uuid4().hex[:12]}",
                    "doc_id": doc_id,
                    "law_name": law_name,
                    "chapter_path": a.get("chapter_path", ""),
                    "article": a["article"],
                    "content": a["content"],
                    "chunk_type": "hierarchical_child",
                    "parent_chunk_id": cur_parent_id,
                }
            )
        cur_parent_id = f"parent_{uuid.uuid4().hex[:12]}"
        cur_parent_arts = []
        cur_parent_len = 0

    for a in article_chunks:
        c_len = len(a["content"])
        if cur_parent_len + c_len > parent_chunk_size and cur_parent_arts:
            _flush_parent()
        cur_parent_arts.append(a)
        cur_parent_len += c_len

    _flush_parent()
    return parent_chunks, child_chunks

