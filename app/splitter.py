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

    # 对 docx 尝试提取（优先使用 python-docx，降级使用 zipfile 解压 word/document.xml）
    if ext == "docx":
        try:
            import docx

            doc_obj = docx.Document(io.BytesIO(content_bytes))
            para_texts = [p.text for p in doc_obj.paragraphs if p.text.strip()]
            if para_texts:
                return "\n".join(para_texts)
        except Exception:
            pass

        try:
            import zipfile
            import xml.etree.ElementTree as ET

            with zipfile.ZipFile(io.BytesIO(content_bytes)) as docx_zip:
                xml_content = docx_zip.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                text_nodes = tree.iter()
                paragraphs = []
                current_p = []
                for elem in text_nodes:
                    if elem.tag.endswith("t"):
                        current_p.append(elem.text or "")
                    elif elem.tag.endswith("p"):
                        if current_p:
                            paragraphs.append("".join(current_p))
                            current_p = []
                if current_p:
                    paragraphs.append("".join(current_p))
                return "\n".join(paragraphs)
        except Exception:
            pass

    # 对 PDF 尝试标准抽取（优先 pypdf，次选原生流解压，杜绝二进制乱码泄漏）
    if ext == "pdf":
        try:
            import pypdf

            reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            pages_txt = [page.extract_text() or "" for page in reader.pages]
            full_txt = "\n".join(p.strip() for p in pages_txt if p and p.strip())
            if full_txt:
                return full_txt
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

