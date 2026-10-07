import io
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".xml",
    ".yaml", ".yml", ".html", ".htm", ".css", ".js", ".jsx",
    ".ts", ".tsx", ".py", ".c", ".cpp", ".h", ".cs", ".java",
    ".go", ".rs", ".php", ".rb", ".sql", ".sh", ".bat", ".ps1",
    ".log", ".ini", ".env", ".rst", ".toml"
}


def _decode_text_bytes(file_bytes: bytes) -> str:
    """Try several common encodings to decode text bytes, with emphasis on UTF-8 and Arabic encodings."""
    encodings = ["utf-8", "utf-8-sig", "cp1256", "iso-8859-6", "latin-1"]
    for enc in encodings:
        try:
            return file_bytes.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return file_bytes.decode("utf-8", errors="replace")


def _extract_pdf(file_bytes: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages_text.append(f"--- صفحة {i + 1} ---\n{text}")
        except Exception as e:
            logger.warning(f"Error extracting PDF page {i + 1}: {e}")
    return "\n\n".join(pages_text).strip()


def _extract_docx(file_bytes: bytes) -> str:
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    lines = []
    for p in doc.paragraphs:
        t = p.text.strip()
        if t:
            lines.append(t)
    for table in doc.tables:
        for row in table.rows:
            row_cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if row_cells:
                lines.append(" | ".join(row_cells))
    return "\n".join(lines).strip()


def _extract_xlsx(file_bytes: bytes) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    sheets_text = []
    for sheet_name in wb.sheetnames:
        sheet = wb[sheet_name]
        rows_text = []
        for row in sheet.iter_rows(values_only=True):
            if any(cell is not None for cell in row):
                row_str = " | ".join(str(cell).strip() if cell is not None else "" for cell in row)
                if row_str.strip():
                    rows_text.append(row_str)
        if rows_text:
            sheets_text.append(f"--- ورقة العمل: {sheet_name} ---\n" + "\n".join(rows_text))
    return "\n\n".join(sheets_text).strip()


def extract_text_from_file(file_bytes: bytes, filename: str) -> Dict[str, any]:
    """
    Extract readable text from uploaded file based on its extension.
    Returns metadata and full extracted content.
    """
    ext = Path(filename).suffix.lower()
    text = ""

    if ext == ".pdf":
        text = _extract_pdf(file_bytes)
    elif ext in (".docx", ".doc"):
        if ext == ".docx":
            text = _extract_docx(file_bytes)
        else:
            raise ValueError("تنسيق .doc القديم غير مدعوم، يرجى حفظ الملف بصيغة .docx أو .pdf")
    elif ext in (".xlsx", ".xls"):
        if ext == ".xlsx":
            text = _extract_xlsx(file_bytes)
        else:
            raise ValueError("تنسيق .xls القديم غير مدعوم، يرجى حفظ الملف بصيغة .xlsx أو .csv")
    elif ext in TEXT_EXTENSIONS or ext.startswith("."):
        text = _decode_text_bytes(file_bytes)
    else:
        # Fallback: attempt to decode as text
        try:
            text = _decode_text_bytes(file_bytes)
        except Exception:
            raise ValueError(f"نوع الملف '{ext}' غير مدعوم لاستخراج النصوص.")

    text = text.strip()
    if not text:
        raise ValueError(
            f"تعذر استخراج نص من الملف '{filename}'. قد يكون الملف فارغاً أو يحتوي على صور ممسوحة ضوئياً بدون نصوص قابلة للتعرف."
        )

    words = text.split()
    preview = text[:300].strip() + ("..." if len(text) > 300 else "")

    return {
        "filename": filename,
        "extension": ext,
        "text": text,
        "char_count": len(text),
        "word_count": len(words),
        "preview": preview,
    }


def _extract_arabic_keywords(text: str) -> set[str]:
    stop_words = {
        "في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "تلك", "التي", "الذي",
        "الذين", "ما", "ماذا", "هل", "كيف", "أين", "متى", "لماذا", "هو", "هي", "هم", "هن",
        "أنا", "نحن", "أنت", "كان", "كانت", "يكون", "أن", "إن", "ثم", "أو", "أم", "لا", "نعم",
        "كل", "بعض", "غير", "بين", "حول", "خلال", "عند", "قبل", "بعد", "حيث", "لو", "إذا"
    }
    words = re.findall(r"[\u0600-\u06FFa-zA-Z0-9]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in stop_words}


def build_document_context_block(
    documents: List[Dict[str, str]],
    question: str,
    count_tokens_fn,
    token_budget: int
) -> str:
    """
    Format attached documents for LLM context.
    If total text fits within token_budget, injects full documents.
    If documents exceed token_budget, applies smart chunking and keyword relevance retrieval.
    """
    if not documents:
        return ""

    # Build full raw context first
    full_blocks = []
    for doc in documents:
        fname = doc.get("filename", "مستند مرفق")
        content = (doc.get("content") or "").strip()
        if content:
            full_blocks.append(
                f"━━━ بداية محتوى الملف المرفق: {fname} ━━━\n"
                f"{content}\n"
                f"━━━ نهاية محتوى الملف: {fname} ━━━"
            )

    combined_full = "\n\n".join(full_blocks)
    if not combined_full.strip():
        return ""

    if count_tokens_fn(combined_full) <= token_budget:
        return combined_full

    # Exceeds budget: chunk documents into sections of ~300 words and rank by relevance
    logger.info("Documents exceed token budget. Performing relevance ranking for context...")
    keywords = _extract_arabic_keywords(question)
    
    chunks = []
    for doc in documents:
        fname = doc.get("filename", "مستند")
        content = doc.get("content", "")
        # Split by double newlines or pages
        paras = re.split(r"\n{2,}|(?=--- صفحة \d+ ---)", content)
        
        current_chunk = []
        current_words = 0
        for p in paras:
            p = p.strip()
            if not p:
                continue
            p_words = len(p.split())
            if current_words + p_words > 350 and current_chunk:
                chunks.append((fname, "\n\n".join(current_chunk)))
                current_chunk = [p]
                current_words = p_words
            else:
                current_chunk.append(p)
                current_words += p_words
        if current_chunk:
            chunks.append((fname, "\n\n".join(current_chunk)))

    if not chunks:
        # Fallback: slice raw text up to budget
        return combined_full[:token_budget * 2]

    # Score chunks
    scored = []
    for i, (fname, chunk_text) in enumerate(chunks):
        score = 0
        if keywords:
            chunk_tokens = set(re.findall(r"[\u0600-\u06FFa-zA-Z0-9]+", chunk_text.lower()))
            score = len(keywords & chunk_tokens)
        scored.append((score, i, fname, chunk_text))

    # Sort descending by score, then ascending by original position
    scored.sort(key=lambda x: (-x[0], x[1]))

    selected = []
    current_tokens = 0
    header_overhead = 100

    for score, original_idx, fname, chunk_text in scored:
        c_block = f"[{fname} - مقطع {original_idx + 1}]:\n{chunk_text}"
        tok = count_tokens_fn(c_block)
        if current_tokens + tok + header_overhead <= token_budget:
            selected.append((original_idx, c_block))
            current_tokens += tok
        elif not selected:
            # At least take the first chunk even if slightly truncated
            selected.append((original_idx, c_block[:token_budget * 2]))
            break

    # Re-order chunks in original chronological sequence
    selected.sort(key=lambda x: x[0])
    selected_texts = [item[1] for item in selected]

    return (
        "━━━ مقتطفات من محتوى الملفات المرفقة (الأكثر صلة بسؤالك) ━━━\n\n"
        + "\n\n---\n\n".join(selected_texts)
        + "\n\n━━━ نهاية مقتطفات الملفات المرفقة ━━━"
    )
