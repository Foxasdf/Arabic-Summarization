"""
Speaker-attributed, hierarchical (map-reduce) summarization.

1. Build a transcript where every line is prefixed by the speaker's *display
   name* (e.g. "د. ياسر الحزيمي: ...").
2. If it fits in the model's context → single pass with the final prompt.
3. Otherwise:
     MAP    – each chunk (split on line boundaries, names re-attached if a long
              turn is cut) → notes grouped by speaker.
     REDUCE – merge notes repeatedly until they fit, always keeping names.
     FINAL  – structured summary with a dedicated "speaker contributions" section.
"""
import logging
from typing import Callable, Dict, List, Optional

from ..models.llm_engine import LLMEngine
from ..models.registry import registry
from ..config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "أنت مساعد وخبير متخصص في تلخيص الاجتماعات والحوارات باللغة العربية الفصحى.\n"
    "- اكتب الملخص كاملاً باللغة العربية الفصحى حصراً وبشكل قطعي، ويُمنع منعاً باتاً استخدام اللغة الإنجليزية أو الصينية أو أي لغات أجنبية أخرى.\n"
    "- النص مُفرَّغ آلياً من الصوت؛ افهم المعنى المقصود وصحح الأخطاء الناتجة عن التفريغ.\n"
    "- كل سطر في النص يبدأ باسم المتحدث. انسب كل فكرة إلى قائلها بالاسم كما هو مكتوب حرفياً.\n"
    "- حافظ على تنسيق Markdown الواضح والعناوين القياسية باللغة العربية فقط بدون أي رموز أو لغات إضافية."
)


def _display(speaker_id: str, speakers: Dict[str, str]) -> str:
    if speaker_id in speakers and speakers[speaker_id].strip():
        return speakers[speaker_id].strip()
    try:
        return f"المتحدث {int(speaker_id.split('_')[-1]) + 1:02d}"
    except ValueError:
        return speaker_id


def build_transcript_lines(segments: List[Dict], speakers: Dict[str, str]) -> List[str]:
    lines = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        name = _display(seg.get("speaker", ""), speakers)
        if lines and lines[-1].startswith(name + ": "):
            lines[-1] += " " + text
        else:
            lines.append(f"{name}: {text}")
    return lines


def _chunk_lines(lines: List[str], engine: LLMEngine, budget: int) -> List[str]:
    """Pack lines into chunks ≤ budget tokens; split oversized turns and re-prefix the name."""
    chunks, cur, cur_tok = [], [], 0
    for line in lines:
        tok = engine.count_tokens(line)
        if tok > budget:
            name, _, body = line.partition(": ")
            words, piece = body.split(), []
            for w in words:
                piece.append(w)
                if engine.count_tokens(" ".join(piece)) > budget * 0.9:
                    sub = f"{name}: {' '.join(piece)}"
                    if cur:
                        chunks.append("\n".join(cur)); cur, cur_tok = [], 0
                    chunks.append(sub); piece = []
            if piece:
                line, tok = f"{name}: {' '.join(piece)}", engine.count_tokens(" ".join(piece)) + 5
            else:
                continue
        if cur_tok + tok > budget and cur:
            chunks.append("\n".join(cur)); cur, cur_tok = [], 0
        cur.append(line); cur_tok += tok
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def _names_hint(names: List[str]) -> str:
    return "قائمة المتحدثين بالاجتماع: " + "، ".join(names) + "\n\n" if names else ""


def _clean_summary_markdown(text: str) -> str:
    """Standardize markdown headings, strip foreign text/translations and stray artifacts."""
    import re
    fixes = [
        (r'##\s*مساهمات[^\n]*', '## مساهمات المتحدثين'),
        (r'##\s*النقاط[^\n]*', '## النقاط الرئيسية'),
        (r'##\s*القرارات[^\n]*', '## القرارات والتوصيات'),
        (r'##\s*المهام[^\n]*', '## المهام والإجراءات'),
    ]
    for pat, rep in fixes:
        text = re.sub(pat, rep, text)
    # Filter any stray CJK characters occasionally emitted by multilingual vocab
    text = re.sub(
        r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\u31f0-\u31ff\u2e80-\u2eff\u3000-\u303f\uff00-\uffef]+",
        "",
        text,
    )
    text = text.replace('according to', 'حسب ما ذكر')
    text = re.sub(r"\s+([.,،؛:!؟])", r"\1", text)
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)

    # Truncate at common English translation sections
    split_patterns = [
        r"(?i)\n+\s*(?:English\s+translation|Translation\s+in\s+English|In\s+English|English\s+version|Summary\s+in\s+English)\s*[:：\-]?",
        r"(?i)\n+\s*###?\s*(?:English|Translation|In\s+English)",
    ]
    for pattern in split_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()].strip()

    # Drop trailing paragraphs that are mostly English
    paragraphs = text.split("\n\n")
    cleaned_paras = []
    for p in paragraphs:
        p_str = p.strip()
        if not p_str:
            continue
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", p_str))
        english_chars = len(re.findall(r"[a-zA-Z]", p_str))
        if english_chars > 20 and english_chars > 2 * (arabic_chars + 1):
            continue
        cleaned_paras.append(p_str)

    res = "\n\n".join(cleaned_paras).strip()
    return res if res else text.strip()


def _final_prompt(body: str, names: List[str], from_notes: bool) -> str:
    src = "ملاحظات مرتبة لأجزاء الاجتماع" if from_notes else "نص الاجتماع"
    contrib = "\n".join(f"- **{n}:** (لخص مساهمته وأفكاره هنا بالعربية الفصحى)" for n in names) or "- **اسم المتحدث:** (مساهمته بالعربية الفصحى)"
    return (
        f"{_names_hint(names)}"
        f"━━━ بداية {src} ━━━\n{body}\n━━━ نهاية {src} ━━━\n\n"
        "المطلوب: كتابة ملخص متكامل باللغة العربية الفصحى حصراً وفق الأقسام التالية:\n\n"
        "## ملخص عام\n"
        "فقرة من 2-4 جمل باللغة العربية الفصحى توضح فكرة الاجتماع ومحاوره الأساسية.\n\n"
        "## مساهمات المتحدثين\n"
        "لكل متحدث سطر يبدأ باسمه بالعربية ثم ما طرحه وركز عليه:\n"
        f"{contrib}\n\n"
        "## النقاط الرئيسية\n"
        "- نقاط واضحة باللغة العربية الفصحى لأهم الأفكار مع ذكر اسم صاحب الفكرة بين قوسين (اسم المتحدث).\n\n"
        "## القرارات والتوصيات\n"
        "القرارات الصادرة أو كتابة: لا توجد قرارات واضحة.\n\n"
        "## المهام والإجراءات\n"
        "المهام المحددة مع اسم المسؤول أو كتابة: لا توجد مهام محددة.\n\n"
        "⚠️ تنبيه إلزامي وحاسم:\n"
        "- يجب أن يكون الملخص كاملاً باللغة العربية الفصحى فقط.\n"
        "- يُمنع منعاً باتاً كتابة أي فقرة أو ترجمة أو عبارة باللغة الإنجليزية."
    )


def _map_prompt(chunk: str, names: List[str], idx: int, total: int) -> str:
    return (
        f"{_names_hint(names)}"
        f"هذا الجزء {idx} من {total} من نص اجتماع:\n\n{chunk}\n\n"
        "استخرج ملاحظات موجزة مجمّعة حسب المتحدث باللغة العربية الفصحى حصراً بهذا الشكل:\n"
        "### <اسم المتحدث كما هو>\n- فكرة أو رأي أو سؤال طرحه\n\n"
        "ثم في النهاية:\n### قرارات ومهام\n- ... (أو: لا يوجد)\n"
        "لا تذكر متحدثاً لم يتكلم في هذا الجزء. يُمنع منعاً باتاً استخدام اللغة الإنجليزية."
    )


def _reduce_prompt(notes: str, names: List[str]) -> str:
    return (
        f"{_names_hint(names)}"
        f"فيما يلي ملاحظات من أجزاء متتالية لنفس الاجتماع:\n\n{notes}\n\n"
        "ادمجها في ملاحظات أقصر باللغة العربية الفصحى حصراً بنفس الشكل (### اسم المتحدث ثم نقاطه)، "
        "واحذف التكرار، وحافظ على نسبة كل فكرة لصاحبها بالاسم. يُمنع منعاً باتاً استخدام اللغة الإنجليزية."
    )


def summarize_meeting(
    segments: List[Dict],
    speakers: Dict[str, str],
    model_key: str = "falcon-h1-7b-instruct",
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> str:
    def progress(p, m):
        logger.info(m)
        if on_progress:
            on_progress(p, m)

    model_info = registry.get_model(model_key) or (registry.list_llm_models() or [None])[0]
    if not model_info:
        raise ValueError("No LLM models registered.")
    engine = LLMEngine(settings.MODELS_DIR / model_info.gguf_filename, max_context=model_info.context_size)

    lines = build_transcript_lines(segments, speakers)
    if not lines:
        raise ValueError("النص فارغ")
    # names in order of first appearance
    names = list(dict.fromkeys(l.split(": ", 1)[0] for l in lines))

    try:
        progress(5, "جاري تحميل نموذج التلخيص...")
        engine.load()
        overhead = engine.count_tokens(SYSTEM_PROMPT + _final_prompt("", names, False)) + 64
        budget = engine.n_ctx - engine.max_tokens - overhead
        transcript = "\n".join(lines)
        tokens = engine.count_tokens(transcript)
        logger.info(f"Transcript tokens={tokens}, budget={budget}, n_ctx={engine.n_ctx}")

        if tokens <= budget:
            progress(30, "جاري توليد الملخص...")
            raw = engine.generate(SYSTEM_PROMPT, _final_prompt(transcript, names, False))
            return _clean_summary_markdown(raw)

        # ---- MAP
        map_budget = engine.n_ctx - 768 - overhead
        chunks = _chunk_lines(lines, engine, map_budget)
        notes = []
        for i, ch in enumerate(chunks, 1):
            progress(10 + int(60 * i / len(chunks)), f"تلخيص الجزء {i} من {len(chunks)}...")
            notes.append(engine.generate(SYSTEM_PROMPT, _map_prompt(ch, names, i, len(chunks)), max_tokens=768))

        # ---- REDUCE (hierarchical)
        level = 0
        while engine.count_tokens("\n\n".join(notes)) > budget and len(notes) > 1:
            level += 1
            groups, cur = [], []
            for n in notes:
                if cur and engine.count_tokens("\n\n".join(cur + [n])) > map_budget:
                    groups.append(cur); cur = []
                cur.append(n)
            groups.append(cur)
            if len(groups) == len(notes):  # can't pack → pair-wise
                groups = [notes[i:i + 2] for i in range(0, len(notes), 2)]
            progress(75, f"دمج الملاحظات (المستوى {level})...")
            notes = [engine.generate(SYSTEM_PROMPT, _reduce_prompt("\n\n---\n\n".join(g), names), max_tokens=768)
                     for g in groups]

        progress(90, "جاري صياغة الملخص النهائي...")
        raw = engine.generate(SYSTEM_PROMPT, _final_prompt("\n\n---\n\n".join(notes), names, True))
        return _clean_summary_markdown(raw)
    finally:
        engine.unload()


def summarize_text(transcript: str, model_key: str = "falcon-h1-7b-instruct") -> str:
    """Backward-compatible helper for plain text (single unnamed speaker)."""
    return summarize_meeting([{"speaker": "SPEAKER_00", "text": transcript}], {}, model_key)
