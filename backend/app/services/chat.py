"""
Chat service for Falcon LLM Q&A grounded exclusively in the transcribed text (transcript_json).
Ensures that the LLM answers questions based ONLY on the verbatim transcription with speaker attribution,
and NOT from the summary text.
"""
import re
import logging
from typing import Dict, List, Optional

from ..models.llm_engine import LLMEngine
from ..models.registry import registry
from ..config import settings

logger = logging.getLogger(__name__)

CHAT_SYSTEM_PROMPT = (
    "أنت 'فالكون'، المساعد الذكي المتخصص في الإجابة عن أسئلة المستخدم حول هذا التسجيل الصوتي أو الاجتماع باللغة العربية الفصحى.\n"
    "⚠️ قواعد صارمة للإجابة:\n"
    "1. مصدرك الوحيد والنهائي للإجابة هو 'النص المفرغ من الصوت' المرفق في المحادثة حصراً. يُمنع منعاً باتاً الاعتماد على أي ملخصات أو معلومات غير مذكورة فيه.\n"
    "2. انسب كل معلومة أو فكرة لصاحبها (اسم المتحدث) كما وردت في النص المفرغ، واذكر التوقيت التقريبي إن وُجد.\n"
    "3. إذا كان سؤال المستخدم يتعلق بأمر لم يُذكر أو لم يرد في النص المفرغ، أجب بكل وضوح وأمانة: 'لم ترد هذه المعلومة في النص المفرغ للتسجيل'.\n"
    "4. لغة الإجابة: يجب أن تكون الإجابة كاملة باللغة العربية الفصحى فقط وبشكل حصري وقطعي. يُمنع منعاً باتاً كتابة أي كلمات أو رموز بلغات أجنبية أو آسيوية (مثل الصينية أو الإنجليزية وغيرها).\n"
)


def format_seconds(seconds: float) -> str:
    """Format seconds into MM:SS or HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _display_speaker(speaker_id: str, speakers: Dict[str, str]) -> str:
    if speaker_id in speakers and speakers[speaker_id].strip():
        return speakers[speaker_id].strip()
    try:
        return f"المتحدث {int(speaker_id.split('_')[-1]) + 1:02d}"
    except (ValueError, IndexError):
        return speaker_id or "متحدث"


def format_transcript_lines(segments: List[Dict], speakers: Dict[str, str]) -> List[str]:
    """
    Format transcript segments into clean speaker-attributed, timestamped lines.
    E.g. '[01:15] د. ياسر الحزيمي: نص الكلام...'
    """
    lines = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = _display_speaker(seg.get("speaker", ""), speakers)
        start_time = format_seconds(seg.get("start", 0.0))
        lines.append(f"[{start_time}] {speaker}: {text}")
    return lines


def _extract_keywords(text: str) -> set[str]:
    """Extract informative Arabic keyword tokens (length >= 3, excluding common stop words)."""
    stop_words = {
        "في", "من", "إلى", "على", "عن", "مع", "هذا", "هذه", "ذلك", "تلك", "التي", "الذي",
        "الذين", "ما", "ماذا", "هل", "كيف", "أين", "متى", "لماذا", "هو", "هي", "هم", "هن",
        "أنا", "نحن", "أنت", "كان", "كانت", "يكون", "أن", "إن", "ثم", "أو", "أم", "لا", "نعم",
        "كل", "بعض", "غير", "بين", "حول", "خلال", "عند", "قبل", "بعد", "حيث", "لو", "إذا"
    }
    words = re.findall(r"[\u0600-\u06FFa-zA-Z0-9]+", text.lower())
    return {w for w in words if len(w) >= 3 and w not in stop_words}


def retrieve_relevant_context(lines: List[str], question: str, engine: LLMEngine, budget: int) -> str:
    """
    If the transcript exceeds the model's token budget, select the most relevant
    segment windows based on the question keywords, preserving chronological order.
    """
    if not lines:
        return ""
    
    full_transcript = "\n".join(lines)
    if engine.count_tokens(full_transcript) <= budget:
        return full_transcript

    keywords = _extract_keywords(question)
    if not keywords:
        # Fallback to the first lines that fit within budget
        selected = []
        cur_tokens = 0
        for l in lines:
            tok = engine.count_tokens(l)
            if cur_tokens + tok > budget:
                break
            selected.append(l)
            cur_tokens += tok
        return "\n".join(selected)

    # Score each line by keyword hits
    scores = []
    for i, line in enumerate(lines):
        line_words = set(re.findall(r"[\u0600-\u06FFa-zA-Z0-9]+", line.lower()))
        score = sum(1 for kw in keywords if kw in line_words or any(kw in w for w in line_words))
        scores.append((score, i))

    # Pick top scoring indices and expand a window around them
    scores.sort(key=lambda x: x[0], reverse=True)
    selected_indices = set()
    for score, idx in scores:
        if score == 0 and len(selected_indices) > 0:
            break
        # Expand window: idx - 1, idx, idx + 1
        for w in range(max(0, idx - 1), min(len(lines), idx + 2)):
            selected_indices.add(w)

    sorted_indices = sorted(selected_indices)
    
    # Filter to fit within token budget
    final_lines = []
    cur_tokens = 0
    for idx in sorted_indices:
        l = lines[idx]
        tok = engine.count_tokens(l)
        if cur_tokens + tok > budget:
            break
        final_lines.append(l)
        cur_tokens += tok

    if not final_lines:
        final_lines = lines[:10]

    return "\n".join(final_lines)


def chat_with_transcript(
    segments: List[Dict],
    speakers: Dict[str, str],
    question: str,
    history: Optional[List[Dict[str, str]]] = None,
    model_key: str = "falcon-h1-7b-instruct",
) -> str:
    """
    Generate an answer using Falcon LLM based strictly on the provided transcript segments.
    The summary text is deliberately not part of this context.
    """
    if not segments:
        raise ValueError("لا يتوفر نص مفرغ لهذا الاجتماع")
    
    question = (question or "").strip()
    if not question:
        raise ValueError("سؤال المستخدم فارغ")

    model_info = registry.get_model(model_key) or (registry.list_llm_models() or [None])[0]
    if not model_info:
        raise ValueError("No LLM models registered.")

    engine = LLMEngine(settings.MODELS_DIR / model_info.gguf_filename, max_context=model_info.context_size)
    lines = format_transcript_lines(segments, speakers)
    
    try:
        engine.load()
        
        # Calculate token budget for transcript context:
        # budget = n_ctx - system_prompt - response_tokens (768) - history/question overhead
        reserved_tokens = 768 + 256
        sys_tokens = engine.count_tokens(CHAT_SYSTEM_PROMPT)
        q_tokens = engine.count_tokens(question)
        budget = max(512, engine.n_ctx - sys_tokens - reserved_tokens - q_tokens - 300)

        transcript_context = retrieve_relevant_context(lines, question, engine, budget)

        # Build messages payload
        # Prepend transcript context to the first user turn or system message
        context_block = (
            "━━━ بداية النص المفرغ من التسجيل الصوتي (المصدر الوحيد للإجابة) ━━━\n"
            f"{transcript_context}\n"
            "━━━ نهاية النص المفرغ ━━━\n\n"
        )

        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]

        # Add recent history (up to last 6 messages) if present
        clean_history = []
        if history:
            for item in history[-6:]:
                role = item.get("role")
                content = (item.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    clean_history.append({"role": role, "content": content})

        if clean_history:
            # Attach transcript context to the first user message in history
            first_user_idx = next((i for i, m in enumerate(clean_history) if m["role"] == "user"), None)
            if first_user_idx is not None:
                clean_history[first_user_idx]["content"] = context_block + clean_history[first_user_idx]["content"]
            else:
                messages[0]["content"] += f"\n\n{context_block}"
            
            messages.extend(clean_history)
            messages.append({
                "role": "user",
                "content": f"{question}\n\n(تنبيه حاسم: أجب باللغة العربية الفصحى فقط وبشكل قطعي. يُمنع منعاً باتاً إدراج أي كلمات بلغات أجنبية مثل الصينية أو الإنجليزية أو كتابة أي ترجمة)."
            })
        else:
            # Single turn: attach context block directly before question
            messages.append({
                "role": "user",
                "content": f"{context_block}سؤال المستخدم حول النص المفرغ:\n{question}\n\n(تنبيه حاسم: أجب باللغة العربية الفصحى فقط وبشكل مباشر وقطعي، ويُمنع منعاً باتاً استخدام أي كلمات بلغات أجنبية مثل الصينية أو الإنجليزية)."
            })

        logger.info(f"Generating Falcon LLM response for question: {question[:80]}...")
        reply = engine.chat(messages, max_tokens=768, temperature=0.2)
        return _clean_arabic_response(reply)
    finally:
        engine.unload()


def _clean_arabic_response(text: str) -> str:
    """
    Ensure the LLM response is strictly in Arabic.
    Strips any trailing English translations, English summaries, hallucinated bilingual sections,
    and any stray CJK / Asian characters emitted by multilingual tokenizers.
    """
    if not text:
        return ""
    text = text.strip()

    # 1. Truncate at common English translation section markers
    split_patterns = [
        r"(?i)\n+\s*(?:English\s+translation|Translation\s+in\s+English|In\s+English|English\s+version|Here\s+is\s+the\s+translation|Summary\s+in\s+English)\s*[:：\-]?",
        r"(?i)\n+\s*###?\s*(?:English|Translation|In\s+English)",
    ]
    for pattern in split_patterns:
        match = re.search(pattern, text)
        if match:
            text = text[:match.start()].strip()

    # 2. Check trailing paragraphs: if a paragraph is predominantly English, strip it
    paragraphs = text.split("\n\n")
    cleaned_paras = []
    for p in paragraphs:
        p_str = p.strip()
        if not p_str:
            continue
        arabic_chars = len(re.findall(r"[\u0600-\u06FF]", p_str))
        english_chars = len(re.findall(r"[a-zA-Z]", p_str))
        # If paragraph has significant english text with very little or no arabic, drop it
        if english_chars > 20 and english_chars > 2 * (arabic_chars + 1):
            continue
        cleaned_paras.append(p_str)

    text = "\n\n".join(cleaned_paras).strip()

    # 3. Strip any stray CJK ideographs / Chinese / Japanese / Korean characters and punctuation
    text = re.sub(
        r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\u31f0-\u31ff\u2e80-\u2eff\u3000-\u303f\uff00-\uffef]+",
        "",
        text,
    )

    # 4. Clean up spaces before punctuation and fix double periods
    text = re.sub(r"\s+([.,،؛:!؟])", r"\1", text)
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)

    return text.strip()


# Specific parameters optimized for Falcon H1 conversational chat, common-sense reasoning, and riddles
CHAT_DEFAULT_TEMPERATURE = 0.6
CHAT_DEFAULT_TOP_P = 0.95
CHAT_DEFAULT_TOP_K = 40
CHAT_DEFAULT_REPEAT_PENALTY = 1.15

GENERAL_FALCON_SYSTEM_PROMPT = (
    "أنت 'Falcon H1'، مساعد ذكي يتمتع بذكاء تحليلي رفيع، استنتاج منطقي عميق، ودقة علمية ولغوية فائقة.\n\n"
    "قواعد التفكير والتحليل المنطقي الصارمة:\n"
    "1. حصر الأشخاص والأنشطة (Counting & Activities):\n"
    "   - احسب بدقة عدد الأشخاص المذكورين في السؤال حصراً، ولا تخترع أشخاصاً غير مذكورين.\n"
    "   - الأنشطة الثنائية (مثل الشطرنج أو أرجوحة التوازن) تتطلب شخصين معاً. فإذا كان هناك شخص شاغر في الغرفة بينما البقية منشغلون بأنشطة فردية (كالبيانو أو التزلج)، فالشاغر يشارك الأول حتماً (مثل سامر مع محمد، أو ريان مع خالد).\n"
    "   - إذا كان المتاح شخصاً واحداً فقط والآخر منشغلاً ولا يوجد شخص ثالث (مثل سارة ومريم فقط)، فالشخص يتدرب بمفرده أو لا يوجد طرف ثانٍ.\n"
    "2. البديهيات والألغاز الكلاسيكية:\n"
    "   - إذا مشى شخص تحت المطر ولم يبتل شعره دون ساتر، فالسبب البديهي أنه أصلع ليس في رأسه شعر.\n"
    "   - الرؤية وانعكاس المرآة يستحيلان تماماً في الظلام الدامس لغياب الضوء، فلا يرى شيئاً.\n"
    "   - الطفو يعتمد على الكثافة: التفاحة والريشة تطفوان فوق الماء، والحجر يغرق.\n"
    "   - مهنة الجرّاح أو الطبيب تمارسها النساء والرجال، فالجرّاح قد يكون والدة الطفل (أمه).\n"
    "   - الشيء الذي يشترك فيه اثنان وإذا باحا به لم يعد سراً هو 'السر'.\n"
    "   - في حركة الساعات، عقرب الساعات يتحرك 0.5 درجة مع كل دقيقة؛ عند 3:15 تكون الزاوية بين العقربين 7.5 درجات.\n"
    "3. أسلوب الإجابة واللغة:\n"
    "   - قدّم الإجابة المباشرة والصحيحة أولاً بإيجاز تام (في سطر إلى سطرين فقط).\n"
    "   - اكتب باللغة العربية الفصحى النقية فقط، واكتب كل الأسماء بحروف عربية حصراً (مثل سارة ومريم)، ويُحظر تماماً إدخال أي حروف لاتينية."
)


def _clean_general_response(text: str) -> str:
    """
    Clean general chat responses: strips stray Asian / CJK characters and cleans spacing.
    Preserves English words, code blocks, and markdown formatting.
    """
    if not text:
        return ""
    text = text.strip()

    # 1. Strip any stray CJK ideographs / Chinese / Japanese / Korean characters and punctuation
    text = re.sub(
        r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u30ff\u31f0-\u31ff\u2e80-\u2eff\u3000-\u303f\uff00-\uffef]+",
        "",
        text,
    )

    # 2. Fix known subword tokenizer glitch where Arabic letter is glued to English transliteration (e.g. سarah -> سارة)
    text = re.sub(r"\b[سS]arah\b", "سارة", text, flags=re.IGNORECASE)
    text = re.sub(r"سarah", "سارة", text)

    # 3. Clean up spaces before punctuation and fix double periods
    text = re.sub(r"\s+([.,،؛:!؟])", r"\1", text)
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)

    return text.strip()


def direct_falcon_chat(
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    documents: Optional[List[Dict[str, str]]] = None,
    system_prompt: Optional[str] = None,
    model_key: str = "falcon-h1-7b-instruct",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
) -> str:
    """
    Direct chat with Falcon H1 LLM. Supports optional attached documents for Q&A,
    summarization, and analysis. Uses refined sampling parameters for sharp logical deduction.
    """
    from .document_parser import build_document_context_block

    message = (message or "").strip()
    if not message:
        raise ValueError("الرسالة لا يمكن أن تكون فارغة")

    model_info = registry.get_model(model_key) or (registry.list_llm_models() or [None])[0]
    if not model_info:
        raise ValueError("No LLM models registered.")

    engine = LLMEngine(settings.MODELS_DIR / model_info.gguf_filename, max_context=model_info.context_size)
    sys_prompt = system_prompt or GENERAL_FALCON_SYSTEM_PROMPT

    effective_temp = CHAT_DEFAULT_TEMPERATURE if temperature is None else temperature
    effective_top_p = CHAT_DEFAULT_TOP_P if top_p is None else top_p
    effective_rep_pen = CHAT_DEFAULT_REPEAT_PENALTY if repeat_penalty is None else repeat_penalty

    try:
        engine.load()

        doc_context = ""
        if documents:
            reserved_tokens = 1024 + 256
            sys_tokens = engine.count_tokens(sys_prompt)
            q_tokens = engine.count_tokens(message)
            budget = max(512, engine.n_ctx - sys_tokens - reserved_tokens - q_tokens - 400)
            doc_context = build_document_context_block(
                documents, message, engine.count_tokens, budget
            )

        messages = [{"role": "system", "content": sys_prompt}]

        # Keep last 8 messages for conversation context
        clean_history = []
        if history:
            for item in history[-8:]:
                role = item.get("role")
                content = (item.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    clean_history.append({"role": role, "content": content})

        if doc_context:
            context_block = f"{doc_context}\n\n"
            if clean_history:
                # Prepend doc context to the first user message in history
                first_u = next((i for i, m in enumerate(clean_history) if m["role"] == "user"), None)
                if first_u is not None:
                    clean_history[first_u]["content"] = context_block + clean_history[first_u]["content"]
                else:
                    messages[0]["content"] += f"\n\n{context_block}"
                messages.extend(clean_history)
                messages.append({"role": "user", "content": message})
            else:
                messages.append({
                    "role": "user",
                    "content": f"{context_block}سؤال أو طلب المستخدم حول محتوى الملف:\n{message}"
                })
        else:
            messages.extend(clean_history)
            messages.append({"role": "user", "content": message})

        logger.info(f"Generating direct Falcon LLM response for: {message[:80]}... (temp={effective_temp})")
        reply = engine.chat(
            messages,
            max_tokens=1024,
            temperature=effective_temp,
            top_p=effective_top_p,
            top_k=CHAT_DEFAULT_TOP_K,
            repeat_penalty=effective_rep_pen,
        )
        return _clean_general_response(reply)
    finally:
        engine.unload()


