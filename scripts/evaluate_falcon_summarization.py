import sys
import time
import json
import re
import gc
from pathlib import Path

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.models.llm_engine import LLMEngine
from app.config import settings

LONG_ARABIC_MEETING_TEXT = """
المشرفة (د. ندى غنيم): ما بين صوتين، فأول من اشتغل بهذا المجال في الجمهورية العربية السورية كانت الدكتورة أميمة الدكاك، وأطروحتها للدكتوراه كانت عبارة عن مركب كلام باللغة الفرنسية. هي كانت المشرفة على رسالتي في الدكتوراه والماجستير ومشروع التخرج، يعني هي أستاذتي منذ أن كنت طالبة في البرامكة بالسنة الثالثة. اشتغلوا هناك على مركب كلام باللغة الفرنسية بما نسميه مركبات الكلام بالبواني (Formant Synthesis)، وتعتمد على استخراج البواني (Formants) للأصوات بالكامل. ويقال إنه إذا استطعنا استخراج البواني بدقة، نستطيع تركيب أي صوت من خلالها. وكان هذا من أصعب أنواع المركبات الصوتية سابقاً. وعندما عادت الدكتورة أميمة إلى هنا، تعاونت معي، وكانت أطروحة الدكتوراه الخاصة بي حول تحويل النص إلى فونيمات (Text-to-Phoneme). هل عرفتكم عنها؟
الطالب الأول: ما عرفنا عنها بالتفصيل.. التكست تو فونيم؟
المشرفة (د. ندى غنيم): تماماً، شو يعني تكست تو فونيم؟ وكيف يرتبط بمشروعكم؟
الطالب الأول: يعني نخرج الفونيمات الصوتية المقابلة للحروف المكتوبة. فإذا أردنا توليد صوت نحتاج الفونيمات لبناء إشارة الصوت.
المشرفة (د. ندى غنيم): عظيم، ولماذا تريدون استخدام Text-to-Phoneme في مشروعكم أنتم تحديداً؟
الطالب الأول: نحن نريد بناء نظام يساعد المذيعين على النطق بشكل صحيح عند قراءة الأخبار، لأن المذيع أحياناً يخطئ في الحركات.
المشرفة (د. ندى غنيم): لحظة، هل سكرتم الموبايل؟ ممتاز عشان نركز. أول ملاحظة عندي: أغلب الطلاب لديهم مشكلة كبيرة في إيصال أفكارهم بأسلوب هندسي واضح. يجب أن تبدأ دائماً من الهدف الأساسي (Objective) وتستخدم المصطلحات العلمية الدقيقة (Terms) الخاصة باختصاصكم. الآن أنتم قلتم مساعدة المذيعين على التحدث بالعربية الفصحى، أين دور الـ Text-to-Phoneme هنا؟ عاصف، أنت معهم، تفضل اشرح.
الطالب عاصف: المشكلة الأساسية تكمن في تشكيل الكلمات. في نشرات الأخبار، بعض المذيعين يتهربون من التشكيل السليم، مثلاً بدل أن يقول: "استقبلَ السيدُ المحافظُ" يقع في اللحن والخطأ. الحل الذي وجدناه يتكون من خطوتين: أولاً تشكيل النص آلياً، وثانياً مقارنة نطق المذيع أثناء تدريبه مع النص المشكول، لنرى هل ينطق النص بحركاته الصحيحة أم يشكل عشوائياً. فنحن نحتاج تحويل النص المشكول إلى فونيمات، ونقارنها مع فونيمات الصوت الفعلي للمذيع.
المشرفة (د. ندى غنيم): جميل، هذه الصياغة قربت الفكرة أكثر. والآن السؤال للطالب الثالث: ما هو المدخل (Input) والمخرج في مشروعكم؟
الطالب الثالث: الفكرة أن إذاعة الأخبار يعانون من أن المذيع المتخرج لا ينطق العربية بشكل صحيح، ويأخذ المدربون وقتاً طويلاً جداً في تدريبه. فطلبوا المساعدة بنظام ذكي. المدخل هو تسجيل صوت المذيع وهو يقرأ نصاً غير مشكول، ونظامنا يقوم بتصويب النطق وإظهار مواضع الخطأ.
المشرفة (د. ندى غنيم): لاحظتم كيف أن كل واحد منكم عرض زاوية مختلفة؟ صار وكأنه لدينا ثلاثة مشاريع منفصلة! إذن الهدف الموحد (Objective) لمشروع التخرج هو: مساعدة المذيعين في الإذاعة والتلفزيون على تصويب نطق اللغة العربية الفصحى وأخطاء التشكيل، عبر مقارنة فونيمات الصوت المسجل مع فونيمات النص بعد التشكيل الآلي، وتقديم تقرير تصويبي فوري.
"""

SYSTEM_PROMPT = (
    "أنت مساعد وخبير متخصص في تلخيص الاجتماعات والحوارات باللغة العربية الفصحى.\n"
    "- اكتب الملخص كاملاً باللغة العربية الفصحى حصراً وبشكل قطعي، ويُمنع منعاً باتاً استخدام اللغة الإنجليزية أو الصينية أو أي لغات أجنبية أخرى.\n"
    "- النص مُفرَّغ آلياً من الصوت؛ افهم المعنى المقصود وصحح الأخطاء الناتجة عن التفريغ.\n"
    "- كل سطر في النص يبدأ باسم المتحدث. انسب كل فكرة إلى قائلها بالاسم كما هو مكتوب حرفياً.\n"
    "- حافظ على تنسيق Markdown الواضح والعناوين القياسية باللغة العربية فقط بدون أي رموز أو لغات إضافية."
)

USER_PROMPT = f"""قائمة المتحدثين بالاجتماع: المشرفة (د. ندى غنيم)، الطالب الأول، الطالب عاصف، الطالب الثالث

━━━ بداية نص الاجتماع ━━━
{LONG_ARABIC_MEETING_TEXT.strip()}
━━━ نهاية نص الاجتماع ━━━

المطلوب: كتابة ملخص متكامل باللغة العربية الفصحى حصراً وفق الأقسام التالية:

## ملخص عام
فقرة من 2-4 جمل باللغة العربية الفصحى توضح فكرة الاجتماع ومحاوره الأساسية.

## مساهمات المتحدثين
لكل متحدث سطر يبدأ باسمه بالعربية ثم ما طرحه وركز عليه:
- **المشرفة (د. ندى غنيم):** (لخص مساهمتها وأفكارها وتوجيهاتها بالعربية الفصحى)
- **الطالب الأول:** (لخص ما طرحه بالعربية الفصحى)
- **الطالب عاصف:** (لخص ما طرحه بالعربية الفصحى)
- **الطالب الثالث:** (لخص ما طرحه بالعربية الفصحى)

## النقاط الرئيسية
- نقاط واضحة باللغة العربية الفصحى لأهم الأفكار العلمية والعملية مع ذكر اسم صاحب الفكرة بين قوسين.

## القرارات والتوصيات
القرارات الصادرة أو التوجيهات الهندسية المعتمدة.

## المهام والإجراءات
المهام المحددة والخطوات التالية للمشروع.

⚠️ تنبيه إلزامي وحاسم:
- يجب أن يكون الملخص كاملاً باللغة العربية الفصحى فقط.
- يُمنع منعاً باتاً كتابة أي فقرة أو ترجمة أو عبارة باللغة الإنجليزية.
"""

def evaluate_summary(text: str, duration_sec: float) -> dict:
    """Evaluate summary quality across multiple linguistic and structural dimensions."""
    words = text.split()
    word_count = len(words)
    char_count = len(text)

    # 1. Structure Check: Headings presence
    expected_headings = [
        "## ملخص عام",
        "## مساهمات المتحدثين",
        "## النقاط الرئيسية",
        "## القرارات والتوصيات",
        "## المهام والإجراءات"
    ]
    headings_found = sum(1 for h in expected_headings if h in text)
    structure_score = round((headings_found / len(expected_headings)) * 10, 1)

    # 2. Key Entities & Information Recall
    key_entities = [
        "ندى غنيم",
        "أميمة الدكاك",
        "بواني" if "بواني" in text else "فورمانت",
        "فونيم",
        "مذيع",
        "تشكيل",
        "تصويب"
    ]
    entities_found = sum(1 for e in key_entities if e in text)
    info_recall_score = round((entities_found / len(key_entities)) * 10, 1)

    # 3. Foreign Language Contamination (English characters)
    english_words = re.findall(r"[a-zA-Z]{2,}", text)
    # Exclude common allowed names if any, but prompt strictly forbids english
    has_english_leak = len(english_words) > 3

    # 4. Repetition / Stutter Loop Detection (repeated 4-grams)
    grams = [" ".join(words[i:i+4]) for i in range(len(words)-3)]
    repetition_count = len(grams) - len(set(grams))
    repetition_ratio = repetition_count / max(1, len(grams))
    has_repetition_loop = repetition_ratio > 0.15

    # 5. Overall Quality Score (out of 10)
    quality_score = (
        structure_score * 0.35 +
        info_recall_score * 0.45 +
        (0.0 if has_english_leak else 1.0) +
        (0.0 if has_repetition_loop else 1.0)
    )
    quality_score = min(10.0, round(quality_score, 1))

    return {
        "word_count": word_count,
        "char_count": char_count,
        "headings_found": f"{headings_found}/{len(expected_headings)}",
        "structure_score": structure_score,
        "entities_found": f"{entities_found}/{len(key_entities)}",
        "info_recall_score": info_recall_score,
        "english_word_count": len(english_words),
        "has_english_leak": has_english_leak,
        "repetition_ratio": round(repetition_ratio, 3),
        "has_repetition_loop": has_repetition_loop,
        "quality_score": quality_score,
        "duration_sec": round(duration_sec, 2),
        "words_per_sec": round(word_count / max(0.1, duration_sec), 2)
    }

def main():
    eval_dir = Path("eval_results/falcon_experiments")
    eval_dir.mkdir(parents=True, exist_ok=True)

    model_path = settings.MODELS_DIR / "Falcon-H1-7B-Instruct-IQ4_XS.gguf"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    print("=" * 75)
    print("Initializing Falcon-H1-7B LLM Engine for Hyperparameter Exploration...")
    print("=" * 75)

    engine = LLMEngine(model_path, max_context=8192)
    t0 = time.time()
    engine.load()
    print(f"Falcon LLM loaded in {time.time()-t0:.2f}s with GPU layers={engine.params.get('n_gpu_layers')}")

    # Experimental parameter configurations to test
    experiments = [
        {
            "id": "exp1_baseline_default",
            "name": "الضبط الافتراضي (Default Baseline)",
            "params": {"temperature": 0.7, "top_p": 0.90, "top_k": 40, "repeat_penalty": 1.05, "min_p": 0.0, "max_tokens": 1200}
        },
        {
            "id": "exp2_high_temp_creative",
            "name": "حرارة مرتفعة / إبداعي (High Temp)",
            "params": {"temperature": 0.9, "top_p": 0.95, "top_k": 50, "repeat_penalty": 1.05, "min_p": 0.0, "max_tokens": 1200}
        },
        {
            "id": "exp3_cold_deterministic",
            "name": "حرارة منخفضة جداً / حتمي (Cold Deterministic)",
            "params": {"temperature": 0.1, "top_p": 0.80, "top_k": 20, "repeat_penalty": 1.05, "min_p": 0.05, "max_tokens": 1200}
        },
        {
            "id": "exp4_zero_repeat_penalty",
            "name": "بدون عقوبة تكرار (Zero Repeat Penalty)",
            "params": {"temperature": 0.2, "top_p": 0.90, "top_k": 40, "repeat_penalty": 1.00, "min_p": 0.05, "max_tokens": 1200}
        },
        {
            "id": "exp5_aggressive_penalty",
            "name": "عقوبة تكرار مفرطة (Aggressive Penalty)",
            "params": {"temperature": 0.3, "top_p": 0.90, "top_k": 40, "repeat_penalty": 1.30, "min_p": 0.05, "max_tokens": 1200}
        },
        {
            "id": "exp6_balanced_factual",
            "name": "توازن واقعي متقن (Balanced Factual)",
            "params": {"temperature": 0.3, "top_p": 0.88, "top_k": 40, "repeat_penalty": 1.12, "min_p": 0.05, "max_tokens": 1200}
        },
        {
            "id": "exp7_optimal_tuned",
            "name": "الضبط المثالي الموصى به (Optimal Tuned)",
            "params": {"temperature": 0.18, "top_p": 0.85, "top_k": 40, "repeat_penalty": 1.15, "min_p": 0.05, "max_tokens": 1200}
        }
    ]

    results = []

    for exp in experiments:
        exp_id = exp["id"]
        exp_name = exp["name"]
        p = exp["params"]

        print("\n" + "-" * 70)
        print(f"Running Experiment: {exp_id} - {exp_name}")
        print(f"Parameters: temp={p['temperature']}, top_p={p['top_p']}, top_k={p['top_k']}, "
              f"rep_pen={p['repeat_penalty']}, min_p={p['min_p']}, max_tok={p['max_tokens']}")
        print("-" * 70)

        t_gen_start = time.time()
        output_text = engine.chat(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            max_tokens=p["max_tokens"],
            temperature=p["temperature"],
            top_p=p["top_p"],
            top_k=p["top_k"],
            min_p=p["min_p"],
            repeat_penalty=p["repeat_penalty"]
        )
        gen_duration = time.time() - t_gen_start

        # Evaluate output
        metrics = evaluate_summary(output_text, gen_duration)

        print(f"Generation completed in {gen_duration:.2f}s ({metrics['words_per_sec']} words/s)")
        print(f"Score: {metrics['quality_score']}/10 | Structure: {metrics['headings_found']} | "
              f"Entities: {metrics['entities_found']} | English Leak: {metrics['has_english_leak']} | "
              f"Loops: {metrics['has_repetition_loop']}")
        print(f"Snippet:\n{output_text[:250]}...\n")

        # Save individual experiment output
        exp_file = eval_dir / f"{exp_id}.md"
        with open(exp_file, "w", encoding="utf-8") as f:
            f.write(f"# تجربة تلخيص: {exp_name} ({exp_id})\n\n")
            f.write("## الإعدادات:\n")
            f.write(f"```json\n{json.dumps(p, indent=2, ensure_ascii=False)}\n```\n\n")
            f.write("## المقاييس والتقييم:\n")
            f.write(f"```json\n{json.dumps(metrics, indent=2, ensure_ascii=False)}\n```\n\n")
            f.write("## نص الملخص المولد:\n\n")
            f.write(output_text)

        results.append({
            "id": exp_id,
            "name": exp_name,
            "params": p,
            "metrics": metrics,
            "summary_snippet": output_text[:300],
            "full_summary": output_text
        })

    # Find the optimal experiment
    sorted_by_score = sorted(results, key=lambda x: (x["metrics"]["quality_score"], -x["metrics"]["repetition_ratio"]), reverse=True)
    best_exp = sorted_by_score[0]

    print("\n" + "=" * 75)
    print(f"EXPERIMENTS COMPLETED! Optimal Configuration Found: {best_exp['id']} ({best_exp['name']})")
    print(f"Optimal Score: {best_exp['metrics']['quality_score']}/10")
    print(f"Optimal Parameters: {json.dumps(best_exp['params'], ensure_ascii=False)}")
    print("=" * 75)

    summary_results_file = Path("eval_results/falcon_experiments_summary.json")
    with open(summary_results_file, "w", encoding="utf-8") as f:
        json.dump({
            "best_configuration": best_exp,
            "all_experiments": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved complete experiment comparison to {summary_results_file}")

    engine.unload()
    gc.collect()

if __name__ == "__main__":
    main()
