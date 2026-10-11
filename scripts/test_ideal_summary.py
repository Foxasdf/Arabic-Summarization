import sys
import json
import time
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
from app.services.summarization import (
    SYSTEM_PROMPT,
    _final_prompt,
    _clean_summary_markdown,
)
from app.config import settings

def main():
    model_path = settings.MODELS_DIR / "Falcon-H1-7B-Instruct-IQ4_XS.gguf"
    engine = LLMEngine(model_path, max_context=8192)
    engine.load()

    # Load the real transcript obtained from the 5-minute audio slice (from Cohere or Whisper)
    cohere_transcript_file = Path("eval_results/transcript_cohere.txt")
    if cohere_transcript_file.exists():
        raw_meeting_text = cohere_transcript_file.read_text(encoding="utf-8")
    else:
        raw_meeting_text = "اجتماع طلاب تخرج مع المشرفة د. ندى غنيم حول مشروع معالجة وتوليد الصوت وتصحيح نطق المذيعين."

    # Setup speaker names identified in the meeting
    names = [
        "د. ندى غنيم (المشرفة)",
        "الطالب الأول",
        "الطالب عاصف",
        "الطالب الثالث"
    ]

    prompt = _final_prompt(raw_meeting_text, names, from_notes=False)

    print("Generating Gold Standard Summary with Optimal Tuned Hyperparameters...")
    # Optimal Hyperparameters discovered from exploration:
    # temperature=0.20, top_p=0.88, top_k=40, min_p=0.05, repeat_penalty=1.10, max_tokens=1200
    t0 = time.time()
    raw_summary = engine.chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.20,
        top_p=0.88,
        top_k=40,
        min_p=0.05,
        repeat_penalty=1.10,
        max_tokens=1200
    )
    gen_time = time.time() - t0

    # Post-process with production cleaning pipeline
    cleaned_summary = _clean_summary_markdown(raw_summary)

    print(f"\nSummary generated in {gen_time:.2f}s:\n")
    print(cleaned_summary)

    # Save to eval_results
    out_file = Path("eval_results/optimal_falcon_meeting_summary.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# الملخص النهائي المعتمد بنموذج Falcon-H1-7B بالإعدادات المثالية\n\n")
        f.write(f"- زمن التوليد: {gen_time:.2f} ثانية\n")
        f.write("- الإعدادات المثالية:\n")
        f.write("  - `temperature`: 0.20\n")
        f.write("  - `top_p`: 0.88\n")
        f.write("  - `top_k`: 40\n")
        f.write("  - `min_p`: 0.05\n")
        f.write("  - `repeat_penalty`: 1.10\n")
        f.write("  - `max_tokens`: 1200\n\n")
        f.write("## نص الملخص:\n\n")
        f.write(cleaned_summary)

    print(f"\nSaved final optimal summary to {out_file}")
    engine.unload()

if __name__ == "__main__":
    main()
