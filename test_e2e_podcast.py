import sys
import os
import json
import time
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure backend is on sys.path
backend_dir = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_dir))

from backend.app.config import settings
from backend.app.services.transcription import transcribe_audio
from backend.app.services.diarization import diarize
from backend.app.services.alignment import align_timestamps
from backend.app.services.summarization import summarize_text
from backend.app.services.audio_processor import get_audio_duration

def main():
    print("=" * 60)
    print("اختبار خط الأنابيب الكامل (End-to-End Test)")
    print("النماذج الأساسية:")
    print("1. تحويل الصوت (ASR): Audar-ASR-V1-Turbo-Q8_0.gguf + mmproj")
    print("2. التلخيص (LLM): Falcon-H1-7B-Instruct-IQ4_XS.gguf")
    print("=" * 60)

    podcast_path = Path("كيف تنجح العلاقات مع ياسر الحزيمي _ بودكاست فنجان(MP3_320K).mp3")
    if not podcast_path.exists():
        print(f"الملف غير موجود: {podcast_path}")
        return

    duration = get_audio_duration(podcast_path)
    print(f"\nملف الصوت: {podcast_path.name}")
    print(f"المدة الكلية للملف: {duration} ثانية (~{round(duration/60, 1)} دقيقة)")

    # We will test on a representative 75-second slice (3 full ASR windows of rich speech)
    test_duration = 75.0
    print(f"\nجاري بدء تفريغ مقطع من البودكاست بطول {test_duration} ثانية عبر Audar ASR...")

    t0 = time.time()
    def progress_callback(pct, msg):
        print(f"[{pct}%] {msg}")

    asr_result = transcribe_audio(
        podcast_path,
        on_progress=progress_callback,
        max_duration_seconds=test_duration
    )
    t_asr = time.time() - t0
    print(f"\nاكتمل التفريغ الصوتي في {round(t_asr, 1)} ثانية.")
    print(f"عدد الكلمات المفردة: {len(asr_result.get('words', []))}")
    print(f"النص المفرغ الكامل:\n\"{asr_result.get('text', '')}\"")

    print("\n" + "-" * 50)
    print("الخطوة 2: فصل المتحدثين ومطابقة الطوابع الزمنية...")
    diarization_result = diarize(podcast_path)
    aligned_segments = align_timestamps(asr_result.get("words", []), diarization_result)
    print(f"تم إنشاء {len(aligned_segments)} مقطع موزعة حسب المتحدثين:")
    for idx, seg in enumerate(aligned_segments[:5], 1):
        print(f"  [{seg['speaker']}] ({seg['start']}s - {seg['end']}s): {seg['text']}")

    print("\n" + "-" * 50)
    print("الخطوة 3: التلخيص الذكي للاجتماع عبر Falcon-H1-7B-Instruct...")
    t_sum_start = time.time()
    full_transcript = asr_result.get('text', '')
    summary = summarize_text(full_transcript, model_key="falcon-h1-7b-instruct")
    t_summary = time.time() - t_sum_start
    print(f"اكتمل التلخيص في {round(t_summary, 1)} ثانية.")

    print("\n" + "=" * 60)
    print("النتيجة النهائية للملخص الذكي:")
    print("=" * 60)
    print(summary)
    print("=" * 60)

    # Save to a report json file for artifact
    report = {
        "audio_file": podcast_path.name,
        "total_duration_seconds": duration,
        "tested_duration_seconds": test_duration,
        "asr_time_seconds": round(t_asr, 2),
        "summary_time_seconds": round(t_summary, 2),
        "transcript": full_transcript,
        "aligned_segments": aligned_segments,
        "summary": summary
    }
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\nتم حفظ التقرير في test_results.json بنجاح.")

if __name__ == "__main__":
    main()
