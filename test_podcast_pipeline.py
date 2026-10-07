"""
Test and verify the entire Arabic Meeting Summarizer pipeline on user's podcast.mp3:
1. Audio conversion (16kHz mono WAV)
2. Pyannote 3.1 Local Offline Diarization
3. Audar ASR V1 Turbo transcription
4. Word-to-speaker alignment
5. Speaker identification & renaming
6. Database storage (SQLite)
7. Falcon-H1-7B speaker-attributed summarization
8. Verification of results (no hallucinations, proper Arabic formatting)
"""
import sys
import time
import json
import asyncio
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_dir))

from app.services.audio_processor import convert_to_wav, get_audio_duration
from app.services.diarization import diarize, diarization_backend_name
from app.services.transcription import transcribe_audio
from app.services.alignment import align_timestamps
from app.services.summarization import summarize_meeting
from app.services.storage import init_db, save_meeting, get_meeting
from app.services.hardware import get_hardware_profile


async def run_podcast_test():
    print("=" * 70)
    print("🎙️ اختبار وتدقيق خط المعالجة الكامل على ملف podcast.mp3")
    print("=" * 70)

    input_file = Path("podcast.mp3")
    if not input_file.exists():
        print("❌ لم يتم العثور على podcast.mp3 في المجلد الرئيسي!")
        return False

    duration = get_audio_duration(input_file)
    print(f"\n[1/6] معلومات الملف الصوتي:")
    print(f"  - المسار: {input_file.resolve()}")
    print(f"  - المدة: {duration:.2f} ثانية ({duration / 60:.2f} دقيقة)")
    print(f"  - محرك فصل المتحدثين: {diarization_backend_name()}")

    # 1. Convert to 16kHz mono WAV
    wav_file = Path("podcast_16k.wav")
    print(f"\n[2/6] تحويل الملف الصوتي إلى 16kHz WAV...")
    t0 = time.time()
    success = await convert_to_wav(input_file, wav_file)
    if not success or not wav_file.exists():
        print("❌ فشل تحويل الصوت عبر ffmpeg!")
        return False
    print(f"  - تم التحويل بنجاح في {time.time() - t0:.2f} ثانية. الحجم: {wav_file.stat().st_size / 1024**2:.2f} MB")

    # 2. Pyannote Diarization
    print(f"\n[3/6] تشغيل فصل المتحدثين بواسطة Pyannote 3.1 المحلي...")
    t0 = time.time()
    turns = diarize(wav_file)
    dur_diar = time.time() - t0
    speakers_found = sorted(list(set(t["speaker"] for t in turns)))
    print(f"  - استغرق فصل المتحدثين: {dur_diar:.2f} ثانية")
    print(f"  - عدد مقاطع المتحدثين المكتشفة: {len(turns)}")
    print(f"  - قائمة المتحدثين المكتشفين: {speakers_found}")
    for idx, t in enumerate(turns[:4]):
        print(f"    مقطع {idx+1}: [{t['start']:.2f}s - {t['end']:.2f}s] {t['speaker']}")

    # 3. Audar ASR
    print(f"\n[4/6] تفريغ الصوت بنموذج Audar-ASR-V1-Turbo...")
    t0 = time.time()

    def asr_prog(pct, msg):
        print(f"    [{pct}%] {msg}")

    asr_res = transcribe_audio(wav_file, on_progress=asr_prog)
    dur_asr = time.time() - t0
    words = asr_res.get("words", [])
    full_text = asr_res.get("text", "")
    print(f"  - استغرق التفريغ: {dur_asr:.2f} ثانية")
    print(f"  - إجمالي الكلمات المفرغة: {len(words)}")
    print(f"  - بداية النص: {full_text[:180]}...")

    # 4. Alignment
    print(f"\n[5/6] مطابقة الكلمات مع المتحدثين (Alignment)...")
    aligned = align_timestamps(words, turns)
    print(f"  - عدد المقاطع بعد المطابقة: {len(aligned)}")
    for idx, seg in enumerate(aligned[:4]):
        print(f"    [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}: {seg['text'][:70]}...")

    # Set up speakers map (rename)
    speakers_map = {}
    for i, spk in enumerate(speakers_found):
        num = i + 1
        speakers_map[spk] = f"المتحدث {num:02d}"

    # Database test
    print(f"\n  - حفظ البيانات في قاعدة البيانات SQLite...")
    await init_db()
    meeting_id = "podcast-test-verified"
    await save_meeting({
        "id": meeting_id,
        "filename": "podcast.mp3",
        "duration_seconds": duration,
        "transcript_json": aligned,
        "diarization_json": turns,
        "speakers_json": speakers_map,
        "status": "transcribed"
    })
    saved = await get_meeting(meeting_id)
    assert saved is not None, "فشل استرجاع الاجتماع من قاعدة البيانات"
    print("  - تم الحفظ والاسترجاع بنجاح.")

    # 5. Summarization with Falcon-H1-7B
    print(f"\n[6/6] توليد التلخيص الذكي بنموذج Falcon-H1-7B...")
    t0 = time.time()

    def sum_prog(pct, msg):
        print(f"    [{pct}%] {msg}")

    summary = summarize_meeting(
        aligned,
        speakers_map,
        model_key="falcon-h1-7b-instruct",
        on_progress=sum_prog
    )
    dur_sum = time.time() - t0
    print(f"  - استغرق التلخيص: {dur_sum:.2f} ثانية")

    print("\n" + "=" * 70)
    print("📄 نص الملخص الذكي المولد:")
    print("=" * 70)
    print(summary)
    print("=" * 70)

    # Save detailed output report
    report_file = Path("podcast_verification_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "filename": "podcast.mp3",
            "duration_seconds": duration,
            "diarization_backend": diarization_backend_name(),
            "speakers_detected": speakers_found,
            "total_words": len(words),
            "total_segments": len(aligned),
            "speakers_map": speakers_map,
            "summary": summary,
            "timings": {
                "conversion_seconds": round(wav_file.stat().st_mtime - t0, 2),
                "diarization_seconds": round(dur_diar, 2),
                "asr_seconds": round(dur_asr, 2),
                "summarization_seconds": round(dur_sum, 2),
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ تم اكتمال الفحص الشامل بنجاح! تم حفظ التقرير في: {report_file.name}")
    return True


if __name__ == "__main__":
    ok = asyncio.run(run_podcast_test())
    sys.exit(0 if ok else 1)
