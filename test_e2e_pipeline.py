"""
End-to-end test script for the complete Arabic Meeting Summarizer pipeline:
1. Hardware detection & configuration
2. Diarization (pyannote or builtin fallback)
3. Audar ASR transcription
4. Word-to-speaker alignment
5. Speaker renaming (inline/modal workflow)
6. SQLite database storage & retrieval
7. Falcon-H1-7B speaker-attributed summarization
"""
import sys
import json
import time
from pathlib import Path

# Add backend to sys.path
backend_path = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.hardware import get_hardware_profile
from app.services.diarization import diarize, diarization_backend_name
from app.services.transcription import transcribe_audio
from app.services.alignment import align_timestamps
from app.services.summarization import summarize_meeting
from app.services.storage import save_meeting, get_meeting, init_db
import asyncio


async def run_pipeline_test():
    print("=" * 60)
    print("🚀 بدء اختبار سلسلة المعالجة الكاملة (E2E Test)")
    print("=" * 60)

    # 1. Hardware Profile
    print("\n[1/7] فحص مواصفات العتاد والتكيف الذكي:")
    hw = get_hardware_profile()
    print(f"  - المعالج (Threads): {hw.cpu_threads}")
    print(f"  - الذاكرة العشوائية (RAM): {hw.ram_available_gb}GB متاح من {hw.ram_total_gb}GB")
    print(f"  - بطاقة الرسوميات (GPU): {hw.gpu_name or 'لا يوجد'}")
    print(f"  - دعم التسريع الرسومي (llama.cpp GPU offload): {hw.llama_gpu_offload}")
    print(f"  - تصنيف العتاد: {hw.tier}")
    asr_params = hw.asr_params()
    llm_params = hw.llm_params(4.0)
    print(f"  - إعدادات ASR: {asr_params}")
    print(f"  - إعدادات LLM (Context window): {llm_params['n_ctx']} tokens")

    # Audio sample
    audio_file = Path("test_sample_2min.wav")
    if not audio_file.exists():
        print("❌ ملف الصوت test_sample_2min.wav غير موجود!")
        return False

    # 2. Diarization
    print("\n[2/7] تشغيل فصل المتحدثين (Speaker Diarization):")
    backend_name = diarization_backend_name()
    print(f"  - المحرك المستخدم: {backend_name}")
    t0 = time.time()
    turns = diarize(audio_file)
    dur_diar = time.time() - t0
    speakers_detected = sorted(list(set(t['speaker'] for t in turns)))
    print(f"  - استغرق: {dur_diar:.2f} ثانية")
    print(f"  - عدد المقاطع الزمنية: {len(turns)}")
    print(f"  - المتحدثون المكتشفون: {speakers_detected}")

    # 3. ASR (Audar ASR Turbo)
    print("\n[3/7] تشغيل تفريغ الصوت بنموذج Audar-ASR-V1-Turbo:")
    t0 = time.time()
    
    def on_asr_progress(pct, msg):
        print(f"    [{pct}%] {msg}")

    asr_result = transcribe_audio(audio_file, on_progress=on_asr_progress)
    dur_asr = time.time() - t0
    print(f"  - استغرق: {dur_asr:.2f} ثانية")
    print(f"  - عدد الكلمات المفرغة: {len(asr_result.get('words', []))}")
    print(f"  - عينة من النص: {asr_result.get('text', '')[:150]}...")

    # 4. Alignment
    print("\n[4/7] مطابقة الطوابع الزمنية بين الكلمات والمتحدثين (Alignment):")
    aligned = align_timestamps(asr_result.get("words", []), turns)
    print(f"  - عدد المقاطع بعد المطابقة: {len(aligned)}")
    for i, seg in enumerate(aligned[:3]):
        print(f"    [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}: {seg['text'][:60]}...")

    # 5. Speaker Renaming
    print("\n[5/7] اختبار تعديل وتسمية المتحدثين:")
    # Map default IDs to custom names
    speakers_map = {}
    if len(speakers_detected) >= 2:
        speakers_map[speakers_detected[0]] = "عبدالرحمن أبو مالح"
        speakers_map[speakers_detected[1]] = "د. ياسر الحزيمي"
    else:
        for i, s in enumerate(speakers_detected):
            speakers_map[s] = f"ضيف {i+1}"
    print(f"  - خريطة الأسماء الجديدة: {speakers_map}")

    # 6. Database Storage & Retrieval
    print("\n[6/7] اختبار الحفظ والاسترجاع في قاعدة البيانات (SQLite):")
    await init_db()
    test_id = "test-meeting-e2e"
    save_data = {
        "id": test_id,
        "filename": "test_sample_2min.wav",
        "duration_seconds": 150.0,
        "transcript_json": aligned,
        "diarization_json": turns,
        "speakers_json": speakers_map,
        "status": "transcribed"
    }
    await save_meeting(save_data)
    loaded = await get_meeting(test_id)
    assert loaded is not None, "فشل استرجاع الاجتماع من قاعدة البيانات"
    assert loaded["speakers_json"] == speakers_map, "الأسماء المحفوظة غير متطابقة"
    print("  - تم حفظ واسترجاع بيانات الاجتماع والمتحدثين بنجاح.")

    # 7. Speaker-Attributed Summarization
    print("\n[7/7] تشغيل التلخيص الذكي بنموذج Falcon-H1-7B مع عزو النقاط للمتحدثين:")
    t0 = time.time()
    
    def on_sum_progress(pct, msg):
        print(f"    [{pct}%] {msg}")

    summary = summarize_meeting(
        aligned,
        speakers_map,
        model_key="falcon-h1-7b-instruct",
        on_progress=on_sum_progress
    )
    dur_sum = time.time() - t0
    print(f"  - استغرق التلخيص: {dur_sum:.2f} ثانية")

    print("\n" + "=" * 60)
    print("📄 نتيجة الملخص الذكي المولد:")
    print("=" * 60)
    print(summary)
    print("=" * 60)

    # Save results to a file for review
    out_file = Path("e2e_test_output.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "hardware": hw.as_dict(),
            "diarization_turns": turns,
            "aligned_segments": aligned,
            "speakers_map": speakers_map,
            "summary": summary,
            "timings": {
                "diarization_seconds": dur_diar,
                "asr_seconds": dur_asr,
                "summarization_seconds": dur_sum
            }
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ اكتمل الاختبار بنجاح وتم حفظ النتائج في {out_file.name}")
    return True

if __name__ == "__main__":
    success = asyncio.run(run_pipeline_test())
    sys.exit(0 if success else 1)
