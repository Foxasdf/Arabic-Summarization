import sys
import os
import json
import time
import subprocess
import tempfile
from pathlib import Path

# Force UTF-8 output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Ensure backend path
backend_dir = Path(__file__).resolve().parent / "backend"
sys.path.insert(0, str(backend_dir))

from backend.app.config import settings
from backend.app.models.asr_engine import ASREngine
from backend.app.services.diarization import diarize
from backend.app.services.alignment import align_timestamps
from backend.app.services.summarization import summarize_text
from backend.app.services.audio_processor import get_audio_duration, get_ffmpeg_binary

CHECKPOINT_FILE = Path("podcast_checkpoint.json")
TRANSCRIPT_TXT_FILE = Path("podcast_full_transcript.txt")
FINAL_RESULTS_FILE = Path("podcast_full_results.json")
FINAL_SUMMARY_FILE = Path("podcast_full_summary.md")

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_chunks": 0, "chunks_data": []}

def save_checkpoint(data):
    try:
        with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save checkpoint: {e}")

def main():
    print("=" * 70)
    print("بدء المعالجة الكاملة لبودكاست فنجان (3 ساعات كاملة)")
    print("النماذج الأساسية:")
    print(" - تحويل الصوت: Audar-ASR-V1-Turbo-Q8_0.gguf + mmproj")
    print(" - التلخيص الذكي: Falcon-H1-7B-Instruct-IQ4_XS.gguf")
    print("=" * 70)

    podcast_path = Path("كيف تنجح العلاقات مع ياسر الحزيمي _ بودكاست فنجان(MP3_320K).mp3")
    if not podcast_path.exists():
        print(f"[ERROR] ملف الصوت غير موجود: {podcast_path}")
        return

    total_duration = get_audio_duration(podcast_path)
    print(f"اسم الملف: {podcast_path.name}")
    print(f"المدة الكلية: {total_duration} ثانية (~{round(total_duration/60, 1)} دقيقة / ~{round(total_duration/3600, 2)} ساعة)")

    chunk_len = 25.0
    total_chunks = max(1, int((total_duration + chunk_len - 1) // chunk_len))
    print(f"إجمالي الأجزاء المقسمة: {total_chunks} جزء (كل جزء {chunk_len} ثانية)")

    checkpoint = load_checkpoint()
    start_chunk_idx = checkpoint.get("completed_chunks", 0)
    all_chunks_data = checkpoint.get("chunks_data", [])

    if start_chunk_idx > 0:
        print(f"تم العثور على نقطة حفظ سابقة! استئناف من الجزء {start_chunk_idx + 1} من {total_chunks}...")

    if start_chunk_idx < total_chunks:
        # Load ASR Engine
        asr_model_path = settings.MODELS_DIR / settings.ASR_GGUF_FILENAME
        mmproj_path = settings.MODELS_DIR / settings.ASR_MMPROJ_FILENAME
        asr_engine = ASREngine(asr_model_path, mmproj_path=mmproj_path)

        print("\nتحميل نموذج Audar ASR V1 Turbo في الذاكرة...")
        asr_engine.load()
        print("تم تحميل نموذج الصوت بنجاح.")

        ffmpeg_bin = get_ffmpeg_binary()
        start_time_all = time.time()
        chunk_times = []

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            for i in range(start_chunk_idx, total_chunks):
                start_sec = i * chunk_len
                dur_sec = min(chunk_len, total_duration - start_sec)
                if dur_sec <= 0.5:
                    break

                t_chunk_start = time.time()
                slice_wav = tmp_path / f"chunk_{i:04d}.wav"

                # Extract slice
                cmd = [
                    ffmpeg_bin, "-y",
                    "-ss", str(round(start_sec, 2)),
                    "-t", str(round(dur_sec, 2)),
                    "-i", str(podcast_path),
                    "-ar", "16000", "-ac", "1",
                    "-c:a", "pcm_s16le",
                    str(slice_wav),
                ]
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if proc.returncode != 0 or not slice_wav.exists():
                    print(f"[Warning] فشل تقطيع المقطع عند {start_sec}s")
                    continue

                # Transcribe
                chunk_text = asr_engine._transcribe_slice(slice_wav)
                if slice_wav.exists():
                    slice_wav.unlink(missing_ok=True)

                t_chunk_dur = time.time() - t_chunk_start
                chunk_times.append(t_chunk_dur)
                avg_time = sum(chunk_times) / len(chunk_times)
                remaining_chunks = total_chunks - (i + 1)
                eta_seconds = remaining_chunks * avg_time
                eta_mins = round(eta_seconds / 60, 1)

                progress_pct = round(((i + 1) / total_chunks) * 100, 1)

                print(f"[{progress_pct}%] جزء {i+1}/{total_chunks} ({round(start_sec + dur_sec)}/{round(total_duration)}s) [الوقت المتبقي المقدر: ~{eta_mins} دقيقة]")
                if chunk_text:
                    print(f"  -> \"{chunk_text}\"")

                chunk_record = {
                    "chunk_index": i,
                    "start": round(start_sec, 2),
                    "end": round(start_sec + dur_sec, 2),
                    "text": chunk_text,
                }
                all_chunks_data.append(chunk_record)

                # Update checkpoint
                save_checkpoint({
                    "completed_chunks": i + 1,
                    "chunks_data": all_chunks_data
                })

                # Append text incrementally to text file
                with open(TRANSCRIPT_TXT_FILE, "a", encoding="utf-8") as tf:
                    if chunk_text:
                        tf.write(chunk_text + " ")

        # Unload ASR Engine to free VRAM/RAM for Falcon
        print("\nإنهاء تفريغ الصوت. جاري تفريغ نموذج Audar ASR من الذاكرة تماماً...")
        asr_engine.unload()
        print("تم تفريغ نموذج الصوت بنجاح.")
    else:
        print(f"\nتم تفريغ كامل الأجزاء مسبقاً ({total_chunks}/{total_chunks})! الانتقال مباشرة لمرحلة التلخيص...")

    # Reconstruct words and full text
    full_text_list = []
    all_words = []
    for c in all_chunks_data:
        c_text = c.get("text", "").strip()
        if c_text:
            full_text_list.append(c_text)
            c_words = c_text.split()
            c_dur = c["end"] - c["start"]
            w_dur = c_dur / max(1, len(c_words))
            for w_i, word in enumerate(c_words):
                all_words.append({
                    "word": word,
                    "start": round(c["start"] + w_i * w_dur, 2),
                    "end": round(c["start"] + (w_i + 1) * w_dur, 2),
                })

    full_transcript = " ".join(full_text_list)
    print(f"\nإجمالي الكلمات المفرغة: {len(all_words)} كلمة.")
    print(f"إجمالي الحروف: {len(full_transcript)} حرف.")

    # Diarization & Alignment
    print("\nجاري مطابقة المتحدثين والفواصل الزمنية...")
    diarization_result = diarize(podcast_path)
    aligned_segments = align_timestamps(all_words, diarization_result)
    print(f"تم توليد {len(aligned_segments)} مقطع موزعة على المتحدثين.")

    # Summarization with Falcon 7B
    print("\nجاري تحميل نموذج Falcon-H1-7B-Instruct لتلخيص كامل البودكاست...")
    t_sum_start = time.time()
    summary = summarize_text(full_transcript, model_key="falcon-h1-7b-instruct")
    t_sum_dur = time.time() - t_sum_start
    print(f"تم إنجاز التلخيص بنجاح في {round(t_sum_dur/60, 2)} دقيقة.")

    # Save summary markdown
    with open(FINAL_SUMMARY_FILE, "w", encoding="utf-8") as sf:
        sf.write(f"# ملخص شامل لبودكاست: {podcast_path.name}\n\n")
        sf.write(f"- المدة الكلية: {round(total_duration/60, 1)} دقيقة\n")
        sf.write(f"- عدد الكلمات: {len(all_words)}\n")
        sf.write(f"- النموذج المستخدم: Falcon-H1-7B-Instruct (IQ4_XS)\n\n---\n\n")
        sf.write(summary)

    # Save complete JSON
    full_output = {
        "audio_file": podcast_path.name,
        "duration_seconds": total_duration,
        "total_words": len(all_words),
        "transcript": full_transcript,
        "aligned_segments": aligned_segments,
        "summary": summary
    }
    with open(FINAL_RESULTS_FILE, "w", encoding="utf-8") as rf:
        json.dump(full_output, rf, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print("اكتملت المعالجة الشاملة بنجاح!")
    print(f"1. النص الكامل محفوظ في: {TRANSCRIPT_TXT_FILE}")
    print(f"2. الملخص النهائي محفوظ في: {FINAL_SUMMARY_FILE}")
    print(f"3. بيانات النتائج كاملة محفوظة في: {FINAL_RESULTS_FILE}")
    print("=" * 70)

if __name__ == "__main__":
    main()
