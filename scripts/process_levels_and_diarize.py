"""
Complete pipeline for:
1. Audio preprocessing with calibrated denoise levels (Production v3.1, tn=0).
2. Pyannote 3.1 speaker diarization on unprocessed canonical audio.
3. Cohere Transcribe Arabic ASR with word-level speaker alignment.
4. Saving all audio files, transcripts with speaker labels, and a comprehensive summary.
"""
import sys
import time
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List

# Setup path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.config import settings
from app.services.audio_processor import get_ffmpeg_binary, get_audio_duration
from app.services.preprocessing import (
    PreprocessingEngine,
    AudioProcessingRequest,
    OutputProfile,
    DenoiseMode,
    PRESET_VERSION,
)
from app.services.diarization import diarize
from app.services.alignment import align_timestamps
from app.models.asr.cohere_engine import CohereASREngine
from app.services.preprocessing.normalizer import ArabicTextNormalizer, calculate_wer_cer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("process_levels")

INPUT_AUDIO = PROJECT_ROOT / "20261011_133306.m4a"
OUTPUT_DIR = PROJECT_ROOT / "denoised_audio_levels"
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"

LEVELS = [
    {
        "id": "raw",
        "order": 1,
        "name_ar": "بدون معالجة - الصوت الخام الأصلي",
        "file_prefix": "1_raw_بدون_معالجة",
        "denoise_mode": DenoiseMode.RAW,
        "desc": "الصوت الأصلي محول إلى 16kHz mono مع فحص الطور؛ يمرر الإشارة الخام 100% دون أي فلاتر.",
    },
    {
        "id": "light",
        "order": 2,
        "name_ar": "خفيفة (Light - خفض 5dB)",
        "file_prefix": "2_light_معالجة_خفيفة",
        "denoise_mode": DenoiseMode.LIGHT,
        "desc": "تنقية طيفية هادئة (تخفيض 5dB) تحافظ بدقة فائقة على نبرة المتحدث والهمسات.",
    },
    {
        "id": "balanced",
        "order": 3,
        "name_ar": "متوازنة (Balanced - خفض 10dB)",
        "file_prefix": "3_balanced_معالجة_متوازنة",
        "denoise_mode": DenoiseMode.BALANCED,
        "desc": "المستوى القياسي المتوازن (تخفيض 10dB) بدون بوابات كتم وبدون قطع للترددات، مثالي للمكيفات والغرف العادية.",
    },
    {
        "id": "strong",
        "order": 4,
        "name_ar": "قوية (Strong - خفض 15dB)",
        "file_prefix": "4_strong_معالجة_قوية",
        "denoise_mode": DenoiseMode.STRONG,
        "desc": "تنقية طيفية مكثفة (تخفيض 15dB) للبيئات الصاخبة ومراوح التبريد؛ تخمد تشويش الفواصل تماماً دون تشويه نبرة المتحدث.",
    },
    {
        "id": "auto",
        "order": 5,
        "name_ar": "تلقائية ذكية (Auto - معايرة ديناميكية)",
        "file_prefix": "5_auto_معالجة_تلقائية",
        "denoise_mode": DenoiseMode.AUTO,
        "desc": "نظام التحليل التلقائي المعاير؛ يفحص نسبة الإشارة للضوضاء ومدة الصوت لاختيار البروفايل القياسي الأمثل.",
    },
]


def format_seconds_to_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def run_pipeline():
    if not INPUT_AUDIO.exists():
        logger.error(f"Input file not found: {INPUT_AUDIO}")
        sys.exit(1)

    ffmpeg_bin = get_ffmpeg_binary()
    audio_dur = get_audio_duration(INPUT_AUDIO)
    logger.info(f"=== بدء تنفيذ خط المعالجة الشامل لملف: {INPUT_AUDIO.name} ({audio_dur:.1f} ثانية) ===")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    engine_prep = PreprocessingEngine(ffmpeg_bin=ffmpeg_bin)
    normalizer = ArabicTextNormalizer()

    # -------------------------------------------------------------
    # 1. توليد الملفات الصوتية بمستويات التنقية المختلفة
    # -------------------------------------------------------------
    logger.info("\n--- [المرحلة 1] توليد وحفظ الملفات الصوتية بمستويات التنقية ---")
    audio_files: Dict[str, Dict[str, Path]] = {}

    for lvl in LEVELS:
        wav_path = OUTPUT_DIR / f"{lvl['file_prefix']}.wav"
        mp3_path = OUTPUT_DIR / f"{lvl['file_prefix']}.mp3"

        logger.info(f"معالجة المستوى {lvl['order']}: {lvl['name_ar']}...")
        temp_job_dir = OUTPUT_DIR / f".job_{lvl['id']}"
        req = AudioProcessingRequest(
            output_profile=OutputProfile.ASR,
            denoise_mode=lvl["denoise_mode"],
        )
        res = engine_prep.process(INPUT_AUDIO, req, temp_job_dir, job_id=f"job_{lvl['id']}")
        if not res.success or "asr" not in res.output_files:
            raise RuntimeError(f"Preprocessing failed for {lvl['id']}: {res.errors}")

        staged_asr = Path(res.output_files["asr"])
        shutil.move(str(staged_asr), str(wav_path))
        if temp_job_dir.exists():
            shutil.rmtree(temp_job_dir, ignore_errors=True)

        # Generate MP3 192kbps for convenient listening
        import subprocess
        cmd_mp3 = [
            ffmpeg_bin, "-y",
            "-i", str(wav_path),
            "-c:a", "libmp3lame", "-b:a", "192k",
            str(mp3_path)
        ]
        subprocess.run(cmd_mp3, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        audio_files[lvl["id"]] = {
            "wav": wav_path,
            "mp3": mp3_path,
            "manifest": res.manifest,
        }
        logger.info(f"  -> تم حفظ: {wav_path.name} و {mp3_path.name}")

    # -------------------------------------------------------------
    # 2. فصل المتحدثين (Speaker Diarization) عبر Pyannote
    # -------------------------------------------------------------
    logger.info("\n--- [المرحلة 2] فصل وتحديد المتحدثين عبر Pyannote 3.1 ---")
    turns_file = TRANSCRIPTS_DIR / "speaker_turns_pyannote.json"
    if turns_file.exists():
        logger.info(f"تم العثور على مقاطع المتحدثين المحسوبة مسبقاً في: {turns_file.name}. جاري تحميلها مباشرة...")
        with open(turns_file, "r", encoding="utf-8") as f:
            turns = json.load(f)
    else:
        # Invariant: Diarization runs on canonical unprocessed audio (Level 1: Raw)
        raw_wav_path = audio_files["raw"]["wav"]
        logger.info(f"تشغيل Pyannote Diarization على الصوت الخام المرجعي: {raw_wav_path.name}...")
        t0_diar = time.time()
        turns = diarize(raw_wav_path)
        diar_duration = round(time.time() - t0_diar, 2)
        logger.info(f"اكتمل فصل المتحدثين في {diar_duration} ثانية. عدد المقاطع المكتشفة: {len(turns)}")

        with open(turns_file, "w", encoding="utf-8") as f:
            json.dump(turns, f, ensure_ascii=False, indent=2)

    speakers = sorted(list({t["speaker"] for t in turns}))
    logger.info(f"المتحدثون المكتشفون ({len(speakers)} متحدثين): {speakers}")

    # -------------------------------------------------------------
    # 3. تفريغ الصوت بنموذج Cohere Transcribe ومطابقته مع المتحدثين
    # -------------------------------------------------------------
    logger.info("\n--- [المرحلة 3] تفريغ الصوت بنموذج Cohere ومطابقة الحوار ---")
    cohere_model_path = settings.MODELS_DIR / getattr(
        settings, "COHERE_ASR_FILENAME", "cohere-transcribe-arabic-07-2026-Q5_K_M.gguf"
    )
    cohere_engine = CohereASREngine(cohere_model_path)
    cohere_engine.load()

    all_transcripts_meta: Dict[str, Any] = {}
    baseline_text = ""

    try:
        for lvl in LEVELS:
            lvl_id = lvl["id"]
            wav_file = audio_files[lvl_id]["wav"]
            logger.info(f"تفريغ مستوى {lvl['order']} ({lvl['name_ar']}) عبر Cohere ASR...")

            t0_asr = time.time()
            asr_res = cohere_engine.transcribe(wav_file)
            asr_dur = round(time.time() - t0_asr, 2)
            full_text = asr_res.get("text", "").strip()

            if lvl_id == "raw":
                baseline_text = full_text

            # Align words with speaker turns
            aligned_segments = align_timestamps(asr_res.get("words", []), turns)
            logger.info(f"  -> تم تفريغ {len(full_text.split())} كلمة ومطابقة {len(aligned_segments)} دور حواري ({asr_dur} ثانية)")

            # Save readable text transcript with speaker tags
            txt_path = TRANSCRIPTS_DIR / f"transcript_{lvl['file_prefix']}_مع_المتحدثين.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"==========================================================\n")
                f.write(f"تفريغ صوتي وحواري - مستوى: {lvl['name_ar']}\n")
                f.write(f"الملف: {wav_file.name} | المدة: {audio_dur:.1f} ثانية\n")
                f.write(f"النموذج: Cohere Transcribe Arabic (GGML Q5_K_M)\n")
                f.write(f"فصل المتحدثين: Pyannote 3.1 Neural Diarization\n")
                f.write(f"==========================================================\n\n")

                for seg in aligned_segments:
                    spk_name = seg["speaker"].replace("SPEAKER_00", "المتحدث 1 (الرئيسي)").replace("SPEAKER_01", "المتحدث 2").replace("SPEAKER_02", "المتحدث 3")
                    st_str = format_seconds_to_timestamp(seg["start"])
                    en_str = format_seconds_to_timestamp(seg["end"])
                    f.write(f"[{st_str} - {en_str}] {spk_name}:\n{seg['text']}\n\n")

            # Save complete JSON data
            json_path = TRANSCRIPTS_DIR / f"transcript_{lvl['file_prefix']}.json"
            lvl_meta = {
                "level_id": lvl_id,
                "name_ar": lvl["name_ar"],
                "audio_wav": str(wav_file.name),
                "audio_mp3": str(audio_files[lvl_id]["mp3"].name),
                "asr_duration_sec": asr_dur,
                "word_count": len(full_text.split()),
                "char_count": len(full_text),
                "full_text": full_text,
                "segments": aligned_segments,
                "wer_cer": calculate_wer_cer(baseline_text, full_text, normalizer=normalizer) if baseline_text else {"raw_wer": 0.0, "norm_wer": 0.0, "norm_cer": 0.0},
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(lvl_meta, f, ensure_ascii=False, indent=2)

            all_transcripts_meta[lvl_id] = lvl_meta
            logger.info(f"  -> تم حفظ النصوص في:\n     {txt_path.name}\n     {json_path.name}")

    finally:
        cohere_engine.unload()

    # -------------------------------------------------------------
    # 4. حساب القياسات الصوتية والتقرير الشامل
    # -------------------------------------------------------------
    logger.info("\n--- [المرحلة 4] كتابة التقرير الشامل وإحصائيات المتحدثين ---")
    import wave
    import numpy as np

    def get_pause_noise_rms(wav_p: Path) -> tuple[float, float]:
        with wave.open(str(wav_p), "rb") as wf:
            data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(float)
        overall_rms = np.sqrt(np.mean(data**2))
        w = 1600 # 100ms
        p_list = []
        for i in range(0, min(len(data), 120 * 16000) - w, w):
            sub = data[i:i+w]
            sub_rms = np.sqrt(np.mean(sub**2))
            if sub_rms < 250:
                p_list.append(sub_rms)
        pause_rms = np.mean(p_list) if p_list else 1.0
        return overall_rms, pause_rms

    raw_overall, raw_pause = get_pause_noise_rms(audio_files["raw"]["wav"])

    # Calculate speaker statistics
    spk_time: Dict[str, float] = {}
    spk_words: Dict[str, int] = {}
    for seg in all_transcripts_meta["raw"]["segments"]:
        s = seg["speaker"]
        spk_time[s] = spk_time.get(s, 0.0) + (seg["end"] - seg["start"])
        spk_words[s] = spk_words.get(s, 0) + len(seg["text"].split())

    tot_spk_time = max(0.1, sum(spk_time.values()))

    # Build RESULTS_SUMMARY.md
    summary_path = OUTPUT_DIR / "RESULTS_SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# تقرير نتائج خفض الضوضاء وفصل المتحدثين والتفريغ الصوتي عبر Cohere\n\n")
        f.write(f"- **الملف المصدر:** `{INPUT_AUDIO.name}`\n")
        f.write(f"- **المدة الزمنية:** {audio_dur:.1f} ثانية (~{audio_dur/60:.2f} دقيقة)\n")
        f.write(f"- **محرك فصل المتحدثين:** Pyannote 3.1 Neural Diarization (NVIDIA GPU)\n")
        f.write(f"- **محرك التفريغ الصوتي (ASR):** Cohere Transcribe Arabic (07-2026 GGML Q5_K_M)\n")
        f.write(f"- **إصدار خط المعالجة الصوتي:** Production v{PRESET_VERSION}\n\n")

        f.write("---\n\n")
        f.write("## 1. نتائج فصل وتوزيع المتحدثين (Speaker Diarization)\n\n")
        f.write("| المتحدث | المسمى في التقرير | مدة الحديث الإجمالية | نسبة المشاركة | عدد الكلمات المنطوقة |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: |\n")
        for spk_id, dur_val in sorted(spk_time.items()):
            label = spk_id.replace("SPEAKER_00", "المتحدث 1 (الرئيسي)").replace("SPEAKER_01", "المتحدث 2").replace("SPEAKER_02", "المتحدث 3")
            pct = (dur_val / tot_spk_time) * 100
            w_cnt = spk_words.get(spk_id, 0)
            f.write(f"| **{spk_id}** | {label} | {dur_val:.1f} ثانية | {pct:.1f}% | {w_cnt} كلمة |\n")

        f.write("\n---\n\n")
        f.write("## 2. جدول مقارنة مستويات خفض الضوضاء والأثر الصوتي\n\n")
        f.write("| المستوى | ملف WAV | ملف MP3 | تشويش الفواصل | خفض الضوضاء | عدد الكلمات | تطابق التفريغ |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |\n")

        for lvl in LEVELS:
            lvl_id = lvl["id"]
            wav_name = f"`{lvl['file_prefix']}.wav`"
            mp3_name = f"`{lvl['file_prefix']}.mp3`"
            _, p_rms = get_pause_noise_rms(audio_files[lvl_id]["wav"])
            p_dbfs = 20 * np.log10(p_rms / 32768)
            red_db = -20 * np.log10(raw_pause / p_rms) if p_rms > 0 else 0.0
            red_str = f"{red_db:.1f} dB" if lvl_id != "raw" else "0.0 dB (الأصل)"
            w_cnt = all_transcripts_meta[lvl_id]["word_count"]
            wer_info = all_transcripts_meta[lvl_id]["wer_cer"]
            wer_pct = f"{wer_info['norm_wer'] * 100:.1f}%" if lvl_id != "raw" else "0.0% (المرجع)"

            f.write(f"| **{lvl['name_ar']}** | {wav_name} | {mp3_name} | {p_dbfs:.1f} dBFS | **{red_str}** | {w_cnt} | {wer_pct} |\n")

        f.write("\n---\n\n")
        f.write("## 3. عينة من الحوار المفرغ والمفصول حسب المتحدثين\n\n")
        f.write("```text\n")
        sample_segs = all_transcripts_meta["raw"]["segments"][:8]
        for seg in sample_segs:
            spk_label = seg["speaker"].replace("SPEAKER_00", "المتحدث 1").replace("SPEAKER_01", "المتحدث 2").replace("SPEAKER_02", "المتحدث 3")
            st_s = format_seconds_to_timestamp(seg["start"])
            en_s = format_seconds_to_timestamp(seg["end"])
            f.write(f"[{st_s} - {en_s}] {spk_label}: {seg['text']}\n")
        f.write("...\n```\n\n")

        f.write("---\n\n")
        f.write("## 4. الملفات المتوفرة في المجلد\n\n")
        f.write("### أ) الملفات الصوتية المجهزة للاستماع والتفريغ:\n")
        for lvl in LEVELS:
            f.write(f"- `{lvl['file_prefix']}.wav` (نسخة الاستوديو 16kHz PCM)\n")
            f.write(f"- `{lvl['file_prefix']}.mp3` (نسخة الاستماع السريع 192kbps)\n")

        f.write("\n### ب) التفريغ النصي مع المتحدثين (في مجلد `transcripts/`):\n")
        for lvl in LEVELS:
            f.write(f"- `transcript_{lvl['file_prefix']}_مع_المتحدثين.txt` (نص مقروء كامل موزع على المتحدثين مع التوقيت)\n")
            f.write(f"- `transcript_{lvl['file_prefix']}.json` (بيانات JSON منظمة تشمل المقاطع والكلمات والتوقيت الدقيق)\n")

    logger.info(f"تم حفظ التقرير الشامل بنجاح في: {summary_path}")
    logger.info("=== اكتملت كافة مراحل التنفيذ بنجاح 100% ===")


if __name__ == "__main__":
    run_pipeline()
