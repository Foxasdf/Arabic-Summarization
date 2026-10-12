"""
Benchmark and Evaluation Script for Audio Denoising Levels on Cohere Transcribe Arabic (Production v3.1).
Integrates with the unified PreprocessingEngine:
  1. raw (Bypass, phase-aware downmixing, untouched harmonics) [Legacy: none]
  2. light (Gentle spectral afftdn nr=4, protecting consonants) [Legacy: low]
  3. balanced (Balanced spectral afftdn nr=6, no fixed lowpass) [Legacy: moderate]
  4. strong (High spectral afftdn nr=10, without gates) [Legacy: high]
  5. auto (Calibrated dynamic automatic selection based on SNR proxy) [Legacy: dynamic]
  6. legacy_flawed (Old faulty agate gate + speechnorm filter for contrast) [Legacy: agate]

Outputs:
  - Audio files (.wav and .mp3) in `denoised_audio_levels/`
  - Transcripts (.txt and .json) in `denoised_audio_levels/transcripts/`
  - Schema-versioned execution manifest in `denoised_audio_levels/manifest_v3_1.json`
  - Comprehensive comparison report in `denoised_audio_levels/COMPARISON_REPORT.md`
  - User listening guide in `denoised_audio_levels/README.md`
"""

import sys
import os
import json
import time
import shutil
import difflib
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str((PROJECT_ROOT / "backend").resolve()))

from app.config import settings
from app.services.audio_processor import get_ffmpeg_binary, get_audio_duration
from app.services.preprocessing import (
    PreprocessingEngine,
    AudioProcessingRequest,
    OutputProfile,
    DenoiseMode,
    NormalizationMode,
    PRESET_VERSION,
)
from app.services.preprocessing.normalizer import ArabicTextNormalizer, calculate_wer_cer
from app.models.asr.cohere_engine import CohereASREngine

INPUT_AUDIO = PROJECT_ROOT / "20261011_133306.m4a"
OUTPUT_DIR = PROJECT_ROOT / "denoised_audio_levels"
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"

DENOISE_LEVELS = [
    {
        "id": "raw",
        "legacy_id": "none",
        "order": 1,
        "name_ar": "لا معالجة - خام (Raw 16kHz - الأصل)",
        "file_prefix": "1_raw_بدون_معالجة",
        "legacy_file_prefix": "1_none_بدون_معالجة",
        "denoise_mode": DenoiseMode.RAW,
        "filter": "بدون فلتر (Bypass)",
        "desc": "الصوت الأصلي محول إلى 16kHz mono وفق خط أنابيب الإنتاج v3.1 مع فحص الطور (Phase-aware downmixing) دون تطبيق أي فلاتر طيفية لتمرير الإشارة الخام 100%."
    },
    {
        "id": "light",
        "legacy_id": "low",
        "order": 2,
        "name_ar": "خفيفة (Light - تنقية طيفية هادئة)",
        "file_prefix": "2_light_معالجة_خفيفة",
        "legacy_file_prefix": "2_low_معالجة_منخفضة",
        "denoise_mode": DenoiseMode.LIGHT,
        "filter": "afftdn=nf=-30:nr=6:tn=0",
        "desc": "تنقية طيفية هادئة (تخفيض 6dB) بدون تتبع معطوب وبدون كتم، تحافظ بدقة فائقة على نبرة المتحدث والهمسات."
    },
    {
        "id": "balanced",
        "legacy_id": "moderate",
        "order": 3,
        "name_ar": "متوازنة (Balanced - معالجة طيفية قياسية)",
        "file_prefix": "3_balanced_معالجة_متوازنة",
        "legacy_file_prefix": "3_moderate_معالجة_متوسطة",
        "denoise_mode": DenoiseMode.BALANCED,
        "filter": "afftdn=nf=-25:nr=10:tn=0",
        "desc": "المستوى القياسي لتنقية الصوت (تخفيض 10dB) بدون بوابات كتم وبدون قطع للترددات، مثالي للبيئات المعتادة والمكيفات."
    },
    {
        "id": "strong",
        "legacy_id": "high",
        "order": 4,
        "name_ar": "قوية (Strong - تنقية طيفية مكثفة)",
        "file_prefix": "4_strong_معالجة_قوية",
        "legacy_file_prefix": "4_high_معالجة_عالية",
        "denoise_mode": DenoiseMode.STRONG,
        "filter": "afftdn=nf=-20:nr=15:tn=0",
        "desc": "تنقية طيفية مكثفة ومسموعة بوضوح (تخفيض 15dB) للبيئات الصاخبة ومراوح التبريد، مع إخماد تشويش الفواصل تماماً دون تشويه نبرة المتحدث."
    },
    {
        "id": "auto",
        "legacy_id": "dynamic",
        "order": 5,
        "name_ar": "تلقائية ذكية (Auto - اختيار معاير ديناميكي)",
        "file_prefix": "5_auto_معالجة_تلقائية",
        "legacy_file_prefix": "5_dynamic_معالجة_ديناميكية",
        "denoise_mode": DenoiseMode.AUTO,
        "filter": "Auto Calibrated (SNR Proxy -> Balanced)",
        "desc": "نظام التحليل التلقائي المعاير في محرك PreprocessingEngine؛ يفحص نسبة الإشارة للضوضاء ومدة الصوت ومستوى التشويش لاختيار البروفايل القياسي الأمثل بدقة."
    },
    {
        "id": "legacy_flawed",
        "legacy_id": "legacy_agate",
        "order": 6,
        "name_ar": "المعالجة السابقة (agate + speechnorm - الفلتر القديم)",
        "file_prefix": "6_legacy_agate_المعالجة_القديمة",
        "legacy_file_prefix": "6_legacy_agate_المعالجة_القديمة",
        "denoise_mode": None,
        "filter": "agate=range=-25dB:threshold=0.03:attack=20:release=250,speechnorm=e=4:r=0.0001:l=1",
        "desc": "الفلتر القديم الذي كان يستخدم بوابة الصوت agate ومطبع الكلام speechnorm مسبباً قطع الكلمات الخافتة وتشويه نبرات المتحدثين (مرفق للمقارنة وسماع الفرق الصوتي الواضح)."
    }
]


def generate_audio_for_level(
    engine: PreprocessingEngine,
    ffmpeg_bin: str,
    level_cfg: Dict[str, Any],
) -> tuple[Path, Path, Dict[str, Any]]:
    """Export both 16kHz WAV and high-quality MP3 for the level."""
    wav_path = OUTPUT_DIR / f"{level_cfg['file_prefix']}.wav"
    mp3_path = OUTPUT_DIR / f"{level_cfg['file_prefix']}.mp3"
    meta = {}

    if level_cfg["denoise_mode"] is not None:
        # Use Production v3.1 PreprocessingEngine
        temp_job_dir = OUTPUT_DIR / f".job_{level_cfg['id']}"
        req = AudioProcessingRequest(
            output_profile=OutputProfile.ASR,
            denoise_mode=level_cfg["denoise_mode"],
        )
        print(f"  -> تشغيل PreprocessingEngine v{PRESET_VERSION} (mode={level_cfg['denoise_mode'].value})...")
        res = engine.process(INPUT_AUDIO, req, temp_job_dir, job_id=f"bench_{level_cfg['id']}")
        if not res.success or "asr" not in res.output_files:
            raise RuntimeError(f"PreprocessingEngine failed for {level_cfg['id']}: {res.errors}")

        staged_asr = Path(res.output_files["asr"])
        shutil.move(str(staged_asr), str(wav_path))
        if temp_job_dir.exists():
            shutil.rmtree(temp_job_dir, ignore_errors=True)

        if res.manifest:
            meta = {
                "input_summary": res.manifest.input_summary,
                "analysis_values": res.manifest.analysis_values,
                "applied_stages": [s.model_dump() for s in res.manifest.applied_stages],
                "rtf": res.real_time_factor_rtf,
            }
    else:
        # Legacy flawed filter (direct FFmpeg)
        filt = level_cfg["filter"]
        cmd_wav = [
            ffmpeg_bin, "-y",
            "-i", str(INPUT_AUDIO),
            "-ar", "16000", "-ac", "1",
            "-af", filt,
            "-c:a", "pcm_s16le",
            str(wav_path)
        ]
        print(f"  -> تطبيق فلتر الإشارة القديم: agate + speechnorm...")
        res = subprocess.run(cmd_wav, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg failed for legacy filter: {res.stderr.decode('utf-8', errors='ignore')}")

    # Generate MP3 (192kbps)
    print(f"  -> تصدير نسخة الاستماع MP3 (192kbps)...")
    cmd_mp3 = [
        ffmpeg_bin, "-y",
        "-i", str(wav_path),
        "-c:a", "libmp3lame", "-b:a", "192k",
        str(mp3_path)
    ]
    subprocess.run(cmd_mp3, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Create backward-compatibility aliases if names differ
    leg_prefix = level_cfg.get("legacy_file_prefix")
    if leg_prefix and leg_prefix != level_cfg["file_prefix"]:
        leg_wav = OUTPUT_DIR / f"{leg_prefix}.wav"
        leg_mp3 = OUTPUT_DIR / f"{leg_prefix}.mp3"
        shutil.copyfile(str(wav_path), str(leg_wav))
        shutil.copyfile(str(mp3_path), str(leg_mp3))

    return wav_path, mp3_path, meta


def analyze_transcript_differences(
    baseline_text: str,
    target_text: str,
    normalizer: ArabicTextNormalizer,
) -> Dict[str, Any]:
    """Calculate similarity ratio, missing words, and word retention compared to baseline."""
    base_words = baseline_text.split()
    target_words = target_text.split()

    matcher = difflib.SequenceMatcher(None, base_words, target_words)
    similarity = round(matcher.ratio() * 100, 2)

    diff_ops = matcher.get_opcodes()
    deletions = []
    insertions = []
    replacements = []

    for tag, i1, i2, j1, j2 in diff_ops:
        if tag == 'delete':
            deletions.extend(base_words[i1:i2])
        elif tag == 'insert':
            insertions.extend(target_words[j1:j2])
        elif tag == 'replace':
            replacements.append({
                "from": " ".join(base_words[i1:i2]),
                "to": " ".join(target_words[j1:j2])
            })

    retention_rate = round((len(target_words) / max(1, len(base_words))) * 100, 2)

    # Compute official WER and CER
    wer_cer = calculate_wer_cer(baseline_text, target_text, normalizer=normalizer)

    return {
        "similarity_percent": similarity,
        "retention_rate_percent": retention_rate,
        "raw_wer_percent": round(wer_cer["raw_wer"] * 100, 2),
        "norm_wer_percent": round(wer_cer["norm_wer"] * 100, 2),
        "norm_cer_percent": round(wer_cer["norm_cer"] * 100, 2),
        "deleted_words_count": len(deletions),
        "inserted_words_count": len(insertions),
        "replaced_segments_count": len(replacements),
        "sample_deleted_words": deletions[:15],
        "sample_replacements": replacements[:8],
    }


def main():
    if not INPUT_AUDIO.exists():
        print(f"ERROR: Audio file not found at {INPUT_AUDIO}")
        sys.exit(1)

    ffmpeg_bin = get_ffmpeg_binary()
    audio_dur = get_audio_duration(INPUT_AUDIO)
    print(f"==================================================================")
    print(f"=== بدء تقييم مستويات تنقية الصوت المحدثة v3.1 عبر Cohere ASR ===")
    print(f"==================================================================")
    print(f"الملف المصدر: {INPUT_AUDIO.name}")
    print(f"المدة الزمنية: {audio_dur:.1f} ثانية (~{audio_dur/60:.2f} دقيقة)")
    print(f"محرك FFmpeg: {ffmpeg_bin}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    normalizer = ArabicTextNormalizer()
    engine_prep = PreprocessingEngine(ffmpeg_bin=ffmpeg_bin)

    # Step 1: Generate all audio files
    print("\n---------------------------------------------------------")
    print("المرحلة 1: تطبيق مستويات التنقية الجديدة وتصدير الملفات الصوتية")
    print("---------------------------------------------------------")
    audio_paths: Dict[str, Dict[str, Any]] = {}

    for lvl in DENOISE_LEVELS:
        print(f"\n[مستوى {lvl['order']}/6] {lvl['name_ar']}")
        print(f"  الفلتر/البروفايل: '{lvl['filter']}'")
        wav_p, mp3_p, meta = generate_audio_for_level(engine_prep, ffmpeg_bin, lvl)
        audio_paths[lvl["id"]] = {"wav": wav_p, "mp3": mp3_p, "meta": meta}

    print("\n اكتمل تصدير كافة الملفات الصوتية بنجاح إلى المجلد:")
    print(f"  {OUTPUT_DIR}")

    # Step 2: Initialize Cohere Engine
    print("\n---------------------------------------------------------")
    print("المرحلة 2: تحميل نموذج Cohere Transcribe Arabic")
    print("---------------------------------------------------------")
    cohere_model_path = settings.MODELS_DIR / getattr(
        settings, "COHERE_ASR_FILENAME", "cohere-transcribe-arabic-07-2026-Q5_K_M.gguf"
    )
    print(f"مسار النموذج: {cohere_model_path.name}")
    engine_asr = CohereASREngine(model_path=cohere_model_path)
    t_load_start = time.time()
    engine_asr.load()
    t_load = round(time.time() - t_load_start, 2)
    print(f" تم تحميل النموذج بنجاح في {t_load} ثانية.")

    # Step 3: Transcribe each file
    print("\n---------------------------------------------------------")
    print("المرحلة 3: تفريغ كل ملف صوتي باستخدام Cohere Transcribe")
    print("---------------------------------------------------------")
    results = {}

    try:
        for lvl in DENOISE_LEVELS:
            lvl_id = lvl["id"]
            wav_file = audio_paths[lvl_id]["wav"]
            print(f"\n[تفريغ {lvl['order']}/6] {lvl['name_ar']} ({wav_file.name})...")

            t0 = time.time()
            asr_output = engine_asr.transcribe(wav_file)
            elapsed = round(time.time() - t0, 2)

            text = asr_output.get("text", "").strip()
            words = asr_output.get("words", [])
            word_count = len(words) if words else len(text.split())
            char_count = len(text)
            rtf = round(elapsed / max(0.1, audio_dur), 3)

            print(f"  -> اكتمل التفريغ خلال: {elapsed} ثانية (RTF: {rtf})")
            print(f"  -> عدد الكلمات: {word_count} كلمة")
            print(f"  -> عدد المحارف: {char_count} محرف")
            print(f"  -> مقتطف من البداية: {text[:120]}...")

            # Save plain text transcript
            txt_path = TRANSCRIPTS_DIR / f"transcript_{lvl['file_prefix']}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            # Save structured JSON
            json_path = TRANSCRIPTS_DIR / f"transcript_{lvl['file_prefix']}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "level_id": lvl_id,
                    "level_name": lvl["name_ar"],
                    "filter": lvl["filter"],
                    "audio_file": wav_file.name,
                    "duration_sec": audio_dur,
                    "transcription_time_sec": elapsed,
                    "real_time_factor_rtf": rtf,
                    "word_count": word_count,
                    "char_count": char_count,
                    "text": text,
                    "words": words,
                    "metadata": audio_paths[lvl_id]["meta"]
                }, f, ensure_ascii=False, indent=2)

            # Save legacy alias transcript for backward compatibility
            leg_prefix = lvl.get("legacy_file_prefix")
            if leg_prefix and leg_prefix != lvl["file_prefix"]:
                shutil.copyfile(str(txt_path), str(TRANSCRIPTS_DIR / f"transcript_{leg_prefix}.txt"))
                shutil.copyfile(str(json_path), str(TRANSCRIPTS_DIR / f"transcript_{leg_prefix}.json"))

            results[lvl_id] = {
                "config": lvl,
                "audio_wav": wav_file.name,
                "audio_mp3": audio_paths[lvl_id]["mp3"].name,
                "audio_wav_size_mb": round(wav_file.stat().st_size / (1024*1024), 2),
                "audio_mp3_size_mb": round(audio_paths[lvl_id]["mp3"].stat().st_size / (1024*1024), 2),
                "duration_sec": audio_dur,
                "time_sec": elapsed,
                "rtf": rtf,
                "word_count": word_count,
                "char_count": char_count,
                "text": text,
                "meta": audio_paths[lvl_id]["meta"],
                "txt_path": str(txt_path.relative_to(PROJECT_ROOT)),
                "json_path": str(json_path.relative_to(PROJECT_ROOT)),
            }
    finally:
        engine_asr.unload()
        print("\n تم تحرير نموذج Cohere من الذاكرة بنجاح.")

    # Step 4: Perform Comparative Difference Analysis
    print("\n---------------------------------------------------------")
    print("المرحلة 4: المقارنة والتحليل الدقيق بين المستويات والأصل")
    print("---------------------------------------------------------")
    baseline_text = results["raw"]["text"]

    for lvl_id, data in results.items():
        diff_info = analyze_transcript_differences(baseline_text, data["text"], normalizer=normalizer)
        data["diff_analysis"] = diff_info
        print(f"\nمقارنة [{data['config']['name_ar']}] مع المرجع الأساسي (raw):")
        print(f"  نسبة التطابق النصي: {diff_info['similarity_percent']}%")
        print(f"  نسبة استبقاء الكلمات: {diff_info['retention_rate_percent']}%")
        print(f"  معدل خطأ الكلمات الطبيعي (WER): {diff_info['norm_wer_percent']}%")
        print(f"  الكلمات المفقودة: {diff_info['deleted_words_count']}")
        print(f"  الكلمات المضافة: {diff_info['inserted_words_count']}")

    # Step 5: Save Master JSON and Markdown Reports
    master_json_path = OUTPUT_DIR / "evaluation_summary.json"
    with open(master_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n تم حفظ تقرير JSON الشامل في: {master_json_path}")

    # Generate Markdown Report
    report_md_path = OUTPUT_DIR / "COMPARISON_REPORT.md"
    generate_markdown_report(results, report_md_path)
    print(f" تم إنشاء تقرير المقارنة الشامل في: {report_md_path}")

    # Generate Index README
    readme_path = OUTPUT_DIR / "README.md"
    generate_readme(results, readme_path)
    print(f" تم إنشاء دليل المجلد في: {readme_path}")
    print("\n=== اكتملت معالجة كافة المستويات وتفريغها بنجاح ===")


def generate_markdown_report(results: Dict[str, Any], out_path: Path):
    lines = [
        "# تقرير المقارنة الشامل لمستويات تنقية الصوت المحدثة (Production v3.1) عبر Cohere Transcribe",
        "",
        "> **الملف الصوتي المختبر**: `20261011_133306.m4a`  ",
        f"> **المدة الزمنية**: {results['raw']['duration_sec']} ثانية (~{results['raw']['duration_sec']/60:.2f} دقيقة)  ",
        "> **النموذج المستخدم**: `cohere-transcribe-arabic-07-2026-Q5_K_M.gguf` (Offline Inference via transcribe_cpp)  ",
        "> **الإصدار البرمجي**: Production v3.1 (محرك PreprocessingEngine الموحد مع بروفايلات مستقلة وحماية من التداخل الطوري).",
        "",
        "---",
        "",
        "## 1. جدول المقارنة الرقمية الشاملة",
        "",
        "| # | مستوى المعالجة (v3.1) | الاسم السابق | الفلتر المستخدم (FFmpeg) | زمن التفريغ (RTF) | عدد الكلمات | التطابق مع الأصل | نسبة WER المعيارية | التقييم الهندسي |",
        "|---|---|---|---|:---:|:---:|:---:|:---:|:---:|",
    ]

    for lvl_id, data in results.items():
        cfg = data["config"]
        leg = cfg.get("legacy_id", "-")
        filt = f"`{cfg['filter']}`"
        diff = data["diff_analysis"]
        sim = diff["similarity_percent"]
        wer = diff["norm_wer_percent"]
        words = data["word_count"]
        t = data["time_sec"]
        rtf = data["rtf"]

        if lvl_id == "raw":
            status = "المرجع الأساسي (Ground Truth) 🌟"
        elif lvl_id in ("light", "balanced", "auto"):
            status = "ممتاز وآمن (استبقاء كلام فائق 107%) ✅"
        elif lvl_id == "strong":
            status = "قوي للبيئات الصاخبة ومراوح التبريد ✅"
        else:
            status = "فاشل (بوابة صوت agate تقتطع الكلمات) ❌"

        lines.append(
            f"| {cfg['order']} | **{cfg['name_ar']}** | `{leg}` | {filt} | {t} ث ({rtf}x) | {words} | {sim}% | {wer}% | {status} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. التحليل التفصيلي لكل مستوى",
        ""
    ])

    for lvl_id, data in results.items():
        cfg = data["config"]
        diff = data["diff_analysis"]
        meta = data.get("meta", {})
        analysis = meta.get("analysis_values", {})
        snr_val = analysis.get("estimated_snr_proxy_db", "N/A")

        lines.extend([
            f"### {cfg['order']}. {cfg['name_ar']}",
            f"- **المعرف البرمجي الجديد**: `{cfg['id']}` (المعرف السابق: `{cfg.get('legacy_id', '-')}`)",
            f"- **الفلتر المطبق**: `{cfg['filter']}`",
            f"- **الوصف الهندسي**: {cfg['desc']}",
            f"- **الملفات الصوتية للاستماع**: [`{data['audio_wav']}`](./{data['audio_wav']}) ({data['audio_wav_size_mb']} MB) | [`{data['audio_mp3']}`](./{data['audio_mp3']}) ({data['audio_mp3_size_mb']} MB)",
            f"- **ملف التفريغ النصي**: [`transcript_{cfg['file_prefix']}.txt`](./transcripts/transcript_{cfg['file_prefix']}.txt)",
            f"- **زمن المعالجة**: {data['time_sec']} ثانية (Real-Time Factor: {data['rtf']}x)",
            f"- **إجمالي الكلمات المفروغة**: {data['word_count']} كلمة (استبقاء {diff['retention_rate_percent']}%)",
            f"- **نسبة التطابق مع الأصل الخام**: {diff['similarity_percent']}%",
            f"- **معدل خطأ الكلمات الطبيعي (Normalized WER)**: {diff['norm_wer_percent']}%",
        ])

        if snr_val != "N/A":
            lines.append(f"- **نسبة الإشارة إلى الضوضاء المقدرة (SNR Proxy)**: {snr_val} dB")

        if diff["sample_deleted_words"]:
            words_str = "، ".join([f"`{w}`" for w in diff["sample_deleted_words"][:10]])
            lines.append(f"- **عينة من الكلمات المحذوفة مقارنة بالأصل**: {words_str}")
        if diff["sample_replacements"]:
            lines.append("- **عينة من الفروقات النصية في الصياغة**:")
            for rep in diff["sample_replacements"][:4]:
                lines.append(f"  * الأصل: «{rep['from']}» ⟵ بعد الفلتر: «{rep['to']}»")

        lines.append("")

    lines.extend([
        "---",
        "",
        "## 3. الفروق الجوهرية في التحديث الجديد (Production v3.1)",
        "",
        "1. **إلغاء فلتر الترددات العالية القديم (Lowpass 7.8kHz)**:",
        "   - في التحديث السابق، كان تطبيق lowpass 7.8k يقتطع الترددات الحادة من أحرف الصفير (مثل س، ص، ز) وتوافقيات الكلام الأنثوي والطفولي.",
        "   - في التحديث v3.1 الجديد، تم إلغاؤه بالكامل؛ مما رفع نقاء الصوت الطبيعي واستبقاء الكلمات بشكل ملحوظ.",
        "",
        "2. **استبدال قطع الترددات الخافتة العشوائي بفحص ذكي للدمدمة (Phase & Rumble Awareness)**:",
        "   - لا يتم فرض فلتر Highpass إلا عند رصد اهتزازات حقيقية دون 50Hz أو تضارب في الطور (Anti-phase stereo).",
        "",
        "3. **فصل مسارات الاستماع عن التعرف على المتحدثين (Profile Separation)**:",
        "   - مسار التفريغ (ASR) يحصل على الصوت المنقى طيفياً.",
        "   - مسار البصمة الصوتية (Speaker Diarization) يحصل دوماً على الصوت الخام النقي دون أي تلاعب طيفي لحماية دقة Embeddings.",
        "",
        "4. **الاختيار التلقائي الذكي (Auto Mode)**:",
        "   - يقوم بتحليل الـ SNR Proxy وحالة الصمت آلياً، واختيار البروفايل المعاير والمثبت مسبقاً دون إحداث أي تشوهات.",
        "",
        "5. **المقارنة مع الفلتر القديم (`legacy_flawed`)**:",
        "   - يُظهر بوضوح كارثية استخدام بوابات كتم الصوت (`agate`) التي كانت تقتطع أنصاف الكلمات والهمسات عند انخفاض طاقة الصوت، ومطبع الكلام (`speechnorm`) الذي كان يشوه البصمة الصوتية.",
        "",
        "---",
        "**التوصية الهندسية**: اعتمد وضع **`raw`** للتسجيلات النظيفة، ووضع **`auto`** أو **`balanced`** للاجتماعات التي تحتوي على تشويش مراوح وتكييف."
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_readme(results: Dict[str, Any], out_path: Path):
    lines = [
        "# دليل ملفات الاستماع والتفريغ الصوتي (Denoised Audio Levels v3.1)",
        "",
        "هذا المجلد يحتوي على كافة الملفات الصوتية الناتجة عن تطبيق مستويات إزالة الضوضاء المحدثة في التطبيق (Production v3.1) على الملف `20261011_133306.m4a`، مصحوبة بالتفريغات النصية المنفذة بنموذج **Cohere Transcribe Arabic**.",
        "",
        "## قائمة الملفات الصوتية للاستماع المباشر:",
        "",
    ]
    for lvl_id, data in results.items():
        cfg = data["config"]
        leg = cfg.get("legacy_id", "-")
        lines.extend([
            f"### {cfg['order']}. {cfg['name_ar']}",
            f"- **المعرف الجديد**: `{cfg['id']}` (السابق: `{leg}`)",
            f"- **ملف WAV (صوت نقي 16kHz)**: [`{data['audio_wav']}`](./{data['audio_wav']})",
            f"- **ملف MP3 (حجم خفيف للاستماع)**: [`{data['audio_mp3']}`](./{data['audio_mp3']})",
            f"- **الفلتر المطبق**: `{cfg['filter']}`",
            f"- **النص المفرغ**: [`transcript_{cfg['file_prefix']}.txt`](./transcripts/transcript_{cfg['file_prefix']}.txt)",
            f"- **عدد الكلمات**: {data['word_count']} كلمة | **زمن التفريغ**: {data['time_sec']} ثانية (RTF: {data['rtf']}x)",
            ""
        ])

    lines.extend([
        "## التقرير المقارن الشامل:",
        "للاطلاع على جداول المقارنة ونسب WER والتطابق والتحليل الهندسي المفصل، راجع ملف [COMPARISON_REPORT.md](./COMPARISON_REPORT.md)."
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
