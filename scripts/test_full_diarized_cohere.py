import asyncio
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT / "scripts"))

import os
for _base in [
    BACKEND / ".venv" / "Lib" / "site-packages" / "torch" / "lib",
    Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib",
]:
    if _base.exists():
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(str(_base))
            except Exception:
                pass
        if str(_base) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = str(_base) + ";" + os.environ.get("PATH", "")

from app.config import settings
from app.services.audio_processor import process_dual_profile_audio
from app.services.diarization import diarize
from app.services.transcription import transcribe_with_speakers
from app.services.preprocessing.normalizer import ArabicTextNormalizer, calculate_wer_cer
from der_scorer import score_der

AUDIO_SOURCE = ROOT / "20261011_133306.m4a"
GOLD_REF_PATH = ROOT / "transcription_20261011_133306.json"
OUT_JSON = ROOT / "eval_results" / "cohere_diarized_eval_133306.json"
OUT_MD = ROOT / "eval_results" / "cohere_diarized_eval_133306.md"

def ts(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

async def main():
    print(f"=== Running Full Pipeline on {AUDIO_SOURCE.name} with Cohere + Pyannote ===")
    t0 = time.time()
    
    # 1. Preprocessing dual profiles
    tmp_dir = ROOT / "scratch" / "bench_run"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    asr_wav = tmp_dir / "asr.16k.wav"
    spk_wav = tmp_dir / "speaker.16k.wav"
    
    print("\n[Step 1] Dual-profile audio preprocessing...")
    ok, err = await process_dual_profile_audio(AUDIO_SOURCE, asr_wav, spk_wav, denoise_mode="raw")
    if not ok:
        print(f"Preprocessing failed: {err}")
        return
    print(f"Preprocessing completed in {time.time() - t0:.1f}s")
    
    # 2. Pyannote Diarization
    t_diar_start = time.time()
    print("\n[Step 2] Pyannote 3.1 Speaker Diarization...")
    def diar_progress(pct, msg):
        print(f"  Diarization progress: {pct}% - {msg}", flush=True)
    turns = diarize(spk_wav, on_progress=diar_progress)
    n_speakers = len(set(t["speaker"] for t in turns))
    print(f"Diarization finished in {time.time() - t_diar_start:.1f}s with {len(turns)} turns and {n_speakers} speakers.")
    
    # 3. Cohere Turn Transcription
    t_asr_start = time.time()
    print("\n[Step 3] Cohere Turn-Based Transcription...")
    def asr_progress(pct, msg):
        print(f"  ASR progress: {pct}% - {msg}", flush=True)
    segments = transcribe_with_speakers(
        audio_path=asr_wav,
        turns=turns,
        on_progress=asr_progress,
        model_identifier="cohere-transcribe-arabic-07-2026-Q5_K_M.gguf",
    )
    print(f"ASR completed in {time.time() - t_asr_start:.1f}s with {len(segments)} transcribed segments.")
    
    total_time = time.time() - t0
    print(f"\nPipeline Total Time: {total_time:.1f}s")
    
    # 4. Evaluation against Gold Reference
    print("\n[Step 4] Scoring against ground truth...")
    gold = json.loads(GOLD_REF_PATH.read_text(encoding="utf-8"))
    gold_segments = gold["segments"]
    
    der_result = score_der(gold_segments, segments)
    print(f"\nDER: {der_result['der']:.4f} | JER: {der_result['jer']:.4f}")
    print(f"Confusion: {der_result['conf_s']}s | Missed Speech: {der_result['miss_s']}s | False Alarm: {der_result['fa_s']}s")
    print(f"Speakers Detected: {der_result['n_hyp_spk']} (Ref: {der_result['n_ref_spk']})")
    
    # Text metrics
    normalizer = ArabicTextNormalizer()
    ref_full_text = " ".join(s["text"] for s in gold_segments)
    hyp_full_text = " ".join(s["text"] for s in segments)
    text_metrics = calculate_wer_cer(ref_full_text, hyp_full_text, normalizer)
    print(f"\nWER (Normalized): {text_metrics['norm_wer']:.4f} | CER (Normalized): {text_metrics['norm_cer']:.4f}")
    print(f"Ref Words: {text_metrics['ref_word_count']} | Hyp Words: {text_metrics['hyp_word_count']}")
    
    # Save JSON results
    eval_payload = {
        "audio_file": AUDIO_SOURCE.name,
        "asr_engine": "Cohere Transcribe Arabic Q5_K_M",
        "asr_mode": getattr(settings, "ASR_MODE", "continuous_vad"),
        "diarization_engine": "Pyannote 3.1",
        "pipeline_duration_seconds": round(total_time, 1),
        "der_metrics": der_result,
        "text_metrics": text_metrics,
        "segments_count": len(segments),
        "segments": segments,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(eval_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote evaluation payload to {OUT_JSON}")
    
    # Generate human-readable Markdown comparison
    md_lines = [
        f"# Cohere + Pyannote Evaluation Results — {AUDIO_SOURCE.name}",
        "",
        f"- **ASR Engine**: Cohere Transcribe Arabic Q5_K_M (`per_turn` diarization-driven)",
        f"- **Diarization Engine**: Pyannote 3.1 Neural Diarization (100% Offline)",
        f"- **Processing Time**: {total_time:.1f}s for 397.8s audio (RTF: {total_time/397.8:.2f}x)",
        f"- **Speakers Found**: {der_result['n_hyp_spk']} (Ground Truth: {der_result['n_ref_spk']})",
        f"- **DER**: {der_result['der']:.4f} | **JER**: {der_result['jer']:.4f}",
        f"- **Missed Speech**: {der_result['miss_s']}s | **Confusion**: {der_result['conf_s']}s",
        f"- **WER (Normalized)**: {text_metrics['norm_wer']:.4f} | **CER (Normalized)**: {text_metrics['norm_cer']:.4f}",
        "",
        "---",
        "",
        "## Transcribed Segments by Speaker",
        "",
    ]
    for seg in segments:
        st_str = ts(seg["start"])
        en_str = ts(seg["end"])
        spk = seg["speaker"]
        txt = seg["text"]
        md_lines.append(f"**[{st_str} - {en_str}] {spk}:**")
        md_lines.append(f"{txt}\n")
        
    OUT_MD.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote readable transcript to {OUT_MD}")

if __name__ == "__main__":
    asyncio.run(main())
