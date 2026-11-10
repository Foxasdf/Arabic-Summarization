#!/usr/bin/env python3
"""
End-to-End Diarization + Cohere Arabic ASR Benchmark on `20261011_133306.m4a`.
Compares:
  1. `pyannote` (Pyannote 3.1 Speaker Diarization)
  2. `nemotron_3` (`nvidia/Nemotron-3-Diarization` via Hugging Face `transformers`)
with Cohere Transcribe Arabic (`cohere-transcribe-arabic-07-2026-Q5_K_M.gguf`).

Follows strict memory hygiene:
- Runs `pyannote` diarization first, then explicitly unloads Pyannote & frees GPU/CPU memory.
- Runs `nemotron_3` diarization second, then explicitly unloads Nemotron-3 & frees GPU/CPU memory.
- Loads Cohere Arabic ASR once, transcribes both turn sets using identical turn cleaning
  and +-200ms boundary padding, then unloads Cohere ASR.
- Compares both pipelines against `transcription_20261011_133306.json` framed as
  "vs. previous baseline" (not human ground truth).
"""
import gc
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = WORKSPACE_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings
from app.models.asr import get_asr_engine
from app.models.asr.base import BaseASREngine
from app.services.audio_processor import ensure_16k_mono_wav, load_audio_16k_mono
from app.services.diarization import (
    _normalize_turns,
    diarization_backend_name,
    diarize,
    filter_phantom_speakers,
    unload_diarization_models,
)

AUDIO_M4A = WORKSPACE_ROOT / "20261011_133306.m4a"
BASELINE_JSON = WORKSPACE_ROOT / "transcription_20261011_133306.json"
COMPARISON_DIR = WORKSPACE_ROOT / "diarization_comparison"
COHERE_MODEL_NAME = "cohere-transcribe-arabic-07-2026-Q5_K_M.gguf"


def _free_memory() -> None:
    """Force garbage collection and empty PyTorch CUDA/MPS caches."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    except Exception:
        pass


def _turn_stats(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not turns:
        return {
            "count": 0,
            "unique_speakers": 0,
            "speakers": [],
            "avg_duration_sec": 0.0,
            "min_duration_sec": 0.0,
            "max_duration_sec": 0.0,
            "sub_0_5s_count": 0,
            "total_speech_sec": 0.0,
        }
    durs = [max(0.0, float(t["end"]) - float(t["start"])) for t in turns]
    speakers = sorted({str(t["speaker"]) for t in turns})
    return {
        "count": len(turns),
        "unique_speakers": len(speakers),
        "speakers": speakers,
        "avg_duration_sec": round(sum(durs) / len(durs), 2),
        "min_duration_sec": round(min(durs), 2),
        "max_duration_sec": round(max(durs), 2),
        "sub_0_5s_count": sum(1 for d in durs if d < 0.5),
        "total_speech_sec": round(sum(durs), 2),
    }


def _normalize_arabic_words(text: str) -> List[str]:
    text = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return [w for w in text.split() if w]


def _compare_with_previous_baseline(
    segments: List[Dict[str, Any]],
    baseline_data: Dict[str, Any],
) -> Dict[str, Any]:
    base_segs = baseline_data.get("segments") or []
    base_text = baseline_data.get("full_text") or " ".join(s.get("text", "") for s in base_segs)
    cand_text = " ".join(s.get("text", "") for s in segments)

    base_words = _normalize_arabic_words(base_text)
    cand_words = _normalize_arabic_words(cand_text)
    base_vocab = set(base_words)
    cand_vocab = set(cand_words)
    jaccard = (
        round(len(base_vocab & cand_vocab) / max(1, len(base_vocab | cand_vocab)) * 100.0, 1)
        if (base_vocab or cand_vocab)
        else 0.0
    )

    base_durs = [max(0.0, float(s.get("end", 0)) - float(s.get("start", 0))) for s in base_segs]
    cand_durs = [max(0.0, float(s.get("end", 0)) - float(s.get("start", 0))) for s in segments]

    base_leaked_tags = len(re.findall(r"(@@@\S+|#{2,}\S+)", base_text))
    cand_leaked_tags = len(re.findall(r"(@@@\S+|#{2,}\S+)", cand_text))

    return {
        "previous_baseline_turns": len(base_segs),
        "candidate_turns": len(segments),
        "previous_baseline_avg_turn_sec": round(sum(base_durs) / max(1, len(base_durs)), 2),
        "candidate_avg_turn_sec": round(sum(cand_durs) / max(1, len(cand_durs)), 2),
        "previous_baseline_words": len(base_words),
        "candidate_words": len(cand_words),
        "vocabulary_jaccard_pct": jaccard,
        "previous_baseline_leaked_special_tags": base_leaked_tags,
        "candidate_leaked_special_tags": cand_leaked_tags,
    }


def _save_backend_outputs(
    backend_key: str,
    backend_label: str,
    audio_name: str,
    diar_time_sec: float,
    asr_time_sec: float,
    raw_stats: Dict[str, Any],
    merged_stats: Dict[str, Any],
    window_stats: Dict[str, Any],
    segments: List[Dict[str, Any]],
    vs_baseline: Dict[str, Any],
) -> Tuple[Path, Path]:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    json_path = COMPARISON_DIR / f"{backend_key}_transcript.json"
    txt_path = COMPARISON_DIR / f"{backend_key}_transcript.txt"

    full_text = " ".join(s.get("text", "").strip() for s in segments if s.get("text", "").strip())
    payload = {
        "audio_file": audio_name,
        "diarization_backend": backend_key,
        "diarization_engine": backend_label,
        "asr_engine": f"Cohere Transcribe Arabic ({COHERE_MODEL_NAME})",
        "diarization_inference_time_sec": round(diar_time_sec, 2),
        "asr_inference_time_sec": round(asr_time_sec, 2),
        "total_pipeline_runtime_sec": round(diar_time_sec + asr_time_sec, 2),
        "raw_turns_before_merging": raw_stats,
        "cleaned_turns_after_merging": merged_stats,
        "asr_windows": window_stats,
        "transcribed_segments_count": len(segments),
        "speakers_detected": sorted({s["speaker"] for s in segments}),
        "vs_previous_baseline": vs_baseline,
        "full_text": full_text,
        "segments": segments,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Audio File: {audio_name}\n")
        f.write(f"Diarization Backend: {backend_label} ({backend_key})\n")
        f.write(f"ASR Engine: Cohere Transcribe Arabic ({COHERE_MODEL_NAME})\n")
        f.write(f"Diarization Time: {diar_time_sec:.2f}s | ASR Time: {asr_time_sec:.2f}s | Total: {diar_time_sec + asr_time_sec:.2f}s\n")
        f.write(f"Unique Speakers Detected: {raw_stats['unique_speakers']} ({', '.join(raw_stats['speakers'])})\n")
        f.write(
            f"Turns (Before Merging): {raw_stats['count']} (avg {raw_stats['avg_duration_sec']:.2f}s) -> "
            f"Turns (After Merging): {merged_stats['count']} (avg {merged_stats['avg_duration_sec']:.2f}s) -> "
            f"Final Utterances: {len(segments)}\n"
        )
        f.write("=" * 80 + "\n\n")
        for idx, seg in enumerate(segments, 1):
            f.write(f"[{seg['start']:06.2f}s - {seg['end']:06.2f}s] {seg['speaker']}: {seg['text']}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("FULL TEXT:\n")
        f.write(full_text + "\n")

    return json_path, txt_path


def main():
    if not AUDIO_M4A.exists():
        raise FileNotFoundError(f"Benchmark audio file not found: {AUDIO_M4A}")

    # Clean up any leftover whisper-raw directory if present to preserve disk space
    whisper_raw = WORKSPACE_ROOT / "models" / "whisper-raw"
    if whisper_raw.exists():
        shutil.rmtree(whisper_raw, ignore_errors=True)

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    wav_16k_path = COMPARISON_DIR / "20261011_133306_16k_mono.wav"

    print("=" * 80)
    print("STEP 1: Decoding & Verifying 20261011_133306.m4a -> 16 kHz Mono float32 WAV")
    print("=" * 80)
    t_prep0 = time.perf_counter()
    ensure_16k_mono_wav(AUDIO_M4A, target_wav_path=wav_16k_path)
    sr = 16000
    waveform = load_audio_16k_mono(wav_16k_path, target_sr=sr)
    prep_sec = time.perf_counter() - t_prep0
    audio_dur = len(waveform) / float(sr)
    print(
        f"[+] Audio verified: sr={sr} Hz, channels=1 (mono), dtype={waveform.dtype}, "
        f"duration={audio_dur:.2f}s, decode_time={prep_sec:.2f}s"
    )
    del waveform
    _free_memory()

    # Run both diarization backends FIRST and explicitly unload each before loading Cohere ASR
    diar_results: Dict[str, Dict[str, Any]] = {}
    force_rerun = "--force" in sys.argv
    for backend_key in ("pyannote", "nemotron_3"):
        label = diarization_backend_name(backend_key)
        cache_file = COMPARISON_DIR / f".diar_cache_{backend_key}.json"
        print("\n" + "=" * 80)
        print(f"STEP 2 ({backend_key.upper()}): Running {label} on {AUDIO_M4A.name}")
        print("=" * 80)

        if cache_file.exists() and not force_rerun:
            with open(cache_file, "r", encoding="utf-8") as cf:
                cached = json.load(cf)
            raw_turns = cached["raw_turns"]
            diar_elapsed = float(cached["diar_time_sec"])
            # Re-apply the CURRENT post-processing to the cached raw diarizer output
            post_raw = filter_phantom_speakers(raw_turns) if backend_key == "nemotron_3" else raw_turns
            merged_turns = _normalize_turns(post_raw)
            print(f"[*] Loaded cached raw diarization from {cache_file.name} (inference time: {diar_elapsed:.2f}s); re-applied current turn post-processing")
        else:
            def _diar_cb(pct: int, msg: str):
                print(f"    [{pct:3d}%] {msg}", flush=True)

            t0 = time.perf_counter()
            merged_turns, raw_turns = diarize(
                wav_16k_path,
                on_progress=_diar_cb,
                backend=backend_key,
                return_raw_turns=True,
            )
            diar_elapsed = time.perf_counter() - t0
            with open(cache_file, "w", encoding="utf-8") as cf:
                json.dump(
                    {
                        "backend": backend_key,
                        "diar_time_sec": round(diar_elapsed, 2),
                        "raw_turns": raw_turns,
                        "merged_turns": merged_turns,
                    },
                    cf,
                    ensure_ascii=False,
                    indent=2,
                )

        # Also compute ASR window stats after BaseASREngine.plan_windows
        windows = BaseASREngine.plan_windows(merged_turns)
        raw_stats = _turn_stats(raw_turns)
        merged_stats = _turn_stats(merged_turns)
        window_stats = _turn_stats(windows)

        print(
            f"[+] {label} finished in {diar_elapsed:.2f}s:\n"
            f"    - Unique speakers: {merged_stats['unique_speakers']} ({merged_stats['speakers']})\n"
            f"    - Raw turns (before merging):   {raw_stats['count']} turns "
            f"(avg {raw_stats['avg_duration_sec']:.2f}s, <0.5s micro-turns={raw_stats['sub_0_5s_count']})\n"
            f"    - Cleaned turns (after merging): {merged_stats['count']} turns "
            f"(avg {merged_stats['avg_duration_sec']:.2f}s, preserved isolated short replies={merged_stats['sub_0_5s_count']})\n"
            f"    - ASR windows (mono-deduped):    {window_stats['count']} windows "
            f"(avg {window_stats['avg_duration_sec']:.2f}s)"
        )

        diar_results[backend_key] = {
            "label": label,
            "diar_time_sec": diar_elapsed,
            "raw_turns": raw_turns,
            "merged_turns": merged_turns,
            "windows": windows,
            "raw_stats": raw_stats,
            "merged_stats": merged_stats,
            "window_stats": window_stats,
        }

        print(f"[*] Unloading {label} and freeing GPU/CPU memory...")
        unload_diarization_models(backend_key)
        _free_memory()

    # Load previous baseline for comparison ("vs. previous baseline")
    baseline_data = {}
    if BASELINE_JSON.exists():
        with open(BASELINE_JSON, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)

    # Now load Cohere Transcribe Arabic ONCE and transcribe both diarization turn sets
    print("\n" + "=" * 80)
    print(f"STEP 3: Loading Cohere Transcribe Arabic ({COHERE_MODEL_NAME}) once for both backends")
    print("=" * 80)
    engine = get_asr_engine(COHERE_MODEL_NAME)
    engine_loaded = False

    try:
        for backend_key in ("pyannote", "nemotron_3"):
            info = diar_results[backend_key]
            label = info["label"]
            asr_cache_file = COMPARISON_DIR / f".asr_cache_{backend_key}.json"
            print("\n" + "-" * 80)
            print(f"[*] Transcribing `{backend_key}` turns ({len(info['windows'])} windows) with Cohere ASR...")
            print("-" * 80)

            if asr_cache_file.exists() and not force_rerun:
                with open(asr_cache_file, "r", encoding="utf-8") as af:
                    asr_cached = json.load(af)
                segments = asr_cached["segments"]
                asr_elapsed = float(asr_cached["asr_time_sec"])
                print(f"[*] Loaded cached ASR transcript from {asr_cache_file.name} (ASR time: {asr_elapsed:.2f}s)")
            else:
                if not engine_loaded:
                    engine.load()
                    engine_loaded = True

                def _asr_cb(pct: int, msg: str):
                    print(f"    [{pct:3d}%] {msg}", flush=True)

                t_asr0 = time.perf_counter()
                segments = engine.transcribe_turns(
                    wav_16k_path,
                    info["merged_turns"],
                    on_progress=_asr_cb,
                    progress_range=(0, 100),
                )
                asr_elapsed = time.perf_counter() - t_asr0
                with open(asr_cache_file, "w", encoding="utf-8") as af:
                    json.dump(
                        {
                            "backend": backend_key,
                            "asr_time_sec": round(asr_elapsed, 2),
                            "segments": segments,
                        },
                        af,
                        ensure_ascii=False,
                        indent=2,
                    )

            info["asr_time_sec"] = asr_elapsed
            info["total_time_sec"] = info["diar_time_sec"] + asr_elapsed
            info["segments"] = segments
            info["final_segment_stats"] = _turn_stats(segments)
            info["vs_baseline"] = _compare_with_previous_baseline(segments, baseline_data)

            json_p, txt_p = _save_backend_outputs(
                backend_key=backend_key,
                backend_label=label,
                audio_name=AUDIO_M4A.name,
                diar_time_sec=info["diar_time_sec"],
                asr_time_sec=asr_elapsed,
                raw_stats=info["raw_stats"],
                merged_stats=info["merged_stats"],
                window_stats=info["window_stats"],
                segments=segments,
                vs_baseline=info["vs_baseline"],
            )
            info["json_path"] = str(json_p)
            info["txt_path"] = str(txt_p)
            print(
                f"[✓] `{backend_key}` ASR completed in {asr_elapsed:.2f}s "
                f"(Total pipeline: {info['total_time_sec']:.2f}s, {len(segments)} final utterances).\n"
                f"    Saved -> {json_p.name}, {txt_p.name}"
            )
            _free_memory()
    finally:
        if engine_loaded:
            print("[*] Unloading Cohere Transcribe Arabic engine and freeing memory...")
            engine.unload()
        _free_memory()

    # Build summary comparison JSON & Markdown report
    pya = diar_results["pyannote"]
    nem = diar_results["nemotron_3"]

    summary_payload = {
        "audio_file": AUDIO_M4A.name,
        "audio_duration_sec": round(audio_dur, 2),
        "preprocessing_decode_sec": round(prep_sec, 2),
        "turn_merging_config": {
            "TURN_MERGE_SAME_SPEAKER_GAP": settings.TURN_MERGE_SAME_SPEAKER_GAP,
            "TURN_MIN_DURATION_SEC": settings.TURN_MIN_DURATION_SEC,
            "TURN_ISOLATED_MIN_DURATION_SEC": settings.TURN_ISOLATED_MIN_DURATION_SEC,
            "TURN_DROP_BLIP_DURATION_SEC": settings.TURN_DROP_BLIP_DURATION_SEC,
            "TURN_PAD_MS": settings.TURN_PAD_MS,
        },
        "pyannote": {
            "engine": pya["label"],
            "diarization_time_sec": round(pya["diar_time_sec"], 2),
            "asr_time_sec": round(pya["asr_time_sec"], 2),
            "total_pipeline_time_sec": round(pya["total_time_sec"], 2),
            "raw_turns": pya["raw_stats"],
            "cleaned_turns": pya["merged_stats"],
            "asr_windows": pya["window_stats"],
            "final_utterances": pya["final_segment_stats"],
            "vs_previous_baseline": pya["vs_baseline"],
        },
        "nemotron_3": {
            "engine": nem["label"],
            "diarization_time_sec": round(nem["diar_time_sec"], 2),
            "asr_time_sec": round(nem["asr_time_sec"], 2),
            "total_pipeline_time_sec": round(nem["total_time_sec"], 2),
            "raw_turns": nem["raw_stats"],
            "cleaned_turns": nem["merged_stats"],
            "asr_windows": nem["window_stats"],
            "final_utterances": nem["final_segment_stats"],
            "vs_previous_baseline": nem["vs_baseline"],
        },
    }

    summary_json_path = COMPARISON_DIR / "comparison_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    # Generate Markdown comparison report
    md_lines = [
        f"# Diarization Backend Benchmark (`20261011_133306.m4a` — {audio_dur:.2f}s)",
        "",
        "## 1. Summary Metrics (`pyannote` vs `nemotron_3` vs Previous Baseline)",
        "",
        "| Metric | Previous Baseline (`transcription_20261011_133306.json`) | `pyannote` (Cleaned + Padded) | `nemotron_3` (`nvidia/Nemotron-3-Diarization`) |",
        "| :--- | :---: | :---: | :---: |",
        f"| **Diarization Inference Time** | N/A | `{pya['diar_time_sec']:.2f}s` | `{nem['diar_time_sec']:.2f}s` |",
        f"| **Cohere ASR Inference Time** | N/A | `{pya['asr_time_sec']:.2f}s` | `{nem['asr_time_sec']:.2f}s` |",
        f"| **Total Pipeline Runtime** | N/A | `{pya['total_time_sec']:.2f}s` | `{nem['total_time_sec']:.2f}s` |",
        f"| **Unique Speakers Detected** | `{len(baseline_data.get('speakers_detected') or list((baseline_data.get('speakers') or {}).keys()))}` (`{', '.join(baseline_data.get('speakers_detected') or list((baseline_data.get('speakers') or {}).keys()))}`) | `{pya['merged_stats']['unique_speakers']}` (`{', '.join(pya['merged_stats']['speakers'])}`) | `{nem['merged_stats']['unique_speakers']}` (`{', '.join(nem['merged_stats']['speakers'])}`) |",
        f"| **Raw Turns (Before Merging)** | `102` (unfiltered `>=0.25s`) | `{pya['raw_stats']['count']}` | `{nem['raw_stats']['count']}` |",
        f"| **Avg Duration (Before Merging)** | `4.19s` | `{pya['raw_stats']['avg_duration_sec']:.2f}s` | `{nem['raw_stats']['avg_duration_sec']:.2f}s` |",
        f"| **Cleaned Turns (After Merging)** | N/A | `{pya['merged_stats']['count']}` | `{nem['merged_stats']['count']}` |",
        f"| **Avg Duration (After Merging)** | N/A | `{pya['merged_stats']['avg_duration_sec']:.2f}s` | `{nem['merged_stats']['avg_duration_sec']:.2f}s` |",
        f"| **ASR Windows (`plan_windows`)** | `110` | `{pya['window_stats']['count']}` (avg `{pya['window_stats']['avg_duration_sec']:.2f}s`) | `{nem['window_stats']['count']}` (avg `{nem['window_stats']['avg_duration_sec']:.2f}s`) |",
        f"| **Final Transcribed Turns** | `{pya['vs_baseline']['previous_baseline_turns']}` (avg `{pya['vs_baseline']['previous_baseline_avg_turn_sec']:.2f}s`) | `{len(pya['segments'])}` (avg `{pya['final_segment_stats']['avg_duration_sec']:.2f}s`) | `{len(nem['segments'])}` (avg `{nem['final_segment_stats']['avg_duration_sec']:.2f}s`) |",
        f"| **Total Arabic Words** | `{pya['vs_baseline']['previous_baseline_words']}` | `{pya['vs_baseline']['candidate_words']}` | `{nem['vs_baseline']['candidate_words']}` |",
        f"| **Lexical Overlap vs Baseline (Jaccard)** | `100.0%` | `{pya['vs_baseline']['vocabulary_jaccard_pct']:.1f}%` | `{nem['vs_baseline']['vocabulary_jaccard_pct']:.1f}%` |",
        f"| **Leaked Special Tokens (`@@@` / `###`)** | `{pya['vs_baseline']['previous_baseline_leaked_special_tags']}` | `{pya['vs_baseline']['candidate_leaked_special_tags']}` | `{nem['vs_baseline']['candidate_leaked_special_tags']}` |",
        "",
        "## 2. First 15 Speaker Turns — Side-by-Side Arabic Transcript",
        "",
        "### `pyannote` (Pyannote 3.1 + Cohere Arabic ASR)",
        "",
    ]
    for idx, seg in enumerate(pya["segments"][:15], 1):
        md_lines.append(
            f"{idx:02d}. `[{seg['start']:06.2f} - {seg['end']:06.2f}] {seg['speaker']}:` {seg['text']}"
        )

    md_lines.extend([
        "",
        "### `nemotron_3` (`nvidia/Nemotron-3-Diarization` + Cohere Arabic ASR)",
        "",
    ])
    for idx, seg in enumerate(nem["segments"][:15], 1):
        md_lines.append(
            f"{idx:02d}. `[{seg['start']:06.2f} - {seg['end']:06.2f}] {seg['speaker']}:` {seg['text']}"
        )

    report_md_path = COMPARISON_DIR / "comparison_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n" + "=" * 80)
    print("\n".join(md_lines))
    print("=" * 80)
    print(f"[✓] All comparison files written to: {COMPARISON_DIR}")


if __name__ == "__main__":
    main()
