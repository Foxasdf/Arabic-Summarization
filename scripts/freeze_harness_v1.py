"""Freeze harness v1: generate held-out RTTM + dataset split + baseline stub.
Run: .venv/Scripts/python.exe scripts/freeze_harness_v1.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "eval_results" / "frozen_v1"
EVAL.mkdir(parents=True, exist_ok=True)

HELD_JSON = ROOT / "transcription_20261011_133306.json"
RTTM_OUT = EVAL / "heldout_133306.rttm"

def json_to_rttm():
    d = json.loads(HELD_JSON.read_text(encoding="utf-8"))
    lines = []
    # order-of-appearance mapping already SPEAKER_00.. — keep stable
    for s in d["segments"]:
        spk = s["speaker"]
        st = float(s["start"])
        dur = max(0.01, float(s["end"]) - st)
        # RTTM: SPEAKER <file> 1 <start> <dur> <NA> <NA> <speaker> <NA> <NA>
        lines.append(f"SPEAKER 20261011_133306 1 {st:.2f} {dur:.2f} <NA> <NA> {spk} <NA> <NA>")
    RTTM_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)

def write_dataset():
    dataset = {
        "version": "frozen_v1",
        "normalizer": "backend/app/services/preprocessing/normalizer.py v1.0.0 (raw=normalize_soft, norm=normalize full). Emission must preserve orthography; unify_* only for scoring/DP.",
        "collar_seconds": 0.25,
        "dev_set": [
            {"id": "quality_setutt", "desc": "synthetic Arabic quality corpus 40 items (clean/fan/crowd/quiet) with refs in backend/tools/eval_pipeline.py:146-157", "dir": "eval_results/eval_data/quality_set"},
            {"id": "audio_nlp_5min", "file": "audio_nlp_1_5min.wav", "use": "stability + speaker-count sanity (no word-level ref frozen yet)"},
            {"id": "scratch_45s", "file": "scratch_45s.wav", "use": "single-speaker check for min_speakers=1 test"},
        ],
        "held_out": [
            {"id": "133306", "audio": "20261011_133306.m4a", "reference_json": "transcription_20261011_133306.json",
             "reference_rttm": "eval_results/frozen_v1/heldout_133306.rttm",
             "segments": 105, "speakers": 3,
             "policy": "HELD OUT — run only at milestones, never for keep/revert per-iteration decisions."}
        ],
        "metrics": {
            "primary": ["norm_WER", "norm_CER", "DER_0.25s", "JER", "speaker_count_error"],
            "sanity_only": ["segment_count", "RTF"],
            "rule": "KEEP iff mean(dev) improves AND no single dev file regresses >0.02 absolute on any primary metric. Held-out never votes."
        },
        "chunk_grid_preregistered": {
            "cohere": [[18, 2.0], [20, 2.0], [25, 2.5], [28, 3.0]],
            "note": "Grid fixed here so chunking sweep cannot become open-ended search. VAD silence 250-350ms explored inside same runs."
        }
    }
    (EVAL / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")

def write_baseline_stub(n_rttm: int):
    baseline = {
        "code_version": "pre-deoverfit (guardrails proper-noun + dish maps + file-specific whisper prompt/hotwords still present)",
        "timestamp": "2026-10-13",
        "held_out_reference": {"segments": 105, "rttm_lines": n_rttm},
        "reported_numbers": {
            "note": "From transcription_20261011_133306_IMPROVED.md header + eval_results/ablation_summary.json. Full pipeline rerun after de-overfit will be milestone M1.",
            "IMPROVED_vs_ref": {"WER": 0.4557, "CER": 0.308, "hyp_segments": 70, "prev_segments": 44, "ref_segments": 105},
            "ablation_A_raw_continuous": {"WER": 0.6322, "CER": 0.3658},
            "ablation_B_filtered": {"WER": 0.6322, "CER": 0.3569},
        },
        "transcribe_cpp_probe": {
            "Session.run": "(pcm, *, task, language, timestamps, pnc, itn, diarize, keep_special_tags, spec_k_drafts, family, vocabulary, prompt, prefix)",
            "temperature_beam": "NOT exposed on Session.run. Only via WhisperRunOptions family ext IF model accepts it. Cohere family almost certainly rejects it -> DO NOT add dead config. Verified 2026-10-13.",
            "native_word_timestamps": "AVAILABLE via timestamps='word' -> Result.words[t0_ms,t1_ms]. Current cohere_worker.py discards them (text only). Alignment prototype must use these, not a second Whisper system."
        }
    }
    (EVAL / "baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=2), encoding="utf-8")

def write_rules():
    (EVAL / "RULES.md").write_text(
        "# Frozen v1 rules (from review)\n"
        "- 133306 is HELD OUT. Tune on dev (quality_set + audio_nlp + scratch). Run held-out at milestones only.\n"
        "- KEEP iff mean(dev primary) improves AND max single-file regression <= 0.02 absolute.\n"
        "- Collar fixed 0.25s for DER scoring. Never tune collar.\n"
        "- Segment count is sanity only, not keep/revert. Use DER/JER + speaker-count error.\n"
        "- Hallucination list must be gated on low-energy/non-speech (VAD) before deleting; else keep text.\n"
        "- unify_alef/yaa/taa_marbuta ONLY in scorer + DP matcher. Emitted transcript preserves orthography (verbatim-but-wrong > silent correction).\n"
        "- min_speakers: test min=1 (scratch_45s is single-speaker probe).\n"
        "- De-overfit lands ALONE (this commit). Timing/alignment is a separate offline-evaluated prototype with boundary-MAE before integration. No energy-weighted fallback; uniform split flagged low-confidence is the honest fallback.\n",
        encoding="utf-8")

if __name__ == "__main__":
    n = json_to_rttm()
    write_dataset()
    write_baseline_stub(n)
    write_rules()
    print(f"frozen_v1 written: {n} rttm lines")
