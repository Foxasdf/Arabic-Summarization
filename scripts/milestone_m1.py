"""M1 milestone: full production pipeline on HELD-OUT 133306 + gold 45s rebuild.
Steps: dual-profile preprocess -> diarize(auto + oracle3) -> Cohere continuous
-> align -> score (WER/CER/DER/JER/counts) vs frozen baseline.
Saves eval_results/frozen_v1/m1_results.json
"""
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(ROOT / "scripts"))

from der_scorer import score_der  # noqa: E402

AUDIO = ROOT / "20261011_133306.m4a"
REF_JSON = ROOT / "transcription_20261011_133306.json"
M1DIR = ROOT / "eval_results" / "frozen_v1" / "m1"
M1DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[M1 {time.strftime('%H:%M:%S')}] {msg}", flush=True)

def main():
    from app.services.audio_processor import process_dual_profile_audio, get_audio_duration
    from app.services import diarization as D
    from app.services.transcription import transcribe_with_speakers
    from app.services.preprocessing.normalizer import ArabicTextNormalizer, calculate_wer_cer

    ref = json.loads(REF_JSON.read_text(encoding="utf-8"))
    ref_segments = ref["segments"]
    ref_text = " ".join(s["text"] for s in ref_segments)
    normalizer = ArabicTextNormalizer()

    # 1. dual profile
    asr_wav = M1DIR / "asr.16k.wav"
    spk_wav = M1DIR / "speaker.16k.wav"
    if not (asr_wav.exists() and spk_wav.exists()):
        log("preprocessing dual profile...")
        ok, err = asyncio.run(process_dual_profile_audio(AUDIO, asr_wav, spk_wav, denoise_mode="raw"))
        assert ok, f"preprocess failed: {err}"
    log(f"asr={asr_wav.stat().st_size} spk={spk_wav.stat().st_size} dur={get_audio_duration(asr_wav)}")

    # 2. diarization
    t0 = time.time()
    log("diarizing auto...")
    turns_auto = D.diarize(spk_wav)
    log(f"auto done in {time.time()-t0:.0f}s: {len(turns_auto)} turns, speakers={sorted(set(t['speaker'] for t in turns_auto))}")
    t0 = time.time()
    log("diarizing oracle num=3...")
    turns_or3 = D.diarize(spk_wav, num_speakers=3)
    log(f"oracle done in {time.time()-t0:.0f}s: {len(turns_or3)} turns")
    (M1DIR / "turns_auto.json").write_text(json.dumps(turns_auto, ensure_ascii=False, indent=1), encoding="utf-8")
    (M1DIR / "turns_oracle3.json").write_text(json.dumps(turns_or3, ensure_ascii=False, indent=1), encoding="utf-8")

    der_auto = score_der(ref_segments, turns_auto)
    der_or3 = score_der(ref_segments, turns_or3)
    log(f"DER auto={der_auto['der']} jer={der_auto['jer']} nspk={der_auto['n_hyp_spk']}")
    log(f"DER or3 ={der_or3['der']} jer={der_or3['jer']} nspk={der_or3['n_hyp_spk']}")

    # 3+4. transcribe + align via production path (auto turns)
    t0 = time.time()
    log("Cohere continuous transcription + alignment (production path)...")
    حال = None
    segments = transcribe_with_speakers(
        asr_wav, turns_auto,
        on_progress=lambda p, m: log(f"asr {p}% {m}"),
        model_identifier="cohere",
    )
    log(f"transcribe done in {time.time()-t0:.0f}s: {len(segments)} segments")
    (M1DIR / "segments_hyp.json").write_text(json.dumps(segments, ensure_ascii=False, indent=1), encoding="utf-8")

    hyp_text = " ".join(s["text"] for s in segments)
    m = calculate_wer_cer(ref_text, hyp_text, normalizer)
    log(f"WER raw={m['raw_wer']} norm={m['norm_wer']} CER raw={m['raw_cer']} norm={m['norm_cer']} (baseline IMPROVED: 0.4557/0.308)")

    out = {
        "dev_run_on_held_out": True,
        "note": "M1 milestone — held-out run permitted once per milestone per RULES.md",
        "wer_raw": m["raw_wer"], "wer_norm": m["norm_wer"], "cer_raw": m["raw_cer"], "cer_norm": m["norm_cer"],
        "ref_words": m["ref_word_count"], "hyp_words": m["hyp_word_count"],
        "subs": m["substitutions"], "dels": m["deletions"], "inss": m["insertions"],
        "der_auto": der_auto, "der_oracle3": der_or3,
        "n_seg_ref": len(ref_segments), "n_seg_hyp": len(segments),
        "baseline_IMPROVED": {"WER": 0.4557, "CER": 0.308, "hyp_segments": 70},
    }
    (M1DIR.parent / "m1_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote m1_results.json")

if __name__ == "__main__":
    main()
