"""Denoise A/B (frozen_v1 step 5, ASR branch only, Speaker stays RAW).
Dev: scratch_45s.wav vs gold ref. Presets: raw vs light(highpass50) vs balanced.
Speaker-branch invariance asserted (not run): process_dual_profile_audio builds
spk_req with RAW/OFF defaults.
"""
import asyncio
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.models.asr import get_asr_engine  # noqa: E402
from app.services.preprocessing.normalizer import ArabicTextNormalizer, calculate_wer_cer  # noqa: E402
from app.services.audio_processor import process_dual_profile_audio  # noqa: E402
from tests.eval.gold_dataset import GOLD_TEST_SET  # noqa: E402

OUT = ROOT / "eval_results" / "frozen_v1" / "denoise_ab.json"

async def denoise_variants():
    from app.services.preprocessing import PreprocessingEngine, AudioProcessingRequest, OutputProfile, DenoiseMode
    eng = PreprocessingEngine()
    outs = {}
    keepdir = ROOT / "eval_results" / "frozen_v1" / "denoise_wavs"
    keepdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        for mode in [DenoiseMode.RAW, DenoiseMode.LIGHT, DenoiseMode.BALANCED]:
            req = AudioProcessingRequest(output_profile=OutputProfile.ASR, denoise_mode=mode)
            res = eng.process(ROOT / "scratch_45s.wav", req, Path(td) / mode.value)
            src = Path(res.output_files["asr"])
            # copy out of temp into persistent dir before cleanup
            dst = keepdir / f"scratch45_{mode.value}.wav"
            shutil.copy2(str(src), str(dst))
            outs[mode.value] = dst
    return outs

def main():
    ref = GOLD_TEST_SET["sample_meeting_45s"]["ground_truth_text"]
    norm = ArabicTextNormalizer()
    # 1. assert dual-profile speaker invariance in code path ( dry: inspect request defaults )
    from app.services.preprocessing.contracts import AudioProcessingRequest as R, OutputProfile as P
    spk_req = R(output_profile=P.SPEAKER)
    assert spk_req.denoise_mode.value == "raw", f"speaker default changed: {spk_req.denoise_mode}"
    print(f"speaker-branch default: denoise={spk_req.denoise_mode.value} norm={spk_req.normalization.value} (RAW invariant holds)")
    # 2. ASR A/B
    wavs = asyncio.run(denoise_variants())
    rows = []
    for mode, wav in wavs.items():
        eng = get_asr_engine("cohere")
        eng.load()
        t0 = time.time()
        res = eng.transcribe(wav, on_progress=None)
        dt = time.time() - t0
        try:
            eng.unload()
        except Exception:
            pass
        m = calculate_wer_cer(ref, res.get("text", ""), norm)
        rows.append({"asr_preset": mode, "norm_wer": m["norm_wer"], "norm_cer": m["norm_cer"],
                     "words": m["hyp_word_count"], "trans_s": round(dt, 1)})
        print(f"{mode}: WER={m['norm_wer']:.4f} CER={m['norm_cer']:.4f} words={m['hyp_word_count']}", flush=True)
    OUT.write_text(json.dumps({"dev": "scratch_45s", "speaker_invariant": True, "rows": rows,
                               "rule": "Promote to auto only if a preset beats raw by >=0.02 with no other regression; else keep raw default."},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
