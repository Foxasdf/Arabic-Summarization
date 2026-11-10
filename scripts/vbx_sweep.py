"""VBxClustering + PLDA calibration sweep on 133306.
Diagnosis so far: SPEAKER_02 (rarest voice, 27 windows) is absorbed into SPEAKER_01.
VBx models speaker+channel variability and is better at minority speakers, but the
auto-instantiated defaults (Fa=0.07 Fb=0.8) under-clustered to 2 speakers.
Sweep Fa (lower => more willing to spawn a new speaker) x Fb (higher => more merging).
Reference: transcription_20261011_133306.json, collar 0.25s.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from der_scorer import score_der  # noqa: E402

SPK_WAV = ROOT / "eval_results" / "frozen_v1" / "m1_M1old" / "speaker.16k.wav"
OUT = ROOT / "eval_results" / "frozen_v1" / "vbx_sweep.json"


def rename(turns):
    m, out = {}, []
    for t in sorted(turns, key=lambda x: x["start"]):
        if t["speaker"] not in m:
            m[t["speaker"]] = f"SPEAKER_{len(m):02d}"
        out.append({"speaker": m[t["speaker"]], "start": round(float(t["start"]), 2),
                    "end": round(float(t["end"]), 2)})
    return out


def main():
    import numpy as np
    import torch
    import yaml
    from pyannote.audio import Pipeline
    from app.services.diarization import _load_wav

    ref = json.loads((ROOT / "transcription_20261011_133306.json").read_text(encoding="utf-8"))["segments"]
    wav = torch.from_numpy(np.array(_load_wav(SPK_WAV), dtype=np.float32)).unsqueeze(0)
    if torch.cuda.is_available():
        wav = wav.cuda()

    rows = []
    for fa in (0.005, 0.01, 0.02, 0.05):
        for fb in (0.5, 0.65, 0.8):
            cfg = {
                "params": {"clustering": {"Fa": fa, "Fb": fb}},
                "pipeline": {
                    "name": "pyannote.audio.pipelines.SpeakerDiarization",
                    "params": {
                        "clustering": "VBxClustering",
                        "segmentation": str(ROOT / "models/pyannote/segmentation-3.0/pytorch_model.bin"),
                        "embedding": str(ROOT / "models/pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin"),
                        "segmentation_batch_size": 32,
                        "embedding_batch_size": 32,
                        "plda": str(ROOT / "models/pyannote/plda"),
                    },
                },
            }
            tmp = ROOT / "eval_results/frozen_v1/_cfg_vbx.yaml"
            tmp.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
            pipe = Pipeline.from_pretrained(str(tmp), token=os.getenv("HF_TOKEN") or None)
            if torch.cuda.is_available():
                pipe.to(torch.device("cuda:0"))
            with torch.inference_mode():
                res = pipe({"waveform": wav, "sample_rate": 16000}, min_speakers=2, max_speakers=8)
            ann = getattr(res, "speaker_diarization", res)
            turns = [{"speaker": spk, "start": float(t.start), "end": float(t.end)}
                     for t, _, spk in ann.itertracks(yield_label=True)]
            s = score_der(ref, rename(turns))
            rows.append({"Fa": fa, "Fb": fb, "n_turns": len(turns), **s})
            print(f"Fa={fa} Fb={fb}: DER={s['der']:.4f} JER={s['jer']:.4f} nspk={s['n_hyp_spk']} turns={len(turns)}", flush=True)
            del pipe
            torch.cuda.empty_cache()

    rows.sort(key=lambda x: x["der"])
    print("\nTop 5:")
    for r in rows[:5]:
        print(f"  Fa={r['Fa']} Fb={r['Fb']}: DER={r['der']:.4f} JER={r['jer']:.4f} nspk={r['n_hyp_spk']}")
    OUT.write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()