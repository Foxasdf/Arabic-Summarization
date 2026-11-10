"""Diarization clustering experiment on 133306 (user-sanctioned reference for this track).
Compares:
  A. AgglomerativeClustering, threshold 0.7046 (current)
  B. AgglomerativeClustering, threshold sweep
  C. VBxClustering + PLDA (pyannote 3.1 default, PLDA files present locally)
Scores DER (collar 0.25s) + JER + count error vs transcription_20261011_133306.json.
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
OUT = ROOT / "eval_results" / "frozen_v1" / "diar_clustering_exp.json"


def build_pipeline(clustering, threshold=None, min_cluster_size=None, plda=True):
    import yaml
    from app.services import diarization as D
    from pyannote.audio import Pipeline
    cfg_path = ROOT / "models" / "pyannote" / "speaker-diarization-3.1" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["pipeline"]["params"]["segmentation"] = str(ROOT / "models" / "pyannote" / "segmentation-3.0" / "pytorch_model.bin")
    cfg["pipeline"]["params"]["embedding"] = str(ROOT / "models" / "pyannote" / "wespeaker-voxceleb-resnet34-LM" / "pytorch_model.bin")
    cfg["pipeline"]["params"]["segmentation_batch_size"] = 32
    cfg["pipeline"]["params"]["embedding_batch_size"] = 32
    cfg["pipeline"]["params"]["clustering"] = clustering
    if clustering == "AgglomerativeClustering":
        if threshold is not None:
            cfg["params"]["clustering"]["threshold"] = float(threshold)
        if min_cluster_size is not None:
            cfg["params"]["clustering"]["min_cluster_size"] = int(min_cluster_size)
    else:
        # VBxClustering takes no method/threshold params (plda + metric only)
        cfg["params"]["clustering"] = {}
    if plda:
        cfg["pipeline"]["params"]["plda"] = str(ROOT / "models" / "pyannote" / "plda")
    else:
        cfg["pipeline"]["params"].pop("plda", None)
    tmp = ROOT / "eval_results" / "frozen_v1" / f"_cfg_{clustering}_{threshold}.yaml"
    tmp.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
    token = os.getenv("HF_TOKEN") or None
    return Pipeline.from_pretrained(str(tmp), token=token)


def run(pipe, num_speakers=None):
    import torch
    import numpy as np
    from app.services.diarization import _load_wav
    wav = _load_wav(SPK_WAV)
    waveform = torch.from_numpy(np.array(wav, dtype=np.float32)).unsqueeze(0)
    if torch.cuda.is_available():
        pipe.to(torch.device("cuda:0"))
    io_input = {"waveform": waveform, "sample_rate": 16000}
    kw = {}
    if num_speakers:
        kw["num_speakers"] = num_speakers
    else:
        kw["min_speakers"] = 2
        kw["max_speakers"] = 8
    with torch.inference_mode():
        res = pipe(io_input, **kw)
    ann = getattr(res, "speaker_diarization", res)
    return [{"speaker": spk, "start": float(t.start), "end": float(t.end)}
            for t, _, spk in ann.itertracks(yield_label=True)]


def main():
    import torch
    ref = json.loads((ROOT / "transcription_20261011_133306.json").read_text(encoding="utf-8"))["segments"]
    rows = []

    experiments = [
        ("Agglomerative_thr0.7046", "AgglomerativeClustering", 0.7045654963945799, None, False),
        ("VBx_PLDA_default", "VBxClustering", None, None, True),
        ("VBx_PLDA_oracle3", "VBxClustering", None, None, True),
    ]
    for name, clustering, thr, mcs, use_plda in experiments:
        pipe = build_pipeline(clustering, threshold=thr, min_cluster_size=mcs, plda=use_plda)
        if name.endswith("oracle3"):
            turns = run(pipe, num_speakers=3)
        else:
            turns = run(pipe)
        s = score_der(ref, turns)
        rows.append({"exp": name, "n_turns": len(turns), **s})
        print(f"{name}: DER={s['der']:.4f} JER={s['jer']:.4f} miss={s['miss_s']} conf={s['conf_s']} nspk={s['n_hyp_spk']} turns={len(turns)}", flush=True)
        del pipe
        torch.cuda.empty_cache()

    OUT.write_text(json.dumps({"reference": "transcription_20261011_133306.json", "collar": 0.25, "rows": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()