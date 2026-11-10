"""Two targeted tests for 133306 speaker separation:
T1: min_cluster_size sweep (config param). A high floor can force-merge rare speakers.
T2: embedding window length vs separability (short windows -> unreliable embeddings).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from der_scorer import score_der  # noqa: E402

SPK_WAV = ROOT / "eval_results" / "frozen_v1" / "m1_M1old" / "speaker.16k.wav"
OUT = ROOT / "eval_results" / "frozen_v1" / "mincluster_window.json"
SR = 16000


def t1_min_cluster_size():
    import yaml
    import torch
    from pyannote.audio import Pipeline
    from app.services.diarization import _load_wav
    cfg_path = ROOT / "models" / "pyannote" / "speaker-diarization-3.1" / "config.yaml"
    orig = cfg_path.read_text(encoding="utf-8")
    ref = json.loads((ROOT / "transcription_20261011_133306.json").read_text(encoding="utf-8"))["segments"]
    wav = torch.from_numpy(np.array(_load_wav(SPK_WAV), dtype=np.float32)).unsqueeze(0)
    rows = []
    try:
        for mcs in (4, 8, 12, 20):
            cfg = yaml.safe_load(orig)
            cfg["params"]["clustering"]["min_cluster_size"] = mcs
            cfg["pipeline"]["params"]["segmentation"] = str(ROOT / "models/pyannote/segmentation-3.0/pytorch_model.bin")
            cfg["pipeline"]["params"]["embedding"] = str(ROOT / "models/pyannote/wespeaker-voxceleb-resnet34-LM/pytorch_model.bin")
            cfg["pipeline"]["params"]["segmentation_batch_size"] = 32
            cfg["pipeline"]["params"]["embedding_batch_size"] = 32
            cfg["pipeline"]["params"].pop("plda", None)
            tmp = ROOT / "eval_results/frozen_v1/_cfg_mcs.yaml"
            tmp.write_text(yaml.dump(cfg, default_flow_style=False), encoding="utf-8")
            pipe = Pipeline.from_pretrained(str(tmp), token=os.getenv("HF_TOKEN") or None)
            if torch.cuda.is_available():
                pipe.to(torch.device("cuda:0"))
            with torch.inference_mode():
                res = pipe({"waveform": wav, "sample_rate": SR}, min_speakers=2, max_speakers=8)
            ann = getattr(res, "speaker_diarization", res)
            turns = [{"speaker": spk, "start": float(t.start), "end": float(t.end)}
                     for t, _, spk in ann.itertracks(yield_label=True)]
            m, out = {}, []
            for t in sorted(turns, key=lambda x: x["start"]):
                if t["speaker"] not in m:
                    m[t["speaker"]] = f"SPEAKER_{len(m):02d}"
                out.append({"speaker": m[t["speaker"]], "start": round(t["start"], 2), "end": round(t["end"], 2)})
            s = score_der(ref, out)
            rows.append({"min_cluster_size": mcs, "n_turns": len(out), **s})
            print(f"mcs={mcs}: DER={s['der']:.4f} JER={s['jer']:.4f} nspk={s['n_hyp_spk']} turns={len(out)}", flush=True)
            del pipe
            torch.cuda.empty_cache()
    finally:
        cfg_path.write_text(orig, encoding="utf-8")
    return rows


def t2_window_length():
    from scipy.io import wavfile
    import torch
    from pyannote.audio.models.embedding import WeSpeakerResNet34
    ref = json.loads((ROOT / "transcription_20261011_133306.json").read_text(encoding="utf-8"))["segments"]
    spk_map = {"SPEAKER_00": 0, "SPEAKER_01": 1, "SPEAKER_02": 2}
    sr, data = wavfile.read(str(SPK_WAV))
    audio = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
    m = WeSpeakerResNet34.from_pretrained(str(ROOT / "models/pyannote/wespeaker-voxceleb-resnet34-LM"))
    m.eval()
    if torch.cuda.is_available():
        m = m.to(torch.device("cuda:0"))
    out = []
    for win_sec in (1.0, 1.5, 2.5, 4.0):
        WIN = int(win_sec * SR)
        wins, labels = [], []
        for seg in ref:
            a, b = int(seg["start"] * SR), int(seg["end"] * SR)
            lab = spk_map[seg["speaker"]]
            pos = a
            while pos + WIN <= b:
                wins.append(audio[pos:pos + WIN])
                labels.append(lab)
                pos += WIN + int(0.3 * SR)
        if len(wins) < 12:
            out.append({"win_sec": win_sec, "n": len(wins), "acc": None}); continue
        embs = []
        with torch.inference_mode():
            for i in range(0, len(wins), 16):
                arr = torch.from_numpy(np.stack([np.ascontiguousarray(w) for w in wins[i:i+16]]).astype(np.float32)).unsqueeze(1)
                if torch.cuda.is_available():
                    arr = arr.to(torch.device("cuda:0"))
                e = m(arr)
                e = e["embedding"] if isinstance(e, dict) else e
                e = e.detach().cpu().numpy()
                embs.append(e.squeeze(1) if e.ndim == 3 else e)
        E = np.concatenate(embs, 0)
        labels = np.array(labels)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        cents = np.stack([En[labels == k].mean(0) for k in range(3)])
        cents /= (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
        acc = float((np.argmax(En @ cents.T, 1) == labels).mean())
        S = En @ En.T
        same = labels[:, None] == labels[None, :]
        np.fill_diagonal(same, False)
        intra_p95 = float(np.percentile(S[same], 95))
        inter_p05 = float(np.percentile(S[~same], 5))
        out.append({"win_sec": win_sec, "n": len(wins), "acc": round(acc, 4),
                    "gap": round(intra_p95 - inter_p05, 4)})
        print(f"win={win_sec}s n={len(wins)} acc={acc:.4f} gap={intra_p95-inter_p05:.4f}", flush=True)
    return out


if __name__ == "__main__":
    print("=== T1: min_cluster_size ===")
    t1 = t1_min_cluster_size()
    print("\n=== T2: embedding window length ===")
    t2 = t2_window_length()
    OUT.write_text(json.dumps({"T1_min_cluster_size": t1, "T2_window": t2}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")