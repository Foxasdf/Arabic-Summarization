"""Speaker-identity investigation on 133306 (reference = label truth).
Uses the pyannote pipeline's own embedding model to embed windows whose TRUE speaker
is known from transcription_20261011_133306.json, then measures intra/inter cosine
similarity and centroid accuracy — i.e. are the three voices actually separable?
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

SPK_WAV = ROOT / "eval_results" / "frozen_v1" / "m1_M1old" / "speaker.16k.wav"
SR = 16000

def main():
    import torch
    from scipy.io import wavfile
    from app.services.diarization import get_pyannote_pipeline

    ref = json.loads((ROOT / "transcription_20260611_133306.json").read_text(encoding="utf-8"))["segments"] \
        if False else json.loads((ROOT / "transcription_20261011_133306.json").read_text(encoding="utf-8"))["segments"]
    spk_map = {"SPEAKER_00": 0, "SPEAKER_01": 1, "SPEAKER_02": 2}

    sr, data = wavfile.read(str(SPK_WAV))
    if data.dtype == np.int16:
        audio = (data.astype(np.float32) / 32768.0)
    else:
        audio = data.astype(np.float32)

    pipeline = get_pyannote_pipeline()
    # Load the WeSpeaker embedding model directly (pipeline.embedding stays a string
    # until the pipeline runs; we want an independent measurement anyway).
    import torch
    from pyannote.audio.models.embedding import WeSpeakerResNet34
    emb_dir = ROOT / "models" / "pyannote" / "wespeaker-voxceleb-resnet34-LM"
    emb_model = WeSpeakerResNet34.from_pretrained(str(emb_dir))
    emb_model.eval()
    if torch.cuda.is_available():
        emb_model = emb_model.to(torch.device("cuda:0"))

    # Build non-overlapping 1.5s windows fully inside a single reference turn
    WIN = int(1.5 * SR)
    windows, labels = [], []
    for seg in ref:
        a, b = int(seg["start"] * SR), int(seg["end"] * SR)
        lab = spk_map[seg["speaker"]]
        pos = a
        while pos + WIN <= b:
            windows.append(audio[pos:pos + WIN])
            labels.append(lab)
            pos += WIN + int(0.3 * SR)  # 0.3s stride gap to avoid boundary bleed
    print(f"windows: {len(windows)}  label counts: {np.bincount(labels, minlength=3)}")
    if len(windows) < 12:
        print("not enough windows"); return

    embs = []
    B = 16
    with torch.inference_mode():
        for i in range(0, len(windows), B):
            batch = windows[i:i + B]
            # WeSpeakerResNet34.forward expects (batch, samples) float tensor
            arr = torch.from_numpy(np.stack([np.ascontiguousarray(w) for w in batch]).astype(np.float32)).unsqueeze(1)
            if torch.cuda.is_available():
                arr = arr.to(torch.device("cuda:0"))
            out = emb_model(arr)
            e = out["embedding"] if isinstance(out, dict) else out
            if hasattr(e, "detach"):
                e = e.detach()
            e = e.cpu().numpy()
            e = e.squeeze(1) if e.ndim == 3 else e
            embs.append(e)
    E = np.concatenate(embs, axis=0)
    labels = np.array(labels)
    if E.ndim > 2:
        E = E.mean(axis=1)
    print("embedding matrix:", E.shape)

    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    S = En @ En.T
    same = (labels[:, None] == labels[None, :])
    np.fill_diagonal(same, False)
    intra = S[same]
    inter = S[~same]
    print(f"intra cos  mean={intra.mean():.4f} p95={np.percentile(intra,95):.4f}")
    print(f"inter cos  mean={inter.mean():.4f} p05={np.percentile(inter,5):.4f}")
    print(f"SEPARATION GAP (intra_p95 - inter_p05) = {np.percentile(intra,95) - np.percentile(inter,5):.4f}")
    for i in range(3):
        for j in range(i + 1, 3):
            m = (labels[:, None] == i) & (labels[None, :] == j)
            if m.sum():
                v = S[m]
                print(f"  S{i}<->S{j} inter: mean={v.mean():.4f} p95={np.percentile(v,95):.4f} n={m.sum()//2}")
    cents = np.stack([En[labels == k].mean(axis=0) for k in range(3)])
    cents /= (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-9)
    pred = np.argmax(En @ cents.T, axis=1)
    print(f"centroid-NN accuracy (ideal clustering ceiling): {(pred == labels).mean():.4f}")
    for k in range(3):
        print(f"  true S{k} n={(labels==k).sum():3d} -> predicted {np.bincount(pred[labels==k], minlength=3)}")

if __name__ == "__main__":
    main()