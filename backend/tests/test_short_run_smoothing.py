import unittest
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.alignment import (
    align_timestamps,
    smooth_short_runs_with_embeddings,
)
from app.services.diarization import build_speaker_embedding_context


def _turn(speaker, start, end):
    return {"speaker": speaker, "start": start, "end": end}


def _unit(vec):
    vec = np.asarray(vec, dtype=np.float32)
    return vec / np.linalg.norm(vec)


class TestShortRunSmoothing(unittest.TestCase):
    def _words(self):
        return [
            {"word": "يعني", "start": 80.0, "end": 80.4, "speaker": "DOC"},
            {"word": "تكست", "start": 80.6, "end": 80.9, "speaker": "STU"},
            {"word": "تو", "start": 80.9, "end": 81.2, "speaker": "STU"},
            {"word": "فونيم", "start": 81.2, "end": 81.6, "speaker": "DOC"},
        ]

    def _stats(self, speakers, values):
        turns = [_turn(spk, i * 1.0, (i + 1) * 1.0) for i, spk in enumerate(speakers)]
        embeddings = [_unit(np.array(v, dtype=np.float32)) for v in values]
        return turns, embeddings

    def _centroids(self):
        return {
            "DOC": _unit(np.array([1.0, 0.0])),
            "STU": _unit(np.array([0.0, 1.0])),
        }

    def test_fragment_absorbed_into_surrounding_speaker(self):
        turns = [
            _turn("DOC", 80.0, 80.55),
            _turn("STU", 80.55, 81.25),
            _turn("DOC", 81.25, 84.0),
        ]
        embeddings = [
            _unit(np.array([1.0, 0.0])),
            _unit(np.array([0.9, 0.1])),  # short turn's audio matches the surrounding speaker
            _unit(np.array([1.0, 0.0])),
        ]

        words = self._words()
        smoothed = smooth_short_runs_with_embeddings(
            words, turns, embeddings, self._centroids()
        )

        self.assertEqual(
            [w["speaker"] for w in smoothed],
            ["DOC", "DOC", "DOC", "DOC"],
        )
        # Input must not be mutated.
        self.assertEqual(words[1]["speaker"], "STU")

    def test_genuine_long_interjection_is_kept(self):
        turns, embeddings = self._stats(
            ["DOC", "STU", "DOC"],
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
        )
        # Supporting STU turn is 1.4s (> max_support_sec): a real interjection.
        turns[1] = {"speaker": "STU", "start": 0.0, "end": 1.4}
        turns[2] = {"speaker": "DOC", "start": 2.0, "end": 3.0}

        words = self._words()
        smoothed = smooth_short_runs_with_embeddings(
            words, turns, embeddings, self._centroids()
        )
        self.assertEqual(
            [w["speaker"] for w in smoothed],
            ["DOC", "STU", "STU", "DOC"],
        )

    def test_ambiguous_embedding_is_kept(self):
        turns, embeddings = self._stats(
            ["DOC", "STU", "DOC"],
            [
                [1.0, 0.0],
                [0.6, 0.8],  # only marginally closer to DOC's neighbour centroid
                [1.0, 0.0],
            ],
        )
        # Make DOC's centroid much closer so the STU embedding is ambiguous.
        centroids = {
            "DOC": _unit(np.array([1.0, 0.1])),
            "STU": _unit(np.array([0.0, 1.0])),
        }

        words = self._words()
        smoothed = smooth_short_runs_with_embeddings(
            words, turns, embeddings, centroids
        )
        self.assertEqual(
            [w["speaker"] for w in smoothed],
            ["DOC", "STU", "STU", "DOC"],
        )

    def test_run_too_long_is_kept(self):
        words = [
            {"word": "يعني", "start": 0.0, "end": 0.5, "speaker": "DOC"},
            {"word": "واحد", "start": 0.9, "end": 1.3, "speaker": "STU"},
            {"word": "اثنين", "start": 1.6, "end": 2.0, "speaker": "STU"},
            {"word": "ثلاثة", "start": 2.3, "end": 2.7, "speaker": "STU"},
            {"word": "اربعة", "start": 3.1, "end": 3.5, "speaker": "STU"},
            {"word": "خمسة", "start": 5.0, "end": 5.5, "speaker": "DOC"},
        ]
        turns, embeddings = self._stats(
            ["DOC", "STU", "DOC"],
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
            ],
        )
        smoothed = smooth_short_runs_with_embeddings(
            words, turns, embeddings, self._centroids()
        )
        # 3.9s and 4 words: far beyond a short fragment.
        self.assertEqual(
            [w["speaker"] for w in smoothed],
            ["DOC", "STU", "STU", "STU", "STU", "DOC"],
        )

    def test_mismatched_embedding_length_is_ignored(self):
        words = self._words()
        turns, embeddings = self._stats(
            ["DOC", "STU", "DOC"],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        smoothed = smooth_short_runs_with_embeddings(
            words, turns, embeddings, self._centroids()
        )
        self.assertEqual(smoothed, words)

    def test_different_neighbour_speakers_are_ignored(self):
        words = [
            {"word": "يعني", "start": 0.0, "end": 0.5, "speaker": "DOC"},
            {"word": "تكست", "start": 0.6, "end": 0.9, "speaker": "STU"},
            {"word": "تو", "start": 0.95, "end": 1.2, "speaker": "STU"},
            {"word": "فونيم", "start": 1.25, "end": 1.6, "speaker": "OTHER"},
        ]
        turns, embeddings = self._stats(
            ["DOC", "STU", "OTHER"],
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        )
        centroids = {
            "DOC": _unit(np.array([1.0, 0.0])),
            "STU": _unit(np.array([0.0, 1.0])),
            "OTHER": _unit(np.array([0.7, 0.7])),
        }
        smoothed = smooth_short_runs_with_embeddings(words, turns, embeddings, centroids)
        self.assertEqual(
            [w["speaker"] for w in smoothed],
            ["DOC", "STU", "STU", "OTHER"],
        )

    def test_none_supporting_embedding_is_kept(self):
        turns, embeddings = self._stats(
            ["DOC", "STU", "DOC"],
            [
                [1.0, 0.0],
                None,
                [1.0, 0.0],
            ],
        )
        words = self._words()
        smoothed = smooth_short_runs_with_embeddings(
            words, turns, embeddings, self._centroids()
        )
        self.assertEqual(
            [w["speaker"] for w in smoothed],
            ["DOC", "STU", "STU", "DOC"],
        )

    def test_align_timestamps_uses_context(self):
        words = [
            {"word": "يعني", "start": 80.0, "end": 80.4},
            {"word": "تكست", "start": 80.6, "end": 80.9},
            {"word": "تو", "start": 80.9, "end": 81.2},
            {"word": "فونيم", "start": 81.2, "end": 81.6},
            {"word": "نحكي", "start": 81.6, "end": 82.0},
        ]
        turns = [
            _turn("DOC", 80.0, 80.45),
            _turn("STU", 80.45, 81.15),
            _turn("DOC", 81.15, 82.0),
        ]
        context = {
            "embeddings": [
                _unit(np.array([1.0, 0.0])),
                _unit(np.array([0.9, 0.1])),
                _unit(np.array([1.0, 0.0])),
            ],
            "centroids": {
                "DOC": _unit(np.array([1.0, 0.0])),
                "STU": _unit(np.array([0.0, 1.0])),
            },
        }

        res = align_timestamps(words, turns, speaker_context=context)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["speaker"], "DOC")
        for token in ("يعني", "تكست", "فونيم"):
            self.assertIn(token, res[0]["text"])

    def test_align_timestamps_without_context_keeps_interjection(self):
        words = self._words()
        turns = [
            _turn("DOC", 80.0, 80.45),
            _turn("STU", 80.45, 81.15),
            _turn("DOC", 81.15, 82.0),
        ]
        res = align_timestamps(words, turns)
        # Without embeddings the temporal behaviour must stay unchanged.
        speakers = []
        for seg in res:
            speakers.extend([seg["speaker"]] * len(seg["text"].split()))
        self.assertEqual(speakers.count("DOC"), 2)
        self.assertEqual(speakers.count("STU"), 2)


class TestBuildSpeakerEmbeddingContext(unittest.TestCase):
    def _save_wav(self, tmp_dir, samples):
        path = tmp_dir / "speaker.wav"
        wavfile.write(str(path), 16000, (np.zeros(samples, dtype=np.float32) * 32767).astype(np.int16))
        return path

    def test_missing_embedding_model_returns_none(self):
        import tempfile
        pipeline = type("FakePipeline", (), {"_inferences": {}})()
        with tempfile.TemporaryDirectory() as tmp:
            path = self._save_wav(Path(tmp), 16000 * 4)
            turns = [_turn("A", 0.0, 2.0), _turn("B", 2.0, 4.0)]
            self.assertIsNone(build_speaker_embedding_context(path, turns, pipeline=pipeline))

    def test_single_speaker_centroid_returns_none(self):
        import tempfile
        sr = 16000
        audio = np.zeros(sr * 4, dtype=np.float32)
        audio[sr:3 * sr] = 1.0

        def fake_embed(chunk):
            mean = float(chunk.mean().item())
            return np.array([[mean, 1.0]], dtype=np.float32)

        pipeline = type("FakePipeline", (), {"_inferences": {"_embedding": fake_embed}})()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wav_path = tmp_path / "speaker.wav"
            wavfile.write(str(wav_path), sr, (audio * 32767).astype(np.int16))
            turns = [_turn("A", 0.5, 3.5)]
            self.assertIsNone(build_speaker_embedding_context(wav_path, turns, pipeline=pipeline))

    def test_builds_centroids_for_multiple_speakers(self):
        import tempfile
        sr = 16000
        audio = np.zeros(sr * 10, dtype=np.float32)
        audio[sr:3 * sr] = 1.0          # speaker A: silence + noise pattern
        audio[5 * sr:7 * sr] = -1.0     # speaker B: opposite signal

        def fake_embed(chunk):
            mean = float(chunk.mean().item())
            return np.array([[mean, 1.0]], dtype=np.float32)

        pipeline = type("FakePipeline", (), {"_inferences": {"_embedding": fake_embed}})()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            wav_path = tmp_path / "speaker.wav"
            wavfile.write(str(wav_path), sr, (audio * 32767).astype(np.int16))
            turns = [
                _turn("A", 1.0, 3.0),
                _turn("B", 5.0, 7.0),
                _turn("A", 7.5, 8.0),
            ]
            context = build_speaker_embedding_context(wav_path, turns, pipeline=pipeline)

        self.assertIsNotNone(context)
        self.assertEqual(len(context["embeddings"]), 3)
        self.assertIsNotNone(context["embeddings"][2])  # short turn still embedded
        self.assertEqual(set(context["centroids"]), {"A", "B"})

    def test_empty_turns_returns_none(self):
        self.assertIsNone(build_speaker_embedding_context(Path("unused.wav"), []))

    def test_missing_audio_returns_none(self):
        pipeline = type("FakePipeline", (), {"_inferences": {"_embedding": lambda chunk: None}})()
        turns = [_turn("A", 0.0, 2.0), _turn("B", 2.0, 4.0)]
        self.assertIsNone(build_speaker_embedding_context(Path("nonexistent.wav"), turns, pipeline=pipeline))


if __name__ == "__main__":
    unittest.main()
