import unittest
import sys
from pathlib import Path

import numpy as np

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parents[1]
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.diarization import (
    _relabel_turns_from_embeddings,
    _refine_speaker_labels,
)


def _turn(speaker, start, end):
    return {"speaker": speaker, "start": start, "end": end}


class TestSpeakerLabelRefinement(unittest.TestCase):
    def test_empty_turns(self):
        self.assertEqual(_relabel_turns_from_embeddings([], []), [])

    def test_single_speaker_unchanged(self):
        turns = [_turn("A", 0.0, 3.0), _turn("A", 3.0, 6.0)]
        embeddings = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
        refined = _relabel_turns_from_embeddings(turns, embeddings)
        self.assertEqual([t["speaker"] for t in refined], ["A", "A"])

    def test_short_turn_never_relabeled(self):
        turns = [
            _turn("A", 0.0, 3.0),
            _turn("A", 3.0, 6.0),
            _turn("B", 6.0, 9.0),
            _turn("B", 9.0, 12.0),
            _turn("A", 12.0, 12.5),  # short, embedding clearly matches B
        ]
        embeddings = [
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ]
        refined = _relabel_turns_from_embeddings(turns, embeddings)
        self.assertEqual(refined[-1]["speaker"], "A")

    def test_long_turn_relabeled_to_nearest_centroid(self):
        turns = [
            _turn("A", 0.0, 3.0),
            _turn("A", 3.0, 6.0),
            _turn("B", 6.0, 9.0),
            _turn("B", 9.0, 12.0),
            _turn("A", 12.0, 15.0),  # mislabelled long turn, audio is B
        ]
        embeddings = [
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ]
        refined = _relabel_turns_from_embeddings(turns, embeddings)
        self.assertEqual(refined[-1]["speaker"], "B")

    def test_margin_prevents_ambiguous_relabel(self):
        turns = [
            _turn("A", 0.0, 3.0),
            _turn("A", 3.0, 6.0),
            _turn("B", 6.0, 9.0),
            _turn("B", 9.0, 12.0),
            _turn("A", 12.0, 15.0),
        ]
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.02])
        b = b / np.linalg.norm(b)
        ambiguous = np.array([1.0, 0.01])
        ambiguous = ambiguous / np.linalg.norm(ambiguous)
        embeddings = [a, a, b, b, ambiguous]
        refined = _relabel_turns_from_embeddings(turns, embeddings, margin=0.01)
        self.assertEqual(refined[-1]["speaker"], "A")

    def test_speaker_without_long_centroid_is_not_relabelled(self):
        turns = [
            _turn("A", 0.0, 3.0),
            _turn("B", 3.0, 6.0),
            _turn("C", 6.0, 6.5),  # only a short turn for C, no centroid
            _turn("C", 6.5, 9.5),  # long turn, embedding looks like B
        ]
        embeddings = [
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            None,
            np.array([0.0, 1.0]),
        ]
        refined = _relabel_turns_from_embeddings(turns, embeddings)
        self.assertEqual(refined[-1]["speaker"], "C")

    def test_input_not_mutated(self):
        turns = [
            _turn("A", 0.0, 3.0),
            _turn("A", 3.0, 6.0),
            _turn("B", 6.0, 9.0),
            _turn("B", 9.0, 12.0),
            _turn("A", 12.0, 15.0),
        ]
        embeddings = [
            np.array([1.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        ]
        _relabel_turns_from_embeddings(turns, embeddings)
        self.assertEqual(turns[-1]["speaker"], "A")


class TestRefineSpeakerLabelsAudioPath(unittest.TestCase):
    def test_refines_long_turns_and_preserves_short_turns(self):
        sr = 16000
        audio = np.zeros(sr * 16, dtype=np.float32)
        # Speaker B's audio is a constant 1.0; speaker A's audio is silence.
        for start, end in [(3, 6), (9, 12), (12, 15), (15, 15.5)]:
            audio[int(start * sr):int(end * sr)] = 1.0

        def fake_embed(chunk):
            mean = float(chunk.mean().item())
            return np.array([[mean, 1.0]], dtype=np.float32)

        pipeline = type("FakePipeline", (), {"_inferences": {"_embedding": fake_embed}})()

        turns = [
            _turn("A", 0.0, 3.0),
            _turn("B", 3.0, 6.0),
            _turn("A", 6.0, 9.0),
            _turn("B", 9.0, 12.0),
            _turn("A", 12.0, 15.0),  # mislabelled, audio is B
            _turn("A", 15.0, 15.5),  # short interjection, left alone
        ]
        refined = _refine_speaker_labels(turns, audio, pipeline)

        self.assertEqual(refined[4]["speaker"], "B")
        self.assertEqual(refined[5]["speaker"], "A")
        self.assertEqual(turns[4]["speaker"], "A")

    def test_missing_embedding_model_returns_turns_unchanged(self):
        turns = [_turn("A", 0.0, 3.0), _turn("B", 3.0, 6.0)]
        pipeline = type("FakePipeline", (), {"_inferences": {}})()
        refined = _refine_speaker_labels(turns, np.zeros(16000 * 6, dtype=np.float32), pipeline)
        self.assertEqual(refined, turns)


if __name__ == "__main__":
    unittest.main()
