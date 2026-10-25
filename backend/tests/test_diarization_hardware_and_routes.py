import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.hardware import HardwareProfile
from app.api.routes import _active_transcriptions


class TestHardwareDiarizationBatchParams(unittest.TestCase):
    """Verify hardware profile assigns safe batch sizes preventing Pyannote CUDA OOM."""

    def test_rtx_2080_ti_11gb_safe_batch_size(self):
        # 11GB RTX 2080 Ti profile from user's logs
        prof = HardwareProfile(
            cpu_threads=16,
            ram_total_gb=31.2,
            ram_available_gb=22.0,
            gpu_name="NVIDIA GeForce RTX 2080 Ti",
            vram_total_gb=11.0,
            vram_free_gb=10.0,
            llama_gpu_offload=True,
            tier="high",
        )
        params = prof.diarization_params()
        # Must NOT be 192/96 (which caused CUDA OOM on 11GB VRAM)
        self.assertEqual(params["segmentation_batch_size"], 64)
        self.assertEqual(params["embedding_batch_size"], 32)
        self.assertTrue(params["use_gpu"])

    def test_24gb_gpu_safe_batch_size(self):
        prof = HardwareProfile(
            cpu_threads=24,
            ram_total_gb=64.0,
            ram_available_gb=48.0,
            gpu_name="NVIDIA GeForce RTX 4090",
            vram_total_gb=24.0,
            vram_free_gb=22.0,
            llama_gpu_offload=True,
            tier="high",
        )
        params = prof.diarization_params()
        self.assertEqual(params["segmentation_batch_size"], 128)
        self.assertEqual(params["embedding_batch_size"], 64)

    def test_8gb_gpu_safe_batch_size(self):
        prof = HardwareProfile(
            cpu_threads=8,
            ram_total_gb=16.0,
            ram_available_gb=10.0,
            gpu_name="NVIDIA GeForce RTX 3070",
            vram_total_gb=8.0,
            vram_free_gb=7.0,
            llama_gpu_offload=True,
            tier="medium",
        )
        params = prof.diarization_params()
        self.assertEqual(params["segmentation_batch_size"], 48)
        self.assertEqual(params["embedding_batch_size"], 24)

    def test_low_free_vram_dynamic_downscale(self):
        # 11GB GPU but free VRAM is low (< 4GB)
        prof_under_4gb = HardwareProfile(
            cpu_threads=16,
            ram_total_gb=32.0,
            ram_available_gb=20.0,
            gpu_name="NVIDIA GeForce RTX 2080 Ti",
            vram_total_gb=11.0,
            vram_free_gb=3.5,
            llama_gpu_offload=True,
            tier="high",
        )
        params_4gb = prof_under_4gb.diarization_params()
        self.assertEqual(params_4gb["segmentation_batch_size"], 32)
        self.assertEqual(params_4gb["embedding_batch_size"], 16)

        # 11GB GPU but critically low free VRAM (< 2GB)
        prof_under_2gb = HardwareProfile(
            cpu_threads=16,
            ram_total_gb=32.0,
            ram_available_gb=20.0,
            gpu_name="NVIDIA GeForce RTX 2080 Ti",
            vram_total_gb=11.0,
            vram_free_gb=1.2,
            llama_gpu_offload=True,
            tier="high",
        )
        params_2gb = prof_under_2gb.diarization_params()
        self.assertEqual(params_2gb["segmentation_batch_size"], 16)
        self.assertEqual(params_2gb["embedding_batch_size"], 8)

    def test_cpu_fallback_batch_size(self):
        prof = HardwareProfile(
            cpu_threads=8,
            ram_total_gb=16.0,
            ram_available_gb=12.0,
            gpu_name=None,
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            llama_gpu_offload=False,
            tier="medium",
        )
        params = prof.diarization_params()
        self.assertEqual(params["segmentation_batch_size"], 32)
        self.assertEqual(params["embedding_batch_size"], 16)
        self.assertFalse(params["use_gpu"])


class TestTranscriptionLocking(unittest.IsolatedAsyncioTestCase):
    """Verify active transcription tracking to prevent duplicate concurrent jobs for the same file."""

    def test_active_transcriptions_set_tracking(self):
        test_file_id = "test-active-uuid-123"
        self.assertNotIn(test_file_id, _active_transcriptions)

        _active_transcriptions.add(test_file_id)
        self.assertIn(test_file_id, _active_transcriptions)

        # Simulate cleanup in finally block
        _active_transcriptions.discard(test_file_id)
        self.assertNotIn(test_file_id, _active_transcriptions)

    async def test_run_with_progress_cancellation_stops_worker(self):
        import asyncio
        import time
        from app.api.routes import _run_with_progress

        worker_stopped = False

        def long_job(on_progress=None):
            nonlocal worker_stopped
            try:
                for i in range(10):
                    time.sleep(0.05)
                    if on_progress:
                        on_progress(i * 10, f"step {i}")
            except InterruptedError:
                worker_stopped = True
                raise

        # Start generator and break after first progress event to simulate client disconnect
        gen = _run_with_progress(long_job)
        async for kind, val in gen:
            if kind == "progress":
                break

        # Closing generator triggers finally block with cancel_event.set()
        await gen.aclose()
        # Give worker a moment to observe cancellation
        await asyncio.sleep(0.1)
        self.assertTrue(worker_stopped)

    def test_diarization_has_asyncio_imported(self):
        import app.services.diarization as diar_mod
        self.assertTrue(hasattr(diar_mod, "asyncio"))

    def test_vectorized_vad_chunk_planning_long_audio(self):
        import numpy as np
        from app.models.asr.base import BaseASREngine

        sr = 16000
        # 120 seconds of synthetic int16 audio with silent pauses
        audio = np.random.randint(-3000, 3000, size=sr * 120, dtype=np.int16)
        # Insert 0.5s silence around 25s, 50s, 75s, 100s
        for sec in (25, 50, 75, 100):
            audio[sec * sr : int((sec + 0.5) * sr)] = 0

        chunks = BaseASREngine.plan_vad_chunks(
            audio_data=audio,
            sr=sr,
            total_duration=120.0,
            target_chunk_len=25.0,
            min_chunk_len=18.0,
            max_chunk_len=30.0,
            overlap=2.5,
            min_silence_sec=0.3,
        )
        self.assertGreaterEqual(len(chunks), 4)
        self.assertEqual(chunks[0]["start"], 0.0)


if __name__ == "__main__":
    unittest.main()
