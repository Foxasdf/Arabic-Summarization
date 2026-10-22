import unittest
import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.storage import init_db, save_meeting, get_meeting, list_meetings, delete_meeting


class TestStorage(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db = Path(self.temp_dir.name) / "test_meetings.db"
        self.orig_db = settings.DB_PATH
        settings.DB_PATH = self.test_db
        await init_db()

    async def asyncTearDown(self):
        settings.DB_PATH = self.orig_db
        self.temp_dir.cleanup()

    async def test_save_and_get_meeting(self):
        meeting_data = {
            "id": "meeting-123",
            "filename": "meeting_test.mp3",
            "duration_seconds": 125.5,
            "status": "uploaded",
            "transcript_json": [
                {"speaker": "متحدث 1", "text": "مرحباً بكم", "start": 0.0, "end": 1.0}
            ]
        }
        saved_id = await save_meeting(meeting_data)
        self.assertEqual(saved_id, "meeting-123")

        fetched = await get_meeting("meeting-123")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["filename"], "meeting_test.mp3")
        self.assertEqual(fetched["duration_seconds"], 125.5)
        self.assertEqual(len(fetched["transcript_json"]), 1)
        self.assertEqual(fetched["transcript_json"][0]["text"], "مرحباً بكم")

    async def test_update_meeting_preserves_upload_time(self):
        meeting_data = {
            "id": "meeting-456",
            "filename": "team_sync.m4a",
            "duration_seconds": 300.0,
            "status": "uploaded"
        }
        await save_meeting(meeting_data)
        original = await get_meeting("meeting-456")
        original_time = original["upload_time"]
        self.assertIsNotNone(original_time)

        # Update meeting with summary
        meeting_data["summary_text"] = "تم الاتفاق على خطة العمل"
        meeting_data["status"] = "completed"
        await save_meeting(meeting_data)

        updated = await get_meeting("meeting-456")
        self.assertEqual(updated["summary_text"], "تم الاتفاق على خطة العمل")
        self.assertEqual(updated["status"], "completed")
        self.assertEqual(updated["upload_time"], original_time)

    async def test_list_and_delete_meeting(self):
        await save_meeting({"id": "m1", "filename": "m1.wav", "status": "uploaded"})
        await save_meeting({"id": "m2", "filename": "m2.wav", "status": "uploaded"})

        meetings = await list_meetings()
        self.assertEqual(len(meetings), 2)

        await delete_meeting("m1")
        meetings_after = await list_meetings()
        self.assertEqual(len(meetings_after), 1)
        self.assertEqual(meetings_after[0]["id"], "m2")

    async def test_init_db_cleans_corrupt_transcribed_and_stale_processing(self):
        # Save a meeting falsely marked as 'transcribed' with empty transcript
        await save_meeting({"id": "corrupt-1", "filename": "c1.wav", "status": "transcribed", "transcript_json": []})
        # Save a meeting that was in-flight when server stopped
        await save_meeting({"id": "processing-1", "filename": "p1.wav", "status": "processing"})
        # Save a legitimate transcribed meeting
        await save_meeting({
            "id": "valid-1",
            "filename": "v1.wav",
            "status": "transcribed",
            "transcript_json": [{"speaker": "SPEAKER_00", "text": "مرحبا", "start": 0.0, "end": 1.0}]
        })

        # Run init_db again to trigger cleanup migration
        await init_db()

        c1 = await get_meeting("corrupt-1")
        self.assertEqual(c1["status"], "error")

        p1 = await get_meeting("processing-1")
        self.assertEqual(p1["status"], "error")

        v1 = await get_meeting("valid-1")
        self.assertEqual(v1["status"], "transcribed")

    async def test_save_meeting_refuses_transcribed_without_transcript(self):
        # Attempt to save a meeting as transcribed with empty transcript list
        await save_meeting({
            "id": "guard-1",
            "filename": "g1.wav",
            "status": "transcribed",
            "transcript_json": []
        })
        m = await get_meeting("guard-1")
        self.assertEqual(m["status"], "error")

        # In list_meetings, it should also be reported as error
        meetings = await list_meetings()
        guard_item = next(x for x in meetings if x["id"] == "guard-1")
        self.assertEqual(guard_item["status"], "error")


if __name__ == "__main__":
    unittest.main()
