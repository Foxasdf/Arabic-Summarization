import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.chat import (
    format_seconds,
    _display_speaker,
    format_transcript_lines,
    retrieve_relevant_context,
    chat_with_transcript,
    direct_falcon_chat,
    _clean_arabic_response,
    _clean_general_response,
)


class TestFalconChatService(unittest.TestCase):

    def test_format_seconds(self):
        self.assertEqual(format_seconds(45), "00:45")
        self.assertEqual(format_seconds(125), "02:05")
        self.assertEqual(format_seconds(3665), "01:01:05")

    def test_display_speaker(self):
        speakers = {"SPEAKER_00": "د. ياسر الحزيمي"}
        self.assertEqual(_display_speaker("SPEAKER_00", speakers), "د. ياسر الحزيمي")
        self.assertEqual(_display_speaker("SPEAKER_01", speakers), "المتحدث 02")

    def test_format_transcript_lines(self):
        segments = [
            {"speaker": "SPEAKER_00", "text": "العلاقات تحتاج إلى وعي وتفهم.", "start": 10.5, "end": 15.0},
            {"speaker": "SPEAKER_01", "text": "وما هي أولى خطوات هذا الوعي؟", "start": 16.0, "end": 20.0},
        ]
        speakers = {"SPEAKER_00": "د. ياسر الحزيمي", "SPEAKER_01": "المحاور"}
        lines = format_transcript_lines(segments, speakers)
        self.assertEqual(len(lines), 2)
        self.assertIn("[00:10] د. ياسر الحزيمي: العلاقات تحتاج إلى وعي وتفهم.", lines[0])
        self.assertIn("[00:16] المحاور: وما هي أولى خطوات هذا الوعي؟", lines[1])

    def test_chat_grounded_in_transcript_only(self):
        """
        Verify that chat_with_transcript passes the transcript lines to the engine,
        and that the system prompt strictly instructs grounding in the transcript.
        """
        segments = [
            {"speaker": "SPEAKER_00", "text": "مشروعنا الجديد سينطلق في شهر أكتوبر القادم.", "start": 5.0, "end": 10.0}
        ]
        speakers = {"SPEAKER_00": "المدير"}

        mock_engine = MagicMock()
        mock_engine.is_loaded.return_value = True
        mock_engine.n_ctx = 8192
        mock_engine.count_tokens.side_effect = lambda t: len(t) // 2
        mock_engine.chat.return_value = "سينطلق المشروع في شهر أكتوبر القادم حسب ما ذكر المدير."

        with patch("app.services.chat.LLMEngine", return_value=mock_engine):
            reply = chat_with_transcript(
                segments=segments,
                speakers=speakers,
                question="متى سينطلق المشروع؟",
                history=[],
            )

            self.assertIn("أكتوبر", reply)
            mock_engine.chat.assert_called_once()
            call_args = mock_engine.chat.call_args[0][0]
            
            # The prompt messages sent to the engine must contain the transcript text
            full_prompt_text = " ".join(m.get("content", "") for m in call_args)
            self.assertIn("مشروعنا الجديد سينطلق في شهر أكتوبر القادم", full_prompt_text)
            self.assertIn("المدير", full_prompt_text)
            self.assertIn("مصدرك الوحيد والنهائي للإجابة هو 'النص المفرغ من الصوت'", full_prompt_text)

    def test_empty_segments_raises_value_error(self):
        with self.assertRaises(ValueError):
            chat_with_transcript(segments=[], speakers={}, question="هل يوجد اجتماع؟")

    def test_empty_question_raises_value_error(self):
        with self.assertRaises(ValueError):
            chat_with_transcript(
                segments=[{"speaker": "s1", "text": "مرحبا", "start": 0, "end": 1}],
                speakers={},
                question="   ",
            )

    def test_api_chat_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)

        with patch("app.api.routes.get_meeting") as mock_get_meeting, \
             patch("app.api.routes.chat_with_transcript") as mock_chat_fn:
            
            mock_get_meeting.return_value = {
                "id": "meeting-abc",
                "filename": "demo.mp3",
                "transcript_json": [
                    {"speaker": "SPEAKER_00", "text": "نحن نعمل على تطوير التطبيق.", "start": 0, "end": 5}
                ],
                "speakers_json": {"SPEAKER_00": "سارة"},
                "summary_text": "ملخص تجريبي يجب ألا يُستخدم في المحادثة",
            }
            mock_chat_fn.return_value = "تعمل سارة على تطوير التطبيق."

            response = client.post(
                "/api/meetings/meeting-abc/chat",
                json={"message": "على ماذا تعمل سارة؟", "history": []}
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["reply"], "تعمل سارة على تطوير التطبيق.")
            self.assertEqual(data["model_used"], "falcon-h1-7b-instruct")
            self.assertIn("processing_time_seconds", data)

            # Crucial verification: chat_with_transcript was called with segments, NOT summary_text
            mock_chat_fn.assert_called_once()
            called_segments = mock_chat_fn.call_args[0][0]
            self.assertEqual(called_segments, [
                {"speaker": "SPEAKER_00", "text": "نحن نعمل على تطوير التطبيق.", "start": 0, "end": 5}
            ])

    def test_api_chat_meeting_not_found(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        with patch("app.api.routes.get_meeting", return_value=None):
            response = client.post(
                "/api/meetings/non-existent/chat",
                json={"message": "أي سؤال؟"}
            )
            self.assertEqual(response.status_code, 404)

    def test_api_chat_missing_transcript(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        with patch("app.api.routes.get_meeting", return_value={"id": "m1", "transcript_json": []}):
            response = client.post(
                "/api/meetings/m1/chat",
                json={"message": "أي سؤال؟"}
            )
            self.assertEqual(response.status_code, 400)

    def test_clean_arabic_response_removes_chinese_characters(self):
        raw_output = "في بداية التسجيل الصوتي... التقريبه هو 54:16. 月份的."
        cleaned = _clean_arabic_response(raw_output)
        self.assertNotIn("月份的", cleaned)
        self.assertIn("54:16.", cleaned)
        self.assertIn("في بداية التسجيل الصوتي...", cleaned)
        self.assertFalse(cleaned.endswith(".."))

    def test_clean_arabic_response_removes_english_hallucinations(self):
        raw_output = "هذا هو الجواب بالعربية.\n\nEnglish translation:\nThis is the answer in English."
        cleaned = _clean_arabic_response(raw_output)
        self.assertEqual(cleaned, "هذا هو الجواب بالعربية.")

    def test_chat_uses_temperature_point_two(self):
        segments = [{"speaker": "SPEAKER_00", "text": "مرحباً بكم في الاجتماع.", "start": 0.0, "end": 2.0}]
        speakers = {"SPEAKER_00": "المدير"}

        mock_engine = MagicMock()
        mock_engine.is_loaded.return_value = True
        mock_engine.n_ctx = 8192
        mock_engine.count_tokens.side_effect = lambda t: len(t) // 2
        mock_engine.chat.return_value = "مرحباً."

        with patch("app.services.chat.LLMEngine", return_value=mock_engine):
            chat_with_transcript(
                segments=segments,
                speakers=speakers,
                question="ما هي التحية؟",
            )
            mock_engine.chat.assert_called_once()
            _, kwargs = mock_engine.chat.call_args
            self.assertEqual(kwargs.get("temperature"), 0.2)

    def test_direct_falcon_chat_success(self):
        mock_engine = MagicMock()
        mock_engine.is_loaded.return_value = True
        mock_engine.n_ctx = 8192
        mock_engine.count_tokens.side_effect = lambda t: len(t) // 2
        mock_engine.chat.return_value = "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهدف لتمكين الآلات من محاكاة الذكاء البشري."

        with patch("app.services.chat.LLMEngine", return_value=mock_engine):
            reply = direct_falcon_chat(
                message="ما هو الذكاء الاصطناعي؟",
                history=[{"role": "user", "content": "مرحبا"}, {"role": "assistant", "content": "أهلا بك"}],
            )
            self.assertIn("الذكاء الاصطناعي", reply)
            mock_engine.chat.assert_called_once()
            call_messages = mock_engine.chat.call_args[0][0]
            # System prompt should identify Falcon H1
            self.assertIn("Falcon H1", call_messages[0]["content"])
            self.assertEqual(call_messages[-1]["content"], "ما هو الذكاء الاصطناعي؟")

    def test_direct_falcon_chat_empty_message(self):
        with self.assertRaises(ValueError):
            direct_falcon_chat(message="   ")

    def test_clean_general_response_preserves_code_and_cleans_cjk(self):
        raw = "إليك الكود المطلوب: ```python\nprint('hello world')\n```\nشكراً لك. 月份的"
        cleaned = _clean_general_response(raw)
        self.assertNotIn("月份的", cleaned)
        self.assertIn("```python\nprint('hello world')\n```", cleaned)
        self.assertIn("إليك الكود المطلوب:", cleaned)

    def test_api_general_chat_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        with patch("app.api.routes.direct_falcon_chat") as mock_direct_chat:
            mock_direct_chat.return_value = "مرحباً! أنا Falcon H1، كيف أساعدك اليوم؟"

            response = client.post(
                "/api/chat",
                json={"message": "مرحباً", "history": []}
            )

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["reply"], "مرحباً! أنا Falcon H1، كيف أساعدك اليوم؟")
            self.assertEqual(data["model_used"], "falcon-h1-7b-instruct")
            self.assertIn("processing_time_seconds", data)
            mock_direct_chat.assert_called_once()

    def test_api_general_chat_empty_message(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        response = client.post(
            "/api/chat",
            json={"message": "   "}
        )
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()


