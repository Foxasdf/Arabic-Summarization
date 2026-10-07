import io
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.document_parser import (
    extract_text_from_file,
    build_document_context_block,
    _decode_text_bytes,
)
from app.services.chat import direct_falcon_chat


class TestDocumentParser(unittest.TestCase):

    def test_decode_arabic_text_bytes(self):
        arabic_text = "بسم الله الرحمن الرحيم\nهذا تقرير تجريبي باللغة العربية."
        raw_bytes = arabic_text.encode("utf-8")
        decoded = _decode_text_bytes(raw_bytes)
        self.assertEqual(decoded, arabic_text)

    def test_extract_txt_file(self):
        text = "هذا ملف نصي يحتوي على 10 كلمات لاختبار استخراج النصوص."
        res = extract_text_from_file(text.encode("utf-8"), "notes.txt")
        self.assertEqual(res["filename"], "notes.txt")
        self.assertEqual(res["extension"], ".txt")
        self.assertIn("هذا ملف نصي", res["text"])
        self.assertGreater(res["word_count"], 5)
        self.assertIn("preview", res)

    def test_extract_code_file(self):
        code = "def hello():\n    print('مرحبا بالعالم')\n"
        res = extract_text_from_file(code.encode("utf-8"), "script.py")
        self.assertEqual(res["extension"], ".py")
        self.assertIn("def hello():", res["text"])

    def test_extract_empty_file_raises_error(self):
        with self.assertRaises(ValueError):
            extract_text_from_file(b"   ", "empty.txt")

    def test_build_document_context_block_fits_budget(self):
        docs = [
            {"filename": "doc1.txt", "content": "محتوى المستند الأول."},
            {"filename": "doc2.txt", "content": "محتوى المستند الثاني."},
        ]
        context = build_document_context_block(
            documents=docs,
            question="ما هو المحتوى؟",
            count_tokens_fn=lambda t: len(t) // 2,
            token_budget=2000,
        )
        self.assertIn("doc1.txt", context)
        self.assertIn("doc2.txt", context)
        self.assertIn("محتوى المستند الأول", context)
        self.assertIn("محتوى المستند الثاني", context)

    def test_build_document_context_block_exceeds_budget(self):
        # Create a large document with 10 sections
        large_sections = [
            f"--- صفحة {i} ---\nهذه فقرة تجريبية طويلة تتحدث عن الذكاء الاصطناعي في قطاع التعليم رقم {i}."
            for i in range(1, 15)
        ]
        large_content = "\n\n".join(large_sections)
        docs = [{"filename": "large_report.pdf", "content": large_content}]

        # Small budget that forces chunk ranking
        context = build_document_context_block(
            documents=docs,
            question="ماذا عن التعليم؟",
            count_tokens_fn=lambda t: len(t) // 2,
            token_budget=150,
        )
        self.assertIn("مقتطفات من محتوى الملفات المرفقة", context)
        self.assertIn("large_report.pdf", context)

    def test_direct_falcon_chat_with_documents(self):
        mock_engine = MagicMock()
        mock_engine.is_loaded.return_value = True
        mock_engine.n_ctx = 8192
        mock_engine.count_tokens.side_effect = lambda t: len(t) // 2
        mock_engine.chat.return_value = "استناداً إلى المستند المرفق، بلغت المبيعات مليون ريال."

        docs = [{"filename": "sales.csv", "content": "المبيعات: 1,000,000 ريال"}]

        with patch("app.services.chat.LLMEngine", return_value=mock_engine):
            reply = direct_falcon_chat(
                message="كم بلغت المبيعات في المستند؟",
                history=[],
                documents=docs,
            )
            self.assertIn("المبيعات", reply)
            mock_engine.chat.assert_called_once()
            call_messages = mock_engine.chat.call_args[0][0]
            # Ensure the document content was injected into the prompt
            all_content = " ".join(m["content"] for m in call_messages)
            self.assertIn("sales.csv", all_content)
            self.assertIn("1,000,000 ريال", all_content)

    def test_api_upload_file_endpoint(self):
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)
        file_content = "تقرير الأداء الفصلي: حقق الفريق نسبة إنجاز 95%.".encode("utf-8")

        response = client.post(
            "/api/chat/upload-file",
            files={"file": ("report.txt", io.BytesIO(file_content), "text/plain")}
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["filename"], "report.txt")
        self.assertEqual(data["extension"], ".txt")
        self.assertIn("95%", data["content"])
        self.assertGreater(data["word_count"], 3)


if __name__ == "__main__":
    unittest.main()
