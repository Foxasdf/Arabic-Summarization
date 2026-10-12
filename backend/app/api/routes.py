import uuid
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..config import settings
from ..models.registry import registry
from ..services.audio_processor import (
    convert_to_wav,
    process_dual_profile_audio,
    get_audio_duration,
    estimate_processing_time,
)
from ..services.transcription import transcribe_audio, transcribe_with_speakers
from ..services.diarization import diarize, diarization_backend_name
from ..services.alignment import align_timestamps
from ..services.summarization import summarize_meeting
from ..services.chat import chat_with_transcript, direct_falcon_chat
from ..services.document_parser import extract_text_from_file
from ..services.hardware import get_hardware_profile
from ..services.storage import save_meeting, get_meeting, list_meetings, delete_meeting
from ..schemas.responses import (
    UploadResponse,
    MeetingListItem,
    ModelInfoSchema,
    ChatRequest,
    ChatResponse,
    FileUploadResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Only one heavy model job at a time – prevents two GGUFs being loaded together on small machines.
_model_lock = asyncio.Lock()


def default_speaker_name(speaker_id: str) -> str:
    try:
        return f"المتحدث {int(speaker_id.split('_')[-1]) + 1:02d}"
    except ValueError:
        return speaker_id


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _run_with_progress(func, *args, **kwargs):
    """Run blocking `func(*args, on_progress=cb, **kwargs)` in a thread, yielding (pct,msg) events then ('result', value)."""
    loop = asyncio.get_running_loop()
    q: asyncio.Queue = asyncio.Queue()

    def cb(pct, msg):
        loop.call_soon_threadsafe(q.put_nowait, ("progress", (pct, msg)))

    def worker():
        try:
            try:
                res = func(*args, on_progress=cb, **kwargs)
            except TypeError:
                res = func(*args, cb, **kwargs)
            loop.call_soon_threadsafe(q.put_nowait, ("result", res))
        except Exception as e:  # noqa
            logger.exception("Background job failed")
            loop.call_soon_threadsafe(q.put_nowait, ("error", e))

    loop.run_in_executor(None, worker)
    while True:
        try:
            kind, val = await asyncio.wait_for(q.get(), timeout=10.0)
            yield kind, val
            if kind in ("result", "error"):
                return
        except asyncio.TimeoutError:
            yield "heartbeat", None


@router.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix.lower() or ".bin"
    temp_path = settings.UPLOAD_DIR / f"{file_id}{ext}"
    with open(temp_path, "wb") as buffer:
        while chunk := await file.read(1024 * 1024):
            buffer.write(chunk)

    duration = get_audio_duration(temp_path)
    await save_meeting({"id": file_id, "filename": file.filename, "duration_seconds": duration, "status": "uploaded"})
    return UploadResponse(file_id=file_id, filename=file.filename, duration_seconds=duration,
                          estimated_minutes=estimate_processing_time(duration))


@router.get("/transcribe/{file_id}")
async def transcribe_endpoint(
    file_id: str,
    num_speakers: Optional[int] = None,
    denoise_mode: Optional[str] = None,
    asr_model: Optional[str] = None,
):
    meeting = await get_meeting(file_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    async def event_generator():
        try:
            files = [f for f in settings.UPLOAD_DIR.glob(f"{file_id}.*") if not f.name.endswith(".wav")] \
                or list(settings.UPLOAD_DIR.glob(f"{file_id}.*"))
            if not files:
                yield _sse({"error": "File not found"})
                return
            wav_path = settings.UPLOAD_DIR / f"{file_id}.16k.wav"

            async with _model_lock:
                denoise_names = {
                    "dynamic": "ديناميكي",
                    "auto": "ديناميكي",
                    "low": "منخفضة",
                    "moderate": "متوسطة",
                    "medium": "متوسطة",
                    "high": "عالية",
                    "aggressive": "عالية",
                    "none": "لا معالجة",
                    "off": "لا معالجة",
                    "لا معالجة": "لا معالجة",
                    "بدون معالجة": "لا معالجة",
                }
                mode_str = denoise_names.get((denoise_mode or "").lower(), denoise_mode or "")
                if mode_str == "لا معالجة":
                    denoise_label = " (بدون معالجة)"
                elif mode_str:
                    denoise_label = f" (عزل: {mode_str})"
                else:
                    denoise_label = ""
                spk_wav_path = settings.UPLOAD_DIR / f"{file_id}.speaker.16k.wav"
                yield _sse({"step": "preprocessing", "progress": 3, "message": f"جاري تجهيز وتنقية الملف الصوتي{denoise_label}..."})
                ok, err = await process_dual_profile_audio(files[0], wav_path, spk_wav_path, denoise_mode=denoise_mode)
                if not ok:
                    raise RuntimeError(f"فشل تحويل وتنقية الملف الصوتي (ffmpeg): {err or ''}")
                yield _sse({"step": "preprocessing", "progress": 8, "message": "تم تحويل وتنقية وتجهيز مسارات الصوت بنجاح"})

                backend_label = "Pyannote 3.1 (NVIDIA GPU)"
                yield _sse({"step": "diarization", "progress": 10,
                            "message": f"جاري فصل وتحديد المتحدثين ({backend_label})..."})
                turns = None
                # Diarization strictly consumes canonical unprocessed speaker audio
                async for kind, val in _run_with_progress(diarize, spk_wav_path, num_speakers):
                    if kind == "progress":
                        yield _sse({"step": "diarization", "progress": val[0], "message": val[1]})
                    elif kind == "heartbeat":
                        yield ": keepalive\n\n"
                    elif kind == "error":
                        raise val
                    else:
                        turns = val
                n_spk = len({t['speaker'] for t in turns}) if turns else 1
                yield _sse({"step": "diarization", "progress": 25, "message": f"تم اكتشاف {n_spk} متحدث/متحدثين بنجاح عبر Pyannote"})

                # Explicitly flush PyTorch CUDA cache so ASR has 100% of GPU VRAM available
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                # Speaker-aware ASR:
                # Transcribes each speaker turn using the chosen ASR engine (Cohere/Audar/Whisper),
                # guaranteeing exact speaker attribution with zero word mixing or fragmentation
                yield _sse({"step": "asr", "progress": 25, "message": "جاري تفريغ الصوت وتنسيق المتحدثين..."})
                segments = None
                async for kind, val in _run_with_progress(
                    transcribe_with_speakers, wav_path, turns, model_identifier=asr_model, progress_range=(25, 95)
                ):
                    if kind == "progress":
                        yield _sse({"step": "asr", "progress": val[0], "message": val[1]})
                    elif kind == "heartbeat":
                        yield ": keepalive\n\n"
                    elif kind == "error":
                        raise val
                    else:
                        segments = val

                if not segments:
                    segments = []

            yield _sse({"step": "alignment", "progress": 97, "message": "جاري حفظ نتائج التفريغ..."})
            old = meeting.get("speakers_json") or {}
            speakers = {s["speaker"]: old.get(s["speaker"], default_speaker_name(s["speaker"])) for s in segments}
            await save_meeting({"id": file_id, "transcript_json": segments, "diarization_json": turns,
                                "speakers_json": speakers, "status": "transcribed"})
            try:
                wav_path.unlink()
            except OSError:
                pass
            yield _sse({"step": "done", "progress": 100, "result": segments, "speakers": speakers})
        except Exception as e:
            logger.exception("Transcription failed")
            await save_meeting({"id": file_id, "status": "error"})
            yield _sse({"error": str(e)})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


class SpeakersUpdate(BaseModel):
    speakers: Dict[str, str]


@router.put("/meetings/{meeting_id}/speakers")
async def update_speakers(meeting_id: str, body: SpeakersUpdate):
    meeting = await get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    speakers = dict(meeting.get("speakers_json") or {})
    for sid, name in body.speakers.items():
        name = (name or "").strip()[:80]
        speakers[sid] = name or default_speaker_name(sid)
    await save_meeting({"id": meeting_id, "speakers_json": speakers})
    return {"speakers": speakers}


@router.get("/summarize/{meeting_id}")
async def summarize_stream(meeting_id: str, model_key: str = "falcon-h1-7b-instruct"):
    """SSE summarization with progress (long meetings take minutes on CPU)."""
    meeting = await get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    segments = meeting.get("transcript_json") or []
    if not segments:
        raise HTTPException(status_code=400, detail="No transcript found for meeting")
    speakers = meeting.get("speakers_json") or {}

    async def gen():
        start = time.time()
        try:
            async with _model_lock:
                summary = None
                async for kind, val in _run_with_progress(summarize_meeting, segments, speakers, model_key):
                    if kind == "progress":
                        yield _sse({"step": "summarize", "progress": val[0], "message": val[1]})
                    elif kind == "heartbeat":
                        yield ": keepalive\n\n"
                    elif kind == "error":
                        raise val
                    else:
                        summary = val
            await save_meeting({"id": meeting_id, "summary_text": summary, "status": "completed"})
            yield _sse({"step": "done", "progress": 100, "summary": summary,
                        "model_used": model_key, "processing_time_seconds": round(time.time() - start, 1)})
        except Exception as e:
            logger.exception("Summarization failed")
            yield _sse({"error": str(e)})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/summarize")
async def summarize_endpoint(meeting_id: str, model_key: str = "falcon-h1-7b-instruct"):
    """Blocking variant kept for compatibility / scripts."""
    meeting = await get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    if not meeting.get("transcript_json"):
        raise HTTPException(status_code=400, detail="No transcript found for meeting")
    start = time.time()
    async with _model_lock:
        summary = await asyncio.to_thread(summarize_meeting, meeting["transcript_json"],
                                          meeting.get("speakers_json") or {}, model_key)
    await save_meeting({"id": meeting_id, "summary_text": summary, "status": "completed"})
    return {"summary": summary, "model_used": model_key, "processing_time_seconds": time.time() - start}


@router.post("/meetings/{meeting_id}/chat", response_model=ChatResponse)
async def chat_meeting_endpoint(meeting_id: str, body: ChatRequest):
    """
    Chat with Falcon LLM about the meeting.
    Extracts answers strictly from the verbatim transcribed text (transcript_json),
    deliberately excluding summary_text.
    """
    meeting = await get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")

    segments = meeting.get("transcript_json") or []
    if not segments:
        raise HTTPException(status_code=400, detail="لا يتوفر نص مفرغ لهذا الاجتماع لسؤاله عنه")

    speakers = meeting.get("speakers_json") or {}
    model_key = body.model_key or "falcon-h1-7b-instruct"
    history_dicts = [
        h.model_dump() if hasattr(h, "model_dump") else h.dict()
        for h in (body.history or [])
    ]

    start = time.time()
    async with _model_lock:
        try:
            reply = await asyncio.to_thread(
                chat_with_transcript,
                segments,
                speakers,
                body.message,
                history_dicts,
                model_key,
            )
        except Exception as e:
            logger.exception("Chat generation failed")
            raise HTTPException(status_code=500, detail=f"فشل توليد الإجابة من Falcon LLM: {str(e)}")

    return ChatResponse(
        reply=reply,
        model_used=model_key,
        processing_time_seconds=round(time.time() - start, 2),
    )


@router.post("/chat", response_model=ChatResponse)
async def general_chat_endpoint(body: ChatRequest):
    """
    Direct chat endpoint with Falcon LLM (general assistant), independent of any meeting.
    Supports optional attached documents.
    """
    if not body.message or not body.message.strip():
        raise HTTPException(status_code=400, detail="الرسالة لا يمكن أن تكون فارغة")

    model_key = body.model_key or "falcon-h1-7b-instruct"
    history_dicts = [
        h.model_dump() if hasattr(h, "model_dump") else h.dict()
        for h in (body.history or [])
    ]
    documents_dicts = [
        d.model_dump() if hasattr(d, "model_dump") else d.dict()
        for d in (body.documents or [])
    ]

    start = time.time()
    async with _model_lock:
        try:
            reply = await asyncio.to_thread(
                direct_falcon_chat,
                body.message,
                history_dicts,
                documents_dicts,
                None,
                model_key,
            )
        except Exception as e:
            logger.exception("General Falcon chat generation failed")
            raise HTTPException(status_code=500, detail=f"فشل توليد الإجابة من Falcon LLM: {str(e)}")

    return ChatResponse(
        reply=reply,
        model_used=model_key,
        processing_time_seconds=round(time.time() - start, 2),
    )


@router.post("/chat/upload-file", response_model=FileUploadResponse)
async def chat_upload_file_endpoint(file: UploadFile = File(...)):
    """
    Extract readable text from an uploaded document (PDF, Word docx, Excel xlsx, TXT, CSV, Code, etc.)
    for analysis and chatting in Falcon H1.
    """
    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(status_code=400, detail="الملف المرفوع فارغ")

        parsed = await asyncio.to_thread(
            extract_text_from_file,
            content_bytes,
            file.filename or "uploaded_file.txt",
        )

        return FileUploadResponse(
            filename=parsed["filename"],
            extension=parsed["extension"],
            char_count=parsed["char_count"],
            word_count=parsed["word_count"],
            preview=parsed["preview"],
            content=parsed["text"],
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception("Failed to extract text from uploaded document")
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء قراءة الملف: {str(e)}")




@router.get("/meetings", response_model=list[MeetingListItem])
async def get_meetings_list():
    return [MeetingListItem(**m) for m in await list_meetings()]


@router.get("/meetings/{meeting_id}")
async def get_meeting_detail(meeting_id: str):
    meeting = await get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@router.delete("/meetings/{meeting_id}")
async def delete_meeting_endpoint(meeting_id: str):
    await delete_meeting(meeting_id)
    for f in settings.UPLOAD_DIR.glob(f"{meeting_id}*"):
        try:
            if f.is_file():
                f.unlink()
        except Exception:
            pass
    return {"status": "success"}


@router.get("/models", response_model=list[ModelInfoSchema])
async def list_models(type: Optional[str] = None):
    if type:
        return [ModelInfoSchema(**m.__dict__) for m in registry.get_models_by_type(type)]
    return [ModelInfoSchema(**m.__dict__) for m in registry.get_models_by_type("llm")]


@router.get("/models/asr", response_model=list[ModelInfoSchema])
async def list_asr_models():
    return [ModelInfoSchema(**m.__dict__) for m in registry.get_models_by_type("asr")]


@router.get("/system")
async def system_info():
    hw = await asyncio.to_thread(get_hardware_profile)
    llm_file = settings.MODELS_DIR / settings.LLM_GGUF_FILENAME
    size = llm_file.stat().st_size / 1024**3 if llm_file.exists() else 4.0
    return {"hardware": hw.as_dict(), "asr": hw.asr_params(), "llm": hw.llm_params(size),
            "diarization_backend": diarization_backend_name()}


@router.get("/health")
async def health_check():
    return {"status": "healthy"}
