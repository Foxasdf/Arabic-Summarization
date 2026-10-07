from pydantic import BaseModel
from typing import List, Optional

class UploadResponse(BaseModel):
    file_id: str
    filename: str
    duration_seconds: float
    estimated_minutes: float

class TranscriptSegment(BaseModel):
    speaker: str
    text: str
    start: float
    end: float

class TranscriptResponse(BaseModel):
    segments: List[TranscriptSegment]
    full_text: str
    duration: float

class SummaryResponse(BaseModel):
    summary: str
    model_used: str
    processing_time_seconds: float

class MeetingListItem(BaseModel):
    id: str
    filename: Optional[str] = None
    upload_time: Optional[str] = None
    duration_seconds: Optional[float] = 0.0
    status: str = "uploaded"

class MeetingDetail(BaseModel):
    id: str
    filename: Optional[str] = None
    upload_time: Optional[str] = None
    duration_seconds: Optional[float] = 0.0
    transcript_json: Optional[List[TranscriptSegment]] = []
    diarization_json: Optional[List[dict]] = []
    summary_text: Optional[str] = ""
    status: str = "uploaded"

class ModelInfoSchema(BaseModel):
    key: str
    name: str
    description: str
    model_type: str

class ErrorResponse(BaseModel):
    detail: str
    error_code: str


class ChatMessageSchema(BaseModel):
    role: str
    content: str


class DocumentAttachment(BaseModel):
    filename: str
    content: str


class FileUploadResponse(BaseModel):
    filename: str
    extension: str
    char_count: int
    word_count: int
    preview: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessageSchema]] = []
    documents: Optional[List[DocumentAttachment]] = []
    model_key: Optional[str] = "falcon-h1-7b-instruct"


class ChatResponse(BaseModel):
    reply: str
    model_used: str
    processing_time_seconds: float

