export interface TranscriptSegment {
  speaker: string;
  text: string;
  start: number;
  end: number;
}

export interface Meeting {
  id: string;
  filename: string;
  upload_time: string;
  duration_seconds: number;
  status: 'uploaded' | 'processing' | 'transcribed' | 'completed' | 'error';
  transcript?: TranscriptSegment[];
  transcript_json?: TranscriptSegment[];
  speakers?: Record<string, string>;
  speakers_json?: Record<string, string>;
  full_text?: string;
  summary?: string;
  summary_text?: string;
}

export interface UpdateSpeakersPayload {
  speakers: Record<string, string>;
}

export interface ModelInfo {
  key: string;
  name: string;
  description: string;
}

export interface ProcessingStatus {
  step: string;
  progress: number;
  message: string;
  error?: string;
  result?: TranscriptSegment[];
}

export type AppView = 'home' | 'processing' | 'meeting' | 'falcon-chat';

export type MainTab = 'meetings' | 'falcon-chat';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface AttachedDocument {
  id: string;
  filename: string;
  extension: string;
  content: string;
  char_count: number;
  word_count: number;
  preview: string;
  upload_time?: string;
}

export interface ChatRequest {
  message: string;
  history?: { role: string; content: string }[];
  documents?: { filename: string; content: string }[];
  model_key?: string;
}

export interface ChatResponse {
  reply: string;
  model_used: string;
  processing_time_seconds: number;
}


