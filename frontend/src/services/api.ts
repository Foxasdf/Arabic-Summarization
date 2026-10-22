import { Meeting, ModelInfo, ProcessingStatus, TranscriptSegment, ChatResponse, AttachedDocument } from '../types';

const API_BASE = '/api';

export const api = {
  uploadAudio(
    file: File,
    onProgress?: (percent: number) => void
  ): Promise<{ file_id: string; filename: string; duration_seconds: number; estimated_minutes: number }> {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      const formData = new FormData();
      formData.append('file', file);

      xhr.open('POST', `${API_BASE}/upload`);

      if (xhr.upload && onProgress) {
        xhr.upload.onprogress = (event) => {
          if (event.lengthComputable) {
            const percent = Math.round((event.loaded / event.total) * 100);
            onProgress(percent);
          }
        };
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText));
          } catch {
            reject(new Error('استجابة غير صحيحة من الخادم'));
          }
        } else {
          try {
            const errData = JSON.parse(xhr.responseText);
            reject(new Error(errData?.detail || `فشل رفع الملف (${xhr.status})`));
          } catch {
            reject(new Error(`فشل رفع الملف (${xhr.status})`));
          }
        }
      };

      xhr.onerror = () => reject(new Error('فشل الاتصال بالخادم أثناء رفع الملف'));
      xhr.ontimeout = () => reject(new Error('انتهت مهلة رفع الملف'));

      xhr.send(formData);
    });
  },

  transcribe(
    fileId: string,
    onProgress: (status: ProcessingStatus) => void,
    numSpeakers?: number,
    denoiseMode?: string,
    asrModel?: string,
    signal?: AbortSignal
  ): Promise<TranscriptSegment[]> {
    return new Promise((resolve, reject) => {
      if (signal?.aborted) {
        return reject(new Error('تم إلغاء عملية التفريغ'));
      }

      const params = new URLSearchParams();
      if (numSpeakers) params.append('num_speakers', numSpeakers.toString());
      if (denoiseMode) params.append('denoise_mode', denoiseMode);
      if (asrModel) params.append('asr_model', asrModel);
      const qs = params.toString();
      const url = qs ? `${API_BASE}/transcribe/${fileId}?${qs}` : `${API_BASE}/transcribe/${fileId}`;
      const eventSource = new EventSource(url);

      let errorTimer: ReturnType<typeof setTimeout> | null = null;

      const abortHandler = () => {
        if (errorTimer) clearTimeout(errorTimer);
        eventSource.close();
        reject(new Error('تم إلغاء عملية التفريغ'));
      };

      if (signal) {
        signal.addEventListener('abort', abortHandler, { once: true });
      }

      const cleanup = () => {
        if (errorTimer) {
          clearTimeout(errorTimer);
          errorTimer = null;
        }
        if (signal) {
          signal.removeEventListener('abort', abortHandler);
        }
        eventSource.close();
      };

      eventSource.onopen = () => {
        if (errorTimer) {
          clearTimeout(errorTimer);
          errorTimer = null;
        }
      };

      eventSource.onmessage = (event) => {
        if (errorTimer) {
          clearTimeout(errorTimer);
          errorTimer = null;
        }
        try {
          const data: ProcessingStatus = JSON.parse(event.data);
          onProgress(data);

          // Handle errors sent via SSE
          if (data.error) {
            cleanup();
            reject(new Error(data.error));
            return;
          }

          // Backend sends step: 'done' when complete
          if (data.step === 'done') {
            cleanup();
            if (data.result && data.result.length > 0) {
              resolve(data.result);
            } else {
              reject(new Error('لم يتم استخراج أي نصوص مفرغة من الملف الصوتي'));
            }
            return;
          }
        } catch (err) {
          console.error('Failed to parse SSE data', err);
        }
      };

      eventSource.onerror = () => {
        // If readyState is CONNECTING (0), browser is automatically attempting to reconnect
        if (eventSource.readyState === EventSource.CONNECTING) {
          if (!errorTimer) {
            // Allow 60s grace period for heavy operations / network re-handshakes
            errorTimer = setTimeout(() => {
              if (eventSource.readyState !== EventSource.OPEN) {
                cleanup();
                reject(new Error('انقطع الاتصال بخادم المعالجة'));
              }
            }, 60000);
          }
          return;
        }

        if (eventSource.readyState === EventSource.CLOSED) {
          if (errorTimer) clearTimeout(errorTimer);
          cleanup();
          reject(new Error('فشل أو انقطع الاتصال بخادم المعالجة'));
        }
      };
    });
  },

  async updateSpeakers(meetingId: string, speakers: Record<string, string>): Promise<{ speakers: Record<string, string> }> {
    const response = await fetch(`${API_BASE}/meetings/${meetingId}/speakers`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ speakers }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => null);
      throw new Error(errData?.detail || `فشل تحديث أسماء المتحدثين: ${response.statusText}`);
    }

    return response.json();
  },

  async summarize(meetingId: string, modelKey?: string): Promise<{ summary: string; model_used: string; processing_time_seconds: number }> {
    const params = new URLSearchParams({ meeting_id: meetingId });
    if (modelKey) {
      params.append('model_key', modelKey);
    }
    
    const response = await fetch(`${API_BASE}/summarize?${params.toString()}`, {
      method: 'POST',
    });
    
    if (!response.ok) {
      const errData = await response.json().catch(() => null);
      throw new Error(errData?.detail || `فشل التلخيص: ${response.statusText}`);
    }
    
    return response.json();
  },

  summarizeStream(meetingId: string, onProgress: (data: { step: string; progress: number; message?: string; summary?: string; error?: string }) => void): Promise<{ summary: string }> {
    return new Promise((resolve, reject) => {
      const eventSource = new EventSource(`${API_BASE}/summarize/${meetingId}`);
      
      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          onProgress(data);
          
          if (data.step === 'done' && data.summary) {
            eventSource.close();
            resolve({ summary: data.summary });
          }

          if (data.error) {
            eventSource.close();
            reject(new Error(data.error));
          }
        } catch (err) {
          console.error('Failed to parse SSE data', err);
        }
      };
      
      eventSource.onerror = () => {
        eventSource.close();
        // Fallback to standard POST if EventSource fails
        api.summarize(meetingId)
          .then(res => resolve({ summary: res.summary }))
          .catch(reject);
      };
    });
  },

  async listMeetings(): Promise<Meeting[]> {
    const response = await fetch(`${API_BASE}/meetings`);
    if (!response.ok) {
      throw new Error(`فشل جلب قائمة الاجتماعات: ${response.statusText}`);
    }
    return response.json();
  },

  async getMeeting(id: string): Promise<Meeting> {
    const response = await fetch(`${API_BASE}/meetings/${id}`);
    if (!response.ok) {
      throw new Error(`فشل جلب بيانات الاجتماع: ${response.statusText}`);
    }
    const data = await response.json();
    // Normalize backend field names to frontend field names
    return {
      ...data,
      transcript: data.transcript_json || data.transcript || [],
      speakers: data.speakers_json || data.speakers || {},
      summary: data.summary_text || data.summary || null,
    };
  },

  async deleteMeeting(id: string): Promise<void> {
    const response = await fetch(`${API_BASE}/meetings/${id}`, {
      method: 'DELETE',
    });
    if (!response.ok) {
      throw new Error(`فشل حذف الاجتماع: ${response.statusText}`);
    }
  },

  async listModels(): Promise<ModelInfo[]> {
    const response = await fetch(`${API_BASE}/models`);
    if (!response.ok) {
      throw new Error(`فشل جلب قائمة النماذج: ${response.statusText}`);
    }
    return response.json();
  },

  async chatWithTranscript(
    meetingId: string,
    message: string,
    history: { role: string; content: string }[] = [],
    modelKey?: string
  ): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE}/meetings/${meetingId}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        history,
        model_key: modelKey || 'falcon-h1-7b-instruct',
      }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => null);
      throw new Error(errData?.detail || `فشل التواصل مع Falcon LLM: ${response.statusText}`);
    }

    return response.json();
  },

  async uploadChatDocument(file: File): Promise<AttachedDocument> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE}/chat/upload-file`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => null);
      throw new Error(errData?.detail || `فشل قراءة الملف: ${response.statusText}`);
    }

    const data = await response.json();
    return {
      id: `doc-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`,
      filename: data.filename,
      extension: data.extension,
      content: data.content,
      char_count: data.char_count,
      word_count: data.word_count,
      preview: data.preview,
      upload_time: new Date().toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' }),
    };
  },

  async chat(
    message: string,
    history: { role: string; content: string }[] = [],
    documents: { filename: string; content: string }[] = [],
    modelKey?: string
  ): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        history,
        documents,
        model_key: modelKey || 'falcon-h1-7b-instruct',
      }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => null);
      throw new Error(errData?.detail || `فشل التواصل مع Falcon LLM: ${response.statusText}`);
    }

    return response.json();
  }
};


