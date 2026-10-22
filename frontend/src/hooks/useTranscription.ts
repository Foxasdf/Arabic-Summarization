import { useState, useCallback, useRef, useEffect } from 'react';
import { api } from '../services/api';
import { ProcessingStatus, TranscriptSegment } from '../types';

export function useTranscription() {
  const [status, setStatus] = useState<ProcessingStatus | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TranscriptSegment[] | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const cancelTranscription = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsProcessing(false);
  }, []);

  const startTranscription = useCallback(async (
    fileId: string,
    numSpeakers?: number,
    denoiseMode?: string,
    asrModel?: string
  ) => {
    // Abort any previous pending transcription request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsProcessing(true);
    setError(null);
    setStatus(null);
    setResult(null);

    try {
      const meetingResult = await api.transcribe(
        fileId,
        (newStatus) => {
          if (!controller.signal.aborted) {
            setStatus(newStatus);
          }
        },
        numSpeakers,
        denoiseMode,
        asrModel,
        controller.signal
      );
      if (!controller.signal.aborted) {
        setResult(meetingResult);
      }
    } catch (err: any) {
      if (!controller.signal.aborted) {
        setError(err instanceof Error ? err.message : 'حدث خطأ غير معروف أثناء التفريغ');
      }
    } finally {
      if (abortControllerRef.current === controller) {
        setIsProcessing(false);
        abortControllerRef.current = null;
      }
    }
  }, []);

  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
    };
  }, []);

  return {
    status,
    isProcessing,
    error,
    result,
    startTranscription,
    cancelTranscription,
  };
}
