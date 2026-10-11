import { useState, useCallback } from 'react';
import { api } from '../services/api';
import { ProcessingStatus, TranscriptSegment } from '../types';

export function useTranscription() {
  const [status, setStatus] = useState<ProcessingStatus | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TranscriptSegment[] | null>(null);

  const startTranscription = useCallback(async (
    fileId: string,
    numSpeakers?: number,
    denoiseMode?: string,
    asrModel?: string
  ) => {
    setIsProcessing(true);
    setError(null);
    setStatus(null);
    setResult(null);

    try {
      const meetingResult = await api.transcribe(
        fileId,
        (newStatus) => {
          setStatus(newStatus);
        },
        numSpeakers,
        denoiseMode,
        asrModel
      );
      setResult(meetingResult);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'حدث خطأ غير معروف');
    } finally {
      setIsProcessing(false);
    }
  }, []);

  return {
    status,
    isProcessing,
    error,
    result,
    startTranscription,
  };
}
