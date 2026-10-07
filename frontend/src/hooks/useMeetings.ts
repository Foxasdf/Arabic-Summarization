import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import { Meeting } from '../types';

export function useMeetings() {
  const [meetings, setMeetings] = useState<Meeting[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMeetings = useCallback(async () => {
    setIsLoading(true);
    try {
      const data = await api.listMeetings();
      setMeetings(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'فشل في تحميل الاجتماعات');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMeetings();
  }, [fetchMeetings]);

  const deleteMeeting = async (id: string) => {
    try {
      await api.deleteMeeting(id);
      setMeetings((prev) => prev.filter((m) => m.id !== id));
    } catch (err) {
      throw err;
    }
  };

  return {
    meetings,
    isLoading,
    error,
    fetchMeetings,
    deleteMeeting,
  };
}
