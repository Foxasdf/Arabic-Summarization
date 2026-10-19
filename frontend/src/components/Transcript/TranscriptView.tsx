import React, { useState, useEffect, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Download, Sparkles, Loader2, Users } from 'lucide-react';
import { TranscriptSegment } from '../../types';
import { SpeakerBadge, formatDefaultSpeakerName } from './SpeakerBadge';
import { ManageSpeakersModal } from './ManageSpeakersModal';
import { formatTime } from '../../utils/formatters';
import { api } from '../../services/api';

interface TranscriptViewProps {
  segments: TranscriptSegment[];
  speakers?: Record<string, string>;
  meetingId: string;
  onSummaryGenerated: (summary: string) => void;
  onSpeakersUpdated?: (speakers: Record<string, string>) => void;
}

export const TranscriptView: React.FC<TranscriptViewProps> = ({ 
  segments, 
  speakers = {},
  meetingId,
  onSummaryGenerated,
  onSpeakersUpdated,
}) => {
  const [isSummarizing, setIsSummarizing] = useState(false);
  const [summaryMessage, setSummaryMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [localSpeakers, setLocalSpeakers] = useState<Record<string, string>>(speakers);

  const displaySegments = useMemo(() => {
    if (!segments || segments.length === 0) return [];
    const punctuation = ['.', '،', '؟', '!', '؛', ',', '?', ':', '…'];
    const merged: TranscriptSegment[] = [];

    for (const seg of segments) {
      const text = (seg.text || '').trim();
      if (!text) continue;
      const last = merged[merged.length - 1];
      if (last && last.speaker === seg.speaker) {
        const gap = seg.start - last.end;
        last.end = Math.max(last.end, seg.end);
        const lastText = last.text.trim();
        const hasPunctuation = punctuation.some(p => lastText.endsWith(p));
        if (gap > 2.0 && !hasPunctuation) {
          last.text = `${lastText}، ${text}`;
        } else {
          last.text = `${lastText} ${text}`;
        }
      } else {
        merged.push({ ...seg, text });
      }
    }
    return merged;
  }, [segments]);

  useEffect(() => {
    setLocalSpeakers(speakers || {});
  }, [speakers]);

  const handleInlineRename = async (speakerId: string, newName: string) => {
    const updated = { ...localSpeakers, [speakerId]: newName };
    setLocalSpeakers(updated);
    try {
      const res = await api.updateSpeakers(meetingId, updated);
      onSpeakersUpdated?.(res.speakers);
    } catch (err) {
      console.error('Failed to update speaker name inline', err);
      setError('فشل حفظ اسم المتحدث');
    }
  };

  const handleModalSave = async (updatedSpeakers: Record<string, string>) => {
    const res = await api.updateSpeakers(meetingId, updatedSpeakers);
    setLocalSpeakers(res.speakers);
    onSpeakersUpdated?.(res.speakers);
  };

  const handleSummarize = async () => {
    setIsSummarizing(true);
    setError(null);
    setSummaryMessage('بدء التلخيص الذكي...');
    try {
      const result = await api.summarizeStream(meetingId, (status) => {
        if (status.message) {
          setSummaryMessage(status.message);
        }
      });
      onSummaryGenerated(result.summary);
      setSummaryMessage(null);
    } catch (err: any) {
      setError(err?.message || 'حدث خطأ أثناء التلخيص. الرجاء المحاولة مرة أخرى.');
    } finally {
      setIsSummarizing(false);
    }
  };

  const getSpeakerDisplayName = (speakerId: string): string => {
    return localSpeakers[speakerId] || formatDefaultSpeakerName(speakerId);
  };

  const handleDownload = () => {
    const text = displaySegments
      .map(s => `[${formatTime(s.start)}] ${getSpeakerDisplayName(s.speaker)}: ${s.text}`)
      .join('\n\n');
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `transcript-${meetingId}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <>
      <div className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden flex flex-col max-h-[800px]">
        <div className="p-4 border-b border-slate-200 flex flex-wrap items-center justify-between gap-3 bg-slate-50/50">
          <div className="flex items-center gap-3">
            <h3 className="font-bold text-slate-800 text-lg">النص الكامل</h3>
            <span className="text-xs text-slate-500 bg-slate-200/60 px-2 py-0.5 rounded-full font-medium">
              {displaySegments.length} مقطع
            </span>
          </div>
          
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsModalOpen(true)}
              className="p-2 text-indigo-700 bg-indigo-50 hover:bg-indigo-100 rounded-lg transition-colors flex items-center gap-1.5 text-sm font-semibold border border-indigo-200/60"
              title="تسمية وإدارة المتحدثين"
            >
              <Users className="w-4 h-4" />
              <span>إدارة وتسمية المتحدثين</span>
            </button>

            <button 
              onClick={handleDownload}
              className="p-2 text-slate-600 hover:text-indigo-600 hover:bg-indigo-50/50 rounded-lg transition-colors flex items-center gap-1.5 text-sm font-medium border border-slate-200"
              title="تصدير النص مع الأسماء"
            >
              <Download className="w-4 h-4" />
              <span className="hidden sm:inline">تصدير</span>
            </button>
            
            <button 
              onClick={handleSummarize}
              disabled={isSummarizing}
              className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white px-4 py-2 rounded-lg transition-colors flex items-center gap-2 text-sm font-semibold shadow-sm"
            >
              {isSummarizing ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Sparkles className="w-4 h-4" />
              )}
              {isSummarizing ? (summaryMessage || 'جاري التلخيص...') : 'تلخيص الاجتماع'}
            </button>
          </div>
        </div>

        {error && (
          <div className="p-3 bg-rose-50 text-rose-600 text-sm font-medium text-center border-b border-rose-100">
            {error}
          </div>
        )}

        <div className="flex-1 overflow-y-auto p-6 space-y-6 scroll-smooth">
          {displaySegments.map((segment, index) => (
            <motion.div 
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: Math.min(index * 0.03, 0.4) }}
              className="flex gap-4 group"
            >
              <div className="w-16 shrink-0 text-left pt-1">
                <span className="text-xs font-medium text-slate-400 tabular-nums">
                  {formatTime(segment.start)}
                </span>
              </div>
              
              <div className="flex-1">
                <div className="mb-1.5">
                  <SpeakerBadge 
                    speakerId={segment.speaker} 
                    displayName={getSpeakerDisplayName(segment.speaker)}
                    onRename={handleInlineRename}
                    editable={true}
                  />
                </div>
                <p className="text-slate-700 leading-relaxed text-[15px]">
                  {segment.text}
                </p>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      <ManageSpeakersModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        segments={displaySegments}
        speakers={localSpeakers}
        onSave={handleModalSave}
      />
    </>
  );
};
