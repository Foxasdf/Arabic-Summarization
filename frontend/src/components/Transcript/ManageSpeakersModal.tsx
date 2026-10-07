import React, { useState, useEffect, useMemo } from 'react';
import { Users, X, Check, MessageSquare, Loader2 } from 'lucide-react';
import { TranscriptSegment } from '../../types';
import { formatDefaultSpeakerName } from './SpeakerBadge';
import { formatTime } from '../../utils/formatters';

interface ManageSpeakersModalProps {
  isOpen: boolean;
  onClose: () => void;
  segments: TranscriptSegment[];
  speakers: Record<string, string>;
  onSave: (updatedSpeakers: Record<string, string>) => Promise<void>;
}

export const ManageSpeakersModal: React.FC<ManageSpeakersModalProps> = ({
  isOpen,
  onClose,
  segments,
  speakers,
  onSave,
}) => {
  const [localSpeakers, setLocalSpeakers] = useState<Record<string, string>>({});
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Group speaker samples
  const speakerData = useMemo(() => {
    const map = new Map<string, { samples: { text: string; start: number }[]; count: number }>();
    
    segments.forEach(seg => {
      const spk = seg.speaker;
      if (!map.has(spk)) {
        map.set(spk, { samples: [], count: 0 });
      }
      const data = map.get(spk)!;
      data.count += 1;
      // Keep up to 3 good length samples
      if (data.samples.length < 3 && seg.text.trim().length > 10) {
        data.samples.push({ text: seg.text, start: seg.start });
      } else if (data.samples.length === 0 && seg.text.trim().length > 0) {
        data.samples.push({ text: seg.text, start: seg.start });
      }
    });

    return Array.from(map.entries()).map(([speakerId, info]) => ({
      speakerId,
      samples: info.samples,
      count: info.count,
    }));
  }, [segments]);

  // Sync initial state when modal opens
  useEffect(() => {
    if (isOpen) {
      const initial: Record<string, string> = {};
      speakerData.forEach(({ speakerId }) => {
        initial[speakerId] = speakers[speakerId] || formatDefaultSpeakerName(speakerId);
      });
      setLocalSpeakers(initial);
      setError(null);
    }
  }, [isOpen, speakerData, speakers]);

  if (!isOpen) return null;

  const handleChange = (speakerId: string, value: string) => {
    setLocalSpeakers(prev => ({
      ...prev,
      [speakerId]: value,
    }));
  };

  const handleSave = async () => {
    setIsSaving(true);
    setError(null);
    try {
      await onSave(localSpeakers);
      onClose();
    } catch (err: any) {
      setError(err?.message || 'فشل حفظ أسماء المتحدثين');
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs animate-in fade-in duration-200" dir="rtl">
      <div 
        className="bg-white rounded-2xl shadow-xl border border-slate-200 w-full max-w-2xl max-h-[85vh] flex flex-col overflow-hidden animate-in zoom-in-95 duration-200"
        role="dialog"
        aria-modal="true"
      >
        {/* Header */}
        <div className="p-5 border-b border-slate-200 flex items-center justify-between bg-slate-50/70">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-100 text-indigo-700 flex items-center justify-center shadow-xs">
              <Users className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-slate-900 text-lg">إدارة وتسمية المتحدثين</h3>
              <p className="text-xs text-slate-500">عدّل أسماء المتحدثين للتعرف عليهم بسهولة في النص والملخص الذكي</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {error && (
          <div className="p-3 bg-rose-50 text-rose-600 text-sm font-medium text-center border-b border-rose-100">
            {error}
          </div>
        )}

        {/* Speakers List */}
        <div className="flex-1 overflow-y-auto p-6 space-y-5">
          {speakerData.length === 0 ? (
            <div className="text-center py-8 text-slate-400">
              لا يوجد متحدثون مسجلون في هذا الاجتماع
            </div>
          ) : (
            speakerData.map(({ speakerId, samples, count }, index) => {
              const defaultName = formatDefaultSpeakerName(speakerId);
              return (
                <div 
                  key={speakerId}
                  className="bg-slate-50 border border-slate-200/80 rounded-xl p-4.5 space-y-3 hover:border-indigo-200 transition-colors"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <span className="w-6 h-6 rounded-full bg-indigo-600 text-white text-xs font-bold flex items-center justify-center">
                        {index + 1}
                      </span>
                      <span className="text-xs font-semibold text-slate-500 bg-slate-200/70 px-2 py-0.5 rounded">
                        {speakerId}
                      </span>
                      <span className="text-xs text-slate-400">
                        ({count} مداخلة)
                      </span>
                    </div>

                    <div className="flex-1 sm:max-w-xs">
                      <input
                        type="text"
                        value={localSpeakers[speakerId] ?? ''}
                        onChange={(e) => handleChange(speakerId, e.target.value)}
                        placeholder={`مثال: ${defaultName}`}
                        className="w-full px-3 py-1.5 text-sm font-medium border border-slate-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-white"
                      />
                    </div>
                  </div>

                  {/* Utterance Samples */}
                  {samples.length > 0 && (
                    <div className="bg-white/80 rounded-lg p-3 border border-slate-200/60 text-xs text-slate-600 space-y-1.5">
                      <div className="flex items-center gap-1.5 text-slate-400 font-semibold text-[11px] mb-1">
                        <MessageSquare className="w-3 h-3" />
                        <span>عينات مما قاله هذا المتحدث:</span>
                      </div>
                      {samples.map((s, idx) => (
                        <div key={idx} className="flex items-start gap-2 text-slate-700">
                          <span className="text-slate-400 tabular-nums shrink-0 mt-0.5">
                            [{formatTime(s.start)}]
                          </span>
                          <span className="line-clamp-2 italic">
                            "{s.text}"
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-slate-200 flex items-center justify-end gap-3 bg-slate-50/50">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-slate-600 hover:bg-slate-200/60 font-medium text-sm transition-colors"
          >
            إلغاء
          </button>
          <button
            type="button"
            onClick={handleSave}
            disabled={isSaving}
            className="bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white px-5 py-2 rounded-xl font-semibold text-sm transition-colors flex items-center gap-2 shadow-sm"
          >
            {isSaving ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Check className="w-4 h-4" />
            )}
            {isSaving ? 'جاري الحفظ...' : 'حفظ الأسماء'}
          </button>
        </div>
      </div>
    </div>
  );
};
