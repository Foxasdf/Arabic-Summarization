import React, { useState, useRef, useEffect } from 'react';
import { Pencil, Check, X } from 'lucide-react';

const SPEAKER_COLORS = [
  'bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100',
  'bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100',
  'bg-purple-50 text-purple-700 border-purple-200 hover:bg-purple-100',
  'bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-100',
  'bg-rose-50 text-rose-700 border-rose-200 hover:bg-rose-100',
  'bg-indigo-50 text-indigo-700 border-indigo-200 hover:bg-indigo-100',
  'bg-teal-50 text-teal-700 border-teal-200 hover:bg-teal-100',
  'bg-cyan-50 text-cyan-700 border-cyan-200 hover:bg-cyan-100',
];

interface SpeakerBadgeProps {
  speakerId: string;
  displayName?: string;
  editable?: boolean;
  onRename?: (speakerId: string, newName: string) => void;
}

export const formatDefaultSpeakerName = (speakerId: string): string => {
  const match = speakerId.match(/\d+/);
  if (match) {
    const num = parseInt(match[0], 10) + 1;
    return `المتحدث ${num.toString().padStart(2, '0')}`;
  }
  return speakerId;
};

export const SpeakerBadge: React.FC<SpeakerBadgeProps> = ({
  speakerId,
  displayName,
  editable = true,
  onRename,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [name, setName] = useState(displayName || formatDefaultSpeakerName(speakerId));
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setName(displayName || formatDefaultSpeakerName(speakerId));
  }, [displayName, speakerId]);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  const getSpeakerColorIndex = (id: string) => {
    const match = id.match(/\d+/);
    if (match) {
      return parseInt(match[0], 10) % SPEAKER_COLORS.length;
    }
    let hash = 0;
    for (let i = 0; i < id.length; i++) {
      hash = id.charCodeAt(i) + ((hash << 5) - hash);
    }
    return Math.abs(hash) % SPEAKER_COLORS.length;
  };

  const colorClass = SPEAKER_COLORS[getSpeakerColorIndex(speakerId)];

  const handleSave = () => {
    const trimmed = name.trim();
    if (trimmed && trimmed !== displayName && onRename) {
      onRename(speakerId, trimmed);
    } else if (!trimmed) {
      setName(displayName || formatDefaultSpeakerName(speakerId));
    }
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      handleSave();
    } else if (e.key === 'Escape') {
      setName(displayName || formatDefaultSpeakerName(speakerId));
      setIsEditing(false);
    }
  };

  if (isEditing && editable) {
    return (
      <div className="inline-flex items-center gap-1 bg-white border border-indigo-300 rounded-lg px-2 py-0.5 shadow-sm text-xs" dir="rtl">
        <input
          ref={inputRef}
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          onKeyDown={handleKeyDown}
          onBlur={handleSave}
          className="text-xs font-semibold text-slate-800 bg-transparent outline-none w-28 text-right"
          placeholder="اسم المتحدث..."
        />
        <button
          onClick={handleSave}
          className="text-emerald-600 hover:text-emerald-700 p-0.5 rounded"
          title="حفظ"
        >
          <Check className="w-3 h-3" />
        </button>
        <button
          onClick={() => {
            setName(displayName || formatDefaultSpeakerName(speakerId));
            setIsEditing(false);
          }}
          className="text-rose-500 hover:text-rose-700 p-0.5 rounded"
          title="إلغاء"
        >
          <X className="w-3 h-3" />
        </button>
      </div>
    );
  }

  const currentDisplayName = displayName || formatDefaultSpeakerName(speakerId);

  return (
    <button
      type="button"
      onClick={() => editable && setIsEditing(true)}
      className={`group inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold border transition-all ${colorClass} ${
        editable ? 'cursor-pointer hover:shadow-xs' : 'cursor-default'
      }`}
      title={editable ? 'انقر لتعديل اسم المتحدث' : undefined}
    >
      <span>{currentDisplayName}</span>
      {editable && (
        <Pencil className="w-2.5 h-2.5 opacity-40 group-hover:opacity-100 transition-opacity" />
      )}
    </button>
  );
};
