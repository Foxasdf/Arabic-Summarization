import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useTranscription } from '../../hooks/useTranscription';
import { CheckCircle2, Circle, Loader2, FileAudio, Type, Users, Sparkles } from 'lucide-react';

interface ProcessingViewProps {
  fileId: string;
  numSpeakers?: number;
  asrModel?: string;
  denoiseMode?: string;
  onComplete: () => void;
}

export const ProcessingView: React.FC<ProcessingViewProps> = ({ fileId, numSpeakers, asrModel, denoiseMode, onComplete }) => {
  const { status, result, error, startTranscription } = useTranscription();
  const [maxProgress, setMaxProgress] = useState(0);

  const asrLabel = 
    asrModel?.includes('cohere') ? 'Cohere Transcribe' :
    (asrModel?.includes('whisper') || asrModel?.includes('dialectal')) ? 'Whisper Turbo' :
    'Audar ASR';

  const isNoDenoise = denoiseMode === 'none' || denoiseMode === 'off' || denoiseMode === 'لا معالجة';
  const denoiseLabel =
    isNoDenoise ? 'بدون معالجة' :
    denoiseMode === 'low' ? 'منخفضة' :
    denoiseMode === 'moderate' ? 'متوسطة' :
    denoiseMode === 'high' ? 'عالية' : 'ديناميكي';

  const STEPS = [
    { 
      id: 'preprocessing', 
      label: isNoDenoise ? 'تجهيز الملف الصوتي (بدون معالجة)' : `تنقية الصوت (${denoiseLabel}) وتجهيزه`, 
      icon: FileAudio 
    },
    { id: 'diarization', label: 'فصل وتحديد المتحدثين', icon: Users },
    { id: 'asr', label: `تحويل الصوت إلى نص (${asrLabel})`, icon: Type },
    { id: 'alignment', label: 'مطابقة النتائج وحفظ الاجتماع', icon: Sparkles },
  ];

  useEffect(() => {
    startTranscription(fileId, numSpeakers, denoiseMode, asrModel);
  }, [fileId, numSpeakers, denoiseMode, asrModel, startTranscription]);

  useEffect(() => {
    if (result) {
      setTimeout(() => onComplete(), 1000);
    }
  }, [result, onComplete]);

  // Keep progress strictly monotonic so it never jumps backward
  const rawProgress = status?.progress || 0;
  useEffect(() => {
    if (rawProgress > maxProgress) {
      setMaxProgress(rawProgress);
    }
  }, [rawProgress, maxProgress]);

  if (error) {
    return (
      <div className="bg-rose-50 border border-rose-200 rounded-2xl p-8 text-center text-rose-700">
        <div className="w-16 h-16 bg-rose-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <span className="text-3xl">⚠️</span>
        </div>
        <h3 className="text-xl font-bold mb-2">حدث خطأ أثناء المعالجة</h3>
        <p>{error}</p>
        <button 
          onClick={() => {
            setMaxProgress(0);
            startTranscription(fileId, numSpeakers, denoiseMode, asrModel);
          }}
          className="mt-6 bg-rose-600 hover:bg-rose-700 text-white px-6 py-2 rounded-lg font-medium transition-colors cursor-pointer"
        >
          إعادة المحاولة
        </button>
      </div>
    );
  }

  const currentProgress = Math.max(maxProgress, rawProgress);
  const currentStepMessage = status?.message || 'جاري التحضير...';
  
  // Determine active step index based on backend SSE step names in actual pipeline order:
  // preprocessing (0) -> diarization (1) -> asr (2) -> alignment (3) -> done (4)
  let activeStepIndex = 0;
  if (status?.step === 'preprocessing') activeStepIndex = 0;
  else if (status?.step === 'diarization') activeStepIndex = 1;
  else if (status?.step === 'asr') activeStepIndex = 2;
  else if (status?.step === 'alignment') activeStepIndex = 3;
  else if (status?.step === 'done') activeStepIndex = 4;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-white rounded-3xl shadow-sm border border-slate-200 p-8 md:p-10"
    >
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-slate-900 mb-2">جاري معالجة الاجتماع</h2>
        <p className="text-slate-500 font-medium text-sm">يتم معالجة الصوت وتفريغه وتحديد المتحدثين بالذكاء الاصطناعي</p>
      </div>

      <div className="bg-slate-50/80 border border-slate-200/70 rounded-2xl p-5 mb-8">
        <div className="flex items-center justify-between mb-3 text-sm">
          <div className="flex items-center gap-2 font-medium text-slate-700 min-w-0">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-indigo-600 animate-pulse shrink-0" />
            <span className="truncate">{currentStepMessage}</span>
          </div>
          <span className="font-bold text-indigo-600 text-base shrink-0 mr-3">
            {Math.min(100, Math.round(currentProgress))}%
          </span>
        </div>
        <div className="h-2.5 w-full bg-slate-200/80 rounded-full overflow-hidden">
          <motion.div 
            className="h-full bg-indigo-600 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: `${Math.min(100, Math.max(0, currentProgress))}%` }}
            transition={{ ease: "easeOut", duration: 0.3 }}
          />
        </div>
      </div>

      <div className="space-y-4">
        {STEPS.map((step, index) => {
          const isCompleted = index < activeStepIndex || status?.step === 'done';
          const isActive = index === activeStepIndex && status?.step !== 'done';
          const isWaiting = index > activeStepIndex;
          
          const Icon = step.icon;

          return (
            <div 
              key={step.id} 
              className={`flex items-center gap-4 p-4 rounded-2xl transition-colors duration-300 ${
                isActive ? 'bg-indigo-50/60 border border-indigo-100 shadow-xs' : 'border border-transparent'
              }`}
            >
              <div className={`
                w-12 h-12 rounded-xl flex items-center justify-center shrink-0 transition-colors
                ${isCompleted ? 'bg-emerald-100 text-emerald-600' : 
                  isActive ? 'bg-indigo-100 text-indigo-600' : 
                  'bg-slate-100 text-slate-400'}
              `}>
                {isCompleted ? <CheckCircle2 className="w-6 h-6" /> : 
                 isActive ? <Loader2 className="w-6 h-6 animate-spin" /> : 
                 <Icon className="w-6 h-6" />}
              </div>
              
              <div className="flex-1 min-w-0">
                <h4 className={`font-bold text-base ${
                  isCompleted ? 'text-slate-900' : 
                  isActive ? 'text-indigo-900' : 
                  'text-slate-400'
                }`}>
                  {step.label}
                </h4>
                {isActive && (
                  <p className="text-sm text-indigo-600/80 mt-1 truncate">
                    {currentStepMessage}
                  </p>
                )}
              </div>
              
              {isWaiting && (
                <div className="text-slate-300 shrink-0">
                  <Circle className="w-5 h-5" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </motion.div>
  );
};
