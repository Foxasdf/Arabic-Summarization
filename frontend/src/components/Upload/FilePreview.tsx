import React from 'react';
import { motion } from 'framer-motion';
import { FileAudio, X, Upload, Loader2 } from 'lucide-react';
import { formatFileSize } from '../../utils/formatters';

interface FilePreviewProps {
  file: File;
  onUpload: (numSpeakers?: number, asrModel?: string, denoiseMode?: string) => void;
  onCancel: () => void;
  isUploading?: boolean;
  uploadProgress?: number;
}

export const FilePreview: React.FC<FilePreviewProps> = ({
  file,
  onUpload,
  onCancel,
  isUploading = false,
  uploadProgress = 0,
}) => {
  const [numSpeakers, setNumSpeakers] = React.useState<number>(0);
  const [asrModel, setAsrModel] = React.useState<string>('cohere-transcribe-arabic');
  const [denoiseMode, setDenoiseMode] = React.useState<string>('none');
  const [isLocalSubmitting, setIsLocalSubmitting] = React.useState<boolean>(false);


  // Synchronous lock to prevent any double-click within milliseconds
  const isBusy = Boolean(isUploading || isLocalSubmitting);

  // Reset local state if file changes
  React.useEffect(() => {
    setIsLocalSubmitting(false);
  }, [file]);

  const handleStart = () => {
    if (isBusy) return;
    setIsLocalSubmitting(true);
    onUpload(numSpeakers > 0 ? numSpeakers : undefined, asrModel, denoiseMode);
  };

  const handleCancel = () => {
    if (isBusy) return;
    onCancel();
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-white rounded-3xl shadow-sm border border-slate-200 overflow-hidden"
    >
      <div className="p-8">
        <div className="flex items-start gap-5 mb-6">
          <div className="w-16 h-16 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center shrink-0">
            <FileAudio className="w-8 h-8" />
          </div>
          <div className="flex-1 min-w-0 pt-1">
            <h3 className="text-lg font-bold text-slate-900 truncate mb-2" dir="ltr" style={{textAlign: 'right'}}>{file.name}</h3>
            <div className="flex items-center gap-4 text-sm text-slate-500">
              <span className="bg-slate-100 px-2.5 py-1 rounded-md">{formatFileSize(file.size)}</span>
              <span className="bg-slate-100 px-2.5 py-1 rounded-md uppercase">{file.name.split('.').pop()}</span>
            </div>
          </div>
        </div>

        {/* Speaker count configuration */}
        <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-4 mb-4">
          <label className="block text-sm font-semibold text-slate-700 mb-2">
            عدد المتحدثين المتوقع في التسجيل:
          </label>
          <div className="flex flex-wrap items-center gap-2">
            {[
              { val: 0, label: 'تلقائي (اكتشاف ذكي)' },
              { val: 2, label: 'متحدثان (بودكاست / حوار)' },
              { val: 3, label: '3 متحدثين' },
              { val: 4, label: '4 متحدثين' },
              { val: 1, label: 'متحدث واحد' },
            ].map(({ val, label }) => (
              <button
                key={val}
                type="button"
                disabled={isBusy}
                onClick={() => setNumSpeakers(val)}
                className={`px-3.5 py-2 rounded-xl text-xs font-semibold transition-all ${
                  isBusy ? 'opacity-60 cursor-not-allowed ' : ''
                }${
                  numSpeakers === val
                    ? 'bg-indigo-600 text-white shadow-sm'
                    : 'bg-white text-slate-600 border border-slate-200 hover:border-slate-300'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
        </div>

        {/* ASR Model configuration */}
        <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <label className="block text-sm font-semibold text-slate-700">
              محرك التفريغ الصوتي (ASR Engine):
            </label>
            <span className="text-xs text-slate-500 font-medium">3 خيارات متاحة</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
            {[
              {
                val: 'audar-asr-v1-turbo',
                name: 'Audar ASR',
                desc: 'الأعلى سرعة وتوافقاً',
              },
              {
                val: 'cohere-transcribe-arabic',
                name: 'Cohere Transcribe',
                desc: 'دقيق للهجات والخلط',
              },
              {
                val: 'whisper-large-v3-turbo-arabic-dialectal',
                name: 'Whisper Turbo',
                desc: 'عالي الدقة للهجات',
              },
            ].map(({ val, name, desc }) => (
              <button
                key={val}
                type="button"
                disabled={isBusy}
                onClick={() => setAsrModel(val)}
                className={`p-3.5 rounded-xl text-right transition-all flex flex-col justify-between border ${
                  isBusy ? 'opacity-60 cursor-not-allowed ' : 'cursor-pointer '
                }${
                  asrModel === val
                    ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                    : 'bg-white text-slate-700 border-slate-200 hover:border-slate-300'
                }`}
              >
                <div className="mb-1" dir="ltr" style={{ textAlign: 'right' }}>
                  <span className="text-sm font-bold block leading-snug">
                    {name}
                  </span>
                </div>
                <span
                  className={`text-xs block leading-normal ${
                    asrModel === val ? 'text-indigo-100' : 'text-slate-500'
                  }`}
                >
                  {desc}
                </span>
              </button>
            ))}
          </div>
        </div>

        {/* Audio Enhancement / Noise Reduction configuration */}
        <div className="bg-slate-50 border border-slate-200/80 rounded-2xl p-4 mb-6">
          <div className="flex items-center justify-between mb-2">
            <label className="block text-sm font-semibold text-slate-700">
              معالجة الصوت وعزل الضوضاء (Noise Reduction):
            </label>
            <span className="text-xs text-slate-500 font-medium">5 خيارات متاحة</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2">
            {[
              {
                val: 'dynamic',
                label: 'ديناميكي',
                desc: 'توازن ذكي وتلقائي',
              },
              {
                val: 'low',
                label: 'منخفضة',
                desc: 'لتسجيلات الاستوديو الهادئة',
              },
              {
                val: 'moderate',
                label: 'متوسطة',
                desc: 'للقاعات والمكاتب المعتادة',
              },
              {
                val: 'high',
                label: 'عالية',
                desc: 'للأماكن الصاخبة والمشوشة',
              },
              {
                val: 'none',
                label: 'لا معالجة',
                desc: 'الصوت الأصلي دون أي تعديل',
              },
            ].map(({ val, label, desc }) => (
              <button
                key={val}
                type="button"
                disabled={isBusy}
                onClick={() => setDenoiseMode(val)}
                className={`p-3 rounded-xl text-center transition-all flex flex-col justify-between border ${
                  isBusy ? 'opacity-60 cursor-not-allowed ' : 'cursor-pointer '
                }${
                  denoiseMode === val
                    ? 'bg-indigo-600 text-white border-indigo-600 shadow-sm'
                    : 'bg-white text-slate-700 border-slate-200 hover:border-slate-300'
                }`}
              >
                <div className="font-bold text-sm mb-1 leading-snug">
                  {label}
                </div>
                <span
                  className={`text-[11px] leading-normal block ${
                    denoiseMode === val ? 'text-indigo-100' : 'text-slate-500'
                  }`}
                >
                  {desc}
                </span>
              </button>
            ))}
          </div>
        </div>


        {/* Action Buttons */}
        <div className="flex items-center gap-3">
          <button
            type="button"
            disabled={isBusy}
            onClick={handleStart}
            className={`flex-1 font-semibold py-3.5 px-6 rounded-xl transition-all flex items-center justify-center gap-2 shadow-sm select-none ${
              isBusy
                ? 'bg-indigo-500 text-white cursor-not-allowed opacity-90'
                : 'bg-indigo-600 hover:bg-indigo-700 text-white cursor-pointer active:scale-[0.99]'
            }`}
          >
            {isBusy ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>
                  {uploadProgress > 0 && uploadProgress < 100
                    ? `جاري رفع الملف (${uploadProgress}%)...`
                    : 'جاري رفع الملف وتجهيز المعالجة...'}
                </span>
              </>
            ) : (
              <>
                <Upload className="w-5 h-5" />
                <span>بدء المعالجة وفصل المتحدثين</span>
              </>
            )}
          </button>
          <button
            type="button"
            disabled={isBusy}
            onClick={handleCancel}
            className={`font-medium py-3.5 px-6 rounded-xl transition-colors flex items-center justify-center gap-2 select-none ${
              isBusy
                ? 'bg-slate-100 text-slate-400 cursor-not-allowed'
                : 'bg-slate-100 hover:bg-slate-200 text-slate-700 cursor-pointer'
            }`}
          >
            <X className="w-5 h-5" />
            <span>إلغاء</span>
          </button>
        </div>

        {/* Real-time upload progress feedback */}
        {isBusy && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-4 pt-3 border-t border-slate-100"
          >
            <div className="flex items-center justify-between text-xs text-slate-500 mb-1.5 font-medium">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-indigo-600 animate-pulse" />
                يتم رفع الملف إلى الخادم، يُرجى الانتظار ولا تغلق الصفحة...
              </span>
              <span className="font-semibold text-indigo-600">{uploadProgress}%</span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-2 overflow-hidden">
              <div
                className="bg-indigo-600 h-2 rounded-full transition-all duration-200 ease-out"
                style={{ width: `${Math.max(6, uploadProgress)}%` }}
              />
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};
