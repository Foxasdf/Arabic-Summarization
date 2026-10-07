import React, { useRef } from 'react';
import { motion } from 'framer-motion';
import { UploadCloud, Music, FileAudio, Video } from 'lucide-react';
import { useUpload } from '../../hooks/useUpload';

interface DropZoneProps {
  onFileSelect: (file: File) => void;
}

export const DropZone: React.FC<DropZoneProps> = ({ onFileSelect }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { isDragging, error, onDragOver, onDragLeave, onDrop, onFileSelect: handleFileSelect, file } = useUpload();

  React.useEffect(() => {
    if (file) {
      onFileSelect(file);
    }
  }, [file, onFileSelect]);

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full"
    >
      <div
        onClick={handleClick}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className={`
          relative overflow-hidden cursor-pointer rounded-3xl border-2 border-dashed transition-all duration-300
          flex flex-col items-center justify-center p-12 min-h-[320px] bg-white shadow-sm
          ${isDragging ? 'border-indigo-500 bg-indigo-50/50 scale-[1.02]' : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50'}
          ${error ? 'border-rose-400 bg-rose-50/50' : ''}
        `}
      >
        <div className="absolute inset-0 bg-gradient-to-b from-transparent to-slate-50/50 pointer-events-none" />
        
        <div className={`p-5 rounded-2xl mb-6 transition-colors duration-300 ${isDragging ? 'bg-indigo-100 text-indigo-600' : 'bg-slate-100 text-slate-500'}`}>
          <UploadCloud className="w-12 h-12" />
        </div>
        
        <h3 className="text-xl font-bold text-slate-800 mb-2">ارفع ملف الاجتماع الصوتي</h3>
        <p className="text-slate-500 mb-8 text-center max-w-sm">
          اسحب الملف وأفلته هنا، أو اضغط لاختيار ملف من جهازك
        </p>
        
        <div className="flex flex-wrap items-center justify-center gap-4 text-xs font-medium text-slate-500">
          <div className="flex items-center gap-1.5 bg-slate-100 px-3 py-1.5 rounded-lg">
            <Music className="w-3.5 h-3.5" /> MP3
          </div>
          <div className="flex items-center gap-1.5 bg-slate-100 px-3 py-1.5 rounded-lg">
            <FileAudio className="w-3.5 h-3.5" /> WAV
          </div>
          <div className="flex items-center gap-1.5 bg-slate-100 px-3 py-1.5 rounded-lg">
            <Video className="w-3.5 h-3.5" /> MP4
          </div>
          <div className="flex items-center gap-1.5 bg-slate-100 px-3 py-1.5 rounded-lg">
            <span className="font-sans font-bold text-slate-400 leading-none">M4A</span>
          </div>
        </div>

        {error && (
          <div className="mt-6 text-sm text-rose-600 bg-rose-50 px-4 py-2 rounded-lg w-full text-center font-medium">
            {error}
          </div>
        )}

        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileSelect}
          className="hidden"
          accept="audio/*,video/mp4"
        />
      </div>
    </motion.div>
  );
};
