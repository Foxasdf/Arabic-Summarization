import React from 'react';
import { motion } from 'framer-motion';
import { FileAudio, X, Upload } from 'lucide-react';
import { formatFileSize } from '../../utils/formatters';

interface FilePreviewProps {
  file: File;
  onUpload: () => void;
  onCancel: () => void;
}

export const FilePreview: React.FC<FilePreviewProps> = ({ file, onUpload, onCancel }) => {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="bg-white rounded-3xl shadow-sm border border-slate-200 overflow-hidden"
    >
      <div className="p-8">
        <div className="flex items-start gap-5 mb-8">
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

        <div className="flex items-center gap-3">
          <button
            onClick={onUpload}
            className="flex-1 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3.5 px-6 rounded-xl transition-colors flex items-center justify-center gap-2"
          >
            <Upload className="w-5 h-5" />
            رفع ومعالجة
          </button>
          <button
            onClick={onCancel}
            className="bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium py-3.5 px-6 rounded-xl transition-colors flex items-center justify-center gap-2"
          >
            <X className="w-5 h-5" />
            إلغاء
          </button>
        </div>
      </div>
    </motion.div>
  );
};
