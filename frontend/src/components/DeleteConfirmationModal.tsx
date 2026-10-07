import React, { useEffect } from 'react';
import { Trash2, AlertTriangle, Loader2 } from 'lucide-react';

interface DeleteConfirmationModalProps {
  isOpen: boolean;
  meetingTitle: string;
  isDeleting: boolean;
  onConfirm: () => void;
  onClose: () => void;
}

export const DeleteConfirmationModal: React.FC<DeleteConfirmationModalProps> = ({
  isOpen,
  meetingTitle,
  isDeleting,
  onConfirm,
  onClose,
}) => {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !isDeleting) {
        onClose();
      }
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, isDeleting, onClose]);

  if (!isOpen) return null;

  return (
    <div 
      className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/50 backdrop-blur-xs animate-in fade-in duration-200"
      onClick={() => {
        if (!isDeleting) onClose();
      }}
      dir="rtl"
    >
      <div 
        className="bg-white rounded-2xl p-6 max-w-md w-full shadow-2xl border border-slate-100 animate-in zoom-in-95 duration-200 overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start gap-4">
          <div className="w-12 h-12 rounded-2xl bg-rose-50 text-rose-600 flex items-center justify-center shrink-0 border border-rose-100">
            <Trash2 className="w-6 h-6" />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-lg font-bold text-slate-900 mb-1">حذف الاجتماع</h3>
            <p className="text-slate-600 text-sm leading-relaxed mb-2">
              هل أنت متأكد من رغبتك في حذف هذا الاجتماع؟
            </p>
            <div 
              className="bg-slate-50 border border-slate-200/80 rounded-xl px-3.5 py-2.5 text-sm font-medium text-slate-700 break-words [overflow-wrap:anywhere] line-clamp-2 mb-4 leading-relaxed"
              title={meetingTitle}
              dir="auto"
            >
              {meetingTitle}
            </div>
            <div className="flex items-start gap-2.5 text-amber-700 text-xs mb-5 bg-amber-50 border border-amber-200/60 p-3 rounded-xl leading-relaxed">
              <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600 mt-0.5" />
              <span className="flex-1 min-w-0">سيتم حذف التسجيل الصوتي والتفريغ والملخص نهائياً. لا يمكن التراجع عن هذا الإجراء.</span>
            </div>
          </div>
        </div>

        <div className="flex items-center justify-end gap-3 mt-2">
          <button
            type="button"
            onClick={onClose}
            disabled={isDeleting}
            className="px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-800 hover:bg-slate-100 rounded-xl transition-colors disabled:opacity-50 cursor-pointer"
          >
            إلغاء
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isDeleting}
            className="flex items-center gap-2 px-5 py-2 text-sm font-medium text-white bg-rose-600 hover:bg-rose-700 rounded-xl transition-all shadow-sm hover:shadow disabled:opacity-50 cursor-pointer"
          >
            {isDeleting ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>جاري الحذف...</span>
              </>
            ) : (
              <>
                <Trash2 className="w-4 h-4" />
                <span>نعم، احذف</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
