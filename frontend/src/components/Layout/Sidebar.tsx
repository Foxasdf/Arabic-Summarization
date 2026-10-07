import React from 'react';
import { Mic, FileText, CheckCircle, AlertCircle, RefreshCw, Upload, Trash2, Bot, Sparkles } from 'lucide-react';
import { Meeting, MainTab } from '../../types';
import { formatDate } from '../../utils/formatters';

interface SidebarProps {
  meetings: Meeting[];
  currentMeetingId: string | null;
  activeTab: MainTab;
  onSelectMeeting: (id: string) => void;
  onNewMeeting: () => void;
  onOpenFalconChat: () => void;
  onDeleteMeeting: (id: string, filename: string) => void;
  isLoading: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({
  meetings,
  currentMeetingId,
  activeTab,
  onSelectMeeting,
  onNewMeeting,
  onOpenFalconChat,
  onDeleteMeeting,
  isLoading,
}) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'uploaded':
        return <Upload className="w-4 h-4 text-slate-400" />;
      case 'processing':
        return <RefreshCw className="w-4 h-4 text-amber-500 animate-spin" />;
      case 'transcribed':
        return <FileText className="w-4 h-4 text-blue-500" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-emerald-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-rose-500" />;
      default:
        return null;
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'uploaded': return 'تم الرفع';
      case 'processing': return 'جاري المعالجة';
      case 'transcribed': return 'تم التفريغ';
      case 'completed': return 'تم التلخيص';
      case 'error': return 'خطأ';
      default: return status;
    }
  };

  return (
    <div className="w-80 bg-white border-l border-slate-200 h-full flex flex-col shadow-sm hidden md:flex">
      {/* Top Action Buttons */}
      <div className="p-3.5 border-b border-slate-200 space-y-2">
        <button
          type="button"
          onClick={onNewMeeting}
          className={`w-full flex items-center justify-center gap-2 py-2.5 px-4 rounded-xl transition-all font-semibold text-sm shadow-xs cursor-pointer ${
            activeTab === 'meetings' && !currentMeetingId
              ? 'bg-indigo-600 text-white hover:bg-indigo-700 ring-2 ring-indigo-200'
              : 'bg-indigo-50 text-indigo-700 hover:bg-indigo-100 border border-indigo-200/80'
          }`}
        >
          <Mic className="w-4 h-4" />
          <span>اجتماع جديد</span>
        </button>

        <button
          type="button"
          onClick={onOpenFalconChat}
          className={`w-full flex items-center justify-between py-2.5 px-3.5 rounded-xl transition-all font-semibold text-sm cursor-pointer ${
            activeTab === 'falcon-chat'
              ? 'bg-gradient-to-r from-indigo-600 via-indigo-700 to-purple-600 text-white shadow-sm ring-2 ring-indigo-300'
              : 'bg-white hover:bg-slate-50 text-slate-700 border border-slate-200 hover:border-slate-300'
          }`}
        >
          <div className="flex items-center gap-2">
            <Bot className="w-4 h-4 text-inherit" />
            <span>شات Falcon H1</span>
          </div>
          <span className={`text-[10px] px-2 py-0.5 rounded-full font-medium flex items-center gap-1 ${
            activeTab === 'falcon-chat'
              ? 'bg-white/20 text-white'
              : 'bg-emerald-50 text-emerald-700 border border-emerald-200'
          }`}>
            <Sparkles className="w-3 h-3" />
            حر ومباشر
          </span>
        </button>
      </div>
      
      {/* Meetings List */}
      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        <div className="flex items-center justify-between px-2 mb-2">
          <h3 className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
            الاجتماعات المسجلة
          </h3>
          <span className="text-[11px] text-slate-400 font-medium">{meetings.length}</span>
        </div>
        
        {isLoading ? (
          <div className="flex justify-center p-4">
            <RefreshCw className="w-5 h-5 text-slate-400 animate-spin" />
          </div>
        ) : meetings.length === 0 ? (
          <div className="text-center p-6 text-sm text-slate-400 border border-dashed border-slate-200 rounded-xl my-2">
            لا توجد اجتماعات سابقة
          </div>
        ) : (
          meetings.map((meeting) => (
            <div
              key={meeting.id}
              onClick={() => onSelectMeeting(meeting.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                  e.preventDefault();
                  onSelectMeeting(meeting.id);
                }
              }}
              className={`group relative w-full text-right p-3 rounded-xl transition-all cursor-pointer select-none ${
                activeTab === 'meetings' && currentMeetingId === meeting.id
                  ? 'bg-indigo-50 border border-indigo-200 shadow-xs'
                  : 'hover:bg-slate-50 border border-transparent'
              }`}
            >
              <div className="flex items-start justify-between gap-2 mb-1">
                <div className="font-medium text-slate-800 text-sm truncate flex-1 min-w-0" title={meeting.filename}>
                  {meeting.filename}
                </div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteMeeting(meeting.id, meeting.filename);
                  }}
                  className="opacity-70 md:opacity-0 md:group-hover:opacity-100 hover:!opacity-100 p-1 rounded-lg text-slate-400 hover:text-rose-600 hover:bg-rose-50 transition-all cursor-pointer shrink-0"
                  title="حذف الاجتماع"
                  aria-label={`حذف ${meeting.filename}`}
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
              <div className="flex items-center justify-between mt-2">
                <div className="text-xs text-slate-400">
                  {formatDate(meeting.upload_time)}
                </div>
                <div className="flex items-center gap-1.5 text-xs font-medium px-2 py-0.5 bg-slate-50 rounded-md border border-slate-100">
                  {getStatusIcon(meeting.status)}
                  <span className="text-slate-600">{getStatusText(meeting.status)}</span>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
