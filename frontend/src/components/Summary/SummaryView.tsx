import React, { useState } from 'react';
import { Copy, Download, CheckCircle, Sparkles, UserCheck } from 'lucide-react';

interface SummaryViewProps {
  summary: string;
}

export const SummaryView: React.FC<SummaryViewProps> = ({ summary }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(summary);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([summary], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'summary.md';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const renderFormattedLine = (lineText: string) => {
    // Process markdown bold **word**
    const parts = lineText.split(/(\*\*.*?\*\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={i} className="font-bold text-slate-900">
            {part.slice(2, -2)}
          </strong>
        );
      }
      return part;
    });
  };

  // Structured parser for markdown summary
  const parseSummary = (text: string) => {
    if (!text) return null;

    // Split by markdown headings: ## or ###
    const sections = text.split(/(?=^#{2,3}\s+)/m).filter(Boolean);
    
    if (sections.length <= 1) {
      return (
        <div className="prose prose-slate max-w-none text-right rtl leading-relaxed space-y-4">
          {text.split('\n\n').map((paragraph, i) => (
            <p key={i} className="text-slate-700">{renderFormattedLine(paragraph)}</p>
          ))}
        </div>
      );
    }

    return sections.map((section, idx) => {
      const lines = section.trim().split('\n');
      const heading = lines[0].replace(/^#{2,3}\s+/, '').trim();
      const content = lines.slice(1).join('\n').trim();

      const isSpeakerSection = heading.includes('مساهمات المتحدثين') || heading.includes('المتحدث');

      return (
        <div 
          key={idx} 
          className={`mb-6 last:mb-0 p-5 rounded-2xl border transition-all ${
            isSpeakerSection 
              ? 'bg-indigo-50/40 border-indigo-200/80 shadow-xs' 
              : 'bg-slate-50/60 border-slate-200/80'
          }`}
        >
          <div className="flex items-center gap-2 mb-3">
            {isSpeakerSection && (
              <div className="w-6 h-6 rounded-md bg-indigo-100 text-indigo-600 flex items-center justify-center shrink-0">
                <UserCheck className="w-3.5 h-3.5" />
              </div>
            )}
            <h4 className={`text-lg font-bold ${isSpeakerSection ? 'text-indigo-950' : 'text-slate-800'}`}>
              {heading}
            </h4>
          </div>

          <div className="space-y-2 text-right rtl">
            {content.split('\n').map((line, i) => {
              const trimmed = line.trim();
              if (!trimmed) return null;

              if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
                return (
                  <div key={i} className="flex items-start gap-2 text-slate-700 leading-relaxed text-[15px]">
                    <span className="text-indigo-500 font-bold shrink-0 mt-1">•</span>
                    <span className="flex-1">{renderFormattedLine(trimmed.substring(2))}</span>
                  </div>
                );
              }
              if (/^\d+\./.test(trimmed)) {
                return (
                  <div key={i} className="flex items-start gap-2 text-slate-700 leading-relaxed text-[15px]">
                    <span className="text-indigo-600 font-semibold shrink-0">{trimmed.split('.')[0]}.</span>
                    <span className="flex-1">{renderFormattedLine(trimmed.substring(trimmed.indexOf('.') + 1).trim())}</span>
                  </div>
                );
              }
              return (
                <p key={i} className="text-slate-700 leading-relaxed text-[15px]">
                  {renderFormattedLine(trimmed)}
                </p>
              );
            })}
          </div>
        </div>
      );
    });
  };

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-emerald-100 overflow-hidden relative">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-emerald-400 via-teal-500 to-indigo-500" />
      
      <div className="p-4 sm:p-6 border-b border-slate-100 flex items-center justify-between bg-slate-50/40">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-emerald-100 text-emerald-600 flex items-center justify-center">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-bold text-slate-800 text-lg">الملخص الذكي للاجتماع</h3>
            <p className="text-xs text-slate-500">تم التلخيص بنموذج Falcon مع عزو النقاط للمتحدثين</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2">
          <button 
            onClick={handleCopy}
            className="p-2 text-slate-500 hover:text-emerald-600 hover:bg-emerald-50 rounded-lg transition-colors flex items-center gap-1.5 text-sm font-medium border border-slate-200 bg-white"
            title="نسخ الملخص"
          >
            {copied ? <CheckCircle className="w-4 h-4 text-emerald-600" /> : <Copy className="w-4 h-4" />}
            <span className="hidden sm:inline">{copied ? 'تم النسخ' : 'نسخ'}</span>
          </button>
          
          <button 
            onClick={handleDownload}
            className="p-2 text-slate-500 hover:text-emerald-600 hover:bg-emerald-50 rounded-lg transition-colors border border-slate-200 bg-white"
            title="تنزيل كملف Markdown"
          >
            <Download className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="p-6">
        {parseSummary(summary)}
      </div>
    </div>
  );
};
