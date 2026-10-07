import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Bot, 
  Send, 
  User, 
  Sparkles, 
  Copy, 
  CheckCircle, 
  RotateCcw, 
  Loader2, 
  AlertCircle, 
  Cpu,
  Lightbulb,
  Code2,
  FileText,
  Clock,
  Paperclip,
  X,
  FileSpreadsheet,
  FileCode,
  File as FileIcon,
  UploadCloud,
  Eye
} from 'lucide-react';
import { ChatMessage, AttachedDocument } from '../../types';
import { api } from '../../services/api';

const STORAGE_KEY = 'falcon_standalone_chat_history';
const ATTACHED_DOCS_KEY = 'falcon_standalone_chat_attached_docs';

const GENERAL_PROMPTS = [
  {
    icon: Lightbulb,
    title: 'أفكار مشاريع مبتكرة',
    prompt: 'اقترح 5 أفكار لمشاريع تقنية مبتكرة في العالم العربي باستخدام الذكاء الاصطناعي مفتوح المصدر.',
  },
  {
    icon: FileText,
    title: 'صياغة مقال احترافي',
    prompt: 'اكتب مقالاً موجزاً ومؤثراً حول دور الذكاء الاصطناعي التوليدي في رفع إنتاجية فرق العمل الحديثة.',
  },
  {
    icon: Code2,
    title: 'شرح تقني وبرمجي',
    prompt: 'اشرح بنية نماذج المحولات (Transformers) وكيف تختلف نماذج التوليد (Decoder-only) عن غيرها.',
  },
  {
    icon: Sparkles,
    title: 'تخطيط وتطوير الذات',
    prompt: 'ضع لي خطة عملية من 5 خطوات لتعلم مهارة تقنية جديدة وإتقانها في وقت قياسي.',
  },
];

const DOCUMENT_PROMPTS = [
  {
    icon: FileText,
    title: 'تلخيص شامل',
    prompt: 'لخص لي هذا المستند المرفق بالكامل في 5 إلى 7 نقاط رئيسية واضحة.',
  },
  {
    icon: Lightbulb,
    title: 'النتائج والتوصيات',
    prompt: 'ما هي أهم الاستنتاجات، النتائج، أو التوصيات التي وردت في هذا الملف؟',
  },
  {
    icon: Sparkles,
    title: 'استخراج الأرقام والتواريخ',
    prompt: 'استخرج جميع الأرقام، الإحصائيات، التواريخ، أو المبالغ المالية المذكورة في المستند.',
  },
  {
    icon: Code2,
    title: 'نقد وتحليل المحتوى',
    prompt: 'قم بتحليل محتوى هذا الملف ونقده موضوعياً، وما هي النقاط التي تحتاج توضيحاً إضافياً؟',
  },
];

export const FalconStandaloneChat: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [attachedDocs, setAttachedDocs] = useState<AttachedDocument[]>(() => {
    try {
      const saved = localStorage.getItem(ATTACHED_DOCS_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploadingDoc, setIsUploadingDoc] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [lastResponseTime, setLastResponseTime] = useState<number | null>(null);
  const [previewDoc, setPreviewDoc] = useState<AttachedDocument | null>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Persist messages to localStorage
  useEffect(() => {
    try {
      if (messages.length > 0) {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
      } else {
        localStorage.removeItem(STORAGE_KEY);
      }
    } catch (e) {
      console.error('Failed to save chat to localStorage', e);
    }
  }, [messages]);

  // Persist attached docs to localStorage
  useEffect(() => {
    try {
      if (attachedDocs.length > 0) {
        localStorage.setItem(ATTACHED_DOCS_KEY, JSON.stringify(attachedDocs));
      } else {
        localStorage.removeItem(ATTACHED_DOCS_KEY);
      }
    } catch (e) {
      console.error('Failed to save attached docs to localStorage', e);
    }
  }, [attachedDocs]);

  // Scroll to bottom when new messages arrive or loading
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 140)}px`;
    }
  };

  const handleCopyMessage = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleClearChat = () => {
    if (messages.length === 0 && attachedDocs.length === 0) return;
    if (window.confirm('هل أنت متأكد من رغبتك في بدء محادثة جديدة ومسح الرسائل والملفات المرفقة؟')) {
      setMessages([]);
      setAttachedDocs([]);
      setError(null);
      setLastResponseTime(null);
      try {
        localStorage.removeItem(STORAGE_KEY);
        localStorage.removeItem(ATTACHED_DOCS_KEY);
      } catch {}
    }
  };

  const handleFileUpload = async (file: File) => {
    if (!file) return;
    setIsUploadingDoc(true);
    setError(null);

    try {
      const doc = await api.uploadChatDocument(file);
      setAttachedDocs(prev => {
        // Prevent duplicate filenames
        const filtered = prev.filter(d => d.filename !== doc.filename);
        return [...filtered, doc];
      });
    } catch (err: any) {
      console.error('Document upload error:', err);
      setError(err?.message || `فشل استخراج النص من الملف "${file.name}".`);
    } finally {
      setIsUploadingDoc(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleRemoveDoc = (id: string) => {
    setAttachedDocs(prev => prev.filter(d => d.id !== id));
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      await handleFileUpload(file);
    }
  };

  const handleSendMessage = async (textToSendInput?: string) => {
    const textToSend = (textToSendInput || input).trim();
    if (!textToSend || isLoading) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: textToSend,
      timestamp: new Date().toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' }),
    };

    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setInput('');
    setError(null);
    setIsLoading(true);

    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    try {
      const historyPayload = messages.map(m => ({
        role: m.role,
        content: m.content,
      }));

      const docsPayload = attachedDocs.map(d => ({
        filename: d.filename,
        content: d.content,
      }));

      const res = await api.chat(textToSend, historyPayload, docsPayload);
      setLastResponseTime(res.processing_time_seconds);

      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: res.reply,
        timestamp: new Date().toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages([...newMessages, assistantMessage]);
    } catch (err: any) {
      console.error('Chat error:', err);
      setError(err?.message || 'تعذر الحصول على استجابة من Falcon H1. يرجى التأكد من تشغيل الخادم والمحاولة ثانية.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getDocBadgeDetails = (ext: string) => {
    const e = ext.toLowerCase();
    if (e === '.pdf') {
      return {
        icon: FileText,
        color: 'bg-blue-50 text-blue-700 border-blue-200',
        badge: 'PDF',
      };
    }
    if (e === '.docx' || e === '.doc') {
      return {
        icon: FileText,
        color: 'bg-sky-50 text-sky-700 border-sky-200',
        badge: 'Word',
      };
    }
    if (e === '.xlsx' || e === '.xls' || e === '.csv') {
      return {
        icon: FileSpreadsheet,
        color: 'bg-cyan-50 text-cyan-700 border-cyan-200',
        badge: 'Excel/CSV',
      };
    }
    if (['.py', '.js', '.ts', '.tsx', '.jsx', '.html', '.css', '.sql', '.json', '.xml'].includes(e)) {
      return {
        icon: FileCode,
        color: 'bg-indigo-50 text-indigo-700 border-indigo-200',
        badge: 'Code',
      };
    }
    return {
      icon: FileIcon,
      color: 'bg-blue-50/70 text-blue-700 border-blue-200',
      badge: 'مستند',
    };
  };

  const renderFormattedContent = (content: string) => {
    const codeBlockRegex = /```([a-zA-Z0-9]*)\n?([\s\S]*?)```/g;
    type FormattedPart =
      | { type: 'text'; content: string }
      | { type: 'code'; lang: string; code: string };

    const parts: FormattedPart[] = [];
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      if (match.index > lastIndex) {
        parts.push({ type: 'text', content: content.slice(lastIndex, match.index) });
      }
      parts.push({ type: 'code', lang: match[1] || 'code', code: match[2].trim() });
      lastIndex = match.index + match[0].length;
    }

    if (lastIndex < content.length) {
      parts.push({ type: 'text', content: content.slice(lastIndex) });
    }

    if (parts.length === 0) {
      parts.push({ type: 'text', content });
    }

    return (
      <div className="space-y-3">
        {parts.map((part, pIdx) => {
          if (part.type === 'code') {
            return (
              <div key={pIdx} className="rounded-xl overflow-hidden bg-slate-900 text-slate-100 my-2 text-left" dir="ltr">
                <div className="flex items-center justify-between px-3 py-1.5 bg-slate-800 text-xs text-slate-300 font-mono border-b border-slate-700">
                  <span>{part.lang}</span>
                  <button
                    type="button"
                    onClick={() => {
                      navigator.clipboard.writeText(part.code || '');
                    }}
                    className="flex items-center gap-1 hover:text-white transition-colors cursor-pointer"
                  >
                    <Copy className="w-3 h-3" />
                    <span>نسخ</span>
                  </button>
                </div>
                <pre className="p-3 text-xs overflow-x-auto font-mono leading-relaxed">
                  <code>{part.code}</code>
                </pre>
              </div>
            );
          }

          const paragraphs = (part.content || '').split('\n\n').filter(Boolean);
          return (
            <div key={pIdx} className="space-y-2">
              {paragraphs.map((p, i) => {
                const lines = p.split('\n');
                return (
                  <p key={i} className="leading-relaxed">
                    {lines.map((line, lIdx) => {
                      const trimmed = line.trim();
                      const isBullet = trimmed.startsWith('- ') || trimmed.startsWith('* ') || trimmed.startsWith('• ');
                      const isNum = /^\d+[\.\)]\s/.test(trimmed);

                      const boldParts = line.split(/(\*\*[^*]+\*\*)/g);
                      const formattedLine = boldParts.map((bp, bIdx) => {
                        if (bp.startsWith('**') && bp.endsWith('**')) {
                          return <strong key={bIdx} className="font-semibold text-slate-950">{bp.slice(2, -2)}</strong>;
                        }
                        return bp;
                      });

                      if (isBullet) {
                        return (
                          <span key={lIdx} className="flex items-start gap-2 my-1 mr-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 mt-2 shrink-0" />
                            <span className="flex-1">{formattedLine}</span>
                          </span>
                        );
                      }

                      if (isNum) {
                        return (
                          <span key={lIdx} className="block my-1 font-medium text-slate-900">
                            {formattedLine}
                          </span>
                        );
                      }

                      return (
                        <span key={lIdx} className="block">
                          {formattedLine}
                        </span>
                      );
                    })}
                  </p>
                );
              })}
            </div>
          );
        })}
      </div>
    );
  };

  const activePrompts = attachedDocs.length > 0 ? DOCUMENT_PROMPTS : GENERAL_PROMPTS;

  return (
    <div 
      className="flex flex-col h-full bg-slate-50 relative select-text"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* Hidden file input */}
      <input
        type="file"
        ref={fileInputRef}
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
          }
        }}
        accept=".pdf,.docx,.doc,.xlsx,.xls,.txt,.md,.markdown,.csv,.tsv,.json,.xml,.yaml,.yml,.html,.htm,.css,.js,.jsx,.ts,.tsx,.py,.c,.cpp,.h,.cs,.java,.go,.rs,.php,.sql,.sh,.bat,.log"
        className="hidden"
      />

      {/* Drag & Drop Visual Overlay */}
      {isDragging && (
        <div className="absolute inset-0 z-50 bg-indigo-600/90 backdrop-blur-xs flex flex-col items-center justify-center text-white border-4 border-dashed border-white/60 m-4 rounded-3xl animate-in fade-in duration-200 pointer-events-none">
          <UploadCloud className="w-16 h-16 mb-4 animate-bounce" />
          <h3 className="text-xl font-bold mb-2">أفلت الملف هنا لإرفاقه</h3>
          <p className="text-indigo-100 text-sm">
            ندعم ملفات PDF، Word، Excel، الجداول، الأكواد، والنصوص البرمجية
          </p>
        </div>
      )}

      {/* Top Banner / Header for Falcon H1 */}
      <div className="bg-white border-b border-slate-200/80 px-4 sm:px-6 py-3.5 flex items-center justify-between shadow-2xs">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-purple-500 flex items-center justify-center text-white shadow-sm ring-2 ring-indigo-100">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-base sm:text-lg font-bold text-slate-900">Falcon H1</h2>
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-semibold bg-indigo-50 text-indigo-700 border border-indigo-200">
                <Cpu className="w-3 h-3" />
                7B Instruct
              </span>
              <span className="hidden sm:inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[11px] font-medium bg-emerald-50 text-emerald-700 border border-emerald-200">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                دعم الملفات والمستندات
              </span>
            </div>
            <p className="text-xs text-slate-500">
              تحدث مباشرة مع النموذج أو ارفع مستندات (PDF / Word / Excel / كود) لتحليلها وتلخيصها
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* Attach file button in header */}
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploadingDoc}
            className="flex items-center gap-1.5 text-xs text-blue-600 hover:text-blue-700 hover:bg-blue-50 bg-blue-50/60 px-3 py-1.5 rounded-lg border border-blue-200 transition-all cursor-pointer font-semibold"
            title="إرفاق ملف أو مستند للمحادثة"
          >
            {isUploadingDoc ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Paperclip className="w-3.5 h-3.5" />
            )}
            <span className="hidden sm:inline">إرفاق ملف</span>
          </button>

          {(messages.length > 0 || attachedDocs.length > 0) && (
            <button
              type="button"
              onClick={handleClearChat}
              className="flex items-center gap-1.5 text-xs text-slate-500 hover:text-rose-600 hover:bg-rose-50 px-3 py-1.5 rounded-lg border border-slate-200 hover:border-rose-200 transition-all cursor-pointer font-medium"
              title="بدء محادثة جديدة ومسح الرسائل والملفات"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">محادثة جديدة</span>
            </button>
          )}
        </div>
      </div>

      {/* Chat Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
        {messages.length === 0 ? (
          <div className="max-w-2xl mx-auto py-6 sm:py-10 text-center">
            {/* Welcome Avatar */}
            <div className="w-20 h-20 mx-auto mb-5 rounded-3xl bg-gradient-to-tr from-indigo-600 via-indigo-500 to-purple-600 p-0.5 shadow-lg shadow-indigo-100 flex items-center justify-center">
              <div className="w-full h-full bg-white rounded-[22px] flex items-center justify-center text-indigo-600">
                <Bot className="w-10 h-10" />
              </div>
            </div>

            <h3 className="text-2xl font-bold text-slate-800 mb-2">
              أهلاً بك في شات Falcon H1
            </h3>
            <p className="text-slate-600 text-sm sm:text-base max-w-lg mx-auto mb-6 leading-relaxed">
              اسألني عن أي موضوع أو اسحب وأفلت مستنداتك هنا (PDF, Word, Excel, نصوص, كود) لنقرأها ونناقشها سوياً.
            </p>

            {/* If files are already attached */}
            {attachedDocs.length > 0 && (
              <div className="mb-6 bg-blue-50/70 border border-blue-200 p-4 rounded-2xl text-right">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2 text-blue-900 font-bold text-sm">
                    <Paperclip className="w-4 h-4 text-blue-600" />
                    <span>الملفات الجاهزة للتحليل ({attachedDocs.length})</span>
                  </div>
                  <span className="text-xs text-blue-600 font-medium">سيبقى الموديل يتذكرها طوال المحادثة</span>
                </div>
                <div className="flex flex-wrap gap-2">
                  {attachedDocs.map(doc => {
                    const badge = getDocBadgeDetails(doc.extension);
                    const DocIcon = badge.icon;
                    return (
                      <div
                        key={doc.id}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-xl border text-xs font-semibold ${badge.color}`}
                      >
                        <DocIcon className="w-4 h-4 shrink-0 text-blue-600" />
                        <span className="truncate max-w-[200px]" title={doc.filename}>{doc.filename}</span>
                        <span className="text-[10px] opacity-75 font-normal text-blue-600">({doc.word_count.toLocaleString('ar-SA')} كلمة)</span>
                        <button
                          type="button"
                          onClick={() => setPreviewDoc(doc)}
                          className="p-1 hover:bg-blue-100 rounded text-blue-600 transition-colors"
                          title="معاينة النص المستخرج"
                        >
                          <Eye className="w-3 h-3" />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleRemoveDoc(doc.id)}
                          className="p-1 hover:bg-blue-100 rounded text-blue-500 hover:text-blue-700 transition-colors"
                          title="إزالة الملف"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Prompt Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-right">
              {activePrompts.map((item, idx) => {
                const IconComponent = item.icon;
                return (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleSendMessage(item.prompt)}
                    className="p-3.5 rounded-xl bg-white hover:bg-indigo-50/60 border border-slate-200 hover:border-indigo-300 text-right transition-all group shadow-2xs hover:shadow-sm cursor-pointer"
                  >
                    <div className="flex items-center gap-2 mb-1.5 text-indigo-600">
                      <IconComponent className="w-4 h-4" />
                      <span className="text-xs font-semibold text-slate-800 group-hover:text-indigo-700">
                        {item.title}
                      </span>
                    </div>
                    <p className="text-xs text-slate-500 line-clamp-2 leading-relaxed">
                      {item.prompt}
                    </p>
                  </button>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-4">
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2 }}
                  className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {/* Assistant avatar */}
                  {msg.role === 'assistant' && (
                    <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-600 text-white flex items-center justify-center shrink-0 mt-0.5 shadow-xs">
                      <Bot className="w-5 h-5" />
                    </div>
                  )}

                  {/* Message bubble */}
                  <div
                    className={`max-w-[85%] sm:max-w-[80%] rounded-2xl p-4 shadow-2xs relative group ${
                      msg.role === 'user'
                        ? 'bg-indigo-600 text-white rounded-br-xs'
                        : 'bg-white text-slate-800 border border-slate-200/90 rounded-bl-xs'
                    }`}
                  >
                    <div className="text-[14.5px] leading-relaxed whitespace-pre-wrap break-words">
                      {msg.role === 'assistant' ? renderFormattedContent(msg.content) : msg.content}
                    </div>

                    <div
                      className={`flex items-center justify-between gap-3 mt-2.5 pt-2 border-t ${
                        msg.role === 'user'
                          ? 'border-indigo-500/40 text-indigo-200 text-[11px]'
                          : 'border-slate-100 text-slate-400 text-[11px]'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span>{msg.timestamp}</span>
                        {msg.role === 'assistant' && (
                          <span className="text-[10px] bg-slate-100 text-slate-500 px-1.5 py-0.5 rounded">
                            Falcon H1
                          </span>
                        )}
                      </div>

                      {msg.role === 'assistant' && (
                        <button
                          type="button"
                          onClick={() => handleCopyMessage(msg.id, msg.content)}
                          className="opacity-70 group-hover:opacity-100 transition-opacity p-1 text-slate-400 hover:text-indigo-600 hover:bg-slate-100 rounded cursor-pointer flex items-center gap-1 text-[11px]"
                          title="نسخ الإجابة"
                        >
                          {copiedId === msg.id ? (
                            <>
                              <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                              <span className="text-emerald-600 font-medium">تم النسخ</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3.5 h-3.5" />
                              <span>نسخ</span>
                            </>
                          )}
                        </button>
                      )}
                    </div>
                  </div>

                  {/* User avatar */}
                  {msg.role === 'user' && (
                    <div className="w-9 h-9 rounded-xl bg-slate-200 text-slate-700 flex items-center justify-center shrink-0 mt-0.5 shadow-2xs">
                      <User className="w-5 h-5" />
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Loading typing indicator */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center gap-3 justify-start"
              >
                <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-600 to-purple-600 text-white flex items-center justify-center shrink-0 shadow-xs">
                  <Bot className="w-5 h-5" />
                </div>
                <div className="bg-white border border-indigo-100 rounded-2xl rounded-bl-xs p-4 text-xs text-indigo-700 font-medium flex items-center gap-2.5 shadow-2xs">
                  <Loader2 className="w-4 h-4 animate-spin text-indigo-600" />
                  <span>Falcon H1 يحلل المحتوى ويصيغ الإجابة باللغة العربية...</span>
                </div>
              </motion.div>
            )}

            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Error banner */}
      {error && (
        <div className="mx-4 sm:mx-6 mb-2 p-3 bg-rose-50 border border-rose-200 rounded-xl text-rose-700 text-xs font-semibold flex items-center justify-between gap-2 shadow-2xs">
          <div className="flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-500 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            type="button"
            onClick={() => setError(null)}
            className="text-rose-500 hover:text-rose-800 font-bold px-2 py-1 cursor-pointer"
          >
            ✕
          </button>
        </div>
      )}

      {/* Input box bar */}
      <div className="p-3 sm:p-4 bg-white border-t border-slate-200/80 shadow-xs">
        <div className="max-w-3xl mx-auto">
          {/* Active Attached Documents Row above input */}
          {attachedDocs.length > 0 && (
            <div className="flex flex-wrap items-center gap-2 mb-2 p-2 bg-blue-50/70 border border-blue-200/80 rounded-xl">
              <span className="text-[11px] text-blue-900 font-semibold flex items-center gap-1">
                <Paperclip className="w-3.5 h-3.5 text-blue-600" />
                <span>مرفقات الجلسة:</span>
              </span>
              {attachedDocs.map(doc => {
                const badge = getDocBadgeDetails(doc.extension);
                const DocIcon = badge.icon;
                return (
                  <div
                    key={doc.id}
                    className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-xs font-medium ${badge.color}`}
                  >
                    <DocIcon className="w-3.5 h-3.5 text-blue-600" />
                    <span className="max-w-[160px] truncate" title={doc.filename}>{doc.filename}</span>
                    <button
                      type="button"
                      onClick={() => setPreviewDoc(doc)}
                      className="hover:text-blue-900 cursor-pointer p-0.5 text-blue-600 hover:bg-blue-100 rounded"
                      title="معاينة النص"
                    >
                      <Eye className="w-3 h-3" />
                    </button>
                    <button
                      type="button"
                      onClick={() => handleRemoveDoc(doc.id)}
                      className="hover:text-blue-800 hover:bg-blue-100 rounded cursor-pointer p-0.5 text-blue-500"
                      title="إزالة هذا الملف"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </div>
                );
              })}
            </div>
          )}

          <form
            onSubmit={(e) => {
              e.preventDefault();
              handleSendMessage();
            }}
            className="flex items-end gap-2"
          >
            {/* Paperclip attachment button */}
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isUploadingDoc || isLoading}
              className={`h-12 w-12 rounded-2xl border flex items-center justify-center transition-all cursor-pointer shrink-0 shadow-2xs ${
                isUploadingDoc
                  ? 'bg-blue-50 border-blue-200 text-blue-600'
                  : attachedDocs.length > 0
                    ? 'bg-blue-50 border-blue-300 text-blue-600 hover:bg-blue-100 hover:border-blue-400'
                    : 'bg-slate-50 hover:bg-blue-50/80 border-slate-200 hover:border-blue-300 text-slate-600 hover:text-blue-600'
              }`}
              title="إرفاق ملف (PDF, Word, Excel, نصوص، أكواد)"
            >
              {isUploadingDoc ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Paperclip className="w-5 h-5" />
              )}
            </button>

            <div className="flex-1 relative">
              <textarea
                ref={textareaRef}
                rows={1}
                value={input}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                placeholder={
                  attachedDocs.length > 0
                    ? `اطرح سؤالك حول "${attachedDocs[attachedDocs.length - 1].filename}" أو تحدث معه بحرية...`
                    : "اكتب رسالتك لـ Falcon H1 أو ارفق ملفاً لتحليله... (Enter للإرسال)"
                }
                disabled={isLoading}
                className="w-full resize-none rounded-2xl border border-slate-300 focus:border-indigo-500 focus:ring-3 focus:ring-indigo-100 p-3.5 pr-4 text-sm text-slate-800 placeholder-slate-400 outline-none transition-all disabled:bg-slate-50 disabled:text-slate-400 shadow-2xs"
                style={{ minHeight: '48px', maxHeight: '140px' }}
              />
            </div>

            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="h-12 px-5 bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 disabled:from-slate-200 disabled:to-slate-200 text-white disabled:text-slate-400 rounded-2xl font-semibold text-sm transition-all flex items-center gap-2 shadow-sm cursor-pointer disabled:cursor-not-allowed shrink-0"
              title="إرسال"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <>
                  <span>إرسال</span>
                  <Send className="w-4 h-4 rotate-180" />
                </>
              )}
            </button>
          </form>

          <div className="flex items-center justify-between mt-2 px-1 text-[11px] text-slate-400">
            <span>اسحب وأفلت الملفات هنا أو اضغط 📎 للإرفاق • Enter للإرسال</span>
            <div className="flex items-center gap-3">
              {lastResponseTime && (
                <span className="flex items-center gap-1 text-slate-500">
                  <Clock className="w-3 h-3 text-slate-400" />
                  <span>زمن الرد: {lastResponseTime} ثانية</span>
                </span>
              )}
              <span className="text-indigo-600 font-medium">Falcon-H1-7B Instruct (TII)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Document Text Preview Modal */}
      {previewDoc && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl max-w-2xl w-full max-h-[80vh] flex flex-col shadow-xl border border-slate-200 animate-in fade-in zoom-in-95 duration-150">
            <div className="p-4 border-b border-slate-200 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="w-5 h-5 text-blue-600" />
                <h3 className="font-bold text-slate-800 text-sm truncate max-w-md" title={previewDoc.filename}>
                  {previewDoc.filename}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setPreviewDoc(null)}
                className="text-slate-400 hover:text-slate-700 p-1 rounded-lg"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="px-4 py-2 bg-slate-50 border-b border-slate-100 flex items-center gap-4 text-xs text-slate-600">
              <span>عدد الكلمات: <strong>{previewDoc.word_count.toLocaleString('ar-SA')}</strong></span>
              <span>عدد الحروف: <strong>{previewDoc.char_count.toLocaleString('ar-SA')}</strong></span>
              <span>النوع: <strong>{previewDoc.extension}</strong></span>
            </div>

            <div className="p-4 overflow-y-auto flex-1 text-xs font-mono leading-relaxed bg-slate-50 text-slate-800 whitespace-pre-wrap select-text">
              {previewDoc.content}
            </div>

            <div className="p-3 border-t border-slate-200 flex justify-end">
              <button
                type="button"
                onClick={() => setPreviewDoc(null)}
                className="px-4 py-2 bg-blue-600 text-white rounded-xl text-xs font-semibold hover:bg-blue-700 cursor-pointer"
              >
                إغلاق المعاينة
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
