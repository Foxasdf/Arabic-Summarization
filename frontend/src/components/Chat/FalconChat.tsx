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
  ShieldCheck,
  MessageSquare
} from 'lucide-react';
import { ChatMessage } from '../../types';
import { api } from '../../services/api';

interface FalconChatProps {
  meetingId: string;
  hasTranscript: boolean;
}

const SUGGESTED_QUESTIONS = [
  'ما هي أهم الموضوعات التي ناقشها المتحدثون؟',
  'ماذا قيل في بداية التسجيل الصوتي؟',
  'هل ذُكرت أرقام أو تواريخ أو مواعيد محددة؟',
  'من هو المتحدث الأكثر مشاركة وماذا طرح؟',
];

export const FalconChat: React.FC<FalconChatProps> = ({ meetingId, hasTranscript }) => {
  // Load messages specifically for this meeting from localStorage if available
  const [messages, setMessages] = useState<ChatMessage[]>(() => {
    try {
      const saved = localStorage.getItem(`falcon_chat_${meetingId}`);
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Sync chat messages and reset state when meetingId changes
  useEffect(() => {
    try {
      const saved = localStorage.getItem(`falcon_chat_${meetingId}`);
      setMessages(saved ? JSON.parse(saved) : []);
    } catch {
      setMessages([]);
    }
    setInput('');
    setError(null);
    setIsLoading(false);
  }, [meetingId]);

  // Persist messages whenever they change for the current meeting
  useEffect(() => {
    try {
      if (messages.length > 0) {
        localStorage.setItem(`falcon_chat_${meetingId}`, JSON.stringify(messages));
      } else {
        localStorage.removeItem(`falcon_chat_${meetingId}`);
      }
    } catch (e) {
      console.error('Failed to save chat to localStorage', e);
    }
  }, [messages, meetingId]);

  // Auto-scroll disabled per user request: prevent erratic page scroll-down

  // Adjust textarea height automatically
  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  };

  const handleCopyMessage = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleClearChat = () => {
    setMessages([]);
    setError(null);
    try {
      localStorage.removeItem(`falcon_chat_${meetingId}`);
    } catch {}
  };

  const handleSendMessage = async (questionText?: string) => {
    const textToSend = (questionText || input).trim();
    if (!textToSend || isLoading) return;

    if (!hasTranscript) {
      setError('لا يتوفر نص مفرغ لهذا الاجتماع لطرح الأسئلة عنه.');
      return;
    }

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
      // Build history for backend
      const historyPayload = messages.map(m => ({
        role: m.role,
        content: m.content,
      }));

      const res = await api.chatWithTranscript(meetingId, textToSend, historyPayload);

      const assistantMessage: ChatMessage = {
        id: `assistant-${Date.now()}`,
        role: 'assistant',
        content: res.reply,
        timestamp: new Date().toLocaleTimeString('ar-SA', { hour: '2-digit', minute: '2-digit' }),
      };

      setMessages([...newMessages, assistantMessage]);
    } catch (err: any) {
      console.error('Chat error:', err);
      setError(err?.message || 'تعذر الحصول على إجابة من Falcon LLM. يرجى المحاولة مرة أخرى.');
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

  const renderFormattedContent = (content: string) => {
    // Process markdown bold **text**
    const parts = content.split(/(\*\*.*?\*\*)/g);
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

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-indigo-100 overflow-hidden relative transition-all">
      {/* Top gradient highlight bar */}
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-emerald-400" />

      {/* Header */}
      <div className="p-4 sm:p-5 border-b border-slate-100 flex flex-wrap items-center justify-between gap-3 bg-gradient-to-r from-slate-50/80 to-indigo-50/30">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-indigo-500 text-white flex items-center justify-center shadow-xs">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-bold text-slate-900 text-lg">المحادثة مع Falcon LLM</h3>
              <span className="text-xs bg-indigo-100 text-indigo-700 font-semibold px-2 py-0.5 rounded-full">
                Falcon-H1
              </span>
            </div>
            <div className="flex items-center gap-1.5 text-xs text-emerald-700 font-medium mt-0.5">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
              <span>الإجابات تُستخرج حصراً من النص المفرغ الأصلي (وليس من نص التلخيص)</span>
            </div>
          </div>
        </div>

        {messages.length > 0 && (
          <button
            onClick={handleClearChat}
            disabled={isLoading}
            className="flex items-center gap-1.5 text-xs font-semibold text-slate-500 hover:text-indigo-600 bg-white hover:bg-indigo-50 border border-slate-200 px-3 py-1.5 rounded-lg transition-colors cursor-pointer"
            title="بدء محادثة جديدة"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>محادثة جديدة</span>
          </button>
        )}
      </div>

      {/* Chat messages viewport */}
      <div className="p-4 sm:p-6 min-h-[220px] max-h-[500px] overflow-y-auto space-y-4 bg-slate-50/40">
        {messages.length === 0 ? (
          <div className="py-8 px-4 text-center max-w-lg mx-auto">
            <div className="w-12 h-12 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center mx-auto mb-3 border border-indigo-100/80">
              <MessageSquare className="w-6 h-6" />
            </div>
            <h4 className="font-bold text-slate-800 text-base mb-1">
              اسأل أي سؤال عن تفاصيل الاجتماع
            </h4>
            <p className="text-xs sm:text-sm text-slate-500 leading-relaxed mb-6">
              سيتولى نموذج Falcon مراجعة النص الصوتي المفرغ كاملاً والبحث فيه بدقة للإجابة عن أسئلتك ونسب الأقوال لقائليها.
            </p>

            {/* Quick question suggestions */}
            <div className="space-y-2 text-right">
              <span className="text-xs font-semibold text-slate-400 block px-1">أسئلة مقترحة:</span>
              <div className="flex flex-wrap gap-2">
                {SUGGESTED_QUESTIONS.map((q, idx) => (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => handleSendMessage(q)}
                    disabled={isLoading}
                    className="text-xs bg-white hover:bg-indigo-50 text-slate-700 hover:text-indigo-700 border border-slate-200 hover:border-indigo-200 px-3 py-2 rounded-xl transition-all shadow-2xs text-right cursor-pointer flex items-center gap-1.5"
                  >
                    <Sparkles className="w-3 h-3 text-indigo-500 shrink-0" />
                    <span>{q}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.2 }}
                  className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  {/* Assistant avatar */}
                  {msg.role === 'assistant' && (
                    <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center shrink-0 mt-1 shadow-xs">
                      <Bot className="w-4 h-4" />
                    </div>
                  )}

                  {/* Message bubble */}
                  <div
                    className={`max-w-[85%] sm:max-w-[78%] rounded-2xl p-4 shadow-xs relative group ${
                      msg.role === 'user'
                        ? 'bg-indigo-600 text-white rounded-br-xs'
                        : 'bg-white text-slate-800 border border-slate-200/80 rounded-bl-xs'
                    }`}
                  >
                    <div className="text-[14.5px] leading-relaxed whitespace-pre-wrap break-words">
                      {msg.role === 'assistant' ? renderFormattedContent(msg.content) : msg.content}
                    </div>

                    <div
                      className={`flex items-center justify-between gap-3 mt-2 text-[11px] ${
                        msg.role === 'user' ? 'text-indigo-200' : 'text-slate-400'
                      }`}
                    >
                      <span>{msg.timestamp}</span>

                      {msg.role === 'assistant' && (
                        <button
                          type="button"
                          onClick={() => handleCopyMessage(msg.id, msg.content)}
                          className="opacity-0 group-hover:opacity-100 transition-opacity p-1 text-slate-400 hover:text-indigo-600 hover:bg-slate-100 rounded cursor-pointer"
                          title="نسخ الإجابة"
                        >
                          {copiedId === msg.id ? (
                            <CheckCircle className="w-3.5 h-3.5 text-emerald-600" />
                          ) : (
                            <Copy className="w-3.5 h-3.5" />
                          )}
                        </button>
                      )}
                    </div>
                  </div>

                  {/* User avatar */}
                  {msg.role === 'user' && (
                    <div className="w-8 h-8 rounded-lg bg-slate-200 text-slate-700 flex items-center justify-center shrink-0 mt-1">
                      <User className="w-4 h-4" />
                    </div>
                  )}
                </motion.div>
              ))}
            </AnimatePresence>

            {/* Loading typing bubble */}
            {isLoading && (
              <motion.div
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex items-center gap-3 justify-start"
              >
                <div className="w-8 h-8 rounded-lg bg-indigo-600 text-white flex items-center justify-center shrink-0 shadow-xs">
                  <Bot className="w-4 h-4" />
                </div>
                <div className="bg-white border border-indigo-100 rounded-2xl rounded-bl-xs p-3.5 text-xs text-indigo-700 font-medium flex items-center gap-2 shadow-2xs">
                  <Loader2 className="w-4 h-4 animate-spin text-indigo-600" />
                  <span>Falcon LLM يقرأ النص المفرغ ويُعدّ الإجابة...</span>
                </div>
              </motion.div>
            )}

          </div>
        )}
      </div>

      {/* Error banner */}
      {error && (
        <div className="p-3 bg-rose-50 border-t border-b border-rose-100 text-rose-700 text-xs font-semibold flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-500 shrink-0" />
          <span className="flex-1">{error}</span>
          <button
            type="button"
            onClick={() => setError(null)}
            className="text-rose-500 hover:text-rose-700 cursor-pointer font-bold px-1"
          >
            ✕
          </button>
        </div>
      )}

      {/* Input box section */}
      <div className="p-3 sm:p-4 bg-white border-t border-slate-200">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSendMessage();
          }}
          className="flex items-end gap-2"
        >
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={handleInputChange}
              onKeyDown={handleKeyDown}
              placeholder="اطرح سؤالك حول ما دار في التسجيل... (اضغط Enter للإرسال)"
              disabled={isLoading || !hasTranscript}
              className="w-full resize-none rounded-xl border border-slate-300 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 p-3 pr-4 text-sm text-slate-800 placeholder-slate-400 outline-none transition-all disabled:bg-slate-50 disabled:text-slate-400"
              style={{ minHeight: '44px', maxHeight: '120px' }}
            />
          </div>

          <button
            type="submit"
            disabled={!input.trim() || isLoading || !hasTranscript}
            className="h-11 px-4 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-200 text-white disabled:text-slate-400 rounded-xl font-semibold text-sm transition-all flex items-center gap-2 shadow-xs cursor-pointer disabled:cursor-not-allowed shrink-0"
            title="إرسال السؤال"
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
          <span>اضغط Enter للإرسال و Shift + Enter لسطر جديد</span>
          <span className="text-emerald-600 font-medium">المصدر: التفريغ الصوتي المفرغ</span>
        </div>
      </div>
    </div>
  );
};
