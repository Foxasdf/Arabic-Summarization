import { useState, useEffect } from 'react';
import { Trash2 } from 'lucide-react';
import { AppView, MainTab, Meeting } from './types';
import { Header } from './components/Layout/Header';
import { Sidebar } from './components/Layout/Sidebar';
import { useMeetings } from './hooks/useMeetings';
import { DropZone } from './components/Upload/DropZone';
import { FilePreview } from './components/Upload/FilePreview';
import { ProcessingView } from './components/Processing/ProcessingView';
import { TranscriptView } from './components/Transcript/TranscriptView';
import { SummaryView } from './components/Summary/SummaryView';
import { FalconChat } from './components/Chat/FalconChat';
import { FalconStandaloneChat } from './components/Chat/FalconStandaloneChat';
import { DeleteConfirmationModal } from './components/DeleteConfirmationModal';
import { api } from './services/api';

function App() {
  const { meetings, isLoading: meetingsLoading, fetchMeetings, deleteMeeting } = useMeetings();
  const [activeTab, setActiveTab] = useState<MainTab>('meetings');
  const [view, setView] = useState<AppView>('home');
  const [currentMeetingId, setCurrentMeetingId] = useState<string | null>(null);
  const [currentMeeting, setCurrentMeeting] = useState<Meeting | null>(null);
  const [fileId, setFileId] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [numSpeakers, setNumSpeakers] = useState<number | undefined>(undefined);
  const [selectedAsrModel, setSelectedAsrModel] = useState<string>('cohere-transcribe-arabic');
  const [selectedDenoiseMode, setSelectedDenoiseMode] = useState<string>('none');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [meetingToDelete, setMeetingToDelete] = useState<{ id: string; filename: string } | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // Fetch meeting details when selected
  useEffect(() => {
    if (currentMeetingId && view === 'meeting' && activeTab === 'meetings') {
      const fetchMeetingDetails = async () => {
        try {
          const data = await api.getMeeting(currentMeetingId);
          setCurrentMeeting(data);
        } catch (error) {
          console.error("Failed to fetch meeting", error);
        }
      };
      fetchMeetingDetails();
    }
  }, [currentMeetingId, view, activeTab]);

  const handleTabChange = (tab: MainTab) => {
    setActiveTab(tab);
  };

  const handleNewMeeting = () => {
    setActiveTab('meetings');
    setView('home');
    setCurrentMeetingId(null);
    setCurrentMeeting(null);
    setFileId(null);
    setSelectedFile(null);
    setNumSpeakers(undefined);
    setSelectedAsrModel('cohere-transcribe-arabic');
    setSelectedDenoiseMode('none');
    setIsUploading(false);
    setUploadProgress(0);
  };

  const handleSelectMeeting = (id: string) => {
    setActiveTab('meetings');
    if (id !== currentMeetingId) {
      setCurrentMeeting(null);
    }
    setCurrentMeetingId(id);
    setView('meeting');
  };

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setIsUploading(false);
    setUploadProgress(0);
  };

  const handleUploadStart = async (selectedNumSpeakers?: number, asrModel?: string, denoiseMode?: string) => {
    if (!selectedFile || isUploading) return;
    try {
      setIsUploading(true);
      setUploadProgress(0);
      setNumSpeakers(selectedNumSpeakers);
      if (asrModel) setSelectedAsrModel(asrModel);
      if (denoiseMode) setSelectedDenoiseMode(denoiseMode);
      const res = await api.uploadAudio(selectedFile, (percent) => {
        setUploadProgress(percent);
      });
      setFileId(res.file_id);
      setView('processing');
    } catch (error: any) {
      console.error("Upload failed", error);
      alert(error?.message || "فشل رفع الملف، يرجى المحاولة مرة أخرى.");
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleCancelUpload = () => {
    if (isUploading) return;
    setSelectedFile(null);
    setIsUploading(false);
    setUploadProgress(0);
  };

  const handleTranscriptionComplete = async () => {
    if (!fileId) return;
    try {
      const meeting = await api.getMeeting(fileId);
      setCurrentMeeting(meeting);
      setCurrentMeetingId(meeting.id);
      setView('meeting');
      fetchMeetings();
    } catch (error) {
      console.error("Failed to load meeting after transcription", error);
    }
  };

  const handleSummaryUpdated = (summary: string) => {
    if (currentMeeting) {
      setCurrentMeeting({ ...currentMeeting, summary, status: 'completed' });
      fetchMeetings();
    }
  };

  const handleSpeakersUpdated = (speakers: Record<string, string>) => {
    if (currentMeeting) {
      setCurrentMeeting({
        ...currentMeeting,
        speakers,
        speakers_json: speakers,
      });
      fetchMeetings();
    }
  };

  const handleDeleteMeetingClick = (id: string, filename: string) => {
    setMeetingToDelete({ id, filename });
  };

  const handleConfirmDelete = async () => {
    if (!meetingToDelete) return;
    setIsDeleting(true);
    try {
      await deleteMeeting(meetingToDelete.id);
      try {
        localStorage.removeItem(`falcon_chat_${meetingToDelete.id}`);
      } catch {}
      if (currentMeetingId === meetingToDelete.id) {
        handleNewMeeting();
      }
      setMeetingToDelete(null);
    } catch (error) {
      console.error("Failed to delete meeting", error);
      alert("فشل حذف الاجتماع. يرجى المحاولة مرة أخرى.");
    } finally {
      setIsDeleting(false);
    }
  };

  const handleCancelDelete = () => {
    if (!isDeleting) {
      setMeetingToDelete(null);
    }
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-slate-50 text-slate-900 font-sans" dir="rtl">
      <Header activeTab={activeTab} onTabChange={handleTabChange} />
      
      <div className="flex flex-1 overflow-hidden">
        <Sidebar 
          meetings={meetings}
          currentMeetingId={currentMeetingId}
          activeTab={activeTab}
          onSelectMeeting={handleSelectMeeting}
          onNewMeeting={handleNewMeeting}
          onOpenFalconChat={() => setActiveTab('falcon-chat')}
          onDeleteMeeting={handleDeleteMeetingClick}
          isLoading={meetingsLoading}
        />
        
        <main className="flex-1 overflow-y-auto relative flex flex-col">
          <div className="absolute inset-0 bg-gradient-to-br from-slate-50 via-white to-indigo-50/30 -z-10" />
          
          {activeTab === 'falcon-chat' ? (
            <div className="flex-1 h-full overflow-hidden">
              <FalconStandaloneChat />
            </div>
          ) : (
            <div className="max-w-4xl mx-auto p-6 md:p-8 min-h-full flex flex-col w-full">
              {view === 'home' && !selectedFile && (
                <div className="flex-1 flex flex-col justify-center max-w-2xl mx-auto w-full">
                  <div className="text-center mb-10">
                    <h2 className="text-3xl font-bold text-slate-800 mb-4">أهلاً بك في ملخص الاجتماعات</h2>
                    <p className="text-slate-600 text-lg">ارفع تسجيل اجتماعك وسنقوم بتفريغه وتلخيصه بدقة عالية</p>
                  </div>
                  <DropZone onFileSelect={handleFileSelect} />
                </div>
              )}

              {view === 'home' && selectedFile && (
                <div className="flex-1 flex flex-col justify-center max-w-xl mx-auto w-full">
                  <FilePreview 
                    file={selectedFile} 
                    onUpload={handleUploadStart} 
                    onCancel={handleCancelUpload}
                    isUploading={isUploading}
                    uploadProgress={uploadProgress}
                  />
                </div>
              )}

              {view === 'processing' && fileId && (
                <div className="flex-1 flex flex-col justify-center max-w-2xl mx-auto w-full">
                  <ProcessingView 
                    fileId={fileId}
                    numSpeakers={numSpeakers}
                    asrModel={selectedAsrModel}
                    denoiseMode={selectedDenoiseMode}
                    onComplete={handleTranscriptionComplete} 
                  />
                </div>
              )}

              {view === 'meeting' && currentMeeting && (
                <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
                  <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                      <div>
                        <h2 className="text-2xl font-bold text-slate-900 mb-2">{currentMeeting.filename}</h2>
                        <div className="text-sm text-slate-500">
                          تم الرفع في: {new Date(currentMeeting.upload_time).toLocaleDateString('ar-SA')}
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={() => handleDeleteMeetingClick(currentMeeting.id, currentMeeting.filename)}
                        className="flex items-center justify-center gap-2 text-rose-600 hover:text-white bg-rose-50 hover:bg-rose-600 border border-rose-200 hover:border-rose-600 px-4 py-2 rounded-xl transition-all font-medium text-sm shadow-sm cursor-pointer self-start sm:self-auto"
                        title="حذف هذا الاجتماع"
                      >
                        <Trash2 className="w-4 h-4" />
                        <span>حذف الاجتماع</span>
                      </button>
                    </div>
                  </div>

                  {currentMeeting.summary && (
                    <>
                      <SummaryView summary={currentMeeting.summary} />
                      <FalconChat 
                        key={currentMeeting.id}
                        meetingId={currentMeeting.id} 
                        hasTranscript={!!(currentMeeting.transcript && currentMeeting.transcript.length > 0)}
                      />
                    </>
                  )}

                  {currentMeeting.transcript && currentMeeting.transcript.length > 0 && (
                    <TranscriptView 
                      segments={currentMeeting.transcript} 
                      speakers={currentMeeting.speakers || currentMeeting.speakers_json || {}}
                      meetingId={currentMeeting.id}
                      onSummaryGenerated={handleSummaryUpdated}
                      onSpeakersUpdated={handleSpeakersUpdated}
                    />
                  )}
                </div>
              )}

              {view === 'meeting' && !currentMeeting && (
                <div className="flex-1 flex flex-col items-center justify-center py-20 text-slate-500">
                  <div className="w-10 h-10 border-4 border-indigo-600 border-t-transparent rounded-full animate-spin mb-4" />
                  <p className="text-sm font-medium">جاري تحميل تفاصيل الاجتماع...</p>
                </div>
              )}
            </div>
          )}
        </main>
      </div>

      <DeleteConfirmationModal
        isOpen={!!meetingToDelete}
        meetingTitle={meetingToDelete?.filename || ''}
        isDeleting={isDeleting}
        onConfirm={handleConfirmDelete}
        onClose={handleCancelDelete}
      />
    </div>
  );
}

export default App;
