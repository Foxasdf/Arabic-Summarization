import { useState, useCallback } from 'react';

export function useUpload() {
  const [isDragging, setIsDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const validateFile = (selectedFile: File) => {
    const validTypes = ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/x-m4a', 'audio/mp4', 'video/mp4'];
    
    if (!validTypes.includes(selectedFile.type) && !selectedFile.name.match(/\.(mp3|wav|ogg|m4a|mp4)$/i)) {
      setError('صيغة الملف غير مدعومة. الرجاء رفع ملف صوتي.');
      return false;
    }
    
    // 500MB limit
    if (selectedFile.size > 500 * 1024 * 1024) {
      setError('حجم الملف كبير جداً. الحد الأقصى هو 500 ميجابايت.');
      return false;
    }

    setError(null);
    return true;
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFile = e.dataTransfer.files[0];
      if (validateFile(droppedFile)) {
        setFile(droppedFile);
      }
    }
  }, []);

  const onFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      if (validateFile(selectedFile)) {
        setFile(selectedFile);
      }
    }
  }, []);

  const reset = useCallback(() => {
    setFile(null);
    setError(null);
    setIsDragging(false);
  }, []);

  return {
    isDragging,
    file,
    error,
    onDragOver,
    onDragLeave,
    onDrop,
    onFileSelect,
    reset,
  };
}
