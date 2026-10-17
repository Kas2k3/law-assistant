import { useEffect, useRef, useState } from "react";

export function useSTT() {
  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState<Blob | null>(null);
  const [duration,  setDuration]  = useState(0);
  const [supported, setSupported] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef        = useRef<BlobPart[]>([]);
  const timerRef         = useRef<ReturnType<typeof setInterval> | null>(null);
  const canceledRef      = useRef(false);

  useEffect(() => { setSupported(!!navigator.mediaDevices?.getUserMedia); }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current   = [];
      canceledRef.current = false;
      mr.ondataavailable  = (e) => chunksRef.current.push(e.data);
      mr.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        if (canceledRef.current) return;
        setAudioBlob(new Blob(chunksRef.current, { type: "audio/webm" }));
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setRecording(true);
      setDuration(0);
      timerRef.current = setInterval(() => setDuration((d) => d + 1), 1000);
    } catch (err) {
      console.error("[STT] mic error:", err);
    }
  };

  const stopRecording = () => {
    mediaRecorderRef.current?.stop();
    setRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  const cancelRecording = () => {
    canceledRef.current = true;
    mediaRecorderRef.current?.stop();
    setRecording(false);
    setAudioBlob(null);
    setDuration(0);
    if (timerRef.current) clearInterval(timerRef.current);
    chunksRef.current = [];
  };

  const clearBlob = () => setAudioBlob(null);

  useEffect(() => () => { if (timerRef.current) clearInterval(timerRef.current); }, []);

  return { startRecording, stopRecording, cancelRecording, clearBlob, recording, audioBlob, duration, supported };
}