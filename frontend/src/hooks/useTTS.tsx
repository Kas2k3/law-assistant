import { useEffect, useRef, useState } from "react";

export function useTTS() {
  const [speakingIdx, setSpeakingIdx] = useState<number | null>(null);
  const [loadingIdx,  setLoadingIdx]  = useState<number | null>(null);
  const audioRef    = useRef<HTMLAudioElement | null>(null);
  const canceledRef = useRef(false);

  const stop = () => {
    canceledRef.current = true;
    audioRef.current?.pause();
    if (audioRef.current) { audioRef.current.src = ""; audioRef.current = null; }
    setSpeakingIdx(null);
    setLoadingIdx(null);
  };

  const speak = async (
    text: string, idx: number, chatId: string,
    cachedUrl?: string, onUrlReceived?: (url: string) => void,
  ) => {
    if (loadingIdx === idx || speakingIdx === idx) { stop(); return; }
    if (loadingIdx !== null || speakingIdx !== null) { stop(); return; }

    canceledRef.current = false;
    setLoadingIdx(idx);

    try {
      let audioUrl = cachedUrl;

      if (!audioUrl) {
        const res = await fetch("/api/tts", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, voice: "nova", chat_id: chatId, msg_idx: idx }),
        });
        if (!res.ok) throw new Error(`TTS failed: ${res.status}`);
        if (canceledRef.current) return;

        const ct = res.headers.get("Content-Type") ?? "";
        if (ct.includes("application/json")) {
          const data = await res.json() as { url: string };
          audioUrl = data.url;
          onUrlReceived?.(data.url);
        } else {
          const buffer = await res.arrayBuffer();
          audioUrl = URL.createObjectURL(new Blob([buffer], { type: "audio/mpeg" }));
        }
      }

      if (canceledRef.current || !audioUrl) return;

      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      audio.onplay  = () => { setLoadingIdx(null); setSpeakingIdx(idx); };
      audio.onended = () => { setSpeakingIdx(null); audioRef.current = null; };
      audio.onerror = () => { setSpeakingIdx(null); setLoadingIdx(null); audioRef.current = null; };
      await audio.play();
    } catch (err) {
      console.error("[TTS]", err);
      setLoadingIdx(null);
      setSpeakingIdx(null);
    }
  };

  useEffect(() => () => stop(), []); // eslint-disable-line
  return { speak, stop, speakingIdx, loadingIdx };
}