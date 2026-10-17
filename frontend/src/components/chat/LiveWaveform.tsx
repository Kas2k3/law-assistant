"use client";
import { useEffect, useRef } from "react";

const BAR_W_RATIO = 0.55; // bar chiếm 55% slot
const MIN_BAR_W   = 3;    // px tối thiểu
const MAX_BAR_W   = 10;   // px tối đa
const SLOT_W      = 7;    // px mỗi slot (bar + gap), điều chỉnh để thay đổi mật độ

export function LiveWaveform({ recording }: { recording: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef    = useRef<number>(0);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    if (!recording) { cancelAnimationFrame(rafRef.current); return; }

    let cancelled = false;

    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      if (cancelled) { stream.getTracks().forEach(t => t.stop()); return; }
      streamRef.current = stream;

      const actx    = new AudioContext();
      const src     = actx.createMediaStreamSource(stream);
      const analyser = actx.createAnalyser();
      analyser.fftSize = 256;
      src.connect(analyser);

      const data = new Uint8Array(analyser.frequencyBinCount);

      const draw = () => {
        if (cancelled) return;
        rafRef.current = requestAnimationFrame(draw);

        const canvas = canvasRef.current;
        if (!canvas) return;

        const dpr = window.devicePixelRatio || 1;
        const W   = canvas.offsetWidth;
        const H   = canvas.offsetHeight;

        // ── Tính số bar động từ chiều rộng thực tế ──
        const bars   = Math.max(4, Math.floor(W / SLOT_W));
        const barW   = Math.min(MAX_BAR_W, Math.max(MIN_BAR_W, SLOT_W * BAR_W_RATIO));
        const gap    = (W - bars * barW) / (bars - 1); // chia đều khoảng cách

        canvas.width  = W * dpr;
        canvas.height = H * dpr;
        const gc = canvas.getContext("2d");
        if (!gc) return;
        gc.scale(dpr, dpr);
        gc.clearRect(0, 0, W, H);

        analyser.getByteFrequencyData(data);
        const step = Math.max(1, Math.floor(data.length / bars));

        for (let i = 0; i < bars; i++) {
          // Lấy trung bình của một dải tần
          let sum = 0;
          for (let j = 0; j < step; j++) sum += data[i * step + j] ?? 0;
          const amp  = sum / step / 255;

          const barH = Math.max(3, amp * H * 0.88);
          const x    = i * (barW + gap);
          const y    = (H - barH) / 2;

          gc.fillStyle = `rgba(83,74,183,${0.35 + amp * 0.65})`;
          gc.beginPath();
          gc.roundRect(x, y, barW, barH, barW / 2);
          gc.fill();
        }
      };

      draw();
    }).catch(console.error);

    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
      streamRef.current?.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    };
  }, [recording]);

  return (
    <canvas
      ref={canvasRef}
      className="flex-1 block"
      style={{ height: 36 }}
    />
  );
}