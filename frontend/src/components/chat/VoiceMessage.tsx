"use client";
import { useEffect, useRef, useState } from "react";

const SLOT_W   = 6;
const MIN_BARS = 8;
const fmt = (s: number) =>
  `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export function VoiceMessage({ audioBlob, onCancel }: { audioBlob: Blob; onCancel: () => void }) {
  const [playing,  setPlaying]  = useState(false);
  const [current,  setCurrent]  = useState(0);
  const [total,    setTotal]    = useState(0);
  const [bars,     setBars]     = useState(40);
  const [waveform, setWaveform] = useState<number[]>([]);
  const audioRef    = useRef<HTMLAudioElement | null>(null);
  const rafRef      = useRef<number>(0);
  const waveContRef = useRef<HTMLDivElement>(null);
  const rawSamples  = useRef<number[]>([]);

  // ── RAF ──────────────────────────────────────────────────────────────────
  const startRAF = (a: HTMLAudioElement) => {
    const tick = () => { setCurrent(a.currentTime); rafRef.current = requestAnimationFrame(tick); };
    rafRef.current = requestAnimationFrame(tick);
  };
  const stopRAF = () => cancelAnimationFrame(rafRef.current);

  // ── Resample raw → n bars, peak = 100% ───────────────────────────────────
  const resample = (raw: number[], n: number) => {
    if (!raw.length || n < 1) return;
    const step = raw.length / n;

    const buckets = Array.from({ length: n }, (_, i) => {
      const start = Math.floor(i * step);
      const end   = Math.min(Math.floor((i + 1) * step), raw.length);
      let sum = 0;
      for (let j = start; j < end; j++) sum += raw[j];
      return sum / Math.max(1, end - start);
    });

    const peak = Math.max(...buckets, 0.001);
    setWaveform(buckets.map((v) => Math.max(0.06, v / peak)));
  };

  // ── Decode audio + setup HTML Audio ──────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    const url = URL.createObjectURL(audioBlob);

    audioBlob.arrayBuffer()
      .then((buf) => new AudioContext().decodeAudioData(buf))
      .then((decoded) => {
        if (cancelled) return;
        const data = decoded.getChannelData(0);
        const RAW  = 1000;
        const step = Math.floor(data.length / RAW);

        // Lưu raw chưa normalize để resample tự tính peak
        const raw = Array.from({ length: RAW }, (_, i) => {
          let sum = 0;
          for (let j = 0; j < step; j++) sum += Math.abs(data[i * step + j] ?? 0);
          return sum / step;
        });
        rawSamples.current = raw;
        setBars((b) => { resample(raw, b); return b; });
      })
      .catch(() => {});

    const a = new Audio(url);
    a.onloadedmetadata = () => { if (!cancelled) setTotal(isFinite(a.duration) ? a.duration : 0); };
    a.onplay  = () => { if (!cancelled) { setPlaying(true);  startRAF(a); } };
    a.onpause = () => { if (!cancelled) { setPlaying(false); stopRAF(); } };
    a.onended = () => {
      if (!cancelled) { setPlaying(false); stopRAF(); setCurrent(0); a.currentTime = 0; }
    };
    audioRef.current = a;

    return () => { cancelled = true; stopRAF(); a.pause(); URL.revokeObjectURL(url); };
  }, [audioBlob]); // eslint-disable-line

  // ── ResizeObserver ────────────────────────────────────────────────────────
  useEffect(() => {
    const el = waveContRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const n = Math.max(MIN_BARS, Math.floor(el.clientWidth / SLOT_W));
      setBars(n);
      if (rawSamples.current.length) {
        resample(rawSamples.current, n);
      } else {
        setWaveform(
          Array.from({ length: n }, (_, i) => 0.15 + Math.abs(Math.sin(i * 0.4)) * 0.85)
        );
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []); // eslint-disable-line

  // ── Controls ──────────────────────────────────────────────────────────────
  const toggle = () => {
    const a = audioRef.current;
    if (!a) return;
    playing ? a.pause() : a.play().catch(() => setPlaying(false));
  };

  const seek = (barIdx: number) => {
    const a = audioRef.current;
    if (!a || total === 0) return;
    const newTime = (barIdx / bars) * total;
    a.currentTime = newTime;
    setCurrent(newTime);
  };

  const progress   = total > 0 ? current / total : 0;
  const activeBars = Math.floor(progress * bars);

  return (
    <div className="flex flex-1 items-center gap-3 min-w-0">

      {/* Xóa */}
      <button
        type="button" onClick={onCancel} title="Hủy"
        className="shrink-0 flex items-center justify-center rounded-full w-8 h-8 transition"
        style={{ background: "var(--bg-hover)", color: "var(--text-muted)" }}
        onMouseEnter={(e) => (e.currentTarget.style.color = "#ef4444")}
        onMouseLeave={(e) => (e.currentTarget.style.color = "var(--text-muted)")}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
          stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
          <polyline points="3 6 5 6 21 6"/>
          <path d="M19 6l-1 14H6L5 6"/>
          <path d="M9 6V4h6v2"/>
        </svg>
      </button>

      {/* Play / Pause */}
      <button
        type="button" onClick={toggle}
        className="shrink-0 flex items-center justify-center rounded-full w-9 h-9 transition"
        style={{ background: "#534AB7", color: "#fff", boxShadow: "0 2px 8px rgba(83,74,183,0.35)" }}
        onMouseEnter={(e) => (e.currentTarget.style.background = "#3C3489")}
        onMouseLeave={(e) => (e.currentTarget.style.background = "#534AB7")}
      >
        {playing ? (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <rect x="5" y="4" width="4" height="16" rx="1.5"/>
            <rect x="15" y="4" width="4" height="16" rx="1.5"/>
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
            <polygon points="6 3 20 12 6 21 6 3"/>
          </svg>
        )}
      </button>

      {/* Waveform */}
      <div
        ref={waveContRef}
        className="flex flex-1 items-center min-w-0 cursor-pointer"
        style={{ height: 36, gap: `${SLOT_W * 0.35}px` }}
      >
        {waveform.map((amp, i) => (
          <div
            key={i}
            onClick={() => seek(i)}
            className="shrink-0 rounded-full"
            style={{
              flex: "1 1 0",
              maxWidth: SLOT_W * 0.65,
              height: `${Math.max(8, Math.round(amp * 100))}%`,
              background: i < activeBars ? "#534AB7" : "var(--border-strong)",
              opacity: i < activeBars ? 1 : 0.5,
              alignSelf: "center",
            }}
          />
        ))}
      </div>

      {/* Timestamp */}
      <span
        className="shrink-0 text-xs font-mono tabular-nums"
        style={{ color: "var(--text-muted)", minWidth: 68 }}
      >
        {playing || current > 0 ? fmt(current) : fmt(total)} / {total > 0 ? fmt(total) : "--:--"}
      </span>
    </div>
  );
}