"use client";
import { Message, Passage } from "@/store/chat-store";
import Image from "next/image";
import ReactMarkdown from "react-markdown";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";

const BOT_AVATAR_SRC = toSvgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<rect width="64" height="64" rx="32" fill="#0f172a"/>
<path d="M22 18h20a10 10 0 0 1 10 10v10a12 12 0 0 1-12 12H24A12 12 0 0 1 12 38V28a10 10 0 0 1 10-10z" fill="#e2e8f0"/>
<circle cx="26" cy="34" r="4" fill="#0f172a"/>
<circle cx="38" cy="34" r="4" fill="#0f172a"/>
<path d="M28 46c2.5 2 5.5 2 8 0" stroke="#0f172a" stroke-width="3" fill="none" stroke-linecap="round"/>
<path d="M32 12v8" stroke="#e2e8f0" stroke-width="4" stroke-linecap="round"/>
</svg>`);

const USER_AVATAR_SRC = toSvgDataUri(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#534AB7"/><stop offset="1" stop-color="#1d4ed8"/>
</linearGradient></defs>
<rect width="64" height="64" rx="32" fill="url(#g)"/>
<circle cx="32" cy="26" r="12" fill="#fff" opacity="0.95"/>
<path d="M14 54c3.5-11 13.2-16 18-16s14.5 5 18 16" fill="#fff" opacity="0.95"/>
</svg>`);

function toSvgDataUri(svg: string) {
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

export function Avatar({ type }: { type: "bot" | "user" }) {
  return (
    <Image
      src={type === "bot" ? BOT_AVATAR_SRC : USER_AVATAR_SRC}
      alt={type}
      width={28}
      height={28}
      // md:28 desktop, 24 mobile
      className="h-6 w-6 shrink-0 rounded-full object-cover md:h-7 md:w-7"
    />
  );
}

export function BotBubble({
  content,
  isSelected = false,
}: {
  content: string;
  isSelected?: boolean;
}) {
  return (
    <div
      className="w-fit rounded-2xl rounded-bl-md px-3 py-2.5 text-sm leading-6 shadow-sm transition-all md:px-4 md:py-3 md:leading-7"
      style={{
        // Mobile: 90vw max, Desktop: 82% capped at 760px
        maxWidth: "min(90vw, 760px)",
        background: "var(--bg-msg-bot)",
        border: isSelected ? "1px solid #534AB7" : "0.5px solid var(--border)",
        color: "var(--text-primary)",
      }}
    >
      <div className="[&_a]:underline [&_code]:rounded [&_code]:px-1 [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:p-3 [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:list-decimal [&_ol]:pl-4 [&_li]:my-0.5 [&_p]:my-1.5 md:[&_ul]:pl-5 md:[&_ol]:pl-5 md:[&_li]:my-1 md:[&_p]:my-2">
        <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>{content}</ReactMarkdown>
      </div>
    </div>
  );
}

export function MessageRow({
  m,
  idx,
  isSelected,
  onSelect,
  onSpeak,
  isSpeaking,
  isLoadingTTS,
  chatId,
  onSaveTtsUrl,
  isAnyTTSActive,
}: {
  m: Message;
  idx: number;
  isSelected: boolean;
  onSelect: (passages: Passage[]) => void;
  onSpeak: (
    text: string,
    idx: number,
    chatId: string,
    cachedUrl?: string,
    onUrlReceived?: (url: string) => void
  ) => void;
  isSpeaking: boolean;
  isLoadingTTS: boolean;
  chatId: string;
  onSaveTtsUrl: (msgIdx: number, url: string) => void;
  isAnyTTSActive: boolean;
}) {
  const isUser      = m.role === "user";
  const hasPassages = !isUser && (m.passages?.length ?? 0) > 0;

  return (
    <div className={`flex items-end gap-1.5 md:gap-2 ${isUser ? "justify-end" : "justify-start"}`}>
      {!isUser && <Avatar type="bot" />}

      {isUser ? (
        // ── User bubble ──────────────────────────────────────────────────────
        <div
          className="w-fit rounded-2xl rounded-br-md px-3 py-2.5 text-sm leading-6 md:px-4 md:py-3 md:leading-7"
          style={{
            maxWidth: "min(82vw, 640px)",
            background: "#534AB7",
            color: "#ffffff",
          }}
        >
          <div className="whitespace-pre-wrap break-words">{m.content}</div>
        </div>
      ) : (
        // ── Bot bubble ───────────────────────────────────────────────────────
        <div className="relative min-w-0">
          <BotBubble content={m.content} isSelected={isSelected} />

          {/* Action buttons */}
          <div className="mt-1 flex flex-wrap items-center gap-1">

            {/* TTS button */}
            <button
              type="button"
              onClick={() =>
                onSpeak(
                  m.content,
                  idx,
                  chatId,
                  m.tts_url ?? undefined,
                  (url) => onSaveTtsUrl(idx, url)
                )
              }
              disabled={isLoadingTTS}
              title={
                isLoadingTTS
                  ? "Đang tải giọng đọc..."
                  : isSpeaking
                  ? "Dừng đọc"
                  : m.tts_url
                  ? "Phát giọng đọc (đã lưu)"
                  : "Đọc to (AI)"
              }
              className="flex items-center gap-1 rounded-lg px-2 py-1 text-xs transition disabled:opacity-60"
              style={{
                background:
                  isSpeaking || isLoadingTTS
                    ? "#EEEDFE"
                    : m.tts_url
                    ? "rgba(83,74,183,0.08)"
                    : "var(--bg-surface)",
                border:
                  isSpeaking || isLoadingTTS
                    ? "0.5px solid #534AB7"
                    : m.tts_url
                    ? "0.5px solid rgba(83,74,183,0.4)"
                    : "0.5px solid var(--border)",
                color:
                  isSpeaking || isLoadingTTS
                    ? "#3C3489"
                    : m.tts_url
                    ? "#534AB7"
                    : "var(--text-muted)",
              }}
            >
              {isLoadingTTS ? (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2" className="animate-spin">
                  <circle cx="12" cy="12" r="10" strokeOpacity="0.25"/>
                  <path d="M12 2a10 10 0 0 1 10 10" strokeOpacity="1"/>
                </svg>
              ) : isSpeaking ? (
                <span className="flex items-end gap-[2px] h-3">
                  {[1, 2, 3].map((i) => (
                    <span key={i} className="w-[3px] rounded-sm"
                      style={{
                        background: "#534AB7",
                        height: i === 2 ? "100%" : "60%",
                        animation: `eq-bar ${0.4 + i * 0.15}s ease-in-out infinite alternate`,
                      }}
                    />
                  ))}
                </span>
              ) : m.tts_url ? (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                  <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
                  <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
                </svg>
              ) : (
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none"
                  stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/>
                  <path d="M15.54 8.46a5 5 0 0 1 0 7.07"/>
                  <path d="M19.07 4.93a10 10 0 0 1 0 14.14"/>
                </svg>
              )}
              <span>
                {isLoadingTTS ? "Đang tải..." : isSpeaking ? "Dừng" : m.tts_url ? "Phát" : "Đọc"}
              </span>
            </button>

            {/* Passages button */}
            {hasPassages && (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); onSelect(m.passages!); }}
                className="flex items-center gap-1 rounded-lg px-2 py-1 text-xs transition"
                style={{
                  background: isSelected ? "#EEEDFE" : "var(--bg-surface)",
                  border: isSelected ? "0.5px solid #534AB7" : "0.5px solid var(--border)",
                  color: isSelected ? "#3C3489" : "var(--text-muted)",
                }}
              >
                <span>📄</span>
                <span>{m.passages!.length} tài liệu</span>
              </button>
            )}
          </div>
        </div>
      )}

      {isUser && <Avatar type="user" />}
    </div>
  );
}