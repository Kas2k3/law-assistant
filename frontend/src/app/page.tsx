"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import Image from "next/image";
import { chatActions, Chat, Message, useChatStore, Passage } from "@/store/chat-store";
import {
  Plus, Search, MoreHorizontal, Pin, Trash2, Pencil,
  MessageCircle, ChevronLeft, ChevronRight, ChevronDown,
  Clock, BookOpen, Folder, Menu, X,
} from "lucide-react";
import { useTTS } from "@/hooks/useTTS";
import { useSTT } from "@/hooks/useSTT";
import { MessageRow, Avatar, BotBubble } from "@/components/chat/MessageRow";
import { PassagePanel, PassageFullModal } from "@/components/chat/PassagePanel";
import { LiveWaveform } from "@/components/chat/LiveWaveform";
import { VoiceMessage } from "@/components/chat/VoiceMessage";

if (typeof CanvasRenderingContext2D !== "undefined" &&
  !CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(
    x: number, y: number, w: number, h: number, r: number
  ) {
    this.beginPath();
    this.moveTo(x + r, y);
    this.arcTo(x + w, y, x + w, y + h, r);
    this.arcTo(x + w, y + h, x, y + h, r);
    this.arcTo(x, y + h, x, y, r);
    this.arcTo(x, y, x + w, y, r);
    this.closePath();
  };
}

function autoTitle(messages: Message[], currentTitle: string) {
  if (!currentTitle.startsWith("Cuộc trò chuyện")) return currentTitle;
  const firstUser = messages.find((m) => m.role === "user")?.content?.trim();
  if (!firstUser) return currentTitle;
  return firstUser.length > 40 ? `${firstUser.slice(0, 40)}...` : firstUser;
}

function sleep(ms: number) {
  return new Promise<void>((resolve) => setTimeout(resolve, ms));
}

/**
 * Sau khi đã có đủ chuỗi trả lời: hiển thị dần (mặc định từng ký tự) để Markdown render theo prefix.
 */
async function revealAnswerProgressively(
  text: string,
  setShown: (s: string) => void,
  opts?: { charStep?: number; delayMs?: number }
): Promise<void> {
  const t = text || "";
  const charStep = Math.max(1, opts?.charStep ?? 1);
  const delayMs = Math.max(0, opts?.delayMs ?? 11);
  let n = 0;
  while (n < t.length) {
    n = Math.min(t.length, n + charStep);
    setShown(t.slice(0, n));
    if (n < t.length) await sleep(delayMs);
  }
}

export default function Home() {
  const { chats, activeId, loading, error } = useChatStore((s) => s);
  const [search, setSearch]               = useState("");
  const [input, setInput]                 = useState("");
  const [sidebarOpen, setSidebarOpen]     = useState(true);
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false);
  const [openMenuChatId, setOpenMenuChatId] = useState<string | null>(null);
  // Stores the pixel position of the "..." button so we can portal the menu there
  const [menuAnchor, setMenuAnchor] = useState<{ top: number; left: number } | null>(null);
  const [sending, setSending]             = useState(false);
  const [renameChatId, setRenameChatId]   = useState<string | null>(null);
  const [renameValue, setRenameValue]     = useState("");
  const [loadingDots, setLoadingDots]     = useState(".");
  const [pendingStatus, setPendingStatus] = useState<string | null>(null);
  const [pendingChatId, setPendingChatId] = useState<string | null>(null);
  const [agentStep, setAgentStep] = useState<{ cur: number; max: number } | null>(null);
  const [streamingChatId, setStreamingChatId] = useState<string | null>(null);
  /** Đang gõ dần câu trả lời đã nhận đủ (Markdown qua BotBubble). */
  const [revealedStreamText, setRevealedStreamText] = useState("");
  /** true trong lúc reveal (kể cả chưa có ký tự nào — hiện bubble chờ). */
  const [postStreamReveal, setPostStreamReveal] = useState(false);
  const [darkMode, setDarkMode]           = useState(true);
  const [queryMode, setQueryMode]         = useState<"normal" | "situation">("normal");
  const [selectedPassages, setSelectedPassages] = useState<Passage[] | null>(null);
  const [selectedMsgIdx, setSelectedMsgIdx]     = useState<number | null>(null);
  const [fullViewPassage, setFullViewPassage]   = useState<Passage | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  // ref for the portal menu div — used to detect outside clicks
  const menuRef   = useRef<HTMLDivElement | null>(null);
  // tracks whether the current open-gesture already "consumed" a pointerdown
  const openingRef = useRef(false);

  const activeChat = useMemo(
    () => chats.find((c) => c._id === activeId) ?? null,
    [chats, activeId]
  );

  const { speak, speakingIdx, loadingIdx } = useTTS();
  const {
    startRecording, stopRecording, cancelRecording, clearBlob,
    recording, audioBlob, duration, supported: sttSupported,
  } = useSTT();

  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);

  useEffect(() => { void chatActions.ensureChatsLoaded(); }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeChat?.messages, revealedStreamText]);

  // ── Outside-click logic: close the portal menu on any tap/click outside it ──
  useEffect(() => {
    if (!openMenuChatId) return;

    function handleOutside(e: PointerEvent) {
      // If this is the same gesture that opened the menu, ignore it
      if (openingRef.current) {
        openingRef.current = false;
        return;
      }
      if (menuRef.current && menuRef.current.contains(e.target as Node)) return;
      setOpenMenuChatId(null);
      setMenuAnchor(null);
    }

    document.addEventListener("pointerdown", handleOutside);
    return () => document.removeEventListener("pointerdown", handleOutside);
  }, [openMenuChatId]);

  useEffect(() => {
    if (!sending) { setLoadingDots("."); return; }
    const frames = [".", "..", "..."];
    let i = 0;
    const t = setInterval(() => { i = (i + 1) % frames.length; setLoadingDots(frames[i]); }, 350);
    return () => clearInterval(t);
  }, [sending]);

  useEffect(() => { setMobileSidebarOpen(false); }, [activeId]);

  async function createChat() { await chatActions.createChat(); }
  async function patchChat(id: string, patch: Partial<Chat>) { await chatActions.patchChat(id, patch); }
  async function removeChat(id: string) { await chatActions.removeChat(id); }

  const saveTtsUrl = async (msgIdx: number, url: string) => {
    if (!activeChat) return;
    const updated = activeChat.messages.map((m, i) =>
      i === msgIdx ? { ...m, tts_url: url } : m
    );
    await patchChat(activeChat._id, { messages: updated });
  };

  const handleRename = async () => {
    if (!renameChatId || !renameValue.trim()) return;
    await patchChat(renameChatId, { title: renameValue.trim() });
    setRenameChatId(null);
    setRenameValue("");
  };

  async function onSend(e: FormEvent) {
    e.preventDefault();
    const promptSent = input.trim();
    if (!promptSent || !activeChat) return;

    const nextMessages: Message[] = [
      ...activeChat.messages,
      { role: "user", content: promptSent },
    ];
    setInput("");
    setSending(true);
    setPendingChatId(activeChat._id);
    setPendingStatus(null);

    await patchChat(activeChat._id, {
      messages: nextMessages,
      title: autoTitle(nextMessages, activeChat.title),
    });

    let fullText = "";
    let metaPassages: Passage[] = [];

    try {
      const res = await fetch("/api/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: promptSent, history: nextMessages, query_mode: queryMode }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let   buffer  = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          try {
            const evt = JSON.parse(raw) as {
              type: "meta" | "token" | "done" | "error" | "status";
              text?: string; passages?: Passage[]; message?: string;
              iteration?: number; max?: number;
            };
            if (evt.type === "status") {
              setPendingStatus(evt.text ?? "");
              if (evt.iteration != null && evt.max != null)
                setAgentStep({ cur: evt.iteration, max: evt.max });
            } else if (evt.type === "meta") {
              setPendingStatus(null); setPendingChatId(null); setAgentStep(null);
              setStreamingChatId(activeChat._id);
              metaPassages = evt.passages ?? [];
            } else if (evt.type === "token") {
              fullText += evt.text ?? "";
            } else if (evt.type === "error") {
              setPendingStatus(null); setPendingChatId(null); setAgentStep(null);
              setStreamingChatId(activeChat._id);
              fullText += `\n\n⚠️ ${evt.message}`;
              break;
            } else if (evt.type === "done") {
              break;
            }
          } catch { /* ignore */ }
        }
      }

      const answer =
        fullText.trim() ||
        "Mình chưa thể tạo câu trả lời. Vui lòng thử lại.";

      setStreamingChatId(activeChat._id);
      setRevealedStreamText("");
      setPostStreamReveal(true);
      await revealAnswerProgressively(answer, setRevealedStreamText, {
        charStep: 2,
        delayMs: 8,
      });
      setPostStreamReveal(false);
      setRevealedStreamText("");
      setStreamingChatId(null);

      await patchChat(activeChat._id, {
        messages: [...nextMessages, {
          role: "assistant",
          content: answer,
          passages: metaPassages,
          tts_url: null,
        }],
      });
    } catch (err) {
      console.error("[onSend]", err);
      const errMsg = "Mình gặp lỗi kết nối. Vui lòng thử lại sau.";
      setStreamingChatId(activeChat._id);
      setRevealedStreamText("");
      setPostStreamReveal(true);
      await revealAnswerProgressively(errMsg, setRevealedStreamText, { charStep: 1, delayMs: 11 });
      setPostStreamReveal(false);
      setRevealedStreamText("");
      setStreamingChatId(null);
      await patchChat(activeChat._id, {
        messages: [...nextMessages, { role: "assistant", content: errMsg, passages: [], tts_url: null }],
      });
    } finally {
      setSending(false); setPendingStatus(null); setPendingChatId(null);
      setAgentStep(null); setStreamingChatId(null);
      setRevealedStreamText("");
      setPostStreamReveal(false);
    }
  }

  const sendVoice = async () => {
    if (!audioBlob) return;
    setSending(true);
    const formData = new FormData();
    formData.append("file", audioBlob, "audio.webm");
    try {
      const res  = await fetch("/api/stt", { method: "POST", body: formData });
      const data = await res.json() as { text: string };
      if (data.text) { setInput(data.text); clearBlob(); }
    } catch (err) {
      console.error("[STT]", err);
    } finally {
      setSending(false); clearBlob();
    }
  };

  const filtered      = chats.filter((c) => (c.title ?? "").toLowerCase().includes(search.toLowerCase()));
  const pinnedChats   = filtered.filter((c) => c.pinned);
  const unpinnedChats = filtered.filter((c) => !c.pinned);

  // ── Portal-based ChatMenu ────────────────────────────────────────────────────
  // Renders into document.body so it is never clipped by overflow:hidden parents.
  function ChatMenuPortal({ chat }: { chat: Chat }) {
    if (openMenuChatId !== chat._id || !menuAnchor) return null;

    const menuContent = (
      <div
        ref={menuRef}
        style={{
          position: "fixed",
          top: menuAnchor.top,
          left: menuAnchor.left,
          width: 200,
          borderRadius: 12,
          padding: 4,
          background: "var(--bg-surface)",
          border: "0.5px solid var(--border-strong)",
          boxShadow: "0 8px 32px rgba(0,0,0,0.35)",
          zIndex: 9000,
        }}
      >
        {/* Đổi tên */}
        <button
          className="w-full rounded-lg px-3 py-2 text-left text-sm"
          style={{ color: "var(--text-primary)" }}
          onPointerDown={(e) => e.stopPropagation()}
          onClick={(e) => {
            e.stopPropagation();
            setRenameChatId(chat._id);
            setRenameValue(chat.title ?? "");
            setOpenMenuChatId(null);
            setMenuAnchor(null);
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-hover)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
        >
          <div className="flex items-center gap-2"><Pencil size={16} />Đổi tên</div>
        </button>

        {/* Ghim / Bỏ ghim */}
        <button
          className="w-full rounded-lg px-3 py-2 text-left text-sm"
          style={{ color: "var(--text-primary)" }}
          onPointerDown={(e) => e.stopPropagation()}
          onClick={async (e) => {
            e.stopPropagation();
            await patchChat(chat._id, { pinned: !chat.pinned });
            setOpenMenuChatId(null);
            setMenuAnchor(null);
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-hover)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
        >
          <div className="flex items-center gap-2"><Pin size={16} />Ghim / Bỏ ghim</div>
        </button>

        {/* Xóa */}
        <button
          className="w-full rounded-lg px-3 py-2 text-left text-sm"
          style={{ color: "var(--danger-text)", background: "transparent" }}
          onPointerDown={(e) => e.stopPropagation()}
          onClick={async (e) => {
            e.stopPropagation();
            await removeChat(chat._id);
            setOpenMenuChatId(null);
            setMenuAnchor(null);
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "var(--danger-bg)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
        >
          <div className="flex items-center gap-2"><Trash2 size={16} />Xóa</div>
        </button>
      </div>
    );

    if (typeof document === "undefined") return null;
    return createPortal(menuContent, document.body);
  }

  function ChatItem({ chat }: { chat: Chat }) {
    return (
      <div
        className="group relative flex items-center gap-1 rounded-lg"
        style={{ background: activeId === chat._id ? "var(--bg-surface)" : "transparent" }}
        onMouseEnter={(e) => { if (activeId !== chat._id) (e.currentTarget as HTMLElement).style.background = "var(--bg-hover)"; }}
        onMouseLeave={(e) => { if (activeId !== chat._id) (e.currentTarget as HTMLElement).style.background = "transparent"; }}
      >
        <button
          onClick={() => chatActions.setActiveChat(chat._id)}
          className="flex-1 truncate rounded-lg px-3 py-2 text-left text-sm"
          style={{ color: activeId === chat._id ? "var(--text-primary)" : "var(--text-secondary)" }}
        >
          {chat.title ?? "Cuộc trò chuyện mới"}
        </button>

        {/* "..." toggle button — calculates anchor position for the portal */}
        <button
          onPointerDown={(e) => {
            e.stopPropagation();
            // Mark this gesture as the one that opened the menu
            // so the document-level outside-click handler ignores it
            openingRef.current = true;
          }}
          onClick={(e) => {
            e.stopPropagation();
            if (openMenuChatId === chat._id) {
              // Already open → close
              setOpenMenuChatId(null);
              setMenuAnchor(null);
              openingRef.current = false;
              return;
            }
            // Compute position: place menu below-left of this button
            const rect = (e.currentTarget as HTMLElement).getBoundingClientRect();
            const MENU_WIDTH = 200;
            // Try right-aligned to button; clamp so it doesn't go off screen
            let left = rect.right - MENU_WIDTH;
            if (left < 8) left = 8;
            if (left + MENU_WIDTH > window.innerWidth - 8) left = window.innerWidth - MENU_WIDTH - 8;
            setMenuAnchor({ top: rect.bottom + 4, left });
            setOpenMenuChatId(chat._id);
          }}
          className="mr-1 w-8 rounded-md py-1 text-center opacity-100 transition md:opacity-0 md:group-hover:opacity-100"
          style={{ color: "var(--text-muted)" }}
        >
          <MoreHorizontal size={18} />
        </button>

        {/* Portal menu — rendered in document.body, never clipped */}
        <ChatMenuPortal chat={chat} />
      </div>
    );
  }

  const sidebarContent = (
    <>
      <div>
        <div className="mb-3 flex items-center justify-between">
          <div className="flex items-center gap-2 text-lg font-semibold" style={{ color: "var(--text-primary)" }}>
            <Image src="/logo.png" alt="VNLaw Logo" width={52} height={52} className="rounded" />
            <span className="text-base">Chatbot VNLaw</span>
          </div>
          <button
            className="flex h-8 w-8 items-center justify-center rounded-lg md:hidden"
            style={{ background: "var(--bg-hover)", color: "var(--text-muted)" }}
            onClick={() => setMobileSidebarOpen(false)}
          >
            <X size={16} />
          </button>
        </div>

        <button
          className="mb-3 w-full rounded-lg px-3 py-2 text-left text-sm transition"
          style={{ background: "var(--bg-surface)", color: "var(--text-secondary)", border: "0.5px solid var(--border)" }}
          onClick={createChat}
          onMouseEnter={(e) => (e.currentTarget.style.background = "var(--bg-hover)")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "var(--bg-surface)")}
        >
          <div className="flex items-center gap-2"><Plus size={16} />Đoạn chat mới</div>
        </button>

        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Tìm kiếm đoạn chat"
          className="mb-3 w-full rounded-lg px-3 py-2 text-sm outline-none"
          style={{ background: "var(--bg-input)", border: "0.5px solid var(--border)", color: "var(--text-primary)" }}
        />

        {error && (
          <div className="mb-3 rounded-lg px-3 py-2 text-xs"
            style={{ background: "var(--danger-bg)", border: "0.5px solid var(--danger-border)", color: "var(--danger-text)" }}>
            {error}
          </div>
        )}
      </div>

      <div className="flex-1 space-y-1 overflow-y-auto pr-1">
        {pinnedChats.length > 0 && (
          <>
            <div className="flex items-center gap-1 mb-1 px-2">
              <span className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                <div className="flex items-center gap-1"><Pin size={14} /><span>Đã ghim</span></div>
              </span>
            </div>
            {pinnedChats.map((chat) => <ChatItem key={chat._id} chat={chat} />)}
            {unpinnedChats.length > 0 && <div className="my-2" style={{ borderTop: "0.5px solid var(--border)" }} />}
          </>
        )}
        {unpinnedChats.length > 0 && (
          <>
            <div className="flex items-center gap-1 mb-1 px-2">
              <span className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
                <div className="flex items-center gap-1"><Clock size={14} /><span>Gần đây</span></div>
              </span>
            </div>
            {unpinnedChats.map((chat) => <ChatItem key={chat._id} chat={chat} />)}
          </>
        )}
      </div>
    </>
  );

  return (
    <div className="min-h-screen" style={{ background: "var(--bg-base)", color: "var(--text-primary)" }}>
      <div className="mx-auto flex h-screen">

        {/* MOBILE SIDEBAR OVERLAY */}
        {mobileSidebarOpen && (
          <>
            <div
              className="fixed inset-0 z-40 md:hidden"
              style={{ background: "rgba(0,0,0,0.5)", backdropFilter: "blur(2px)" }}
              onClick={() => setMobileSidebarOpen(false)}
            />
            <aside
              className="fixed left-0 top-0 z-50 flex h-full flex-col gap-3 md:hidden"
              style={{
                width: "min(85vw, 320px)",
                padding: 14,
                background: "var(--bg-sidebar)",
                borderRight: "0.5px solid var(--border)",
                overflowY: "auto",
                animation: "slideInLeft 0.22s ease",
              }}
            >
              {sidebarContent}
            </aside>
          </>
        )}

        {/* DESKTOP SIDEBAR */}
        <aside
          className="relative hidden md:flex flex-col gap-3 transition-all duration-200"
          style={{
            width: sidebarOpen ? 300 : 56,
            padding: sidebarOpen ? 12 : 8,
            height: "100vh",
            overflow: "hidden",
            background: "var(--bg-sidebar)",
            borderRight: "0.5px solid var(--border)",
            flexShrink: 0,
          }}
        >
          <button
            onClick={() => setSidebarOpen((p) => !p)}
            className="absolute right-2 top-2 flex h-9 w-9 items-center justify-center rounded-lg text-sm transition"
            style={{ background: "var(--bg-surface)", border: "0.5px solid var(--border-strong)", color: "var(--text-secondary)" }}
            title={sidebarOpen ? "Ẩn sidebar" : "Hiện sidebar"}
          >
            {sidebarOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
          </button>

          {!sidebarOpen && (
            <div className="flex h-full flex-col items-center gap-3 mt-2">
              <div className="h-9 w-9" />
              {[
                { icon: <Plus size={18} />, title: "Đoạn chat mới", action: createChat },
                { icon: <Search size={18} />, title: "Tìm kiếm", action: () => setSidebarOpen(true) },
              ].map(({ icon, title, action }) => (
                <button key={title} onClick={action} title={title}
                  className="flex h-9 w-9 items-center justify-center rounded-lg"
                  style={{ background: "var(--bg-surface)", color: "var(--text-secondary)" }}>
                  {icon}
                </button>
              ))}
              <div className="mt-2 flex w-full flex-1 flex-col items-center gap-2 overflow-y-auto pb-2" style={{ scrollbarWidth: "none" as any }}>
                {filtered.map((chat) => (
                  <button key={chat._id} onClick={() => chatActions.setActiveChat(chat._id)}
                    className="h-9 w-9 rounded-lg text-xs flex items-center justify-center"
                    style={{ background: activeId === chat._id ? "var(--bg-msgbtn)" : "var(--bg-surface)" }}
                    title={chat.title ?? "Chat"}>
                    <MessageCircle size={16} />
                  </button>
                ))}
              </div>
            </div>
          )}

          {sidebarOpen && <div className="flex flex-col gap-3 h-full">{sidebarContent}</div>}
        </aside>

        {/* MAIN CONTENT */}
        <main className="flex flex-1 flex-col overflow-hidden" style={{ background: "var(--bg-main)", minWidth: 0 }}>
          <div className="flex flex-1 overflow-hidden">
            <div className="flex flex-1 flex-col overflow-hidden min-w-0">

              {/* Topbar */}
              <div className="flex items-center justify-between px-3 py-3 md:px-6 md:py-4"
                style={{ borderBottom: "0.5px solid var(--border)" }}>
                <div className="flex items-center gap-2 min-w-0">
                  <button
                    className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg md:hidden"
                    style={{ background: "var(--bg-surface)", color: "var(--text-secondary)", border: "0.5px solid var(--border)" }}
                    onClick={() => setMobileSidebarOpen(true)}
                  >
                    <Menu size={16} />
                  </button>
                  <div className="truncate text-sm font-semibold md:text-base" style={{ color: "var(--text-primary)" }}>
                    {activeChat?.title ?? "Hệ thống hỏi đáp pháp luật"}
                    {selectedPassages && (
                      <span className="ml-2 hidden text-xs font-normal italic md:inline" style={{ color: "var(--text-muted)" }}>
                        — {selectedPassages.length} văn bản liên quan
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-1 md:gap-2">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="hidden h-4 w-4 md:block"
                    style={{ color: !darkMode ? "#f59e0b" : "var(--text-muted)" }}>
                    <path fill="currentColor" d="M6.76 4.84l-1.8-1.79-1.41 1.41 1.79 1.8 1.42-1.42zm10.45-1.79l-1.79 1.79 1.41 1.42 1.8-1.8-1.42-1.41zM12 4V1h-1v3h1zm0 19v-3h-1v3h1zm8-11h3v-1h-3v1zM4 12H1v-1h3v1zm12.24 7.16l1.8 1.79 1.41-1.41-1.79-1.8-1.42 1.42zM4.22 19.54l1.79-1.8-1.41-1.41-1.8 1.79 1.42 1.42zM12 6a6 6 0 100 12 6 6 0 000-12z"/>
                  </svg>
                  <button
                    onClick={() => setDarkMode((p) => !p)}
                    className="relative inline-flex h-6 w-11 items-center rounded-full transition-all duration-300"
                    style={{ background: darkMode ? "#534AB7" : "var(--bg-input)", border: "0.5px solid var(--border)" }}
                  >
                    <span className="inline-block h-4 w-4 transform rounded-full bg-white shadow transition-all duration-300"
                      style={{ transform: darkMode ? "translateX(20px)" : "translateX(4px)" }} />
                  </button>
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" className="hidden h-4 w-4 md:block"
                    style={{ color: darkMode ? "#6366f1" : "var(--text-muted)" }}>
                    <path fill="currentColor" d="M21 12.79A9 9 0 0111.21 3c0-.34.02-.67.05-1A9 9 0 1021 12.79z"/>
                  </svg>
                </div>
              </div>

              {/* Messages */}
              <div className="flex-1 overflow-y-auto px-3 py-3 md:px-6 md:py-4">
                {activeChat?.messages?.length ? (
                  <div className="space-y-3 md:space-y-4">
                    {activeChat.messages.map((m, idx) => (
                      <MessageRow
                        key={idx} m={m} idx={idx} chatId={activeChat._id}
                        isSelected={selectedMsgIdx === idx}
                        onSelect={(passages) => {
                          if (selectedMsgIdx === idx) { setSelectedPassages(null); setSelectedMsgIdx(null); }
                          else { setSelectedPassages(passages); setSelectedMsgIdx(idx); }
                        }}
                        onSpeak={speak}
                        isSpeaking={speakingIdx === idx}
                        isLoadingTTS={loadingIdx === idx}
                        onSaveTtsUrl={saveTtsUrl}
                        isAnyTTSActive={speakingIdx !== null || loadingIdx !== null}
                      />
                    ))}
                    {streamingChatId === activeChat._id && (postStreamReveal || revealedStreamText.length > 0) && (
                      <div className="flex items-end gap-2">
                        <Avatar type="bot" />
                        <BotBubble content={revealedStreamText || "\u00a0"} />
                      </div>
                    )}
                    {pendingChatId === activeChat._id && pendingStatus && (
                      <div className="flex items-end gap-2">
                        <Avatar type="bot" />
                        <div className="rounded-2xl rounded-bl-md px-3 py-2 text-xs md:px-4 md:py-3 md:text-sm animate-pulse"
                          style={{ background: "var(--status-bg)", color: "var(--status-text)" }}>
                          <div>{pendingStatus}</div>
                          {agentStep != null && (
                            <div className="mt-1 text-[10px] opacity-85 md:text-xs">
                              Vòng suy luận {agentStep.cur}/{agentStep.max}
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    <div ref={bottomRef} />
                  </div>
                ) : (
                  <div className="mt-16 text-center text-sm md:mt-24" style={{ color: "var(--text-muted)" }}>
                    Bắt đầu một đoạn chat mới...
                  </div>
                )}
              </div>

              {/* Input form */}
              <form onSubmit={onSend} className="p-2 md:p-4" style={{ borderTop: "0.5px solid var(--border)" }}>
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setQueryMode((m) => m === "normal" ? "situation" : "normal")}
                    style={{
                      display: "inline-flex", alignItems: "center", gap: 6,
                      padding: "4px 10px", borderRadius: 20, fontSize: 12, fontWeight: 500,
                      border: queryMode === "situation" ? "1.5px solid #534AB7" : "0.5px solid var(--border)",
                      background: queryMode === "situation" ? "#EEEDFE" : "transparent",
                      color: queryMode === "situation" ? "#3C3489" : "var(--text-muted)",
                      cursor: "pointer", transition: "all 0.15s",
                    }}
                  >
                    <span style={{
                      width: 7, height: 7, borderRadius: "50%", flexShrink: 0, transition: "background 0.15s",
                      background: queryMode === "situation" ? "#534AB7" : "var(--border-strong)"
                    }} />
                    {queryMode === "situation" ? "Phân tích tình huống" : "Hỏi đáp thường"}
                  </button>
                  {queryMode === "situation" && (
                    <span className="hidden text-xs sm:inline" style={{ color: "var(--text-muted)" }}>
                      Mô tả đầy đủ: ai · làm gì · hoàn cảnh · hậu quả
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-1.5 md:gap-2">
                  {!recording && !audioBlob && (
                    <input
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); onSend(e as any); } }}
                      placeholder={queryMode === "situation" ? "Mô tả tình huống..." : "Nhập câu hỏi..."}
                      className="flex-1 rounded-xl px-3 py-2.5 text-sm outline-none transition md:px-4 md:py-3"
                      style={{
                        background: "var(--bg-input)",
                        border: queryMode === "situation" ? "0.5px solid #534AB7" : "0.5px solid var(--border)",
                        color: "var(--text-primary)", minWidth: 0,
                      }}
                    />
                  )}
                  {recording && (
                    <>
                      <div className="flex flex-1 items-center gap-2 rounded-xl px-2 py-2 md:gap-3 md:px-3"
                        style={{ background: "var(--bg-input)", border: "1px solid #ef4444", minHeight: 44 }}>
                        <span className="shrink-0 relative flex h-3 w-3">
                          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75" />
                          <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500" />
                        </span>
                        <LiveWaveform recording={recording} />
                        <span className="shrink-0 text-xs font-mono tabular-nums" style={{ color: "#ef4444" }}>
                          {String(Math.floor(duration / 60)).padStart(2, "0")}:{String(duration % 60).padStart(2, "0")}
                        </span>
                      </div>
                      <button type="button" onClick={stopRecording} title="Dừng ghi âm"
                        className="shrink-0 flex items-center justify-center rounded-full w-10 h-10 transition"
                        style={{ background: "#ef4444", color: "#fff" }}>
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                          <rect x="5" y="5" width="14" height="14" rx="2" />
                        </svg>
                      </button>
                    </>
                  )}
                  {!recording && audioBlob && (
                    <div className="flex flex-1 items-center gap-2 rounded-xl px-2 py-2 md:px-3"
                      style={{ background: "var(--bg-input)", border: "0.5px solid #534AB7", minHeight: 44 }}>
                      <VoiceMessage audioBlob={audioBlob} onCancel={cancelRecording} />
                    </div>
                  )}
                  {sttSupported && !recording && !audioBlob && (
                    <button type="button" onClick={startRecording} title="Ghi âm"
                      className="shrink-0 flex items-center justify-center rounded-xl transition"
                      style={{ width: 42, height: 42, background: "var(--bg-surface)", border: "0.5px solid var(--border)", color: "var(--text-muted)" }}>
                      <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                        <line x1="12" y1="19" x2="12" y2="23"/>
                        <line x1="8" y1="23" x2="16" y2="23"/>
                      </svg>
                    </button>
                  )}
                  <button
                    type={audioBlob ? "button" : "submit"}
                    onClick={audioBlob ? sendVoice : undefined}
                    disabled={loading || sending}
                    className="shrink-0 rounded-xl text-sm font-medium transition disabled:opacity-60"
                    style={{ height: 42, width: 42, background: "#534AB7", color: "var(--text-on-accent)", display: "flex", alignItems: "center", justifyContent: "center" }}
                  >
                    {sending ? (
                      <span className="text-xs">{loadingDots}</span>
                    ) : (
                      <svg width="17" height="17" viewBox="0 0 24 24" fill="none"
                        stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="22" y1="2" x2="11" y2="13"/>
                        <polygon points="22 2 15 22 11 13 2 9 22 2"/>
                      </svg>
                    )}
                  </button>
                </div>
              </form>
            </div>

            {/* Passage panel */}
            {selectedPassages && (
              <>
                <div className="hidden md:flex">
                  <PassagePanel passages={selectedPassages} onClose={() => { setSelectedPassages(null); setSelectedMsgIdx(null); }} />
                </div>
                <div className="fixed inset-0 z-50 flex flex-col md:hidden" style={{ background: "var(--bg-sidebar)" }}>
                  <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: "0.5px solid var(--border)" }}>
                    <div className="flex items-center gap-2 text-base font-bold" style={{ color: "var(--text-primary)" }}>
                      <BookOpen className="h-5 w-5" style={{ color: "#534AB7" }} />Căn cứ pháp lý
                    </div>
                    <button onClick={() => { setSelectedPassages(null); setSelectedMsgIdx(null); }}
                      className="rounded-lg px-2 py-1 text-sm"
                      style={{ background: "var(--bg-hover)", color: "var(--text-muted)" }}>
                      <X size={18} />
                    </button>
                  </div>
                  <div className="flex-1 overflow-y-auto">
                    <PassagePanel passages={selectedPassages} onClose={() => { setSelectedPassages(null); setSelectedMsgIdx(null); }} mobileEmbedded />
                  </div>
                </div>
              </>
            )}
          </div>
        </main>

        {fullViewPassage && <PassageFullModal passage={fullViewPassage} onClose={() => setFullViewPassage(null)} />}
      </div>

      {/* Rename modal */}
      {renameChatId && (
        <div
          className="fixed inset-0 flex items-center justify-center p-4"
          style={{ background: "rgba(0,0,0,0.5)", zIndex: 300 }}
          onClick={(e) => { if (e.target === e.currentTarget) { setRenameChatId(null); } }}
        >
          <div className="w-full max-w-sm rounded-xl p-4 shadow-xl md:max-w-md"
            style={{ background: "var(--bg-surface)", border: "0.5px solid var(--border-strong)" }}>
            <div className="mb-3 text-base font-semibold" style={{ color: "var(--text-primary)" }}>
              Đổi tên cuộc trò chuyện
            </div>
            <input
              value={renameValue}
              onChange={(e) => setRenameValue(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); handleRename(); } }}
              className="w-full rounded-lg px-3 py-2 text-sm outline-none"
              style={{ background: "var(--bg-input)", border: "0.5px solid var(--border)", color: "var(--text-primary)" }}
              placeholder="Nhập tên mới..."
              autoFocus
            />
            <div className="mt-4 flex justify-end gap-2">
              <button
                onClick={() => { setRenameChatId(null); setRenameValue(""); }}
                className="rounded-lg px-3 py-2 text-sm transition"
                style={{ background: "var(--bg-hover)", color: "var(--text-secondary)" }}
              >Hủy</button>
              <button
                onClick={handleRename}
                className="rounded-lg px-3 py-2 text-sm font-medium transition"
                style={{ background: "var(--accent)", color: "var(--text-on-accent)" }}
              >Lưu</button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes slideInLeft {
          from { transform: translateX(-100%); opacity: 0; }
          to   { transform: translateX(0);    opacity: 1; }
        }
      `}</style>
    </div>
  );
}