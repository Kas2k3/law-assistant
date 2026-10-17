"use client";
import { useState } from "react";
import { Passage } from "@/store/chat-store";
import { BookOpen, ChevronDown, ChevronRight, Folder } from "lucide-react";

function LegalBasisMetaRow({ label, value }: { label: string; value?: string | null }) {
  const v = (value ?? "").trim();
  if (!v) return null;
  return (
    <div className="flex gap-2 py-1.5 text-xs" style={{ borderBottom: "0.5px solid var(--border)" }}>
      <span className="w-16 shrink-0 font-medium md:w-20" style={{ color: "var(--text-muted)" }}>
        {label}
      </span>
      <span className="min-w-0 flex-1 leading-snug" style={{ color: "var(--text-primary)" }}>
        {v}
      </span>
    </div>
  );
}

export function PassageFullModal({
  passage,
  onClose,
}: {
  passage: Passage;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center p-0 md:items-start md:p-10"
      style={{ background: "rgba(0,0,0,0.5)" }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="relative w-full rounded-t-2xl p-4 shadow-xl md:max-w-2xl md:rounded-xl md:p-5"
        style={{
          background: "var(--bg-surface)",
          border: "0.5px solid var(--border-strong)",
          // Mobile: bottom sheet chiếm tối đa 90vh
          maxHeight: "90vh",
          overflowY: "auto",
        }}
      >
        {/* Mobile drag handle */}
        <div className="mx-auto mb-3 h-1 w-10 rounded-full md:hidden" style={{ background: "var(--border-strong)" }} />

        <button
          onClick={onClose}
          className="absolute right-3 top-3 rounded-lg px-2 py-1 text-xs"
          style={{ background: "var(--bg-hover)", color: "var(--text-muted)" }}
        >
          ✕
        </button>

        <div className="mb-1 pr-10 text-sm font-semibold leading-snug" style={{ color: "var(--text-primary)" }}>
          {passage.ten || "(Không có tiêu đề)"}
        </div>

        {passage.tendemuc && (
          <div className="mb-3 text-xs" style={{ color: "var(--text-muted)" }}>
            {passage.tenchude && `${passage.tenchude} › `}{passage.tendemuc}
            {passage.tenchuong && ` › ${passage.tenchuong}`}
          </div>
        )}

        <div
          className="whitespace-pre-wrap break-words text-xs leading-relaxed"
          style={{
            borderTop: "0.5px solid var(--border)",
            paddingTop: 12,
            color: "var(--text-primary)",
            maxHeight: "50vh",
            overflowY: "auto",
          }}
        >
          {passage.noidung || "(Không có nội dung)"}
        </div>

        <div
          className="mt-4 flex justify-end gap-2 pt-3 text-xs"
          style={{ borderTop: "0.5px solid var(--border)" }}
        >
          {passage.url && (
            <a
              href={passage.url} target="_blank" rel="noreferrer"
              className="rounded-lg px-3 py-1.5"
              style={{ border: "0.5px solid var(--border)", color: "var(--text-secondary)" }}
            >
              Mở nguồn ↗
            </a>
          )}
          <button
            onClick={onClose}
            className="rounded-lg px-3 py-1.5"
            style={{ border: "0.5px solid #534AB7", color: "#534AB7" }}
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
}

export function PassagePanel({
  passages,
  onClose,
  mobileEmbedded = false,
}: {
  passages: Passage[];
  onClose: () => void;
  mobileEmbedded?: boolean;
}) {
  const [open, setOpen]         = useState<Record<number, boolean>>({});
  const [fullView, setFullView] = useState<Passage | null>(null);

  return (
    <>
      <aside
        className="flex flex-col overflow-hidden"
        style={{
          // Desktop: fixed side panel width
          // mobileEmbedded: full width, no border
          width: mobileEmbedded ? "100%" : "min(28rem, 32vw)",
          minWidth: mobileEmbedded ? undefined : 280,
          borderLeft: mobileEmbedded ? "none" : "0.5px solid var(--border)",
          background: "var(--bg-sidebar)",
          height: mobileEmbedded ? "100%" : undefined,
        }}
      >
        {/* Header — chỉ hiện khi không phải mobileEmbedded (header đã ở page.tsx) */}
        {!mobileEmbedded && (
          <div
            className="flex shrink-0 items-start justify-between gap-2 px-4 py-4"
            style={{ borderBottom: "0.5px solid var(--border)" }}
          >
            <div className="min-w-0">
              <div className="flex items-center gap-2 text-base font-bold" style={{ color: "var(--text-primary)" }}>
                <BookOpen className="h-5 w-5 shrink-0" style={{ color: "#534AB7" }} />
                <span>Căn cứ pháp lý</span>
              </div>
              <p className="mt-1.5 text-xs leading-relaxed" style={{ color: "var(--text-muted)" }}>
                Dựa trên Bộ Pháp điển do Bộ Tư pháp Việt Nam biên soạn.
              </p>
              <p className="mt-2 text-xs italic" style={{ color: "var(--text-muted)" }}>
                Tìm thấy {passages.length} văn bản liên quan
              </p>
            </div>
            <button
              type="button" onClick={onClose}
              className="shrink-0 rounded-lg px-2 py-1 text-sm"
              style={{ color: "var(--text-muted)", background: "var(--bg-hover)" }}
            >
              ✕
            </button>
          </div>
        )}

        {/* mobileEmbedded: show count */}
        {mobileEmbedded && (
          <div className="px-4 py-3 text-xs italic" style={{ color: "var(--text-muted)", borderBottom: "0.5px solid var(--border)" }}>
            Dựa trên Bộ Pháp điển — {passages.length} văn bản liên quan
          </div>
        )}

        {/* List */}
        <div className="flex-1 overflow-y-auto px-3 py-3 space-y-2.5">
          {passages.map((p, i) => {
            const expanded = !open[i];
            const isWeb    = p.source === "web" || p.source === "web_realtime";

            return (
              <div
                key={`${p.mapc ?? "doc"}-${i}`}
                className="overflow-hidden rounded-xl"
                style={{ background: "var(--bg-hover)", border: "0.5px solid var(--border)" }}
              >
                {/* Accordion header */}
                <button
                  type="button"
                  className="flex w-full items-center gap-2 px-3 py-2.5 text-left transition"
                  style={{
                    background: expanded ? "var(--bg-main)" : "var(--bg-hover)",
                    borderBottom: expanded ? "0.5px solid var(--border)" : "none",
                  }}
                  onClick={() => setOpen((prev) => ({ ...prev, [i]: !prev[i] }))}
                >
                  <ChevronDown
                    className="h-4 w-4 shrink-0"
                    style={{
                      color: "var(--text-muted)",
                      transform: expanded ? "rotate(0deg)" : "rotate(-90deg)",
                      transition: "transform 0.2s",
                    }}
                  />
                  <Folder className="h-4 w-4 shrink-0" style={{ color: "#534AB7" }} />
                  <span className="min-w-0 flex-1 truncate text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
                    Tài liệu {i + 1}
                  </span>
                  <span
                    className="shrink-0 rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase"
                    style={{
                      background: isWeb ? "#EEEDFE" : "#EAF3DE",
                      color: isWeb ? "#3C3489" : "#3B6D11",
                    }}
                  >
                    {isWeb ? "Web" : "Pháp điển"}
                  </span>
                  {p.score !== undefined && (
                    <span className="shrink-0 text-[10px] tabular-nums" style={{ color: "var(--text-muted)" }}>
                      {(p.score * 100).toFixed(0)}%
                    </span>
                  )}
                </button>

                {/* Accordion body */}
                {expanded && (
                  <div className="px-3 pb-3 pt-1">
                    <LegalBasisMetaRow label="Chủ đề"  value={p.tenchude} />
                    <LegalBasisMetaRow label="Đề mục"  value={p.tendemuc} />
                    <LegalBasisMetaRow label="Chương"  value={p.tenchuong} />
                    <LegalBasisMetaRow label="Điều"    value={p.ten} />

                    {p.url && (
                      <div className="pt-2">
                        <a
                          href={p.url} target="_blank" rel="noreferrer"
                          className="inline-flex max-w-full items-center gap-1 break-all text-xs underline"
                          style={{ color: "#534AB7" }}
                        >
                          {p.url.length > 44 ? `${p.url.slice(0, 44)}…` : p.url}
                        </a>
                      </div>
                    )}

                    <div
                      className="mt-2 max-h-36 overflow-y-auto rounded-lg px-2 py-2 text-xs leading-relaxed"
                      style={{ background: "var(--bg-main)", color: "var(--text-secondary)" }}
                    >
                      {p.noidung?.trim() || "—"}
                    </div>

                    <button
                      type="button"
                      className="mt-2.5 flex w-full items-center justify-center gap-2 rounded-lg px-3 py-2 text-xs font-medium"
                      style={{ border: "1.5px solid #534AB7", color: "#534AB7", background: "transparent" }}
                      onMouseEnter={(e) => (e.currentTarget.style.background = "#EEEDFE")}
                      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                      onClick={() => setFullView(p)}
                    >
                      Xem nội dung đầy đủ <ChevronRight className="h-4 w-4" />
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </aside>

      {fullView && <PassageFullModal passage={fullView} onClose={() => setFullView(null)} />}
    </>
  );
}