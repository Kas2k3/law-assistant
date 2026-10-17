import { NextResponse } from "next/server";
import { deleteChatById, getChatById, updateChat } from "@/lib/chats";

type RouteContext = {
  params: Promise<{ id: string }>;
};

export async function GET(_: Request, context: RouteContext) {
  const { id } = await context.params;
  const chat = await getChatById(id);
  if (!chat) {
    return NextResponse.json({ error: "Không tìm thấy hội thoại" }, { status: 404 });
  }
  return NextResponse.json({ chat });
}

export async function PATCH(req: Request, context: RouteContext) {
  const { id } = await context.params;
  const body = (await req.json().catch(() => ({}))) as {
    title?: string;
    pinned?: boolean;
    messages?: { role: "user" | "assistant"; content: string }[];
  };

  const patch: {
    title?: string;
    pinned?: boolean;
    messages?: { role: "user" | "assistant"; content: string }[];
  } = {};

  if ("title" in body) patch.title = body.title;
  if ("pinned" in body) patch.pinned = body.pinned;
  if ("messages" in body) patch.messages = body.messages;

  const chat = await updateChat(id, patch);

  if (!chat) {
    return NextResponse.json({ error: "Không tìm thấy hội thoại" }, { status: 404 });
  }
  return NextResponse.json({ chat });
}

export async function DELETE(_: Request, context: RouteContext) {
  const { id } = await context.params;
  const ok = await deleteChatById(id);
  if (!ok) {
    return NextResponse.json({ error: "Không tìm thấy hội thoại" }, { status: 404 });
  }
  return NextResponse.json({ ok: true });
}
