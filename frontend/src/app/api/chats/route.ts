import { NextResponse } from "next/server";
import { createChat, listChats } from "@/lib/chats";

export async function GET() {
  try {
    const chats = await listChats();
    return NextResponse.json({ chats });
  } catch (error) {
    return NextResponse.json(
      { error: "Không thể tải danh sách hội thoại", detail: String(error) },
      { status: 500 }
    );
  }
}

export async function POST(req: Request) {
  try {
    const body = (await req.json().catch(() => ({}))) as { title?: string };
    const chat = await createChat(body.title || "Cuộc trò chuyện mới");
    return NextResponse.json({ chat }, { status: 201 });
  } catch (error) {
    return NextResponse.json(
      { error: "Không thể tạo hội thoại", detail: String(error) },
      { status: 500 }
    );
  }
}
