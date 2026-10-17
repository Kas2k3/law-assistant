import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const body = await req.json();

  const backendUrl = process.env.RAG_BACKEND_URL ?? "http://localhost:8001";

  const res = await fetch(`${backendUrl}/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    return NextResponse.json({ error: "TTS failed" }, { status: res.status });
  }

  const contentType = res.headers.get("Content-Type") ?? "";

  // Firebase URL trả về JSON
  if (contentType.includes("application/json")) {
    const data = await res.json();
    return NextResponse.json(data);
  }

  // Fallback: stream audio trực tiếp
  const buffer = await res.arrayBuffer();
  return new NextResponse(buffer, {
    headers: {
      "Content-Type": "audio/mpeg",
      "Content-Length": buffer.byteLength.toString(),
    },
  });
}