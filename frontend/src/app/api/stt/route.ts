import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  const formData = await req.formData();
  const file = formData.get("file") as File;
  if (!file) return NextResponse.json({ error: "No file" }, { status: 400 });

  const backendForm = new FormData();
  backendForm.append("file", file, "audio.webm");

  const backendUrl = process.env.RAG_BACKEND_URL ?? "http://localhost:8001";

  const res = await fetch(`${backendUrl}/stt`, {
    method: "POST",
    body: backendForm,
  });

  const data = await res.json();
  return NextResponse.json(data);
}