import { MongoClient } from "mongodb";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type ChatDoc = {
  _id: string;
  title: string;
  pinned: boolean;
  messages: ChatMessage[];
  createdAt?: string;
  updatedAt?: string;
  created_at?: string;
  updated_at?: string;
};

const MONGODB_URI = process.env.MONGODB_URI ?? "";
const MONGODB_DB = process.env.MONGODB_DB ?? "chatbot_rag";
const MONGODB_COLLECTION = process.env.MONGODB_COLLECTION ?? "conversations";

let mongoClient: MongoClient | null = null;

const memoryStore = new Map<string, ChatDoc>();

function nowIso() {
  return new Date().toISOString();
}

function hasMongo() {
  return Boolean(MONGODB_URI);
}

async function getCollection() {
  if (!hasMongo()) return null;
  if (!mongoClient) {
    mongoClient = new MongoClient(MONGODB_URI, {
      serverSelectionTimeoutMS: 8000,
    });
    await mongoClient.connect();
  }
  return mongoClient.db(MONGODB_DB).collection<ChatDoc>(MONGODB_COLLECTION);
}

function sortChats(chats: ChatDoc[]) {
  const getUpdatedValue = (chat: ChatDoc) =>
    chat.updated_at ?? chat.updatedAt ?? chat.created_at ?? chat.createdAt ?? "";

  return chats.sort((a, b) => {
    const aPinned = Boolean(a?.pinned);
    const bPinned = Boolean(b?.pinned);
    if (aPinned !== bPinned) return aPinned ? -1 : 1;

    return getUpdatedValue(b).localeCompare(getUpdatedValue(a));
  });
}

export async function listChats(): Promise<ChatDoc[]> {
  const col = await getCollection();
  if (!col) return sortChats(Array.from(memoryStore.values()));
  const docs = await col.find({}).toArray();
  return sortChats(docs);
}

export async function createChat(title = "Cuộc trò chuyện mới"): Promise<ChatDoc> {
  const doc: ChatDoc = {
    _id: crypto.randomUUID(),
    title,
    pinned: false,
    messages: [],
    createdAt: nowIso(),
    updatedAt: nowIso(),
  };
  const col = await getCollection();
  if (!col) {
    memoryStore.set(doc._id, doc);
    return doc;
  }
  await col.insertOne(doc);
  return doc;
}

export async function getChatById(id: string): Promise<ChatDoc | null> {
  const col = await getCollection();
  if (!col) return memoryStore.get(id) ?? null;
  return col.findOne({ _id: id });
}

export async function updateChat(
  id: string,
  patch: Partial<Pick<ChatDoc, "title" | "pinned" | "messages">>
): Promise<ChatDoc | null> {
  const col = await getCollection();
  const updatedAt = nowIso();
  const safePatch = Object.fromEntries(
    Object.entries(patch).filter(([, value]) => value !== undefined)
  ) as Partial<Pick<ChatDoc, "title" | "pinned" | "messages">>;

  if (!col) {
    const current = memoryStore.get(id);
    if (!current) return null;
    const next: ChatDoc = { ...current, ...safePatch, updatedAt };
    memoryStore.set(id, next);
    return next;
  }

  await col.updateOne({ _id: id }, { $set: { ...safePatch, updatedAt } });
  return col.findOne({ _id: id });
}

export async function deleteChatById(id: string): Promise<boolean> {
  const col = await getCollection();
  if (!col) return memoryStore.delete(id);
  const result = await col.deleteOne({ _id: id });
  return result.deletedCount > 0;
}
