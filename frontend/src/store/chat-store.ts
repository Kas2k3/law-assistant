"use client";

import { useSyncExternalStore } from "react";

export type Passage = {
  mapc: string;
  ten: string;
  tenchuong?: string;
  tendemuc?: string;
  tenchude?: string;
  noidung: string;
  score?: number;
  source?: string;
  url?: string;
};

export interface Message {
  role: "user" | "assistant";
  content: string;
  passages?: Passage[];
  tts_url?: string|null;   // ← thêm
}

export type Chat = {
  _id: string;
  title: string;
  pinned: boolean;
  messages: Message[];
  updatedAt?: string;
  createdAt?: string;
  updated_at?: string;
  created_at?: string;
};

type ChatState = {
  chats: Chat[];
  activeId: string | null;
  loading: boolean;
  initialized: boolean;
  error: string | null;
};

type Action =
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "SET_ACTIVE"; payload: string | null }
  | { type: "SET_CHATS"; payload: Chat[] }
  | { type: "UPSERT_CHAT"; payload: Chat }
  | { type: "REMOVE_CHAT"; payload: string };

const initialState: ChatState = {
  chats: [],
  activeId: null,
  loading: false,
  initialized: false,
  error: null,
};

function getUpdatedValue(chat: Chat) {
  return chat.updated_at ?? chat.updatedAt ?? chat.created_at ?? chat.createdAt ?? "";
}

function sortChats(chats: Chat[]) {
  return [...chats].sort((a, b) => {
    if (a.pinned !== b.pinned) return a.pinned ? -1 : 1;
    return getUpdatedValue(b).localeCompare(getUpdatedValue(a));
  });
}

function reducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case "SET_LOADING":
      return { ...state, loading: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload };
    case "SET_ACTIVE":
      return { ...state, activeId: action.payload };
    case "SET_CHATS": {
      const chats = sortChats(action.payload);
      const hasActive = chats.some((c) => c._id === state.activeId);
      return {
        ...state,
        chats,
        initialized: true,
        activeId: hasActive ? state.activeId : (chats[0]?._id ?? null),
      };
    }
    case "UPSERT_CHAT": {
      const exists = state.chats.some((c) => c._id === action.payload._id);
      const chats = exists
        ? state.chats.map((c) => (c._id === action.payload._id ? action.payload : c))
        : [...state.chats, action.payload];
      return {
        ...state,
        chats: sortChats(chats),
        activeId: state.activeId ?? action.payload._id,
      };
    }
    case "REMOVE_CHAT": {
      const chats = state.chats.filter((c) => c._id !== action.payload);
      return {
        ...state,
        chats,
        activeId: state.activeId === action.payload ? (chats[0]?._id ?? null) : state.activeId,
      };
    }
    default:
      return state;
  }
}

let state = initialState;
const listeners = new Set<() => void>();

function emit() {
  listeners.forEach((listener) => listener());
}

function dispatch(action: Action) {
  state = reducer(state, action);
  emit();
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function getState() {
  return state;
}

async function requestJson<T>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  const res = await fetch(input, init);
  const data = (await res.json().catch(() => ({}))) as T & { error?: string; detail?: string };
  if (!res.ok) {
    throw new Error(data.detail || data.error || `Request failed with status ${res.status}`);
  }
  return data;
}

export const chatActions = {
  setActiveChat(id: string | null) {
    dispatch({ type: "SET_ACTIVE", payload: id });
  },

  async ensureChatsLoaded(force = false) {
    const current = getState();
    if (current.initialized && !force) return;
    dispatch({ type: "SET_LOADING", payload: true });
    dispatch({ type: "SET_ERROR", payload: null });
    try {
      const data = await requestJson<{ chats: Chat[] }>("/api/chats");
      dispatch({ type: "SET_CHATS", payload: data.chats ?? [] });
    } catch (error) {
      dispatch({ type: "SET_ERROR", payload: String(error) });
    } finally {
      dispatch({ type: "SET_LOADING", payload: false });
    }
  },

  async createChat() {
    dispatch({ type: "SET_ERROR", payload: null });
    const data = await requestJson<{ chat: Chat }>("/api/chats", { method: "POST" });
    dispatch({ type: "UPSERT_CHAT", payload: data.chat });
    dispatch({ type: "SET_ACTIVE", payload: data.chat._id });
    return data.chat;
  },

  async patchChat(id: string, patch: Partial<Chat>) {
    dispatch({ type: "SET_ERROR", payload: null });
    const data = await requestJson<{ chat: Chat }>(`/api/chats/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(patch),
    });
    dispatch({ type: "UPSERT_CHAT", payload: data.chat });
    return data.chat;
  },

  async removeChat(id: string) {
    dispatch({ type: "SET_ERROR", payload: null });
    await requestJson<{ ok: boolean }>(`/api/chats/${id}`, { method: "DELETE" });
    dispatch({ type: "REMOVE_CHAT", payload: id });
  },
};

export function useChatStore<T>(selector: (s: ChatState) => T): T {
  return useSyncExternalStore(subscribe, () => selector(getState()), () => selector(initialState));
}
