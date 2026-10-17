import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except Exception:  # pragma: no cover
    MongoClient = None
    PyMongoError = Exception

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

CHAT_DIR = "chat_history"
CHAT_FILE = os.path.join(CHAT_DIR, "conversations.json")
DB_FILE = os.path.join(CHAT_DIR, "conversations.sqlite")
MONGODB_URI = (os.getenv("MONGODB_URI") or "").strip()
MONGODB_DB = (os.getenv("MONGODB_DB") or "chatbot_rag").strip()
MONGODB_COLLECTION = (os.getenv("MONGODB_COLLECTION") or "conversations").strip()
MONGODB_TIMEOUT_MS = int((os.getenv("MONGODB_TIMEOUT_MS") or "8000").strip())


def _mongo_enabled() -> bool:
    return bool(MONGODB_URI and MongoClient is not None)


def _get_mongo_collection():
    if not _mongo_enabled():
        return None
    try:
        client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=MONGODB_TIMEOUT_MS,
            connectTimeoutMS=MONGODB_TIMEOUT_MS,
            socketTimeoutMS=MONGODB_TIMEOUT_MS,
        )
        client.admin.command("ping")
        db = client[MONGODB_DB]
        return db[MONGODB_COLLECTION]
    except Exception as e:
        # Không raise để app vẫn chạy (fallback SQLite/JSON),
        # nhưng in ra lỗi để người dùng biết Mongo đang không truy cập được.
        print(f"[MongoDB] Không kết nối được: {type(e).__name__}: {e}")
        return None


def _ensure_storage():
    os.makedirs(CHAT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                pinned INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT
            )
            """
        )
        # Migration nhẹ: thêm cột pinned nếu DB cũ chưa có.
        try:
            cur.execute("ALTER TABLE sessions ADD COLUMN pinned INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                created_at TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _read_json():
    if not os.path.exists(CHAT_FILE):
        return {}
    try:
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {str(i): chat for i, chat in enumerate(data)}
        if isinstance(data, dict):
            return data
        return {}
    except Exception:
        return {}


def _write_json(chats):
    os.makedirs(CHAT_DIR, exist_ok=True)
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)


def _load_chats_from_mongo():
    col = _get_mongo_collection()
    if col is None:
        return {}
    try:
        docs = list(
            col.find(
                {},
                {
                    "_id": 1,
                    "title": 1,
                    "messages": 1,   # ← đã include messages, tts_url nằm trong này
                    "pinned": 1,
                    "created_at": 1,
                    "updated_at": 1,
                },
            )
        )
        chats = {}
        for d in docs:
            sid = str(d.get("_id"))
            # Giữ nguyên toàn bộ message object kể cả tts_url
            messages = d.get("messages") or []
            chats[sid] = {
                "title":      d.get("title") or "Cuộc trò chuyện mới",
                "messages":   [
                    {**msg, "tts_url": msg.get("tts_url", None)}  # ← đảm bảo field tồn tại
                    for msg in (d.get("messages") or [])
                ],
                "pinned":     bool(d.get("pinned", False)),
                "created_at": d.get("created_at") or datetime.now().isoformat(),
                "updated_at": d.get("updated_at") or datetime.now().isoformat(),
            }
        return chats
    except PyMongoError:
        return {}


def _upsert_chat_mongo(chat_id, chat):
    col = _get_mongo_collection()
    if col is None:
        return False
    now = datetime.now().isoformat()
    payload = {
        "_id": chat_id,
        "title": chat.get("title", "Cuộc trò chuyện mới"),
        "messages": chat.get("messages", []),
        "pinned": bool(chat.get("pinned", False)),
        "created_at": chat.get("created_at", now),
        "updated_at": chat.get("updated_at", now),
    }
    try:
        col.replace_one({"_id": chat_id}, payload, upsert=True)
        return True
    except PyMongoError:
        return False


def _delete_chat_mongo(chat_id):
    col = _get_mongo_collection()
    if col is None:
        return False
    try:
        col.delete_one({"_id": chat_id})
        return True
    except PyMongoError:
        return False


def _sync_all_to_mongo(chats):
    col = _get_mongo_collection()
    if col is None:
        return False
    try:
        # KHÔNG xoá toàn bộ collection trước khi sync.
        # Trước đây delete_many({}) có thể làm mất dữ liệu Mongo nếu nguồn fallback
        # (sqlite/json) chưa đầy đủ hoặc đang lỗi tạm thời.
        now = datetime.now().isoformat()
        if not chats:
            return True
        for chat_id, chat in chats.items():
            payload = {
                "_id": chat_id,
                "title": chat.get("title", "Cuộc trò chuyện mới"),
                "messages": chat.get("messages", []),
                "pinned": bool(chat.get("pinned", False)),
                "created_at": chat.get("created_at", now),
                "updated_at": chat.get("updated_at", now),
            }
            col.replace_one(
                {"_id": chat_id},
                payload,
                upsert=True,
            )
        return True
    except PyMongoError:
        return False


def _load_chats_from_sqlite():
    _ensure_storage()
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        sessions = conn.execute(
            "SELECT id, title, pinned, created_at, updated_at FROM sessions"
        ).fetchall()
        if not sessions:
            return {}

        rows = conn.execute(
            """
            SELECT session_id, role, content, created_at
            FROM messages
            ORDER BY id ASC
            """
        ).fetchall()

        msg_map = {}
        for row in rows:
            sid = row["session_id"]
            msg_map.setdefault(sid, []).append(
                {"role": row["role"], "content": row["content"]}
            )

        chats = {}
        for s in sessions:
            sid = s["id"]
            chats[sid] = {
                "title": s["title"] or "Cuoc tro chuyen moi",
                "messages": msg_map.get(sid, []),
                "pinned": bool(s["pinned"] or 0),
                "created_at": s["created_at"] or datetime.now().isoformat(),
                "updated_at": s["updated_at"] or datetime.now().isoformat(),
            }
        return chats
    finally:
        conn.close()


def _upsert_chat_sqlite(chat_id, chat):
    _ensure_storage()
    conn = sqlite3.connect(DB_FILE)
    now = datetime.now().isoformat()
    try:
        conn.execute(
            """
            INSERT INTO sessions (id, title, pinned, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                pinned=excluded.pinned,
                updated_at=excluded.updated_at
            """,
            (
                chat_id,
                chat.get("title", "Cuoc tro chuyen moi"),
                1 if chat.get("pinned", False) else 0,
                chat.get("created_at", now),
                chat.get("updated_at", now),
            ),
        )
        conn.execute("DELETE FROM messages WHERE session_id = ?", (chat_id,))
        for msg in chat.get("messages", []):
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    chat_id,
                    msg.get("role", "assistant"),
                    msg.get("content", ""),
                    now,
                ),
            )
        conn.commit()
    finally:
        conn.close()


def _delete_chat_sqlite(chat_id):
    _ensure_storage()
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (chat_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (chat_id,))
        conn.commit()
    finally:
        conn.close()


def _sync_all_to_sqlite(chats):
    _ensure_storage()
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM sessions")
        for chat_id, chat in chats.items():
            conn.execute(
                """
                INSERT INTO sessions (id, title, pinned, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    chat.get("title", "Cuoc tro chuyen moi"),
                    1 if chat.get("pinned", False) else 0,
                    chat.get("created_at", datetime.now().isoformat()),
                    chat.get("updated_at", datetime.now().isoformat()),
                ),
            )
            for msg in chat.get("messages", []):
                conn.execute(
                    """
                    INSERT INTO messages (session_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        chat_id,
                        msg.get("role", "assistant"),
                        msg.get("content", ""),
                        datetime.now().isoformat(),
                    ),
                )
        conn.commit()
    finally:
        conn.close()


def load_chats():
    """
    Đọc lịch sử hội thoại. Ưu tiên SQLite; nếu SQLite trống thì fallback JSON.
    Luôn đồng bộ lại để có cả conversations.json và conversations.sqlite.
    """
    # Ưu tiên MongoDB nếu có cấu hình, fallback SQLite/JSON nếu Mongo không sẵn sàng.
    chats = _load_chats_from_mongo()
    if chats:
        _write_json(chats)
        return chats

    chats = _load_chats_from_sqlite()
    if chats:
        _write_json(chats)
        _sync_all_to_mongo(chats)
        return chats

    chats = _read_json()
    if chats:
        _sync_all_to_sqlite(chats)
        _sync_all_to_mongo(chats)
    else:
        _ensure_storage()
    _write_json(chats)
    return chats


def cleanup_empty_chats():
    chats = load_chats()
    to_delete = [cid for cid, chat in chats.items() if not chat.get("messages")]
    for cid in to_delete:
        chats.pop(cid, None)
        _delete_chat_sqlite(cid)
        _delete_chat_mongo(cid)
    _write_json(chats)


def save_chat(chat_id, title, messages):
    chats = load_chats()
    now = datetime.now().isoformat()

    if chat_id in chats:
        old_messages = chats[chat_id].get("messages", [])
        # Merge: giữ tts_url từ messages cũ
        merged = []
        for i, msg in enumerate(messages):
            merged_msg = dict(msg)
            if i < len(old_messages):
                old_tts = old_messages[i].get("tts_url")
                if old_tts and not merged_msg.get("tts_url"):
                    merged_msg["tts_url"] = old_tts
            merged.append(merged_msg)
        messages = merged

        pinned = bool(chats[chat_id].get("pinned", False))
        chats[chat_id]["title"]      = title
        chats[chat_id]["messages"]   = messages
        chats[chat_id]["pinned"]     = pinned
        chats[chat_id]["updated_at"] = now
    else:
        chats[chat_id] = {
            "title":      title,
            "messages":   messages,
            "pinned":     False,
            "created_at": now,
            "updated_at": now,
        }

    _write_json(chats)
    if not _upsert_chat_mongo(chat_id, chats[chat_id]):
        _upsert_chat_sqlite(chat_id, chats[chat_id])


def create_new_chat():
    chats = load_chats()
    new_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    chats[new_id] = {
        "title": "Cuộc trò chuyện mới",
        "messages": [],
        "pinned": False,
        "created_at": now,
        "updated_at": now,
    }
    _write_json(chats)
    if not _upsert_chat_mongo(new_id, chats[new_id]):
        _upsert_chat_sqlite(new_id, chats[new_id])
    return new_id


def rename_chat(chat_id, new_title):
    chats = load_chats()
    if chat_id in chats:
        chats[chat_id]["title"] = new_title
        chats[chat_id]["updated_at"] = datetime.now().isoformat()
        _write_json(chats)
        if not _upsert_chat_mongo(chat_id, chats[chat_id]):
            _upsert_chat_sqlite(chat_id, chats[chat_id])


def toggle_pin_chat(chat_id, pinned: Optional[bool] = None):
    chats = load_chats()
    if chat_id not in chats:
        return False
    current = bool(chats[chat_id].get("pinned", False))
    new_value = (not current) if pinned is None else bool(pinned)
    chats[chat_id]["pinned"] = new_value
    chats[chat_id]["updated_at"] = datetime.now().isoformat()
    _write_json(chats)
    if not _upsert_chat_mongo(chat_id, chats[chat_id]):
        _upsert_chat_sqlite(chat_id, chats[chat_id])
    return True


def delete_chat(chat_id):
    chats = load_chats()
    if chat_id in chats:
        del chats[chat_id]
        _write_json(chats)
        if not _delete_chat_mongo(chat_id):
            _delete_chat_sqlite(chat_id)

def save_tts_url(chat_id: str, msg_idx: int, tts_url: str) -> bool:
    """Lưu URL audio TTS vào message tương ứng trong MongoDB."""
    col = _get_mongo_collection()
    if col is None:
        print("[TTS] MongoDB không khả dụng, bỏ qua lưu URL")
        return False
    try:
        result = col.update_one(
            {"_id": chat_id},
            {"$set": {f"messages.{msg_idx}.tts_url": tts_url}},
        )
        print(f"[TTS] Saved URL to MongoDB: chat={chat_id} msg={msg_idx}")
        return result.modified_count > 0
    except Exception as e:
        print(f"[TTS MongoDB] Lỗi lưu URL: {e}")
        return False

