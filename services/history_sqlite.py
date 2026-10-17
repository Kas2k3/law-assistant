import sqlite3
import uuid
from datetime import datetime
import os
import sys

# Thêm đường dẫn để import module khởi tạo DB
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import từ module khởi tạo
from chat_history import history_db_init

DB_PATH = history_db_init.DB_FILE
history_db_init.init_db()  # đảm bảo DB đã được khởi tạo

def load_chats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, title, created_at, updated_at FROM sessions")
    sessions = cur.fetchall()

    data = {}
    for sid, title, created_at, updated_at in sessions:
        cur.execute("""
            SELECT role, content FROM messages
            WHERE session_id=? ORDER BY created_at ASC
        """, (sid,))
        messages = [{"role": r, "content": c} for r, c in cur.fetchall()]
        data[sid] = {
            "title": title,
            "messages": messages,
            "created_at": created_at,
            "updated_at": updated_at
        }
    conn.close()
    return data


def save_chat(chat_id, title, messages):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = datetime.now().isoformat()

    cur.execute("SELECT 1 FROM sessions WHERE id=?", (chat_id,))
    exists = cur.fetchone()

    if exists:
        cur.execute("""
            UPDATE sessions SET title=?, updated_at=? WHERE id=?
        """, (title, now, chat_id))
        cur.execute("DELETE FROM messages WHERE session_id=?", (chat_id,))
    else:
        cur.execute("""
            INSERT INTO sessions (id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (chat_id, title, now, now))

    for m in messages:
        cur.execute("""
            INSERT INTO messages (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (chat_id, m["role"], m["content"], now))

    conn.commit()
    conn.close()


def create_new_chat():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    new_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    cur.execute("""
        INSERT INTO sessions (id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
    """, (new_id, "Cuộc trò chuyện mới", now, now))
    conn.commit()
    conn.close()
    return new_id


def rename_chat(chat_id, new_title):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = datetime.now().isoformat()
    cur.execute("""
        UPDATE sessions SET title=?, updated_at=? WHERE id=?
    """, (new_title, now, chat_id))
    conn.commit()
    conn.close()


def delete_chat(chat_id):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE session_id=?", (chat_id,))
    cur.execute("DELETE FROM sessions WHERE id=?", (chat_id,))
    conn.commit()
    conn.close()


def cleanup_empty_chats():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        DELETE FROM sessions
        WHERE id NOT IN (SELECT DISTINCT session_id FROM messages)
    """)
    conn.commit()
    conn.close()
