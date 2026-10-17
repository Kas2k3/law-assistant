# Dùng để khởi tạo database SQLite cho chatbot
import os
import sqlite3

# Đường dẫn thực tới file SQLite thật
DB_FILE = os.path.join(os.path.dirname(__file__), "conversations.sqlite")

def init_db():
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()

    # Tạo bảng sessions
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at TEXT,
        updated_at TEXT
    )
    """)

    # Tạo bảng messages
    cur.execute("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()

# Cho phép chạy độc lập để khởi tạo DB
if __name__ == "__main__":
    init_db()
    print(f"✅ Database khởi tạo tại: {DB_FILE}")
