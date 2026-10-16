from __future__ import annotations

import sqlite3
import threading
import time
from typing import Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from config import DB_PATH, logger

_conn: Optional[sqlite3.Connection] = None
_lock = threading.Lock()


def connect_to_database() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    with _lock:
        if _conn is not None:
            return _conn
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        _init_schema(conn)
        _conn = conn
        logger.info("database: 已连接 %s", DB_PATH)
        return _conn


def _init_schema(conn: sqlite3.Connection):
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_number TEXT,
            role TEXT,
            content TEXT,
            ts REAL
        );
        """
    )
    conn.commit()



class SQLiteChatHistory(BaseChatMessageHistory):

    def __init__(self, session_id: str):
        self.session_id = str(session_id)

    @property
    def messages(self) -> list:
        conn = connect_to_database()
        with _lock:
            rows = conn.execute(
                "SELECT role, content FROM chat_history WHERE phone_number=? "
                "ORDER BY id ASC LIMIT 50",
                (self.session_id,),
            ).fetchall()
        msgs = []
        for r in rows:
            if r["role"] == "assistant":
                msgs.append(AIMessage(content=r["content"]))
            else:
                msgs.append(HumanMessage(content=r["content"]))
        return msgs

    def add_message(self, message) -> None:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        conn = connect_to_database()
        with _lock:
            conn.execute(
                "INSERT INTO chat_history (phone_number, role, content, ts) "
                "VALUES (?, ?, ?, ?)",
                (self.session_id, role, message.content, time.time()),
            )
            conn.commit()

    def clear(self) -> None:
        conn = connect_to_database()
        with _lock:
            conn.execute("DELETE FROM chat_history WHERE phone_number=?",
                         (self.session_id,))
            conn.commit()


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return SQLiteChatHistory(session_id)


if __name__ == "__main__":
    h = SQLiteChatHistory("demo_user")
    h.clear()
    h.add_message(HumanMessage(content="你好"))
    h.add_message(AIMessage(content="你好！有什么可以帮你？"))
    print("history:", [(type(m).__name__, m.content) for m in h.messages])
