import os
import sqlite3
from typing import Optional, List, Dict, Any


class PlatformBridge:
    """
    Minimal, synchronous SQLite helper for Project RISE.

    Usage (from async FastAPI endpoints):
        bridge = PlatformBridge()  # or pass a custom db_path
        rows = await asyncio.to_thread(bridge.get_progress, user_id)

    Tables:
      - user_profile(user_id TEXT PRIMARY KEY, name TEXT, created_at TIMESTAMP)
      - progress(user_id TEXT, course_id TEXT, lesson_id TEXT, status TEXT, updated_at TIMESTAMP)

    Notes:
      - This module is intentionally synchronous for simplicity and reliability.
      - Call its methods with `asyncio.to_thread(...)` from async contexts.
      - The DB file path is controlled by PLATFORM_DB_PATH or defaults to ./data/platform.db
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("PLATFORM_DB_PATH", "./data/platform.db")
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._init_db()

    # -------- internal helpers --------
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # return rows as dict-like
        return conn

    def _init_db(self) -> None:
        with self._conn() as cx:
            cx.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    user_id    TEXT PRIMARY KEY,
                    name       TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cx.execute("""
                CREATE TABLE IF NOT EXISTS progress (
                    user_id    TEXT    NOT NULL,
                    course_id  TEXT    NOT NULL,
                    lesson_id  TEXT    NOT NULL,
                    status     TEXT    NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, course_id, lesson_id)
                )
            """)
            cx.commit()

    # -------- user profile --------
    def upsert_user(self, user_id: str, name: Optional[str] = None) -> None:
        """
        Create or update a user profile.
        """
        with self._conn() as cx:
            cx.execute(
                """
                INSERT INTO user_profile (user_id, name)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET name=excluded.name
                """,
                (user_id, name),
            )
            cx.commit()

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Return a single user profile as a dict, or None if not found.
        """
        with self._conn() as cx:
            cur = cx.execute(
                "SELECT user_id, name, created_at FROM user_profile WHERE user_id = ?",
                (user_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # -------- progress tracking --------
    def set_progress(self, user_id: str, course_id: str, lesson_id: str, status: str) -> None:
        """
        Insert or update a user's progress for a lesson.
        Status is an arbitrary string (e.g., 'started', 'completed', 'review').
        """
        with self._conn() as cx:
            cx.execute(
                """
                INSERT INTO progress (user_id, course_id, lesson_id, status)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, course_id, lesson_id)
                DO UPDATE SET status=excluded.status, updated_at=CURRENT_TIMESTAMP
                """,
                (user_id, course_id, lesson_id, status),
            )
            cx.commit()

    def get_progress(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Return all progress rows for a user, newest first.
        """
        with self._conn() as cx:
            cur = cx.execute(
                """
                SELECT user_id, course_id, lesson_id, status, updated_at
                FROM progress
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,),
            )
            return [dict(r) for r in cur.fetchall()]

    def get_course_progress(self, user_id: str, course_id: str) -> List[Dict[str, Any]]:
        """
        Return progress rows for a specific course, newest first.
        """
        with self._conn() as cx:
            cur = cx.execute(
                """
                SELECT user_id, course_id, lesson_id, status, updated_at
                FROM progress
                WHERE user_id = ? AND course_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id, course_id),
            )
            return [dict(r) for r in cur.fetchall()]

    def delete_progress(self, user_id: str, course_id: Optional[str] = None) -> int:
        """
        Delete progress entries for a user; optionally limit to a specific course.
        Returns number of rows deleted.
        """
        with self._conn() as cx:
            if course_id:
                cur = cx.execute(
                    "DELETE FROM progress WHERE user_id = ? AND course_id = ?",
                    (user_id, course_id),
                )
            else:
                cur = cx.execute(
                    "DELETE FROM progress WHERE user_id = ?",
                    (user_id,),
                )
            cx.commit()
            return cur.rowcount
