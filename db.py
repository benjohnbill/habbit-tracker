import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime

DB_PATH = Path("habit_tracker.db")


@contextmanager
def get_conn(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS habits(
                habit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                frequency TEXT NOT NULL CHECK(frequency IN ('daily','weekly')),
                goal INTEGER NOT NULL,
                reminder_text TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS checkins(
                checkin_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL UNIQUE,
                note TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS checkin_items(
                checkin_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                checkin_id INTEGER NOT NULL,
                habit_id INTEGER NOT NULL,
                value INTEGER NOT NULL,
                FOREIGN KEY(checkin_id) REFERENCES checkins(checkin_id),
                FOREIGN KEY(habit_id) REFERENCES habits(habit_id),
                UNIQUE(checkin_id, habit_id)
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS coaching_logs(
                coaching_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                tone TEXT NOT NULL,
                weather_summary TEXT,
                input_summary TEXT,
                output_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )


# ---------- Habits CRUD ----------
def list_habits():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM habits ORDER BY habit_id ASC").fetchall()
        return [dict(r) for r in rows]


def create_habit(name: str, description: str, frequency: str, goal: int, reminder_text: str):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO habits(name, description, frequency, goal, reminder_text, created_at)
            VALUES(?,?,?,?,?,?)
            """,
            (name, description or None, frequency, int(goal), reminder_text or None, now_iso()),
        )


def update_habit(habit_id: int, name: str, description: str, frequency: str, goal: int, reminder_text: str):
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE habits
            SET name=?, description=?, frequency=?, goal=?, reminder_text=?
            WHERE habit_id=?
            """,
            (name, description or None, frequency, int(goal), reminder_text or None, int(habit_id)),
        )


def delete_habit(habit_id: int):
    with get_conn() as conn:
        # cascade manually
        conn.execute("DELETE FROM checkin_items WHERE habit_id=?", (int(habit_id),))
        conn.execute("DELETE FROM habits WHERE habit_id=?", (int(habit_id),))


# ---------- Checkins ----------
def upsert_checkin(date_str: str, note: str):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT checkin_id FROM checkins WHERE date=?", (date_str,))
        row = cur.fetchone()
        if row:
            checkin_id = row["checkin_id"]
            cur.execute("UPDATE checkins SET note=? WHERE checkin_id=?", (note or None, checkin_id))
            return checkin_id
        cur.execute(
            "INSERT INTO checkins(date, note, created_at) VALUES(?,?,?)",
            (date_str, note or None, now_iso()),
        )
        return cur.lastrowid


def upsert_checkin_item(checkin_id: int, habit_id: int, value: int):
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO checkin_items(checkin_id, habit_id, value)
            VALUES(?,?,?)
            ON CONFLICT(checkin_id, habit_id) DO UPDATE SET value=excluded.value
            """,
            (int(checkin_id), int(habit_id), int(value)),
        )


def get_checkin(date_str: str):
    with get_conn() as conn:
        checkin = conn.execute("SELECT * FROM checkins WHERE date=?", (date_str,)).fetchone()
        if not checkin:
            return None
        items = conn.execute(
            """
            SELECT ci.*, h.name, h.goal, h.frequency
            FROM checkin_items ci
            JOIN habits h ON h.habit_id = ci.habit_id
            WHERE ci.checkin_id=?
            ORDER BY h.habit_id ASC
            """,
            (checkin["checkin_id"],),
        ).fetchall()
        return {
            "checkin": dict(checkin),
            "items": [dict(r) for r in items],
        }


def list_checkins_between(start_date: str, end_date: str):
    """inclusive start/end; date is YYYY-MM-DD"""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM checkins
            WHERE date BETWEEN ? AND ?
            ORDER BY date ASC
            """,
            (start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]


def get_items_between(start_date: str, end_date: str):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT c.date, h.habit_id, h.name, h.goal, h.frequency, ci.value
            FROM checkins c
            JOIN checkin_items ci ON ci.checkin_id = c.checkin_id
            JOIN habits h ON h.habit_id = ci.habit_id
            WHERE c.date BETWEEN ? AND ?
            ORDER BY c.date ASC, h.habit_id ASC
            """,
            (start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------- Coaching logs ----------
def add_coaching_log(date_str: str, tone: str, weather_summary: str, input_summary: str, output_text: str):
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO coaching_logs(date, tone, weather_summary, input_summary, output_text, created_at)
            VALUES(?,?,?,?,?,?)
            """,
            (date_str, tone, weather_summary or None, input_summary or None, output_text, now_iso()),
        )


def list_coaching_logs(limit: int = 100):
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT * FROM coaching_logs
            ORDER BY date DESC, coaching_id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]


def get_coaching_log(coaching_id: int):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM coaching_logs WHERE coaching_id=?", (int(coaching_id),)).fetchone()
        return dict(row) if row else None


def seed_sample_habits_if_empty():
    habits = list_habits()
    if habits:
        return
    create_habit("물 8잔 마시기", "하루 수분 섭취 목표", "daily", 8, "물 한 잔 어때요?")
    create_habit("스트레칭", "가벼운 전신 스트레칭", "daily", 1, "5분만 해도 좋아요.")
    create_habit("책 읽기", "집중 독서", "weekly", 3, "이번 주 3번만 읽어도 성공!")
