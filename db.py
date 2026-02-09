import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).with_name("habbit_tracker.db")


def _now_iso() -> str:
    return datetime.utcnow().isoformat()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dog_collection (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            habit_id INTEGER NULL,
            image_url TEXT NOT NULL,
            rarity TEXT NOT NULL,
            earned_by TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dog_milestones (
            date TEXT NOT NULL,
            rate_bucket TEXT NOT NULL,
            claimed INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            PRIMARY KEY (date, rate_bucket)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS habit_records (
            date TEXT NOT NULL,
            nickname TEXT NOT NULL,
            checked_count INTEGER NOT NULL,
            total INTEGER NOT NULL,
            mood INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (date, nickname)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS coach_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS groups (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_code TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS group_members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            group_id INTEGER NOT NULL,
            nickname TEXT NOT NULL,
            joined_at TEXT NOT NULL,
            UNIQUE(group_id, nickname)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS group_streak_logs (
            group_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            achieved INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (group_id, date)
        )
        """
    )
    conn.commit()
    conn.close()


def add_dog_to_collection(
    date_value: str,
    habit_id: Optional[int],
    url: str,
    rarity: str,
    earned_by: str,
) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO dog_collection (date, habit_id, image_url, rarity, earned_by, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (date_value, habit_id, url, rarity, earned_by, _now_iso()),
    )
    conn.commit()
    conn.close()


def list_dog_collection(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    rarity: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    clauses = []
    params: List[Any] = []
    if date_from:
        clauses.append("date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("date <= ?")
        params.append(date_to)
    if rarity:
        clauses.append("rarity = ?")
        params.append(rarity)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    cur.execute(
        f"""
        SELECT id, date, habit_id, image_url, rarity, earned_by, created_at
        FROM dog_collection
        {where}
        ORDER BY created_at DESC
        """,
        params,
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def claim_milestone_if_needed(date_value: str, bucket: str) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT OR IGNORE INTO dog_milestones (date, rate_bucket, claimed, created_at)
        VALUES (?, ?, 1, ?)
        """,
        (date_value, bucket, _now_iso()),
    )
    claimed = cur.rowcount == 1
    conn.commit()
    conn.close()
    return claimed


def get_claimed_buckets(date_value: str) -> set:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rate_bucket FROM dog_milestones
        WHERE date = ? AND claimed = 1
        """,
        (date_value,),
    )
    buckets = {row["rate_bucket"] for row in cur.fetchall()}
    conn.close()
    return buckets


def upsert_habit_record(
    date_value: str,
    nickname: str,
    checked_count: int,
    total: int,
    mood: int,
) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO habit_records (date, nickname, checked_count, total, mood, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(date, nickname) DO UPDATE SET
            checked_count = excluded.checked_count,
            total = excluded.total,
            mood = excluded.mood,
            created_at = excluded.created_at
        """,
        (date_value, nickname, checked_count, total, mood, _now_iso()),
    )
    conn.commit()
    conn.close()


def get_habit_record(date_value: str, nickname: str) -> Optional[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT date, nickname, checked_count, total, mood, created_at
        FROM habit_records
        WHERE date = ? AND nickname = ?
        """,
        (date_value, nickname),
    )
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def add_coach_log(date_value: str, log_type: str, content: str, model: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO coach_logs (date, type, content, model, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (date_value, log_type, content, model, _now_iso()),
    )
    conn.commit()
    conn.close()


def list_coach_logs(
    date_from: Optional[str],
    date_to: Optional[str],
    log_type: Optional[str],
    search: Optional[str],
) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    clauses = []
    params: List[Any] = []
    if date_from:
        clauses.append("date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("date <= ?")
        params.append(date_to)
    if log_type:
        clauses.append("type = ?")
        params.append(log_type)
    if search:
        clauses.append("content LIKE ?")
        params.append(f"%{search}%")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    cur.execute(
        f"""
        SELECT id, date, type, content, model, created_at
        FROM coach_logs
        {where}
        ORDER BY created_at DESC
        """,
        params,
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def _generate_group_code() -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(6))


def create_group(name: str) -> str:
    conn = get_conn()
    cur = conn.cursor()
    while True:
        code = _generate_group_code()
        cur.execute("SELECT 1 FROM groups WHERE group_code = ?", (code,))
        if not cur.fetchone():
            break
    cur.execute(
        """
        INSERT INTO groups (group_code, name, created_at)
        VALUES (?, ?, ?)
        """,
        (code, name, _now_iso()),
    )
    conn.commit()
    conn.close()
    return code


def join_group(group_code: str, nickname: str) -> None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id FROM groups WHERE group_code = ?", (group_code,))
    row = cur.fetchone()
    if not row:
        conn.close()
        raise ValueError("유효하지 않은 그룹 코드입니다.")
    group_id = row["id"]
    cur.execute(
        """
        INSERT OR IGNORE INTO group_members (group_id, nickname, joined_at)
        VALUES (?, ?, ?)
        """,
        (group_id, nickname, _now_iso()),
    )
    conn.commit()
    conn.close()


def list_groups_for_nickname(nickname: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT g.id, g.group_code, g.name, g.created_at
        FROM groups g
        JOIN group_members m ON g.id = m.group_id
        WHERE m.nickname = ?
        ORDER BY g.created_at DESC
        """,
        (nickname,),
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def get_group_members(group_code: str) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.nickname, m.joined_at
        FROM group_members m
        JOIN groups g ON g.id = m.group_id
        WHERE g.group_code = ?
        ORDER BY m.joined_at ASC
        """,
        (group_code,),
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def compute_member_today_achieved(nickname: str, date_value: str, daily_goal_n: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT checked_count, total FROM habit_records
        WHERE date = ? AND nickname = ?
        """,
        (date_value, nickname),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return False
    return row["total"] > 0 and row["checked_count"] >= daily_goal_n


def update_group_daily_status(group_id: int, date_value: str, daily_goal_n: int) -> bool:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT nickname FROM group_members WHERE group_id = ?", (group_id,))
    members = [row["nickname"] for row in cur.fetchall()]
    if not members:
        conn.close()
        return False
    achieved_all = all(compute_member_today_achieved(n, date_value, daily_goal_n) for n in members)
    cur.execute(
        """
        INSERT INTO group_streak_logs (group_id, date, achieved, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(group_id, date) DO UPDATE SET
            achieved = excluded.achieved,
            created_at = excluded.created_at
        """,
        (group_id, date_value, 1 if achieved_all else 0, _now_iso()),
    )
    conn.commit()
    conn.close()
    return achieved_all


def calc_group_streak(group_id: int, daily_goal_n: int) -> int:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT date, achieved
        FROM group_streak_logs
        WHERE group_id = ?
        ORDER BY date DESC
        """,
        (group_id,),
    )
    streak = 0
    for row in cur.fetchall():
        if row["achieved"] == 1:
            streak += 1
        else:
            break
    conn.close()
    return streak


def list_group_logs(group_id: int, days: int) -> List[Dict[str, Any]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT date, achieved, created_at
        FROM group_streak_logs
        WHERE group_id = ?
        ORDER BY date DESC
        LIMIT ?
        """,
        (group_id, days),
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


init_db()
