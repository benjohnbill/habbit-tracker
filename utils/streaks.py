from typing import List, Dict
from datetime import datetime, timedelta


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def compute_daily_streak(items: List[Dict], habit_id: int, goal: int, as_of_date: str) -> int:
    """
    items: list of dict with {date, habit_id, value, goal, frequency}
    streak counts backwards from as_of_date, consecutive days with value >= goal.
    """
    # Filter rows for habit
    rows = [r for r in items if int(r["habit_id"]) == int(habit_id)]
    by_date = {r["date"]: int(r["value"]) for r in rows}

    d = _parse_date(as_of_date)
    streak = 0
    while True:
        ds = d.strftime("%Y-%m-%d")
        val = by_date.get(ds)
        if val is None or val < int(goal):
            break
        streak += 1
        d = d - timedelta(days=1)
    return streak
