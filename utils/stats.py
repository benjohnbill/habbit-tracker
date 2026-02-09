from typing import List, Dict, Tuple
from collections import defaultdict
import pandas as pd


def build_seven_day_summary(items_7d: List[Dict]) -> str:
    """
    items_7d rows: {date, habit_id, name, goal, frequency, value}
    returns human-readable summary in Korean.
    """
    if not items_7d:
        return "- 최근 7일 기록이 없습니다."

    df = pd.DataFrame(items_7d)
    # success = value >= goal
    df["success"] = df["value"].astype(int) >= df["goal"].astype(int)

    out = []
    for (hid, name, freq), g in df.groupby(["habit_id", "name", "frequency"], sort=False):
        rate = g["success"].mean() * 100.0
        n = len(g)
        out.append(f"- {name} ({freq}): 성공률 {rate:.0f}% (기록 {n}건)")

    # weakest habit
    by_habit = df.groupby(["habit_id", "name"], sort=False)["success"].mean().reset_index()
    weakest = by_habit.sort_values("success", ascending=True).iloc[0]
    out.append(f"- 가장 약한 습관: {weakest['name']} (성공률 {weakest['success']*100:.0f}%)")
    return "\n".join(out)


def compute_today_achievement(habits: List[Dict], today_values: Dict[int, int]) -> Tuple[float, int, int]:
    """
    returns (rate, success_count, total_count) based on daily habits only (weekly는 오늘 달성률 계산에서 제외 가능).
    MVP: weekly도 value>=goal이면 성공으로 카운트하되, 입력 단위는 동일(정수).
    """
    total = 0
    success = 0
    for h in habits:
        hid = int(h["habit_id"])
        goal = int(h["goal"])
        val = int(today_values.get(hid, 0))
        total += 1
        if val >= goal:
            success += 1
    rate = (success / total) * 100.0 if total else 0.0
    return rate, success, total


def items_to_dataframe(items: List[Dict]) -> pd.DataFrame:
    if not items:
        return pd.DataFrame(columns=["date", "habit_id", "name", "goal", "frequency", "value"])
    return pd.DataFrame(items)
