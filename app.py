# app.py
# Streamlit: AI Habit Tracker (Weather 401-friendly + key test + trimming + secrets fallback)
from __future__ import annotations

import calendar
import json
import random
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

import db

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")


# -----------------------------
# Constants
# -----------------------------
HABITS = [
    ("🌅", "기상 미션"),
    ("💧", "물 마시기"),
    ("📚", "공부/독서"),
    ("🏃", "운동하기"),
    ("😴", "수면"),
]

CITY_OPTIONS: Dict[str, Dict[str, str]] = {
    "Seoul": {"q": "Seoul,KR"},
@@ -190,381 +193,734 @@ def weather_error_hint(debug: Dict[str, Any]) -> str:
    사용자에게 '정확한 조치'를 안내하기 위한 메시지.
    특히 401(Invalid API key) 케이스를 명확히 설명.
    """
    if not debug:
        return "알 수 없는 오류입니다."

    if debug.get("status_code") == 401:
        return (
            "OpenWeatherMap API Key가 **유효하지 않습니다(401)**.\n\n"
            "- 키 앞뒤 공백/줄바꿈이 섞였는지 확인\n"
            "- OpenWeatherMap에서 발급한 키가 맞는지 확인\n"
            "- 발급 직후라면 활성화까지 5~30분 걸릴 수 있어요\n"
            "- 무료 플랜에서도 Geocoding/Current Weather는 사용 가능합니다"
        )

    if debug.get("reason", "").startswith("timeout"):
        return "네트워크가 느려서 요청이 시간 초과(10초) 되었어요. 잠시 후 다시 시도해 주세요."

    if debug.get("step") == "geocode" and debug.get("reason") == "도시 검색 결과 0개":
        return "도시 검색 결과가 없어요. 도시명을 바꾸거나(Seoul/Busan 등) 다시 시도해 주세요."

    return f"날씨 요청 실패: {debug.get('message') or debug.get('reason') or '원인 불명'}"


# -----------------------------
# Dog CEO
# Dog rewards (cached)
# -----------------------------
def _breed_from_dog_url(url: str) -> str:
    try:
        marker = "/breeds/"
        if marker not in url:
            return "알 수 없음"
        seg = url.split(marker, 1)[1].split("/", 1)[0]
        seg = seg.replace("-", " ").strip()
        return seg if seg else "알 수 없음"
    except Exception:
        return "알 수 없음"


@st.cache_data(ttl=60)
def get_dog_image() -> Optional[Dict[str, str]]:
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        img_url = data.get("message")
        if not img_url or not isinstance(img_url, str):
            return None
        return {"url": img_url, "breed": _breed_from_dog_url(img_url)}
    except Exception:
        return None


# -----------------------------
# OpenAI report
# -----------------------------
def _get_openai_client(api_key: str) -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai 패키지가 설치되어 있지 않습니다. requirements.txt에 openai를 추가해 주세요.")
    return OpenAI(api_key=_clean_key(api_key))


def _style_system_prompt(style: str) -> str:
    base = (
        "너는 사용자의 습관 체크인 데이터를 바탕으로 '코치 리포트'를 작성한다. "
        "의학적/치료적 진단은 하지 말고, 실천 가능한 제안만 한다. "
        "출력 형식을 반드시 지켜라."
    )
    if style == "스파르타 코치":
        return base + " 톤은 엄격하고 직설적. 짧고 명확. 모욕/비난 금지."
    if style == "따뜻한 멘토":
        return base + " 톤은 따뜻하고 공감적. 작은 성취를 인정하고 부담을 낮춘다."
    return base + " 톤은 RPG 게임 마스터. '플레이어', '퀘스트' 같은 표현을 섞어 재미있게."


def generate_report(
    openai_api_key: str,
def build_report_prompt(
    habits_checked: List[str],
    habits_unchecked: List[str],
    mood: int,
    weather: Optional[Dict[str, Any]],
    dog_breed: Optional[str],
    coach_style: str,
) -> Optional[str]:
    openai_api_key = _clean_key(openai_api_key)
    if not openai_api_key:
        return None

) -> Tuple[str, str]:
    system_prompt = _style_system_prompt(coach_style)
    weather_text = "날씨 정보 없음"
    if weather:
        weather_text = (
            f"{weather.get('city')}({weather.get('country')}) | {weather.get('description')} | "
            f"{weather.get('temp_c')}°C(체감 {weather.get('feels_like_c')}°C) | "
            f"습도 {weather.get('humidity')}% | 바람 {weather.get('wind_mps')}m/s"
        )

    breed_text = dog_breed if dog_breed else "알 수 없음"

    user_prompt = f"""
아래 데이터를 기반으로 리포트를 작성해줘.

[오늘 기분 점수]
{mood}/10

[완료한 습관]
{", ".join(habits_checked) if habits_checked else "없음"}

[미완료 습관]
{", ".join(habits_unchecked) if habits_unchecked else "없음"}

[날씨]
{weather_text}

[오늘의 강아지 품종]
{breed_text}

출력 형식(반드시 지켜):
## 컨디션 등급
- 등급: (S/A/B/C/D 중 하나)
- 한 줄 요약: ...

## 습관 분석
- 잘한 점: ...
- 아쉬운 점: ...
- 내일 1% 개선: ...

## 날씨 코멘트
- ...

## 내일 미션
- (체크박스 습관과 연결된 실행 미션 3개)

## 오늘의 한마디
- (짧게 1문장)
""".strip()
    return system_prompt, user_prompt


def generate_report(
    openai_api_key: str,
    habits_checked: List[str],
    habits_unchecked: List[str],
    mood: int,
    weather: Optional[Dict[str, Any]],
    dog_breed: Optional[str],
    coach_style: str,
) -> Optional[str]:
    openai_api_key = _clean_key(openai_api_key)
    if not openai_api_key:
        return None

    system_prompt, user_prompt = build_report_prompt(
        habits_checked=habits_checked,
        habits_unchecked=habits_unchecked,
        mood=mood,
        weather=weather,
        dog_breed=dog_breed,
        coach_style=coach_style,
    )

    try:
        client = _get_openai_client(openai_api_key)
        resp = client.responses.create(
            model=MODEL_NAME,
            input=[
                {"role": "system", "content": [{"type": "text", "text": _style_system_prompt(coach_style)}]},
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
            ],
            temperature=0.7,
        )

        if hasattr(resp, "output_text") and resp.output_text:
            return str(resp.output_text).strip()

        out_texts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) == "output_text":
                    out_texts.append(getattr(c, "text", ""))
        text = "\n".join([t for t in out_texts if t]).strip()
        return text if text else None
    except Exception:
        return None


# -----------------------------
# Session state
# -----------------------------
def _init_demo_records() -> List[Dict[str, Any]]:
    rng = random.Random(20260209)
    today = date.today()
    out: List[Dict[str, Any]] = []
    for i in range(6, 0, -1):
        d = today - timedelta(days=i)
        checked_count = rng.randint(1, 5)
        m = rng.randint(3, 9)
        rate = round(checked_count / len(HABITS) * 100, 1)
        out.append({"date": d.isoformat(), "checked_count": checked_count, "rate": rate, "mood": m})
        out.append(
            {
                "date": d.isoformat(),
                "checked_count": checked_count,
                "rate": rate,
                "mood": m,
                "checked_habits": [name for _, name in HABITS[:checked_count]],
            }
        )
    return out


def ensure_state():
    if "records" not in st.session_state:
        st.session_state.records = _init_demo_records()
    if "last_report" not in st.session_state:
        st.session_state.last_report = None
    if "last_weather" not in st.session_state:
        st.session_state.last_weather = None
    if "last_weather_debug" not in st.session_state:
        st.session_state.last_weather_debug = None
    if "last_dog" not in st.session_state:
        st.session_state.last_dog = None
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = None
    if "last_new_rewards" not in st.session_state:
        st.session_state.last_new_rewards = []
    if "nickname" not in st.session_state:
        st.session_state.nickname = "guest"


def upsert_today_record(checked_count: int, mood: int):
    today_s = date.today().isoformat()
def _find_record(target_date: date) -> Optional[Dict[str, Any]]:
    target = target_date.isoformat()
    for rec in st.session_state.records:
        if rec.get("date") == target:
            return rec
    return None


def upsert_record(target_date: date, checked_habits: List[str], mood: int):
    today_s = target_date.isoformat()
    checked_count = len(checked_habits)
    rate = round(checked_count / len(HABITS) * 100, 1)
    rec = {"date": today_s, "checked_count": checked_count, "rate": rate, "mood": mood}
    rec = {
        "date": today_s,
        "checked_count": checked_count,
        "rate": rate,
        "mood": mood,
        "checked_habits": checked_habits,
    }

    records: List[Dict[str, Any]] = st.session_state.records
    for i, r in enumerate(records):
        if r.get("date") == today_s:
            records[i] = rec
            break
    else:
        records.append(rec)

    st.session_state.records = sorted(records, key=lambda x: x.get("date", ""))[-7:]


def _month_calendar(year: int, month: int, records_map: Dict[str, Dict[str, Any]]) -> List[List[str]]:
    cal = calendar.Calendar(firstweekday=6)
    weeks = cal.monthdatescalendar(year, month)
    out: List[List[str]] = []
    for week in weeks:
        row: List[str] = []
        for d in week:
            key = d.isoformat()
            if d.month != month:
                row.append("")
                continue
            rec = records_map.get(key)
            if rec:
                row.append(f"{d.day}\n✅ {rec.get('checked_count')}/{len(HABITS)}")
            else:
                row.append(str(d.day))
        out.append(row)
    return out


def get_quote() -> Optional[str]:
    try:
        r = requests.get("https://api.quotable.io/random", timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        content = data.get("content")
        author = data.get("author")
        if not content:
            return None
        return f"{content} — {author}" if author else content
    except Exception:
        return None


def get_cat_fact() -> Optional[str]:
    try:
        r = requests.get("https://catfact.ninja/fact", timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("fact")
    except Exception:
        return None


def get_activity() -> Optional[str]:
    try:
        r = requests.get("https://www.boredapi.com/api/activity", timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("activity")
    except Exception:
        return None


def milestone_rarity(bucket: int) -> str:
    if bucket >= 100:
        return "epic"
    if bucket >= 80:
        return "rare"
    if bucket >= 50:
        return random.choice(["common", "rare"])
    return "common"


def award_milestones(target_date: date, achievement_rate: float, total_habits: int) -> List[Dict[str, str]]:
    if total_habits == 0:
        return []
    today_key = target_date.isoformat()
    buckets = [20, 50, 80, 100]
    newly_awarded: List[Dict[str, str]] = []
    dog_card = None
    for bucket in buckets:
        if achievement_rate < bucket:
            continue
        if not db.claim_milestone_if_needed(today_key, str(bucket)):
            continue
        if dog_card is None:
            dog_card = get_dog_image()
        if not dog_card or not dog_card.get("url"):
            continue
        rarity = milestone_rarity(bucket)
        db.add_dog_to_collection(
            today_key,
            None,
            dog_card["url"],
            rarity,
            "milestone",
        )
        newly_awarded.append(
            {
                "bucket": str(bucket),
                "url": dog_card["url"],
                "rarity": rarity,
            }
        )
    return newly_awarded


# -----------------------------
# Sidebar: keys + test
# -----------------------------
ensure_state()

with st.sidebar:
    st.header("🔑 API 키 설정")

    # Secrets fallback (배포 시 편의)
    try:
        default_openai = str(st.secrets.get("OPENAI_API_KEY", ""))  # type: ignore
    except Exception:
        default_openai = ""
    try:
        default_owm = str(st.secrets.get("OPENWEATHER_API_KEY", ""))  # type: ignore
    except Exception:
        default_owm = ""

    openai_api_key = st.text_input("OpenAI API Key", value=default_openai, type="password")
    owm_api_key = st.text_input("OpenWeatherMap API Key", value=default_owm, type="password")

    st.divider()
    st.subheader("🌦️ 날씨 키 테스트")

    test_city = st.selectbox("테스트 도시", options=list(CITY_OPTIONS.keys()), index=0, key="test_city")
    if st.button("날씨 키 테스트 실행", use_container_width=True):
        _, dbg = get_weather(test_city, owm_api_key)
        if dbg.get("ok"):
            st.success("성공! OpenWeatherMap 키가 정상입니다.")
        else:
            st.error("실패! 아래 원인을 확인하세요.")
            st.info(weather_error_hint(dbg))
            with st.expander("디버그 상세"):
                st.write(dbg)

    st.caption("401이면 코드가 아니라 키 문제인 경우가 대부분입니다.")

    st.divider()
    st.subheader("👤 프로필")
    st.session_state.nickname = st.text_input("닉네임", value=st.session_state.nickname)
    daily_goal_n = st.slider("오늘 목표 습관 수", 1, len(HABITS), min(3, len(HABITS)))


# -----------------------------
# Main UI
# -----------------------------
ensure_state()

st.title("📊 AI 습관 트래커")
st.caption("오늘의 습관을 체크하고, AI 코치 리포트로 내일을 준비해요.")

records_map = {rec.get("date"): rec for rec in st.session_state.records}

st.subheader("🗓️ 습관 캘린더")
calendar_date = st.date_input("기록할 날짜 선택", value=date.today())
calendar_rows = _month_calendar(calendar_date.year, calendar_date.month, records_map)
calendar_df = pd.DataFrame(calendar_rows, columns=["일", "월", "화", "수", "목", "금", "토"])
st.table(calendar_df)

st.subheader("✅ 습관 체크인")

existing_record = _find_record(calendar_date) or {}
existing_checked = set(existing_record.get("checked_habits") or [])
existing_mood = int(existing_record.get("mood") or 6)

c1, c2 = st.columns(2)
habit_values: Dict[str, bool] = {}
for i, (emoji, name) in enumerate(HABITS):
    with (c1 if i % 2 == 0 else c2):
        habit_values[name] = st.checkbox(f"{emoji} {name}", value=False)
        habit_values[name] = st.checkbox(f"{emoji} {name}", value=name in existing_checked)

mood = st.slider("😊 오늘 기분 점수", 1, 10, 6)
mood = st.slider("😊 오늘 기분 점수", 1, 10, existing_mood)

c3, c4 = st.columns(2)
with c3:
    city_label = st.selectbox("🏙️ 도시 선택", options=list(CITY_OPTIONS.keys()), index=0)
with c4:
    coach_style = st.radio("🧑‍🏫 코치 스타일", options=COACH_STYLES, horizontal=True)

checked_habits = [name for name, v in habit_values.items() if v]
unchecked_habits = [name for name, v in habit_values.items() if not v]

checked_count = len(checked_habits)
achievement_rate = round(checked_count / len(HABITS) * 100, 1)

# Keep today's record synced
upsert_today_record(checked_count=checked_count, mood=mood)
today_key = calendar_date.isoformat()
save_checkin = st.button("오늘 체크인 저장", type="secondary", use_container_width=True)
if save_checkin:
    upsert_record(target_date=calendar_date, checked_habits=checked_habits, mood=mood)
    db.upsert_habit_record(
        calendar_date.isoformat(),
        st.session_state.nickname,
        checked_count,
        len(HABITS),
        mood,
    )
    if checked_count > 0:
        new_rewards = award_milestones(calendar_date, achievement_rate, len(HABITS))
        st.session_state.last_new_rewards = new_rewards
        for reward in new_rewards:
            st.toast(f"신규 도감 획득! {reward['bucket']}% ({reward['rarity']})")
    else:
        st.session_state.last_new_rewards = []

st.subheader("📌 오늘 요약")
m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{achievement_rate}%")
m2.metric("달성 습관", f"{checked_count}/{len(HABITS)}")
m3.metric("기분", f"{mood}/10")

st.subheader("📈 최근 7일 달성률")
df = pd.DataFrame(st.session_state.records).sort_values("date")
st.bar_chart(df.set_index("date")[["rate"]])

st.subheader("✨ 오늘의 추가 영감")
quote = get_quote()
cat_fact = get_cat_fact()
activity = get_activity()
cols = st.columns(3)
cols[0].info(quote or "명언을 가져오지 못했어요.")
cols[1].info(cat_fact or "고양이 사실을 가져오지 못했어요.")
cols[2].info(activity or "오늘의 활동을 가져오지 못했어요.")

st.subheader("🧠 AI 코치 리포트")
btn = st.button("컨디션 리포트 생성", type="primary", use_container_width=True)

if btn:
    with st.spinner("날씨와 강아지를 불러오는 중..."):
        weather, weather_dbg = get_weather(city_label, owm_api_key)
        dog = get_dog_image()

    st.session_state.last_weather = weather
    st.session_state.last_weather_debug = weather_dbg
    st.session_state.last_dog = dog

    with st.spinner("AI 코치가 리포트를 작성하는 중..."):
        report = generate_report(
            openai_api_key=openai_api_key,
            habits_checked=checked_habits,
            habits_unchecked=unchecked_habits,
            mood=mood,
            weather=weather,
            dog_breed=(dog.get("breed") if dog else None),
            coach_style=coach_style,
        )
    st.session_state.last_report = report
    st.session_state.last_prompt = build_report_prompt(
        habits_checked=checked_habits,
        habits_unchecked=unchecked_habits,
        mood=mood,
        weather=weather,
        dog_breed=(dog.get("breed") if dog else None),
        coach_style=coach_style,
    )
    if report:
        db.add_coach_log(calendar_date.isoformat(), "daily", report, MODEL_NAME)

# Results
weather = st.session_state.last_weather
dog = st.session_state.last_dog
report = st.session_state.last_report
weather_dbg = st.session_state.last_weather_debug
prompt_bundle = st.session_state.get("last_prompt")

left, right = st.columns(2)

with left:
    st.markdown("### 🌦️ 오늘의 날씨")
    if weather:
        st.info(
            f"**{weather.get('city')}**\n\n"
            f"- 상태: {weather.get('description')}\n"
            f"- 기온: {weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C)\n"
            f"- 습도: {weather.get('humidity')}%\n"
            f"- 바람: {weather.get('wind_mps')} m/s"
        )
    else:
        st.warning("날씨 정보를 불러오지 못했어요.")
        st.info(weather_error_hint(weather_dbg or {}))
        with st.expander("🔧 날씨 디버그 상세"):
            st.write(weather_dbg if weather_dbg else {"ok": False, "reason": "no debug"})

with right:
    st.markdown("### 🎁 오늘의 보상")
    today_rewards = db.list_dog_collection(date_from=today_key, date_to=today_key)
    saved_record = db.get_habit_record(today_key, st.session_state.nickname)
    if len(HABITS) == 0:
        st.info("활성 습관이 없어 보상 시스템이 비활성화되어 있어요.")
    elif not saved_record:
        st.caption("체크인을 저장하면 오늘의 보상이 열립니다.")
    elif saved_record.get("checked_count", 0) == 0:
        st.caption("습관을 완료한 뒤 체크인을 저장하면 보상이 지급돼요.")
    elif st.session_state.last_new_rewards:
        for reward in st.session_state.last_new_rewards:
            st.image(reward["url"], use_container_width=True)
            st.caption(f"{reward['bucket']}% 달성 · {reward['rarity']}")
    elif today_rewards:
        st.image(today_rewards[0]["image_url"], use_container_width=True)
        st.caption(f"오늘 획득한 보상 {len(today_rewards)}장")
    else:
        st.caption("아직 오늘의 보상이 없어요. 체크인을 완료해보세요!")

    st.markdown("### 🐶 오늘의 강아지")
    if dog and dog.get("url"):
        st.image(dog["url"], use_container_width=True)
        st.caption(f"품종(추정): {dog.get('breed', '알 수 없음')}")
    else:
        st.warning("강아지 이미지를 불러오지 못했어요. (잠시 후 다시 시도해 주세요)")

st.markdown("### 📝 AI 코치 리포트")
if report:
    st.markdown(report)
else:
    st.caption("아직 리포트가 없어요. 위 버튼을 눌러 생성해보세요. (OpenAI 키 필요)")

st.markdown("### 🔗 공유용 텍스트")
share_text = {
    "date": date.today().isoformat(),
    "city": city_label,
    "coach_style": coach_style,
    "achievement_rate": achievement_rate,
    "checked_habits": checked_habits,
    "mood": mood,
    "weather": weather,
    "dog": dog,
    "report": report,
}
st.code(json.dumps(share_text, ensure_ascii=False, indent=2), language="json")
report_tab, collection_tab, group_tab = st.tabs(["🧠 AI 리포트", "🐶 도감", "👥 그룹"])

with report_tab:
    st.markdown("### 📝 AI 코치 리포트")
    if report:
        st.markdown(report)
    else:
        st.caption("아직 리포트가 없어요. 위 버튼을 눌러 생성해보세요. (OpenAI 키 필요)")

    with st.expander("🧩 생성된 프롬프트 보기"):
        if prompt_bundle:
            system_prompt, user_prompt = prompt_bundle
            st.markdown("**System Prompt**")
            st.code(system_prompt)
            st.markdown("**User Prompt**")
            st.code(user_prompt)
        else:
            st.caption("아직 프롬프트가 없어요. 리포트를 생성하면 표시됩니다.")

    st.markdown("### 🗂️ 코칭 히스토리")
    history_cols = st.columns(3)
    history_from = history_cols[0].date_input("시작일", value=date.today() - timedelta(days=30))
    history_to = history_cols[1].date_input("종료일", value=date.today())
    history_type = history_cols[2].selectbox("타입", options=["all", "daily", "weekly"])
    history_search = st.text_input("검색어(내용 포함)")
    history_rows = db.list_coach_logs(
        history_from.isoformat(),
        history_to.isoformat(),
        None if history_type == "all" else history_type,
        history_search,
    )
    if history_rows:
        history_df = pd.DataFrame(history_rows)
        st.dataframe(history_df[["date", "type", "model", "created_at"]], use_container_width=True)
        for row in history_rows[:10]:
            with st.expander(f"{row['date']} · {row['type']}"):
                st.code(row["content"])
        st.download_button(
            "코칭 기록 CSV 내보내기",
            history_df.to_csv(index=False).encode("utf-8"),
            file_name="coach_logs.csv",
            mime="text/csv",
        )
    else:
        st.caption("저장된 코칭 기록이 없습니다.")

    st.markdown("### 🔗 공유용 텍스트")
    share_text = {
        "date": calendar_date.isoformat(),
        "city": city_label,
        "coach_style": coach_style,
        "achievement_rate": achievement_rate,
        "checked_habits": checked_habits,
        "mood": mood,
        "weather": weather,
        "dog": dog,
        "report": report,
        "quote": quote,
        "cat_fact": cat_fact,
        "activity": activity,
        "reward_cards_today": len(db.list_dog_collection(date_from=today_key, date_to=today_key)),
    }
    st.code(json.dumps(share_text, ensure_ascii=False, indent=2), language="json")

with collection_tab:
    st.markdown("### 🐶 도감")
    filter_cols = st.columns(2)
    date_filter = filter_cols[0].selectbox("날짜 필터", options=["최근 7일", "전체"])
    rarity_filter = filter_cols[1].selectbox("등급", options=["all", "common", "rare", "epic"])
    if date_filter == "최근 7일":
        date_from = (date.today() - timedelta(days=6)).isoformat()
        date_to = date.today().isoformat()
    else:
        date_from = None
        date_to = None
    rarity = None if rarity_filter == "all" else rarity_filter
    collection_rows = db.list_dog_collection(date_from, date_to, rarity)
    if collection_rows:
        grid_cols = st.columns(4)
        for idx, item in enumerate(collection_rows):
            with grid_cols[idx % 4]:
                st.image(item["image_url"], use_container_width=True)
                st.caption(f"{item['date']} · {item['rarity']} · {item['earned_by']}")
                with st.expander("확대 보기"):
                    st.image(item["image_url"], use_container_width=True)
        collection_df = pd.DataFrame(collection_rows)
        st.download_button(
            "도감 기록 CSV 내보내기",
            collection_df.to_csv(index=False).encode("utf-8"),
            file_name="dog_collection.csv",
            mime="text/csv",
        )
    else:
        st.caption("조건에 맞는 도감 기록이 없습니다.")

with group_tab:
    st.markdown("### 👥 함께 streak")
    # Option A 선택: 닉네임 기반 다중 사용자 기록을 SQLite에 저장해 그룹 상태를 계산합니다.
    st.caption(
        "이 앱은 닉네임 기반으로 동작합니다. "
        "다른 멤버는 각자 닉네임을 입력하고 체크인을 해야 달성 여부가 반영됩니다."
    )
    create_cols = st.columns(2)
    group_name = create_cols[0].text_input("그룹 이름")
    if create_cols[1].button("그룹 생성"):
        if group_name.strip():
            code = db.create_group(group_name.strip())
            st.success(f"그룹 생성 완료! 코드: {code}")
        else:
            st.error("그룹 이름을 입력해 주세요.")

    join_cols = st.columns(2)
    join_code = join_cols[0].text_input("참여 코드")
    if join_cols[1].button("그룹 참여"):
        try:
            db.join_group(join_code.strip(), st.session_state.nickname.strip())
            st.success("그룹 참여 완료!")
        except ValueError as exc:
            st.error(str(exc))

    groups = db.list_groups_for_nickname(st.session_state.nickname)
    if groups:
        group_options = {f"{g['name']} ({g['group_code']})": g for g in groups}
        selected = st.selectbox("내 그룹", options=list(group_options.keys()))
        group = group_options[selected]
        members = db.get_group_members(group["group_code"])
        achieved = db.update_group_daily_status(group["id"], date.today().isoformat(), daily_goal_n)
        streak = db.calc_group_streak(group["id"], daily_goal_n)

        st.metric("그룹 streak", f"{streak}일")
        st.info("오늘 그룹 달성" if achieved else "오늘 그룹 미달성")

        member_rows = []
        for member in members:
            member_rows.append(
                {
                    "nickname": member["nickname"],
                    "today": "달성"
                    if db.compute_member_today_achieved(member["nickname"], date.today().isoformat(), daily_goal_n)
                    else "미달성",
                }
            )
        st.dataframe(pd.DataFrame(member_rows), use_container_width=True)

        logs = db.list_group_logs(group["id"], 7)
        if logs:
            log_df = pd.DataFrame(logs)
            st.dataframe(log_df, use_container_width=True)
    else:
        st.caption("아직 참여한 그룹이 없습니다.")

with st.expander("📎 API 안내 / 준비물"):
    st.markdown(
        """
**OpenWeatherMap 401(Invalid API key)일 때**
- 키 오타/공백/줄바꿈이 가장 흔한 원인입니다(이 앱은 자동 trim 처리하지만, 중간에 공백이 섞인 경우는 그대로 실패합니다).
- OpenWeatherMap에서 발급한 키가 맞는지 확인하세요.
- 발급 직후에는 활성화까지 시간이 걸릴 수 있습니다(보통 5~30분).
- 테스트 버튼으로 먼저 확인해보세요.

**OpenAI**
- OpenAI 키가 없으면 리포트 생성이 되지 않습니다.

**Dog CEO**
- 무료 공개 API라 간헐적 실패 가능
"""
    )

st.caption("© AI 습관 트래커 — 오늘의 작은 체크가 내일을 바꿔요.")
