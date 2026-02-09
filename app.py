import streamlit as st
import pandas as pd
from datetime import date, timedelta

import db
from services.weather import fetch_current_weather, weather_to_summary, simple_weather_hint
from services.dog import fetch_random_dog_images
from services.coach import generate_coaching, TONES
from utils.stats import build_seven_day_summary, compute_today_achievement, items_to_dataframe
from utils.streaks import compute_daily_streak

st.set_page_config(page_title="AI Habit Tracker", page_icon="✅", layout="wide")


# ---------- Helpers ----------
def get_secret_or_sidebar(key_name: str, label: str, password: bool = True) -> str:
    # 1) secrets
    if key_name in st.secrets and st.secrets[key_name]:
        return str(st.secrets[key_name])
    # 2) session state
    ss_key = f"__{key_name}"
    if ss_key not in st.session_state:
        st.session_state[ss_key] = ""
    # 3) sidebar input
    t = st.sidebar.text_input(label, value=st.session_state[ss_key], type="password" if password else "default")
    st.session_state[ss_key] = t
    return t


@st.cache_data(ttl=600)
def cached_weather(city: str, api_key: str):
    return fetch_current_weather(city=city, api_key=api_key)


@st.cache_data(ttl=60)
def cached_dogs(n: int):
    return fetch_random_dog_images(n=n)


def ensure_seed():
    db.init_db()
    db.seed_sample_habits_if_empty()


# ---------- UI: Sidebar ----------
ensure_seed()

st.sidebar.title("AI Habit Tracker")

city = st.sidebar.text_input("도시 (기본: Seoul)", value=st.session_state.get("city", "Seoul"))
st.session_state["city"] = city

tone = st.sidebar.selectbox("코칭 톤", options=TONES, index=TONES.index(st.session_state.get("tone", TONES[0])))
st.session_state["tone"] = tone

openai_key = get_secret_or_sidebar("OPENAI_API_KEY", "OpenAI API Key")
owm_key = get_secret_or_sidebar("OPENWEATHER_API_KEY", "OpenWeatherMap API Key")

storage = st.sidebar.radio("저장소", options=["sqlite3 (default)", "json (옵션-미구현)"], index=0)
if storage != "sqlite3 (default)":
    st.sidebar.warning("json 저장소는 옵션이며 현재 예시는 sqlite3만 구현되어 있어요.")

menu = st.sidebar.radio("메뉴", options=["오늘 체크인", "습관 관리", "대시보드/통계", "AI 코칭 기록"])

st.sidebar.divider()
with st.sidebar.expander("고급 설정"):
    model = st.text_input("OpenAI 모델", value=st.session_state.get("model", "gpt-4o-mini"))
    st.session_state["model"] = model


# ---------- Data ----------
habits = db.list_habits()


# ---------- Page: Habits Management ----------
def page_habits():
    st.header("습관 관리")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("습관 추가")
        with st.form("add_habit_form", clear_on_submit=True):
            name = st.text_input("이름", placeholder="예: 물 8잔 마시기")
            description = st.text_area("설명(선택)", height=80)
            frequency = st.selectbox("주기", options=["daily", "weekly"])
            goal = st.number_input("목표(goal, 정수)", min_value=1, value=1, step=1)
            reminder_text = st.text_input("알림 메시지(선택)", placeholder="예: 지금 물 한 잔!")
            submitted = st.form_submit_button("추가")
            if submitted:
                if not name.strip():
                    st.error("이름(name)은 필수입니다.")
                else:
                    db.create_habit(name.strip(), description, frequency, int(goal), reminder_text)
                    st.success("습관을 추가했어요.")
                    st.rerun()

    with col2:
        st.subheader("기존 습관")
        if not habits:
            st.info("아직 습관이 없어요. 왼쪽에서 추가해보세요.")
            return

        for h in habits:
            with st.expander(f"#{h['habit_id']} • {h['name']} ({h['frequency']}, goal={h['goal']})", expanded=False):
                st.caption(f"created_at: {h['created_at']}")
                st.write(h.get("description") or "_설명 없음_")
                st.write(f"알림: {h.get('reminder_text') or '-'}")

                with st.form(f"edit_habit_{h['habit_id']}"):
                    name = st.text_input("이름", value=h["name"], key=f"n_{h['habit_id']}")
                    description = st.text_area("설명", value=h.get("description") or "", height=80, key=f"d_{h['habit_id']}")
                    frequency = st.selectbox(
                        "주기", options=["daily", "weekly"], index=["daily", "weekly"].index(h["frequency"]), key=f"f_{h['habit_id']}"
                    )
                    goal = st.number_input("목표(goal)", min_value=1, value=int(h["goal"]), step=1, key=f"g_{h['habit_id']}")
                    reminder_text = st.text_input("알림 메시지", value=h.get("reminder_text") or "", key=f"r_{h['habit_id']}")
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.form_submit_button("수정 저장"):
                            db.update_habit(int(h["habit_id"]), name.strip(), description, frequency, int(goal), reminder_text)
                            st.success("수정했어요.")
                            st.rerun()
                    with c2:
                        if st.form_submit_button("삭제", type="primary"):
                            db.delete_habit(int(h["habit_id"]))
                            st.success("삭제했어요.")
                            st.rerun()


# ---------- Page: Today Check-in ----------
def page_today():
    st.header("오늘 체크인")

    # date selection
    default_date = st.session_state.get("selected_date", date.today())
    selected_date = st.date_input("날짜 선택", value=default_date)
    st.session_state["selected_date"] = selected_date
    date_str = selected_date.strftime("%Y-%m-%d")

    # weather
    weather = None
    weather_summary = ""
    weather_hint = None
    try:
        if owm_key:
            weather = cached_weather(city, owm_key)
    except Exception as e:
        st.warning(f"날씨 정보를 불러오지 못했어요: {e}")
        weather = None

    weather_summary = weather_to_summary(weather)
    weather_hint = simple_weather_hint(weather)

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.subheader("오늘의 날씨")
        if weather:
            st.write(f"**{city}**")
            st.write(weather_summary)
            if weather_hint:
                st.info(weather_hint)
        else:
            st.info("날씨 정보 없음 (API Key가 없거나 호출 실패)")

    # load existing checkin (if any)
    existing = db.get_checkin(date_str)
    existing_note = existing["checkin"].get("note") if existing else ""
    existing_items = {int(it["habit_id"]): int(it["value"]) for it in (existing["items"] if existing else [])}

    with right:
        st.subheader("습관 체크인")
        if not habits:
            st.warning("습관이 없습니다. 먼저 '습관 관리'에서 습관을 추가하세요.")
            return

        with st.form("checkin_form"):
            values = {}
            for h in habits:
                hid = int(h["habit_id"])
                goal = int(h["goal"])
                # 간단 규칙: goal이 1이면 checkbox UX, 그 외는 number_input
                if goal == 1:
                    checked = existing_items.get(hid, 0) >= 1
                    v = st.checkbox(f"{h['name']} (goal=1)", value=checked, key=f"chk_{date_str}_{hid}")
                    values[hid] = 1 if v else 0
                else:
                    v = st.number_input(
                        f"{h['name']} (목표 {goal})",
                        min_value=0,
                        value=int(existing_items.get(hid, 0)),
                        step=1,
                        key=f"num_{date_str}_{hid}",
                    )
                    values[hid] = int(v)

            note = st.text_area("오늘 메모(선택)", value=existing_note or "", height=100)
            saved = st.form_submit_button("저장")

        if saved:
            try:
                checkin_id = db.upsert_checkin(date_str, note)
                for hid, v in values.items():
                    db.upsert_checkin_item(checkin_id, hid, int(v))
                st.success("오늘 체크인을 저장했어요.")
                st.session_state["last_saved_date"] = date_str
                st.rerun()
            except Exception as e:
                st.error(f"저장 중 오류: {e}")

    # summary + streaks + dog reward + coaching
    st.divider()
    st.subheader("오늘 요약")

    # compute today values from current DB (fresh)
    fresh = db.get_checkin(date_str)
    today_values = {}
    today_items_for_ai = []
    if fresh:
        for it in fresh["items"]:
            hid = int(it["habit_id"])
            today_values[hid] = int(it["value"])
            today_items_for_ai.append(
                {"name": it["name"], "goal": int(it["goal"]), "value": int(it["value"]), "frequency": it["frequency"]}
            )

    rate, success_count, total_count = compute_today_achievement(habits, today_values)
    st.write(f"- 달성률: **{rate:.0f}%** ({success_count}/{total_count})")

    # streak top 3 (daily only)
    start_30 = (selected_date - timedelta(days=60)).strftime("%Y-%m-%d")
    end_30 = date_str
    items_60d = db.get_items_between(start_30, end_30)

    streak_rows = []
    for h in habits:
        if h["frequency"] != "daily":
            continue
        s = compute_daily_streak(items_60d, int(h["habit_id"]), int(h["goal"]), date_str)
        streak_rows.append((h["name"], s))
    streak_rows.sort(key=lambda x: x[1], reverse=True)
    top3 = streak_rows[:3]
    if top3:
        st.write("**streak TOP 3 (daily)**")
        for name, s in top3:
            st.write(f"- {name}: {s}일 연속")

    # Dog reward
    st.divider()
    st.subheader("오늘의 보상 🐶")
    try:
        if total_count == 0:
            st.info("습관이 없어서 보상을 계산할 수 없어요.")
        else:
            if rate >= 100:
                st.success("퍼펙트! 100% 달성 🎉🎉")
                urls = cached_dogs(2)
                cols = st.columns(2)
                for i, u in enumerate(urls[:2]):
                    with cols[i]:
                        st.image(u, use_container_width=True)
            elif rate >= 70:
                st.success("좋아요! 70% 이상 달성 🎉")
                urls = cached_dogs(1)
                if urls:
                    st.image(urls[0], use_container_width=True)
            else:
                st.info("오늘도 기록한 것만으로 충분히 잘했어요. 내일은 조금만 더 가볍게 가볼까요?")
                urls = cached_dogs(1)
                if urls:
                    st.image(urls[0], use_container_width=True)
    except Exception as e:
        st.warning(f"Dog API 호출 실패: {e}")

    # AI coaching
    st.divider()
    st.subheader("AI 코칭")

    # 7-day summary
    start_7 = (selected_date - timedelta(days=6)).strftime("%Y-%m-%d")
    end_7 = date_str
    items_7d = db.get_items_between(start_7, end_7)
    seven_day_summary = build_seven_day_summary(items_7d)

    with st.expander("최근 7일 요약 보기", expanded=False):
        st.markdown(seven_day_summary)

    can_generate = bool(openai_key) and bool(fresh) and bool(today_items_for_ai)
    c1, c2 = st.columns([1, 1])
    with c1:
        gen = st.button("AI 코칭 생성", disabled=not can_generate, type="primary")
    with c2:
        regen = st.button("코칭 다시 생성", disabled=not can_generate)

    if (gen or regen) and not openai_key:
        st.error("OpenAI API Key가 필요해요.")
        return

    if (gen or regen) and not can_generate:
        st.warning("코칭을 생성하려면 먼저 오늘 체크인을 저장해 주세요.")
        return

    if gen or regen:
        try:
            output, input_summary = generate_coaching(
                api_key=openai_key,
                model=st.session_state.get("model", "gpt-4o-mini"),
                tone=tone,
                date_str=date_str,
                city=city,
                weather_summary=weather_summary,
                today_items=today_items_for_ai,
                seven_day_summary=seven_day_summary,
                note=fresh["checkin"].get("note") if fresh else "",
            )
            db.add_coaching_log(
                date_str=date_str,
                tone=tone,
                weather_summary=weather_summary,
                input_summary=input_summary,
                output_text=output,
            )
            st.markdown(output)
        except Exception as e:
            st.error(f"코칭 생성 실패: {e}")


# ---------- Page: Dashboard ----------
def page_dashboard():
    st.header("대시보드 / 통계")

    if not habits:
        st.warning("습관이 없습니다. 먼저 '습관 관리'에서 습관을 추가하세요.")
        return

    preset = st.selectbox("기간", options=["최근 7일", "최근 30일", "커스텀"], index=0)
    today = date.today()
    if preset == "최근 7일":
        start = today - timedelta(days=6)
        end = today
    elif preset == "최근 30일":
        start = today - timedelta(days=29)
        end = today
    else:
        c1, c2 = st.columns(2)
        with c1:
            start = st.date_input("시작일", value=today - timedelta(days=29), key="dash_start")
        with c2:
            end = st.date_input("종료일", value=today, key="dash_end")

    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    items = db.get_items_between(start_s, end_s)
    df = items_to_dataframe(items)

    if df.empty:
        st.info("선택한 기간에 데이터가 없어요.")
        return

    df["success"] = df["value"].astype(int) >= df["goal"].astype(int)

    st.subheader("전체 달성률 추이")
    daily = df.groupby("date")["success"].mean().reset_index()
    daily["success_rate"] = daily["success"] * 100.0
    daily = daily.drop(columns=["success"])
    st.line_chart(daily.set_index("date"))

    st.subheader("습관별 달성률")
    by_habit = df.groupby("name")["success"].mean().reset_index()
    by_habit["success_rate"] = by_habit["success"] * 100.0
    st.bar_chart(by_habit.set_index("name")[["success_rate"]])

    st.subheader("가장 긴 streak TOP 3 (daily)")
    # compute streaks as of end date
    streak_rows = []
    for h in habits:
        if h["frequency"] != "daily":
            continue
        s = compute_daily_streak(items, int(h["habit_id"]), int(h["goal"]), end_s)
        streak_rows.append((h["name"], s))
    streak_rows.sort(key=lambda x: x[1], reverse=True)
    top3 = streak_rows[:3]
    if top3:
        for name, s in top3:
            st.write(f"- {name}: {s}일 연속")
    else:
        st.info("daily 습관이 없거나 streak를 계산할 데이터가 없어요.")

    st.divider()
    st.subheader("AI 한 줄 요약")
    if st.button("AI 한 줄 요약 생성", type="primary"):
        if not openai_key:
            st.error("OpenAI API Key가 필요해요.")
            return
        # 간단 요약 프롬프트
        summary_lines = []
        summary_lines.append(f"기간: {start_s} ~ {end_s}")
        summary_lines.append("습관별 성공률:")
        for _, r in by_habit.sort_values("success_rate", ascending=False).iterrows():
            summary_lines.append(f"- {r['name']}: {r['success_rate']:.0f}%")
        weakest = by_habit.sort_values("success_rate", ascending=True).iloc[0]
        summary_lines.append(f"가장 약한 습관: {weakest['name']} ({weakest['success_rate']:.0f}%)")
        user_prompt = "\n".join(summary_lines) + "\n\n위 통계를 한 줄로 요약해줘. (한국어, 간결, 실행 의지 높이기)"

        try:
            output, _ = generate_coaching(
                api_key=openai_key,
                model=st.session_state.get("model", "gpt-4o-mini"),
                tone=tone,
                date_str=end_s,
                city=city,
                weather_summary="(대시보드 요약에는 날씨 생략)",
                today_items=[],
                seven_day_summary=user_prompt,
                note="(한 줄 요약 요청)",
            )
            # generate_coaching 포맷은 4파트 강제라서, 여기서는 간단히 첫 줄만 표시하도록 처리
            st.markdown("**결과**")
            st.write(output.strip().splitlines()[0] if output.strip() else output)
        except Exception as e:
            st.error(f"요약 생성 실패: {e}")


# ---------- Page: Coaching Logs ----------
def page_logs():
    st.header("AI 코칭 기록")

    logs = db.list_coaching_logs(limit=200)
    if not logs:
        st.info("아직 코칭 기록이 없어요.")
        return

    # select
    options = [f"{l['date']} | {l['tone']} | #{l['coaching_id']}" for l in logs]
    idx = st.selectbox("기록 선택", options=list(range(len(options))), format_func=lambda i: options[i])
    selected = logs[idx]

    st.subheader(f"{selected['date']} • {selected['tone']}")
    st.caption(f"created_at: {selected['created_at']}")
    if selected.get("weather_summary"):
        st.write(f"날씨: {selected['weather_summary']}")

    # show checkin too
    chk = db.get_checkin(selected["date"])
    if chk:
        st.write("**체크인 메모**")
        st.write(chk["checkin"].get("note") or "-")
        st.write("**체크인 항목**")
        df = pd.DataFrame(chk["items"])
        st.dataframe(df[["name", "goal", "value", "frequency"]], use_container_width=True)

    st.divider()
    st.markdown(selected["output_text"])

    st.divider()
    st.subheader("내보내기")
    export_df = pd.DataFrame(logs)
    st.download_button(
        "코칭 로그 CSV 다운로드",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="coaching_logs.csv",
        mime="text/csv",
    )


# ---------- Router ----------
if menu == "습관 관리":
    page_habits()
elif menu == "대시보드/통계":
    page_dashboard()
elif menu == "AI 코칭 기록":
    page_logs()
else:
    page_today()
