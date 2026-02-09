from typing import List, Dict, Optional, Tuple
from openai import OpenAI

TONES = ["따뜻한 코치", "엄격한 트레이너", "친구 같은 응원"]


def build_system_prompt(tone: str) -> str:
    if tone == "엄격한 트레이너":
        return (
            "너는 엄격하지만 공정한 습관 코치다. 변명은 부드럽게 차단하고, 실행 가능한 액션을 강하게 제시한다. "
            "짧고 단호한 문장, 그러나 모욕적이거나 비난하지 않는다. "
            "항상 구체적이고 측정 가능한 행동을 제안한다."
        )
    if tone == "친구 같은 응원":
        return (
            "너는 친근한 친구 같은 습관 코치다. 공감과 유머를 섞어 부담을 줄이고, 작은 성공도 크게 칭찬한다. "
            "너무 과장하지 말고, 현실적인 다음 कदम을 3개로 제시한다."
        )
    # default: 따뜻한 코치
    return (
        "너는 따뜻하고 전문적인 습관 코치다. 공감 기반 피드백 + 현실적인 개선안을 제시한다. "
        "자기효능감을 높이는 표현을 사용하고, 비난/진단/의학적 조언은 하지 않는다."
    )


def build_user_prompt(
    date_str: str,
    city: str,
    weather_summary: str,
    today_items: List[Dict],
    seven_day_summary: str,
    note: str,
) -> str:
    lines = []
    lines.append(f"- 날짜: {date_str}")
    lines.append(f"- 도시: {city}")
    lines.append(f"- 날씨 요약: {weather_summary or '없음'}")
    lines.append("")
    lines.append("## 오늘 체크인 결과")
    if not today_items:
        lines.append("- (오늘 기록된 습관 데이터가 없습니다)")
    else:
        for it in today_items:
            # it: {name, goal, value, frequency}
            lines.append(f"- {it['name']} (목표 {it['goal']}, 달성 {it['value']})")
    lines.append("")
    lines.append("## 최근 7일 요약")
    lines.append(seven_day_summary or "- 요약 없음")
    lines.append("")
    lines.append("## 사용자 메모")
    lines.append(note.strip() if note else "(없음)")
    lines.append("")
    lines.append("아래 형식을 반드시 지켜서 Markdown으로 출력해라:")
    lines.append("## 오늘의 피드백")
    lines.append("## 내일의 추천 액션 (3가지)")
    lines.append("## 습관별 개선 팁")
    lines.append("## 한 줄 격려")
    return "\n".join(lines)


def generate_coaching(
    api_key: str,
    model: str,
    tone: str,
    date_str: str,
    city: str,
    weather_summary: str,
    today_items: List[Dict],
    seven_day_summary: str,
    note: str,
    timeout: int = 20,
) -> Tuple[str, str]:
    """
    returns (output_text, input_summary)
    """
    client = OpenAI(api_key=api_key)
    sys = build_system_prompt(tone)
    user = build_user_prompt(date_str, city, weather_summary, today_items, seven_day_summary, note)

    # input_summary는 디버그/로그용으로 간단히
    input_summary = f"tone={tone}; city={city}; weather={weather_summary}; items={len(today_items)}"

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        timeout=timeout,
    )
    text = resp.choices[0].message.content or ""
    return text, input_summary
