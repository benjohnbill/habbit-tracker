# AI Habit Tracker (Streamlit)

## 기능
- 습관 CRUD (daily/weekly, goal 정수)
- 날짜별 체크인 저장 + 메모
- 달성률/간단 streak(daily) 계산
- OpenWeatherMap 현재 날씨 표시(실패 시 fallback)
- Dog API 보상 이미지(달성률 기준)
- OpenAI 코칭 생성 + 기록 저장 + 기록 조회/내보내기

## 설치
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
