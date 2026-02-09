import requests
from typing import Dict, Any, Optional

BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def _safe_get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def fetch_current_weather(city: str, api_key: str, units: str = "metric", lang: str = "kr", timeout: int = 8) -> Optional[Dict[str, Any]]:
    if not api_key:
        return None
    params = {"q": city, "appid": api_key, "units": units, "lang": lang}
    r = requests.get(BASE_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    weather_desc = None
    w0 = data.get("weather", [{}])[0] if isinstance(data.get("weather"), list) else {}
    weather_desc = w0.get("description")

    main = data.get("main", {}) if isinstance(data.get("main"), dict) else {}
    wind = data.get("wind", {}) if isinstance(data.get("wind"), dict) else {}

    rain_1h = _safe_get(data, "rain", "1h", default=0)
    snow_1h = _safe_get(data, "snow", "1h", default=0)

    return {
        "city": city,
        "description": weather_desc,
        "temp": main.get("temp"),
        "feels_like": main.get("feels_like"),
        "humidity": main.get("humidity"),
        "wind_speed": wind.get("speed"),
        "rain_1h": rain_1h,
        "snow_1h": snow_1h,
        "raw": data,
    }


def weather_to_summary(weather: Optional[Dict[str, Any]]) -> str:
    if not weather:
        return "날씨 정보 없음"
    parts = []
    if weather.get("description"):
        parts.append(str(weather["description"]))
    if weather.get("temp") is not None:
        parts.append(f"기온 {weather['temp']}°C")
    if weather.get("feels_like") is not None:
        parts.append(f"체감 {weather['feels_like']}°C")
    if weather.get("humidity") is not None:
        parts.append(f"습도 {weather['humidity']}%")
    if weather.get("wind_speed") is not None:
        parts.append(f"풍속 {weather['wind_speed']}m/s")
    if (weather.get("rain_1h") or 0) > 0:
        parts.append(f"강수(1h) {weather['rain_1h']}mm")
    if (weather.get("snow_1h") or 0) > 0:
        parts.append(f"적설(1h) {weather['snow_1h']}mm")
    return " / ".join(parts) if parts else "날씨 정보 있음(요약 불가)"


def simple_weather_hint(weather: Optional[Dict[str, Any]]) -> Optional[str]:
    if not weather:
        return None
    desc = (weather.get("description") or "").lower()
    rain = (weather.get("rain_1h") or 0) > 0
    snow = (weather.get("snow_1h") or 0) > 0
    if rain or "rain" in desc or "비" in desc:
        return "오늘은 비가 올 수 있어요. 실내 대체 습관(스트레칭/홈트)로 계획해보세요."
    if snow or "snow" in desc or "눈" in desc:
        return "눈/추위 가능성이 있어요. 이동 습관은 무리하지 말고 실내 루틴을 준비해요."
    if "clear" in desc or "맑" in desc:
        return "날씨가 괜찮다면 가벼운 산책/야외 활동을 습관에 끼워 넣어보세요."
    return None
