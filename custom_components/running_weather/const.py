from __future__ import annotations

DOMAIN = "running_weather"
DEFAULT_NAME = "Running Weather"
CONF_WEATHER_ENTITY = "weather_entity"
CONF_HOURLY_WINDOWS = "hourly_windows"
CONF_WEIGHTS = "weights"
CONF_SCORING = "scoring"
CONF_DAYLIGHT_BONUS = "daylight_bonus"
CONF_DAYLIGHT_START = "daylight_start"
CONF_DAYLIGHT_END = "daylight_end"
DEFAULT_HOURLY_WINDOWS = 72
DEFAULT_SCAN_INTERVAL_MINUTES = 30
DEFAULT_WEIGHTS_PERCENT = {
    "temperature": 30.0,
    "humidity": 20.0,
    "wind_speed": 15.0,
    "precipitation": 15.0,
    "sunshine": 10.0,
    "ground": 10.0,
}
DEFAULT_DAYLIGHT_BONUS_PERCENT = 5.0
DEFAULT_SCORING_FUNCTIONS = {
    "temperature": "relative_center",
    "humidity": "low_is_better",
    "wind_speed": "low_is_better",
    "precipitation": "low_is_better",
    "sunshine": "high_is_better",
    "ground": "low_is_better",
}

CONDITION_PENALTIES = {
    "thunderstorm": 0.4,
    "hail": 0.3,
    "rainy": 0.2,
    "rain": 0.2,
    "pouring": 0.25,
    "snowy": 0.25,
    "snowy-rainy": 0.25,
    "exceptional": 0.25,
}

SUPPORTED_METRICS = [
    "temperature",
    "humidity",
    "wind_speed",
    "precipitation",
    "sunshine",
    "ground",
]
