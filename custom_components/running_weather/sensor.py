from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
import logging
from typing import Any

from astral import LocationInfo
from astral.sun import sun
import homeassistant.util.dt as dt_util
import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.components.sensor import PLATFORM_SCHEMA, SensorEntity
from homeassistant.const import ATTR_TEMPERATURE, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.typing import ConfigType, DiscoveryInfoType
from homeassistant.helpers.update_coordinator import (
    CoordinatorEntity,
    DataUpdateCoordinator,
    UpdateFailed,
)

from .const import (
    CONF_DAYLIGHT_BONUS,
    CONDITION_PENALTIES,
    CONF_HOURLY_WINDOWS,
    CONF_SCORING,
    CONF_WEIGHTS,
    DEFAULT_DAYLIGHT_BONUS_PERCENT,
    CONF_DAYLIGHT_END,
    CONF_DAYLIGHT_START,
    CONF_WEATHER_ENTITY,
    DEFAULT_SCORING_FUNCTIONS,
    DEFAULT_HOURLY_WINDOWS,
    DEFAULT_NAME,
    DEFAULT_SCAN_INTERVAL_MINUTES,
    DEFAULT_WEIGHTS_PERCENT,
    DOMAIN,
    SUPPORTED_METRICS,
)

# Configuration for YAML usage
PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_WEATHER_ENTITY): cv.entity_id,
        vol.Optional(CONF_NAME, default=DEFAULT_NAME): cv.string,
        vol.Optional(CONF_HOURLY_WINDOWS, default=DEFAULT_HOURLY_WINDOWS): vol.All(
            vol.Coerce(int), vol.Range(min=12, max=240)
        ),
        vol.Optional(CONF_WEIGHTS, default={}): vol.Schema(
            {
                vol.Optional("temperature"): vol.All(
                    vol.Coerce(float), vol.Range(min=0, max=100)
                ),
                vol.Optional("humidity"): vol.All(
                    vol.Coerce(float), vol.Range(min=0, max=100)
                ),
                vol.Optional("wind_speed"): vol.All(
                    vol.Coerce(float), vol.Range(min=0, max=100)
                ),
                vol.Optional("precipitation"): vol.All(
                    vol.Coerce(float), vol.Range(min=0, max=100)
                ),
                vol.Optional("sunshine"): vol.All(
                    vol.Coerce(float), vol.Range(min=0, max=100)
                ),
                vol.Optional("ground"): vol.All(
                    vol.Coerce(float), vol.Range(min=0, max=100)
                ),
            }
        ),
        vol.Optional(CONF_SCORING, default={}): vol.Schema(
            {
                vol.Optional("temperature"): vol.In(
                    ["relative_center", "low_is_better", "high_is_better"]
                ),
                vol.Optional("humidity"): vol.In(
                    ["relative_center", "low_is_better", "high_is_better"]
                ),
                vol.Optional("wind_speed"): vol.In(
                    ["relative_center", "low_is_better", "high_is_better"]
                ),
                vol.Optional("precipitation"): vol.In(
                    ["relative_center", "low_is_better", "high_is_better"]
                ),
                vol.Optional("sunshine"): vol.In(
                    ["relative_center", "low_is_better", "high_is_better"]
                ),
                vol.Optional("ground"): vol.In(
                    ["relative_center", "low_is_better", "high_is_better"]
                ),
            }
        ),
        vol.Optional(CONF_DAYLIGHT_BONUS, default=DEFAULT_DAYLIGHT_BONUS_PERCENT): vol.All(
            vol.Coerce(float), vol.Range(min=0, max=30)
        ),
        vol.Optional(CONF_DAYLIGHT_START): cv.time,
        vol.Optional(CONF_DAYLIGHT_END): cv.time,
    }
)


@dataclass
class MetricBounds:
    minimum: float
    maximum: float

    @property
    def spread(self) -> float:
        return max(self.maximum - self.minimum, 0.0001)

    def centered_score(self, value: float) -> float:
        midpoint = (self.maximum + self.minimum) / 2
        distance = abs(value - midpoint)
        return max(0.0, 1.0 - distance / (self.spread / 2))

    def low_is_better(self, value: float) -> float:
        raw = 1.0 - (value - self.minimum) / self.spread
        return max(0.0, min(1.0, raw))

    def high_is_better(self, value: float) -> float:
        raw = (value - self.minimum) / self.spread
        return max(0.0, min(1.0, raw))


LOGGER = logging.getLogger(__name__)

_DEFAULT_FALLBACK_SCORES = {
    "temperature": 0.6,
    "humidity": 0.6,
    "wind_speed": 0.6,
    "precipitation": 0.7,
    "sunshine": 0.6,
    "ground": 0.5,
}


@dataclass
class ScoreProfile:
    weights: dict[str, float]
    scoring: dict[str, str]
    daylight_bonus: float
    daylight_start: time | None
    daylight_end: time | None

    def metric_score(self, metric: str, bounds: MetricBounds, value: float | None) -> float:
        if value is None:
            return _DEFAULT_FALLBACK_SCORES.get(metric, 0.0)
        strategy = self.scoring.get(metric, "relative_center")
        if strategy == "low_is_better":
            return bounds.low_is_better(value)
        if strategy == "high_is_better":
            return bounds.high_is_better(value)
        return bounds.centered_score(value)


def _normalize_weights(raw: dict[str, float]) -> dict[str, float]:
    merged = {**DEFAULT_WEIGHTS_PERCENT, **raw}
    total = sum(merged.values()) or 1.0
    return {key: max(0.0, val) / total for key, val in merged.items()}


def _parse_time(value: Any) -> time | None:
    if value is None or value == "":
        return None
    if isinstance(value, time):
        return value
    return dt_util.parse_time(str(value))


def _build_score_profile(config: ConfigType) -> ScoreProfile:
    weights = _normalize_weights(config.get(CONF_WEIGHTS, {}))
    scoring = {**DEFAULT_SCORING_FUNCTIONS, **config.get(CONF_SCORING, {})}
    daylight_bonus = config.get(CONF_DAYLIGHT_BONUS, DEFAULT_DAYLIGHT_BONUS_PERCENT)
    daylight_start = _parse_time(config.get(CONF_DAYLIGHT_START))
    daylight_end = _parse_time(config.get(CONF_DAYLIGHT_END))
    return ScoreProfile(
        weights=weights,
        scoring=scoring,
        daylight_bonus=daylight_bonus,
        daylight_start=daylight_start,
        daylight_end=daylight_end,
    )


class RunningWeatherCoordinator(DataUpdateCoordinator[dict[str, Any]]):
    """Pulls forecast data and evaluates running windows."""

    def __init__(
        self,
        hass: HomeAssistant,
        weather_entity: str,
        hours_to_inspect: int,
        score_profile: ScoreProfile,
    ) -> None:
        super().__init__(
            hass,
            LOGGER,
            name="Running Weather",
            update_interval=timedelta(minutes=DEFAULT_SCAN_INTERVAL_MINUTES),
        )
        self.weather_entity = weather_entity
        self.hours_to_inspect = hours_to_inspect
        self.score_profile = score_profile

    async def _async_update_data(self) -> dict[str, Any]:
        state = self.hass.states.get(self.weather_entity)
        if state is None:
            raise UpdateFailed(f"Weather entity {self.weather_entity} not found")

        raw_forecast = await self._async_get_forecast(state)
        if not raw_forecast:
            raise UpdateFailed("No forecast data provided by weather entity")

        parsed_forecast = _normalize_forecast(raw_forecast, self.hours_to_inspect)
        baseline = _build_baseline(parsed_forecast)
        forecast_scores = _score_forecast(
            parsed_forecast, baseline, self.score_profile, self.hass
        )
        now = dt_util.utcnow()
        current_snapshot = _build_current_snapshot(
            state, forecast_scores, baseline, self.score_profile, self.hass, now
        )

        return {
            "baseline": baseline,
            "forecast_scores": forecast_scores,
            "current": current_snapshot,
            "best_today": _best_windows(forecast_scores, now, now + timedelta(days=1)),
            "best_week": _best_windows(forecast_scores, now, now + timedelta(days=7)),
        }

    async def _async_get_forecast(self, state: Any) -> list[dict[str, Any]]:
        forecast: list[dict[str, Any]] | None = state.attributes.get("forecast")
        if forecast:
            return forecast

        # Fallback for integrations exposing a service response instead of an attribute
        response = await self.hass.services.async_call(
            "weather",
            "get_forecast",
            {"entity_id": self.weather_entity, "type": "hourly"},
            blocking=True,
            return_response=True,
        )
        if isinstance(response, dict):
            forecast = response.get("forecast")

        return forecast or []


def _normalize_forecast(
    raw_forecast: list[dict[str, Any]], hours: int
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    now = dt_util.utcnow()
    for entry in raw_forecast:
        when = entry.get("datetime") or entry.get("time") or entry.get("dt")
        if when is None:
            continue
        dt_when = when if isinstance(when, datetime) else dt_util.parse_datetime(str(when))
        if dt_when is None:
            continue
        if dt_when < now or dt_when > now + timedelta(hours=hours):
            continue

        precipitation = entry.get("precipitation")
        if precipitation is None:
            precipitation = entry.get("precipitation_probability", 0)

        cloud_coverage = entry.get("cloud_coverage")
        uv_index = entry.get("uv_index")
        sunshine = _derive_sunshine(cloud_coverage, uv_index)
        ground = _derive_ground_wetness(precipitation, condition)

        normalized.append(
            {
                "datetime": dt_when,
                "temperature": entry.get("temperature"),
                "humidity": entry.get("humidity"),
                "wind_speed": entry.get("wind_speed"),
                "precipitation": precipitation,
                "condition": entry.get("condition"),
                "sunshine": sunshine,
                "ground": ground,
            }
        )

    normalized.sort(key=lambda item: item["datetime"])
    return normalized


def _derive_sunshine(
    cloud_coverage: float | None, uv_index: float | None
) -> float | None:
    if cloud_coverage is not None:
        return max(0.0, min(100.0, 100.0 - float(cloud_coverage)))
    if uv_index is not None:
        return max(0.0, min(100.0, (float(uv_index) / 11.0) * 100.0))
    return None


def _derive_ground_wetness(
    precipitation: float | None, condition: str | None
) -> float | None:
    if precipitation is None and not condition:
        return None

    wetness = 0.0
    if precipitation is not None:
        # treat values > 10 as probability percent, smaller numbers as mm/h
        if precipitation > 10:
            wetness += min(precipitation * 0.4, 40.0)
        else:
            wetness += min(precipitation * 12.0, 60.0)

    text = (condition or "").lower()
    if "snow" in text:
        wetness += 30.0
    if "sleet" in text or "snowy-rainy" in text:
        wetness += 25.0
    if "rain" in text or "pouring" in text:
        wetness += 20.0
    if "thunder" in text:
        wetness += 15.0

    return max(0.0, min(100.0, wetness))


def _build_baseline(forecast: list[dict[str, Any]]) -> dict[str, MetricBounds]:
    def bounds_for(key: str, default: float) -> MetricBounds:
        values = [entry.get(key) for entry in forecast if entry.get(key) is not None]
        if not values:
            return MetricBounds(default, default + 1)
        return MetricBounds(min(values), max(values))

    return {
        "temperature": bounds_for("temperature", 15.0),
        "humidity": bounds_for("humidity", 60.0),
        "wind_speed": bounds_for("wind_speed", 10.0),
        "precipitation": bounds_for("precipitation", 0.0),
        "sunshine": bounds_for("sunshine", 50.0),
        "ground": bounds_for("ground", 20.0),
    }


def _get_daylight_window(moment: datetime, hass: HomeAssistant, profile: ScoreProfile):
    local_moment = dt_util.as_local(moment)

    if profile.daylight_start and profile.daylight_end:
        start = datetime.combine(local_moment.date(), profile.daylight_start, tzinfo=local_moment.tzinfo)
        end = datetime.combine(local_moment.date(), profile.daylight_end, tzinfo=local_moment.tzinfo)
        if start >= end:
            return None, None
        return dt_util.as_utc(start), dt_util.as_utc(end)

    if hass.config.latitude is None or hass.config.longitude is None:
        return None, None

    location = LocationInfo(
        name="home",
        region="home",
        timezone=hass.config.time_zone or str(dt_util.DEFAULT_TIME_ZONE),
        latitude=hass.config.latitude,
        longitude=hass.config.longitude,
    )
    solar = sun(location.observer, date=local_moment.date(), tzinfo=location.timezone)
    return solar.get("sunrise"), solar.get("sunset")


def _is_daytime(moment: datetime, hass: HomeAssistant, profile: ScoreProfile) -> bool:
    sunrise, sunset = _get_daylight_window(moment, hass, profile)
    if sunrise and sunset:
        return sunrise <= moment <= sunset
    return 6 <= moment.hour <= 21


def _score_forecast(
    forecast: list[dict[str, Any]], baseline: dict[str, MetricBounds], profile: ScoreProfile, hass: HomeAssistant
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []

    for entry in forecast:
        temp = entry.get("temperature")
        humidity = entry.get("humidity")
        wind = entry.get("wind_speed")
        precip = entry.get("precipitation")
        condition = str(entry.get("condition") or "").lower()
        sunshine = entry.get("sunshine")
        ground = entry.get("ground")
        daylight_bonus = (profile.daylight_bonus / 100.0) if _is_daytime(entry["datetime"], hass, profile) else 0.0

        temp_score = profile.metric_score("temperature", baseline["temperature"], temp) or 0.0
        humidity_score = profile.metric_score("humidity", baseline["humidity"], humidity) or 0.0
        wind_score = profile.metric_score("wind_speed", baseline["wind_speed"], wind) or 0.0
        precip_score = profile.metric_score("precipitation", baseline["precipitation"], precip) or 0.0
        sunshine_score = profile.metric_score("sunshine", baseline["sunshine"], sunshine) or 0.0
        ground_score = profile.metric_score("ground", baseline["ground"], ground) or 0.0

        condition_penalty = 0.0
        for key, penalty in CONDITION_PENALTIES.items():
            if key in condition:
                condition_penalty = max(condition_penalty, penalty)

        base = (
            temp_score * profile.weights["temperature"]
            + humidity_score * profile.weights["humidity"]
            + wind_score * profile.weights["wind_speed"]
            + precip_score * profile.weights["precipitation"]
            + sunshine_score * profile.weights["sunshine"]
            + ground_score * profile.weights["ground"]
            + daylight_bonus
        )
        score = max(0.0, min(100.0, base * (1 - condition_penalty) * 100))

        scored.append({**entry, "score": round(score, 1)})

    return scored


def _best_windows(
    forecast_scores: list[dict[str, Any]], start: datetime, end: datetime
) -> list[dict[str, Any]]:
    filtered = [item for item in forecast_scores if start <= item["datetime"] <= end]
    filtered.sort(key=lambda item: item["score"], reverse=True)
    best = filtered[:5]
    return [
        {
            "time": dt_util.as_local(item["datetime"]).isoformat(),
            "score": item["score"],
            "temperature": item.get("temperature"),
            "humidity": item.get("humidity"),
            "wind_speed": item.get("wind_speed"),
            "condition": item.get("condition"),
            "sunshine": item.get("sunshine"),
            "ground": item.get("ground"),
            "reason": _describe_window(item),
        }
        for item in best
    ]


def _describe_window(window: dict[str, Any]) -> str:
    phrases: list[str] = []
    temperature = window.get("temperature")
    humidity = window.get("humidity")
    wind = window.get("wind_speed")
    precip = window.get("precipitation")
    condition = window.get("condition")
    sunshine = window.get("sunshine")
    ground = window.get("ground")

    if temperature is not None:
        phrases.append(f"~{temperature}° gefühlt")
    if humidity is not None:
        phrases.append(f"{humidity}% r.F.")
    if wind is not None:
        phrases.append(f"Wind {wind} m/s")
    if precip:
        phrases.append(f"Niederschlag {precip}")
    if sunshine is not None:
        phrases.append(f"Sonne {int(round(sunshine))}%")
    if ground is not None:
        phrases.append(f"Boden {(100 - ground):.0f}% trocken")
    if condition:
        phrases.append(str(condition))

    return ", ".join(phrases)


def _build_current_snapshot(
    state: Any,
    forecast_scores: list[dict[str, Any]],
    baseline: dict[str, MetricBounds],
    profile: ScoreProfile,
    hass: HomeAssistant,
    now: datetime,
) -> dict[str, Any]:
    # Prefer the forecast slot closest to now for consistency
    nearest = min(
        forecast_scores,
        key=lambda item: abs((item["datetime"] - now).total_seconds()),
        default=None,
    )

    temperature = state.attributes.get(ATTR_TEMPERATURE)
    humidity = state.attributes.get("humidity")
    wind = state.attributes.get("wind_speed")
    precip = state.attributes.get("precipitation")
    if precip is None:
        precip = state.attributes.get("precipitation_probability")
    condition = state.attributes.get("condition")
    sunshine = _derive_sunshine(
        state.attributes.get("cloud_coverage"), state.attributes.get("uv_index")
    )
    ground = _derive_ground_wetness(precip, condition)

    temp_score = (
        profile.metric_score("temperature", baseline["temperature"], temperature)
        if temperature is not None
        else None
    )
    humidity_score = (
        profile.metric_score("humidity", baseline["humidity"], humidity)
        if humidity is not None
        else None
    )
    wind_score = (
        profile.metric_score("wind_speed", baseline["wind_speed"], wind)
        if wind is not None
        else None
    )
    precip_score = (
        profile.metric_score("precipitation", baseline["precipitation"], precip)
        if precip is not None
        else None
    )
    sunshine_score = (
        profile.metric_score("sunshine", baseline["sunshine"], sunshine)
        if sunshine is not None
        else None
    )
    ground_score = (
        profile.metric_score("ground", baseline["ground"], ground)
        if ground is not None
        else None
    )

    current_score = None
    if nearest is not None:
        current_score = nearest["score"]
    elif None not in (temp_score, humidity_score, wind_score, precip_score):
        current_score = (
            (temp_score or 0.0) * profile.weights["temperature"]
            + (humidity_score or 0.0) * profile.weights["humidity"]
            + (wind_score or 0.0) * profile.weights["wind_speed"]
            + (precip_score or 0.0) * profile.weights["precipitation"]
            + (sunshine_score or 0.0) * profile.weights["sunshine"]
            + (ground_score or 0.0) * profile.weights["ground"]
        )
        daylight_bonus = (
            profile.daylight_bonus / 100.0 if _is_daytime(now, hass, profile) else 0.0
        )
        current_score = round((current_score + daylight_bonus) * 100, 1)

    return {
        "score": current_score,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind,
        "precipitation": precip,
        "sunshine": sunshine,
        "ground": ground,
        "condition": condition,
        "source": state.entity_id,
    }


async def async_setup_platform(
    hass: HomeAssistant,
    config: ConfigType,
    async_add_entities: AddEntitiesCallback,
    discovery_info: DiscoveryInfoType | None = None,
) -> None:
    weather_entity = config[CONF_WEATHER_ENTITY]
    name = config[CONF_NAME]
    hours = config[CONF_HOURLY_WINDOWS]
    score_profile = _build_score_profile(config)

    coordinator = RunningWeatherCoordinator(
        hass, weather_entity, hours, score_profile
    )
    await coordinator.async_refresh()
    if not coordinator.last_update_success:
        raise UpdateFailed("Initial Running Weather update failed")

    async_add_entities([RunningWeatherSensor(coordinator, name)])


async def async_setup_entry(
    hass: HomeAssistant, entry: ConfigEntry, async_add_entities: AddEntitiesCallback
) -> None:
    """Set up Running Weather from a config entry."""

    config: dict[str, Any] = {**entry.data, **entry.options}
    weather_entity = config[CONF_WEATHER_ENTITY]
    name = config.get(CONF_NAME, DEFAULT_NAME)
    hours = config.get(CONF_HOURLY_WINDOWS, DEFAULT_HOURLY_WINDOWS)
    score_profile = _build_score_profile(config)

    coordinator = RunningWeatherCoordinator(
        hass, weather_entity, hours, score_profile
    )
    await coordinator.async_refresh()
    if not coordinator.last_update_success:
        raise UpdateFailed("Initial Running Weather update failed")

    async_add_entities([RunningWeatherSensor(coordinator, name)])


class RunningWeatherSensor(CoordinatorEntity[RunningWeatherCoordinator], SensorEntity):
    """Sensor summarizing the current running conditions."""

    _attr_has_entity_name = True
    _attr_should_poll = False
    _attr_native_unit_of_measurement = None
    _attr_icon = "mdi:run"

    def __init__(self, coordinator: RunningWeatherCoordinator, name: str) -> None:
        super().__init__(coordinator)
        self._attr_name = name
        self._attr_unique_id = f"{DOMAIN}_{coordinator.weather_entity}"

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            identifiers={(DOMAIN, "running_weather")},
            name="Running Weather",
            manufacturer="Running Weather",
            model="Forecast scorer",
        )

    @property
    def native_value(self) -> str | None:
        data = self.coordinator.data
        if not data:
            return None
        current = data.get("current")
        forecast_scores = data.get("forecast_scores", [])
        if not current:
            return None

        current_score = current.get("score")
        if current_score is None:
            return None

        best_score = max((item["score"] for item in forecast_scores), default=current_score)
        ratio = current_score / best_score if best_score else 0
        if ratio >= 0.8:
            return "sehr gut"
        if ratio >= 0.55:
            return "ok"
        return "schlecht"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data or {}
        current = data.get("current", {})
        return {
            "current_score": current.get("score"),
            "current_conditions": current,
            "best_today": data.get("best_today", []),
            "best_week": data.get("best_week", []),
            "baseline": _serialize_baseline(data.get("baseline", {})),
            "forecast_samples": len(data.get("forecast_scores", [])),
        }

    @property
    def available(self) -> bool:
        return self.coordinator.last_update_success


def _serialize_baseline(baseline: dict[str, MetricBounds]) -> dict[str, dict[str, float]]:
    serialized: dict[str, dict[str, float]] = {}
    for key, bounds in baseline.items():
        serialized[key] = {"min": bounds.minimum, "max": bounds.maximum}
    return serialized
