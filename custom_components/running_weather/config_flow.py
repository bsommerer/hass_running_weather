from __future__ import annotations

from typing import Any

import voluptuous as vol
from homeassistant import config_entries
from homeassistant.const import CONF_NAME
from homeassistant.core import callback
from homeassistant.helpers import selector

from .const import (
    CONF_DAYLIGHT_BONUS,
    CONF_HOURLY_WINDOWS,
    CONF_SCORING,
    CONF_WEATHER_ENTITY,
    CONF_WEIGHTS,
    CONF_DAYLIGHT_END,
    CONF_DAYLIGHT_START,
    DEFAULT_DAYLIGHT_BONUS_PERCENT,
    DEFAULT_HOURLY_WINDOWS,
    DEFAULT_NAME,
    DEFAULT_SCORING_FUNCTIONS,
    DEFAULT_WEIGHTS_PERCENT,
    DOMAIN,
    SUPPORTED_METRICS,
)


def _build_weight_schema(current: dict[str, Any]) -> dict[str, selector.NumberSelector]:
    schema: dict[str, selector.NumberSelector] = {}
    for metric in SUPPORTED_METRICS:
        schema[vol.Optional(metric, default=current.get(metric, 0.0))] = selector.NumberSelector(
            selector.NumberSelectorConfig(min=0, max=100, step=1, unit_of_measurement="%")
        )
    return schema


def _build_scoring_schema(current: dict[str, str]) -> dict[str, selector.SelectSelector]:
    schema: dict[str, selector.SelectSelector] = {}
    for metric in SUPPORTED_METRICS:
        schema[vol.Optional(metric, default=current.get(metric, "relative_center"))] = selector.SelectSelector(
            selector.SelectSelectorConfig(
                options=[
                    "relative_center",
                    "low_is_better",
                    "high_is_better",
                ],
                translation_key="scoring_modes",
                mode=selector.SelectSelectorMode.DROPDOWN,
            )
        )
    return schema


class RunningWeatherConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Running Weather."""

    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}

        if user_input is not None:
            return self._create_entry(user_input)

        data_schema = vol.Schema(
            {
                vol.Required(CONF_WEATHER_ENTITY): selector.EntitySelector(
                    selector.EntitySelectorConfig(domain=["weather"])
                ),
                vol.Optional(CONF_NAME, default=DEFAULT_NAME): str,
                vol.Optional(
                    CONF_HOURLY_WINDOWS, default=DEFAULT_HOURLY_WINDOWS
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=12, max=240, step=1)
                ),
            }
        )

        return self.async_show_form(step_id="user", data_schema=data_schema, errors=errors)

    def _create_entry(self, data: dict[str, Any]):
        weather_entity = data[CONF_WEATHER_ENTITY]
        self._async_abort_entries_match({CONF_WEATHER_ENTITY: weather_entity})

        return self.async_create_entry(title=data.get(CONF_NAME, DEFAULT_NAME), data=data)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: config_entries.ConfigEntry):
        return RunningWeatherOptionsFlowHandler(config_entry)


class RunningWeatherOptionsFlowHandler(config_entries.OptionsFlow):
    """Handle Running Weather options."""

    def __init__(self, config_entry: config_entries.ConfigEntry) -> None:
        self.config_entry = config_entry

    async def async_step_init(self, user_input: dict[str, Any] | None = None):
        errors: dict[str, str] = {}
        entry = self.config_entry
        current_weights = entry.options.get(CONF_WEIGHTS, DEFAULT_WEIGHTS_PERCENT)
        current_scoring = entry.options.get(CONF_SCORING, DEFAULT_SCORING_FUNCTIONS)
        daylight_bonus = entry.options.get(CONF_DAYLIGHT_BONUS, DEFAULT_DAYLIGHT_BONUS_PERCENT)
        daylight_start = entry.options.get(CONF_DAYLIGHT_START)
        daylight_end = entry.options.get(CONF_DAYLIGHT_END)

        if user_input is not None:
            weight_input = user_input.get(CONF_WEIGHTS) or current_weights
            weights = {k: v for k, v in weight_input.items() if v is not None}
            if sum(weights.values()) <= 0:
                errors["base"] = "invalid_weights"
            else:
                scoring = user_input.get(CONF_SCORING) or current_scoring
                daylight = user_input.get(CONF_DAYLIGHT_BONUS, daylight_bonus)
                daylight_start_input = user_input.get(CONF_DAYLIGHT_START, entry.options.get(CONF_DAYLIGHT_START))
                daylight_end_input = user_input.get(CONF_DAYLIGHT_END, entry.options.get(CONF_DAYLIGHT_END))

                if (daylight_start_input and not daylight_end_input) or (
                    daylight_end_input and not daylight_start_input
                ):
                    errors["base"] = "invalid_daylight_window"
                else:
                    options = {
                        CONF_WEIGHTS: weights,
                        CONF_SCORING: scoring,
                        CONF_DAYLIGHT_BONUS: daylight,
                        CONF_DAYLIGHT_START: daylight_start_input,
                        CONF_DAYLIGHT_END: daylight_end_input,
                    }
                    return self.async_create_entry(title="", data=options)

        data_schema = vol.Schema(
            {
                vol.Optional(CONF_WEIGHTS, default=current_weights): vol.Schema(
                    _build_weight_schema(current_weights)
                ),
                vol.Optional(CONF_SCORING, default=current_scoring): vol.Schema(
                    _build_scoring_schema(current_scoring)
                ),
                vol.Optional(
                    CONF_DAYLIGHT_BONUS, default=daylight_bonus
                ): selector.NumberSelector(
                    selector.NumberSelectorConfig(min=0, max=30, step=0.5, unit_of_measurement="%")
                ),
                vol.Optional(CONF_DAYLIGHT_START, default=daylight_start): selector.TimeSelector(),
                vol.Optional(CONF_DAYLIGHT_END, default=daylight_end): selector.TimeSelector(),
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=data_schema,
            errors=errors,
        )
