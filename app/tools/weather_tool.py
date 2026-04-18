"""Weather tool — fetches current weather and forecast via wttr.in.

Uses the free wttr.in API (no API key required).
Supports:
- Current conditions (temperature, humidity, wind, description)
- 3-day forecast
- Location-based lookup (city name, lat/lng, or landmark)
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

_TIMEOUT = 10.0
_WTTR_URL = "https://wttr.in/{location}"


class WeatherTool(BaseTool):
    """Fetch current weather and 3-day forecast for a location."""

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return (
            "Get current weather conditions and a 3-day forecast for any location. "
            "Use this when the user asks about weather, climate, or temperature "
            "for trip planning or daily information. "
            "Input a city name, landmark, or 'lat,lng' coordinates."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "location": {
                "type": "string",
                "description": "City name (e.g. 'Tokyo'), landmark, or coordinates (e.g. '35.6762,139.6503')",
            }
        }

    def run(self, **kwargs: Any) -> str:
        """Fetch weather data from wttr.in and return formatted results.

        Args:
            **kwargs: Must include 'location' (str).

        Returns:
            Formatted weather string with current conditions and forecast.
        """
        location = kwargs.get("location", "").strip()
        if not location:
            return "Error: 'location' parameter is required."

        logger.info("WeatherTool fetching weather for: %s", location)

        try:
            # Use wttr.in JSON API for structured data
            url = _WTTR_URL.format(location=httpx.URL(location))
            response = httpx.get(
                f"https://wttr.in/{location}",
                params={"format": "j1"},
                headers={"User-Agent": "curl/7.68.0"},
                timeout=_TIMEOUT,
                follow_redirects=True,
            )
            response.raise_for_status()
            data = response.json()

            return self._format_weather(data, location)

        except httpx.HTTPStatusError as exc:
            logger.error("Weather API HTTP error: %s", exc)
            return f"Weather lookup failed for '{location}': HTTP {exc.response.status_code}"
        except httpx.RequestError as exc:
            logger.error("Weather API network error: %s", exc)
            return f"Weather lookup failed for '{location}': network error"
        except Exception as exc:
            logger.error("Weather parsing error: %s", exc, exc_info=True)
            return f"Weather lookup failed for '{location}': {exc}"

    @staticmethod
    def _format_weather(data: dict, location: str) -> str:
        """Format wttr.in JSON response into readable text."""
        parts: list[str] = []

        # ── Current conditions ──
        current = data.get("current_condition", [{}])[0]
        if current:
            temp_c = current.get("temp_C", "?")
            temp_f = current.get("temp_F", "?")
            feels_c = current.get("FeelsLikeC", "?")
            humidity = current.get("humidity", "?")
            wind_kph = current.get("windspeedKmph", "?")
            wind_dir = current.get("winddir16Point", "?")
            desc_list = current.get("weatherDesc", [{}])
            desc = desc_list[0].get("value", "Unknown") if desc_list else "Unknown"
            visibility = current.get("visibility", "?")
            uv = current.get("uvIndex", "?")

            parts.append(f"## Current Weather in {location}")
            parts.append(f"- Condition: {desc}")
            parts.append(f"- Temperature: {temp_c}°C ({temp_f}°F), feels like {feels_c}°C")
            parts.append(f"- Humidity: {humidity}%")
            parts.append(f"- Wind: {wind_kph} km/h {wind_dir}")
            parts.append(f"- Visibility: {visibility} km")
            parts.append(f"- UV Index: {uv}")

        # ── Nearest area info ──
        area = data.get("nearest_area", [{}])[0]
        if area:
            area_name = area.get("areaName", [{}])[0].get("value", "")
            country = area.get("country", [{}])[0].get("value", "")
            region = area.get("region", [{}])[0].get("value", "")
            if area_name:
                parts.append(f"- Location: {area_name}, {region}, {country}")

        # ── 3-day forecast ──
        forecast = data.get("weather", [])
        if forecast:
            parts.append("\n## 3-Day Forecast")
            for day in forecast[:3]:
                date = day.get("date", "?")
                max_c = day.get("maxtempC", "?")
                min_c = day.get("mintempC", "?")
                avg_temp_c = day.get("avgtempC", "?")
                sun_hours = day.get("sunHour", "?")
                total_rain = day.get("totalSnow_cm", "0")

                # Get hourly description for midday
                hourly = day.get("hourly", [])
                midday_desc = "?"
                for h in hourly:
                    if h.get("time") in ("1200", "1500"):
                        desc_list = h.get("weatherDesc", [{}])
                        midday_desc = desc_list[0].get("value", "?") if desc_list else "?"
                        break

                parts.append(
                    f"- {date}: {midday_desc}, {min_c}°C–{max_c}°C "
                    f"(avg {avg_temp_c}°C), {sun_hours}h sun"
                )

        return "\n".join(parts)
