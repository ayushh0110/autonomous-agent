"""DateTime tool — current time, timezone conversion, date calculations.

Pure Python stdlib — no API needed. Uses the `zoneinfo` module (Python 3.9+).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, available_timezones

from app.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

# Common timezone aliases → IANA names
_TZ_ALIASES: dict[str, str] = {
    "est": "America/New_York", "edt": "America/New_York",
    "cst": "America/Chicago", "cdt": "America/Chicago",
    "mst": "America/Denver", "mdt": "America/Denver",
    "pst": "America/Los_Angeles", "pdt": "America/Los_Angeles",
    "gmt": "Europe/London", "utc": "UTC",
    "bst": "Europe/London",
    "cet": "Europe/Paris", "cest": "Europe/Paris",
    "ist": "Asia/Kolkata",
    "jst": "Asia/Tokyo",
    "kst": "Asia/Seoul",
    "cst_china": "Asia/Shanghai", "cst china": "Asia/Shanghai",
    "hkt": "Asia/Hong_Kong",
    "sgt": "Asia/Singapore",
    "aest": "Australia/Sydney", "aedt": "Australia/Sydney",
    "nzst": "Pacific/Auckland", "nzdt": "Pacific/Auckland",
    # City shortcuts
    "new york": "America/New_York", "nyc": "America/New_York",
    "los angeles": "America/Los_Angeles", "la": "America/Los_Angeles",
    "chicago": "America/Chicago",
    "london": "Europe/London",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "tokyo": "Asia/Tokyo",
    "seoul": "Asia/Seoul",
    "beijing": "Asia/Shanghai", "shanghai": "Asia/Shanghai",
    "hong kong": "Asia/Hong_Kong",
    "singapore": "Asia/Singapore",
    "sydney": "Australia/Sydney",
    "auckland": "Pacific/Auckland",
    "dubai": "Asia/Dubai",
    "mumbai": "Asia/Kolkata", "delhi": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata", "india": "Asia/Kolkata",
    "moscow": "Europe/Moscow",
    "toronto": "America/Toronto",
    "vancouver": "America/Vancouver",
    "denver": "America/Denver",
    "cairo": "Africa/Cairo",
    "nairobi": "Africa/Nairobi",
    "lagos": "Africa/Lagos",
    "sao paulo": "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires",
    "mexico city": "America/Mexico_City",
    "bangkok": "Asia/Bangkok",
    "jakarta": "Asia/Jakarta",
    "kuala lumpur": "Asia/Kuala_Lumpur",
}


class DateTimeTool(BaseTool):
    """Get current date/time and perform date calculations."""

    @property
    def name(self) -> str:
        return "datetime"

    @property
    def description(self) -> str:
        return (
            "Get current date and time, convert between timezones, or calculate "
            "date differences. Actions: "
            "'now' — current time (optionally in a timezone), "
            "'convert' — convert time between timezones, "
            "'diff' — days between two dates or days until a date. "
            "Example: action='now', timezone='tokyo' or "
            "action='diff', date='2025-12-25'."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "action": {
                "type": "string",
                "description": "One of: 'now', 'convert', 'diff'.",
            },
            "timezone": {
                "type": "string",
                "description": "Timezone name or city (e.g. 'tokyo', 'ist', 'America/New_York'). Optional.",
            },
            "date": {
                "type": "string",
                "description": "A date in YYYY-MM-DD format for 'diff' calculations. Optional.",
            },
        }

    def run(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action", "now")).strip().lower()
        timezone = str(kwargs.get("timezone", "")).strip()
        date_str = str(kwargs.get("date", "")).strip()

        logger.info("[DATETIME] action=%s tz=%s date=%s", action, timezone, date_str)

        if action == "now":
            return self._now(timezone)
        elif action in ("convert", "conversion"):
            return self._now(timezone)  # show time in requested TZ
        elif action == "diff":
            return self._diff(date_str, timezone)
        else:
            return self._now(timezone)

    def _now(self, timezone: str) -> str:
        """Get current time, optionally in a specific timezone."""
        tz = self._resolve_tz(timezone) if timezone else None
        now = datetime.now(tz or ZoneInfo("UTC"))

        tz_name = timezone or "UTC"
        return (
            f"Current date and time in {tz_name}:\n"
            f"  Date: {now.strftime('%A, %B %d, %Y')}\n"
            f"  Time: {now.strftime('%I:%M %p')} ({now.strftime('%H:%M')})\n"
            f"  Timezone: {now.tzname()} (UTC{now.strftime('%z')})"
        )

    def _diff(self, date_str: str, timezone: str) -> str:
        """Calculate days between now and a target date."""
        if not date_str:
            return "Error: 'date' is required for diff (format: YYYY-MM-DD)."

        try:
            target = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return f"Error: Invalid date format '{date_str}'. Use YYYY-MM-DD."

        tz = self._resolve_tz(timezone) if timezone else ZoneInfo("UTC")
        now = datetime.now(tz).replace(tzinfo=None)
        delta = target - now.replace(hour=0, minute=0, second=0, microsecond=0)

        days = delta.days
        if days > 0:
            return f"{days} days from now until {date_str}."
        elif days == 0:
            return f"{date_str} is today!"
        else:
            return f"{date_str} was {abs(days)} days ago."

    @staticmethod
    def _resolve_tz(tz_input: str) -> ZoneInfo:
        """Resolve a timezone string to a ZoneInfo object."""
        key = tz_input.lower().strip()

        # Check aliases first
        if key in _TZ_ALIASES:
            return ZoneInfo(_TZ_ALIASES[key])

        # Try as IANA timezone
        try:
            return ZoneInfo(tz_input)
        except (KeyError, Exception):
            pass

        # Fuzzy match against IANA names
        for tz in available_timezones():
            if key in tz.lower():
                return ZoneInfo(tz)

        # Fallback to UTC
        return ZoneInfo("UTC")
