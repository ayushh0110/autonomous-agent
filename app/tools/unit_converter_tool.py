"""Unit converter tool — convert between common units.

Pure math — no API needed. Covers length, weight, temperature,
speed, volume, area, data, and time.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

# ── Conversion tables ──────────────────────────────────────────────────
# All values are relative to a base unit in each category.

_LENGTH = {  # base: meters
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "nm": 1852.0, "nautical mile": 1852.0, "nautical miles": 1852.0,
}

_WEIGHT = {  # base: grams
    "mg": 0.001, "milligram": 0.001, "milligrams": 0.001,
    "g": 1.0, "gram": 1.0, "grams": 1.0,
    "kg": 1000.0, "kilogram": 1000.0, "kilograms": 1000.0,
    "oz": 28.3495, "ounce": 28.3495, "ounces": 28.3495,
    "lb": 453.592, "pound": 453.592, "pounds": 453.592, "lbs": 453.592,
    "ton": 907185.0, "tons": 907185.0, "tonne": 1000000.0, "tonnes": 1000000.0,
}

_VOLUME = {  # base: liters
    "ml": 0.001, "milliliter": 0.001, "milliliters": 0.001,
    "l": 1.0, "liter": 1.0, "liters": 1.0, "litre": 1.0, "litres": 1.0,
    "gal": 3.78541, "gallon": 3.78541, "gallons": 3.78541,
    "qt": 0.946353, "quart": 0.946353, "quarts": 0.946353,
    "pt": 0.473176, "pint": 0.473176, "pints": 0.473176,
    "cup": 0.236588, "cups": 0.236588,
    "fl oz": 0.0295735, "fluid ounce": 0.0295735, "fluid ounces": 0.0295735,
}

_SPEED = {  # base: m/s
    "m/s": 1.0, "mps": 1.0,
    "km/h": 0.277778, "kmh": 0.277778, "kph": 0.277778, "kmph": 0.277778,
    "mph": 0.44704,
    "knot": 0.514444, "knots": 0.514444, "kn": 0.514444,
    "ft/s": 0.3048, "fps": 0.3048,
}

_AREA = {  # base: sq meters
    "sq mm": 1e-6, "mm2": 1e-6,
    "sq cm": 1e-4, "cm2": 1e-4,
    "sq m": 1.0, "m2": 1.0,
    "sq km": 1e6, "km2": 1e6,
    "sq ft": 0.092903, "ft2": 0.092903,
    "sq mi": 2.59e6, "mi2": 2.59e6,
    "acre": 4046.86, "acres": 4046.86,
    "hectare": 10000.0, "hectares": 10000.0, "ha": 10000.0,
}

_DATA = {  # base: bytes
    "b": 1, "byte": 1, "bytes": 1,
    "kb": 1024, "kilobyte": 1024, "kilobytes": 1024,
    "mb": 1048576, "megabyte": 1048576, "megabytes": 1048576,
    "gb": 1073741824, "gigabyte": 1073741824, "gigabytes": 1073741824,
    "tb": 1099511627776, "terabyte": 1099511627776, "terabytes": 1099511627776,
}

_TIME = {  # base: seconds
    "ms": 0.001, "millisecond": 0.001, "milliseconds": 0.001,
    "s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hr": 3600.0, "hour": 3600.0, "hours": 3600.0,
    "d": 86400.0, "day": 86400.0, "days": 86400.0,
    "week": 604800.0, "weeks": 604800.0,
    "month": 2629746.0, "months": 2629746.0,
    "year": 31556952.0, "years": 31556952.0, "yr": 31556952.0,
}

_ALL_CATEGORIES = {
    "length": _LENGTH,
    "weight": _WEIGHT,
    "volume": _VOLUME,
    "speed": _SPEED,
    "area": _AREA,
    "data": _DATA,
    "time": _TIME,
}


class UnitConverterTool(BaseTool):
    """Convert between common units of measurement."""

    @property
    def name(self) -> str:
        return "unit_converter"

    @property
    def description(self) -> str:
        return (
            "Convert between units of measurement. Supports: "
            "length (km, miles, feet, inches, cm, m), "
            "weight (kg, lbs, oz, grams), "
            "temperature (celsius, fahrenheit, kelvin), "
            "volume (liters, gallons, cups), "
            "speed (km/h, mph, m/s, knots), "
            "area (sq m, sq ft, acres, hectares), "
            "data (KB, MB, GB, TB), "
            "time (seconds, minutes, hours, days, years). "
            "Example: value=100, from_unit='fahrenheit', to_unit='celsius'."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "value": {
                "type": "number",
                "description": "The numeric value to convert.",
            },
            "from_unit": {
                "type": "string",
                "description": "The unit to convert from (e.g. 'miles', 'kg', 'fahrenheit').",
            },
            "to_unit": {
                "type": "string",
                "description": "The unit to convert to (e.g. 'km', 'lbs', 'celsius').",
            },
        }

    def run(self, **kwargs: Any) -> str:
        try:
            value = float(kwargs.get("value", 0))
        except (ValueError, TypeError):
            return "Error: 'value' must be a number."

        from_unit = str(kwargs.get("from_unit", "")).strip().lower()
        to_unit = str(kwargs.get("to_unit", "")).strip().lower()

        if not from_unit or not to_unit:
            return "Error: Both 'from_unit' and 'to_unit' are required."

        logger.info("[UNIT-CONVERTER] %s %s → %s", value, from_unit, to_unit)

        # Temperature is special — not ratio-based
        if self._is_temp(from_unit) and self._is_temp(to_unit):
            return self._convert_temp(value, from_unit, to_unit)

        # Find category
        for cat_name, table in _ALL_CATEGORIES.items():
            if from_unit in table and to_unit in table:
                base_value = value * table[from_unit]
                result = base_value / table[to_unit]
                formatted = f"{result:.6g}"
                return f"{value} {from_unit} = {formatted} {to_unit}"

        return (
            f"Cannot convert '{from_unit}' to '{to_unit}'. "
            f"Make sure both units are in the same category "
            f"(length, weight, volume, speed, area, data, time, temperature)."
        )

    @staticmethod
    def _is_temp(unit: str) -> bool:
        return unit in (
            "c", "celsius", "f", "fahrenheit", "k", "kelvin",
        )

    @staticmethod
    def _convert_temp(value: float, from_u: str, to_u: str) -> str:
        # Normalize
        f = from_u[0]  # c, f, or k
        t = to_u[0]

        if f == t:
            return f"{value} {from_u} = {value} {to_u}"

        # Convert to Celsius first
        if f == "f":
            c = (value - 32) * 5 / 9
        elif f == "k":
            c = value - 273.15
        else:
            c = value

        # Convert from Celsius to target
        if t == "f":
            result = c * 9 / 5 + 32
        elif t == "k":
            result = c + 273.15
        else:
            result = c

        return f"{value} {from_u} = {result:.4g} {to_u}"
