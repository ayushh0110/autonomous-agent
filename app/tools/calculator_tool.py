"""Calculator tool — safe math evaluation for the agent.

Handles arithmetic, roots, powers, trig, logarithms, etc.
Uses Python's math module in a sandboxed eval — no arbitrary code execution.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from app.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)

# Safe functions and constants the calculator can use
_SAFE_MATH = {
    # Basic
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    # Powers & roots
    "sqrt": math.sqrt,
    "cbrt": lambda x: x ** (1 / 3),
    "pow": math.pow,
    # Trig
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "radians": math.radians,
    "degrees": math.degrees,
    # Logarithms
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "ln": math.log,
    # Misc
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    # Constants
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "tau": math.tau,
}


class CalculatorTool(BaseTool):
    """Evaluate mathematical expressions safely.

    Supports arithmetic, roots, powers, trigonometry, logarithms,
    factorials, and common constants (pi, e).
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression. Supports: "
            "arithmetic (+, -, *, /, //, %, **), "
            "sqrt(), cbrt(), pow(), abs(), round(), "
            "sin(), cos(), tan(), log(), log2(), log10(), "
            "ceil(), floor(), factorial(), gcd(), "
            "constants: pi, e, tau. "
            "Example inputs: '2 + 2', 'sqrt(44567)', '2**10', "
            "'sin(radians(45))', 'factorial(10)', 'log2(1024)'."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "expression": {
                "type": "string",
                "description": (
                    "The math expression to evaluate. "
                    "Use Python syntax: ** for power, sqrt() for square root, "
                    "factorial() for factorial, etc."
                ),
            }
        }

    def run(self, **kwargs: Any) -> str:
        expression = kwargs.get("expression", "").strip()
        if not expression:
            return "Error: No expression provided."

        logger.info("[CALCULATOR] Evaluating: %s", expression)

        try:
            # Normalize common patterns
            expr = self._normalize(expression)

            # Safety check — reject anything that's not math
            self._validate(expr)

            # Evaluate in sandboxed environment
            result = eval(expr, {"__builtins__": {}}, _SAFE_MATH)  # noqa: S307

            # Format the result nicely
            if isinstance(result, float):
                # Avoid ugly floating point like 4.000000000000001
                if result == int(result) and abs(result) < 1e15:
                    formatted = str(int(result))
                else:
                    formatted = f"{result:.10g}"
            else:
                formatted = str(result)

            logger.info("[CALCULATOR] Result: %s = %s", expression, formatted)
            return f"{expression} = {formatted}"

        except ZeroDivisionError:
            return f"Error: Division by zero in '{expression}'."
        except ValueError as exc:
            return f"Error: Math domain error — {exc}"
        except (SyntaxError, TypeError, NameError) as exc:
            return f"Error: Invalid expression '{expression}' — {exc}"
        except Exception as exc:
            logger.warning("[CALCULATOR] Unexpected error: %s", exc)
            return f"Error: Could not evaluate '{expression}' — {exc}"

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalize(expr: str) -> str:
        """Convert natural-language math patterns to Python syntax."""
        # "square root of X" → "sqrt(X)"
        expr = re.sub(r"square\s*root\s*(?:of\s*)?", "sqrt", expr, flags=re.I)
        # "X squared" → "X**2"
        expr = re.sub(r"(\d+)\s*squared", r"\1**2", expr, flags=re.I)
        # "X cubed" → "X**3"
        expr = re.sub(r"(\d+)\s*cubed", r"\1**3", expr, flags=re.I)
        # "X to the power of Y" → "X**Y"
        expr = re.sub(
            r"(\d+)\s*to\s*the\s*power\s*(?:of\s*)?(\d+)",
            r"\1**\2", expr, flags=re.I,
        )
        # "X!" → "factorial(X)"
        expr = re.sub(r"(\d+)!", r"factorial(\1)", expr)
        # "^" → "**" (caret to Python power)
        expr = expr.replace("^", "**")
        # "×" → "*", "÷" → "/"
        expr = expr.replace("×", "*").replace("÷", "/")
        return expr

    @staticmethod
    def _validate(expr: str) -> None:
        """Reject expressions that could be malicious code."""
        # Only allow: digits, operators, parens, dots, commas, spaces, function names
        allowed = re.compile(
            r'^[0-9a-zA-Z_\s\+\-\*/\.\,\(\)\%]+$'
        )
        if not allowed.match(expr):
            raise ValueError(f"Expression contains disallowed characters: {expr}")

        # Block dangerous patterns
        blocked = ["import", "exec", "eval", "open", "os.", "sys.", "__",
                    "lambda", "class", "def", "print", "input"]
        expr_lower = expr.lower()
        for word in blocked:
            if word in expr_lower:
                raise ValueError(f"Expression contains blocked keyword: {word}")
