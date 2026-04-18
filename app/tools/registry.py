"""Tool registry — central store for tool lookup, validation, and execution.

The agent never interacts with tools directly.  It talks to this registry,
which validates inputs against schemas before executing anything.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from app.tools.base_tool import BaseTool

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Stores tools and provides schema introspection + safe execution."""

    def __init__(self, tools: Sequence[BaseTool] | None = None) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: BaseTool) -> None:
        """Register a tool.  Raises if a tool with the same name exists."""
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s", tool.name)

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return schemas for all registered tools (for prompt injection)."""
        return [tool.schema() for tool in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Validate inputs and execute a tool by name.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Dict of keyword arguments for the tool.

        Returns:
            The tool's string output, or a structured error message.
        """
        # ── Validate tool exists ──
        tool = self._tools.get(tool_name)
        if tool is None:
            available = ", ".join(self._tools.keys()) or "(none)"
            msg = f"Unknown tool '{tool_name}'. Available tools: {available}"
            logger.error(msg)
            return f"Error: {msg}"

        # ── Validate required inputs ──
        schema = tool.input_schema
        missing = [key for key in schema if key not in tool_input]
        if missing:
            msg = f"Missing required input(s) for '{tool_name}': {missing}"
            logger.error(msg)
            return f"Error: {msg}"

        # ── Type validation (lightweight) ──
        for key, spec in schema.items():
            expected_type = spec.get("type", "string")
            value = tool_input.get(key)
            if value is not None and not self._check_type(value, expected_type):
                msg = (
                    f"Invalid type for '{tool_name}.{key}': "
                    f"expected {expected_type}, got {type(value).__name__}"
                )
                logger.error(msg)
                return f"Error: {msg}"

        # ── Execute ──
        logger.info("Executing tool '%s' with input: %s", tool_name, tool_input)
        try:
            result = tool.run(**tool_input)
            logger.info(
                "Tool '%s' returned %d chars",
                tool_name,
                len(result) if result else 0,
            )
            return result
        except Exception as exc:
            msg = f"Tool '{tool_name}' failed: {exc}"
            logger.error(msg, exc_info=True)
            return f"Error: {msg}"

    # ── Helpers ──

    @staticmethod
    def _check_type(value: Any, expected: str) -> bool:
        """Lightweight type check based on schema type strings."""
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
        }
        expected_cls = type_map.get(expected)
        if expected_cls is None:
            return True  # Unknown type → skip validation
        return isinstance(value, expected_cls)
