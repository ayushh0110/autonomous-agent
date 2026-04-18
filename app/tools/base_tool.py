"""Base class for all agent tools.

Every tool exposes a schema so the LLM knows what tools are available
and what inputs they expect.  The agent never hardcodes tool knowledge —
it reads these schemas at runtime.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class that every tool must implement.

    Subclasses define *name*, *description*, *input_schema*, and *run*.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short, unique tool identifier (e.g. ``'web_search'``)."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable explanation of what the tool does."""

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """Declare the expected input parameters.

        Returns a dict mapping parameter names to their type descriptors.
        Example::

            {
                "query": {
                    "type": "string",
                    "description": "The search query to execute"
                }
            }
        """

    def schema(self) -> dict[str, Any]:
        """Return the full tool descriptor for LLM prompt injection.

        This is what the LLM sees when deciding which tool to call.
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    @abstractmethod
    def run(self, **kwargs: Any) -> str:
        """Execute the tool with the given keyword arguments.

        Args:
            **kwargs: Named inputs matching the declared input_schema.

        Returns:
            A string containing the tool's output.
        """
