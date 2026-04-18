"""Application configuration — loads secrets and settings from environment."""

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the app directory (or project root)
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_env_path)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    """Immutable application settings sourced from environment variables."""

    groq_api_key: str = field(repr=False, default="")
    groq_model: str = "llama-3.1-8b-instant"
    groq_api_url: str = "https://api.groq.com/openai/v1/chat/completions"
    max_agent_steps: int = 4
    max_tool_calls: int = 2
    max_plan_steps: int = 5
    max_execution_steps: int = 10
    max_refinements: int = 2

    def validate(self) -> None:
        """Raise if required configuration is missing."""
        if not self.groq_api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )


def get_settings() -> Settings:
    """Build and validate a Settings instance from the current environment."""
    settings = Settings(
        groq_api_key=os.getenv("GROQ_API_KEY", ""),
        max_agent_steps=int(os.getenv("MAX_AGENT_STEPS", "4")),
        max_tool_calls=int(os.getenv("MAX_TOOL_CALLS", "2")),
        max_plan_steps=int(os.getenv("MAX_PLAN_STEPS", "5")),
        max_execution_steps=int(os.getenv("MAX_EXECUTION_STEPS", "10")),
        max_refinements=int(os.getenv("MAX_REFINEMENTS", "2")),
    )
    settings.validate()
    logger.info("Settings loaded successfully (model=%s)", settings.groq_model)
    return settings
