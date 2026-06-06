"""
AzureOpenAITarget — Legacy Azure OpenAI connector (backward-compatible).

Kept so existing code that imports AzureOpenAITarget continues to work.
Internally delegates to OpenAICompatibleTarget.

Env vars (new unified names — preferred):
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_API_VERSION, TARGET_MODEL

Legacy env vars (still supported as fallbacks):
    AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION, AZURE_OPENAI_MODEL

For new code, prefer importing OpenAICompatibleTarget directly:
    from targets.openai_compatible import OpenAICompatibleTarget
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from .openai_compatible import OpenAICompatibleTarget

load_dotenv()


class AzureOpenAITarget(OpenAICompatibleTarget):
    """
    Azure OpenAI connector — thin wrapper around OpenAICompatibleTarget.

    Reads credentials with a unified → legacy env var fallback chain:
        OPENAI_API_KEY  →  AZURE_OPENAI_API_KEY
        OPENAI_BASE_URL →  AZURE_OPENAI_ENDPOINT
        TARGET_MODEL    →  AZURE_OPENAI_MODEL

    All other behaviour (complete, get_sentiment, extra_headers) is
    inherited from OpenAICompatibleTarget.
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        api_version: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        resolved_key      = api_key   or os.getenv("OPENAI_API_KEY")      or os.getenv("AZURE_OPENAI_API_KEY", "")
        resolved_endpoint = endpoint  or os.getenv("OPENAI_BASE_URL")     or os.getenv("AZURE_OPENAI_ENDPOINT")
        resolved_version  = api_version or os.getenv("OPENAI_API_VERSION") or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
        resolved_model    = model     or os.getenv("TARGET_MODEL")         or os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")

        super().__init__(
            api_key=resolved_key,
            base_url=resolved_endpoint,
            api_version=resolved_version,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
