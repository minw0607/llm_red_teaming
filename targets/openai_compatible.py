"""
OpenAICompatibleTarget — Generic connector for any OpenAI-compatible API.

Supports every provider that implements the OpenAI chat-completions spec:
    - Azure OpenAI  (Option A in .env.example)
    - OpenAI direct (Option B)
    - Ollama        (Option C) — local, no API key required
    - Groq          (Option D)
    - Together AI   (Option E)
    - LM Studio     (Option F) — local GUI

All configuration is read from environment variables (via .env).
The minimal required variable is OPENAI_API_KEY; set OPENAI_BASE_URL
to point at any non-default endpoint.

Usage
-----
    from targets.openai_compatible import OpenAICompatibleTarget

    target   = OpenAICompatibleTarget()
    response = target.complete("What is the capital of France?")

    # Override model per-call
    response = target.complete("...", model="gpt-4-turbo")
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI

load_dotenv()


def _is_azure(base_url: str | None) -> bool:
    """Heuristic: Azure endpoints contain 'openai.azure.com' or 'azure.com'."""
    if not base_url:
        return False
    return "openai.azure.com" in base_url or "azure.com" in base_url


class OpenAICompatibleTarget:
    """
    Provider-agnostic target for any OpenAI-compatible chat-completions API.

    Reads all configuration from environment variables. Override any value
    by passing it as a constructor argument.

    Parameters
    ----------
    api_key : str | None
        API key. Reads ``OPENAI_API_KEY`` from env if not provided.
    base_url : str | None
        Base URL. Reads ``OPENAI_BASE_URL`` from env.
        Leave blank for the default OpenAI endpoint.
    api_version : str | None
        Azure-only API version. Reads ``OPENAI_API_VERSION`` from env.
    model : str | None
        Model / deployment name. Reads ``TARGET_MODEL`` from env,
        defaults to ``"gpt-4o"``.
    temperature : float
        Sampling temperature (default 0.0 — deterministic).
    max_tokens : int
        Maximum tokens in the response (default 512).
    extra_headers : dict | None
        Additional HTTP headers, e.g. Azure APIM subscription keys.
        Automatically populated from ``AZURE_APIM_HEADER_NAME`` /
        ``AZURE_APIM_SUBSCRIPTION_KEY`` env vars if present.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        extra_headers: dict | None = None,
    ):
        self.api_key     = api_key  or os.getenv("OPENAI_API_KEY", "")
        self.base_url    = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.api_version = api_version or os.getenv("OPENAI_API_VERSION", "2024-02-15-preview")
        self.model       = model or os.getenv("TARGET_MODEL", "gpt-4o")
        self.temperature = temperature
        self.max_tokens  = max_tokens

        # Build extra headers (APIM gateway support)
        self.extra_headers = extra_headers or {}
        apim_header = os.getenv("AZURE_APIM_HEADER_NAME")
        apim_key    = os.getenv("AZURE_APIM_SUBSCRIPTION_KEY")
        if apim_header and apim_key:
            self.extra_headers[apim_header] = apim_key

        # Instantiate the right client
        if _is_azure(self.base_url):
            self._client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.base_url,
                api_version=self.api_version,
                default_headers=self.extra_headers or None,
            )
            self._provider = "azure"
        else:
            client_kwargs: dict = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            if self.extra_headers:
                client_kwargs["default_headers"] = self.extra_headers
            self._client = OpenAI(**client_kwargs)
            self._provider = "openai"

    # ── Core interface ────────────────────────────────────────────────────────

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: str | None = None,
    ) -> str:
        """
        Send a chat completion request and return the response text.

        Parameters
        ----------
        user_prompt : str
            The user message / adversarial input.
        system_prompt : str
            System message prepended to the conversation.
        model : str | None
            Override the instance-level model for this call only.

        Returns
        -------
        str
            The model's response, stripped of leading/trailing whitespace.
        """
        response = self._client.chat.completions.create(
            model=model or self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def get_sentiment(self, text: str) -> str:
        """
        Binary sentiment classification helper.

        Returns ``"positive"`` or ``"negative"`` (lowercased).
        """
        prompt = (
            "Determine the sentiment of the following sentence. "
            "Reply with only 'positive' or 'negative'.\n\n"
            f"{text}"
        )
        return self.complete(user_prompt=prompt).lower()

    # ── Dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        endpoint = self.base_url or "https://api.openai.com"
        return (
            f"OpenAICompatibleTarget("
            f"provider={self._provider!r}, "
            f"model={self.model!r}, "
            f"endpoint={endpoint!r})"
        )
