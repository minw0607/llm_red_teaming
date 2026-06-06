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
import time
import logging
from dotenv import load_dotenv
from openai import OpenAI, AzureOpenAI, APIStatusError, RateLimitError

load_dotenv()

logger = logging.getLogger(__name__)


def _is_azure(base_url: str | None, api_version: str | None = None) -> bool:
    """
    Decide whether to use the AzureOpenAI client.

    Three signals, any one is sufficient:
    1. Explicit Azure domain  — ``*.openai.azure.com``
    2. ``azure.com`` anywhere in the URL
    3. Custom endpoint (not api.openai.com) AND api_version is set.
       This catches Azure APIM gateways at arbitrary domains
       (e.g. atlas.protiviti.com) which still require the Azure client's
       URL pattern: /openai/deployments/{model}/chat/completions?api-version=…

    Standard OpenAI / Groq / Together / Ollama never need api_version,
    so signal 3 is a safe discriminator.
    """
    if not base_url:
        return False
    if "openai.azure.com" in base_url:
        return True
    if "azure.com" in base_url:
        return True
    # APIM or other Azure proxy: custom domain + api_version present
    is_custom_endpoint = "api.openai.com" not in base_url
    if is_custom_endpoint and api_version:
        return True
    return False


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
    max_retries : int
        Number of retry attempts on transient errors (429, 500, 503).
        Uses exponential backoff: 2, 4, 8 … seconds. Default 3.
    """

    # HTTP status codes treated as transient / retriable
    _RETRIABLE_STATUSES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        extra_headers: dict | None = None,
        max_retries: int = 3,
    ):
        self.api_key     = api_key  or os.getenv("OPENAI_API_KEY", "")
        self.base_url    = base_url or os.getenv("OPENAI_BASE_URL") or None
        self.api_version = api_version or os.getenv("OPENAI_API_VERSION", "2024-02-15-preview")
        self.model       = model or os.getenv("TARGET_MODEL", "gpt-4o")
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.max_retries = max_retries

        # Build extra headers (APIM gateway support)
        self.extra_headers = extra_headers or {}
        apim_header = os.getenv("AZURE_APIM_HEADER_NAME")
        apim_key    = os.getenv("AZURE_APIM_SUBSCRIPTION_KEY")
        if apim_header and apim_key:
            self.extra_headers[apim_header] = apim_key

        # Instantiate the right client
        if _is_azure(self.base_url, self.api_version):
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

        Automatically retries on transient errors (HTTP 429 / 500 / 502 /
        503 / 504) with exponential backoff (2 s, 4 s, 8 s, …).

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

        Raises
        ------
        openai.APIStatusError
            Re-raised after all retry attempts are exhausted, or immediately
            for non-retriable errors (e.g. 401 Unauthorized, 404 Not Found).
        """
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            try:
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

            except APIStatusError as exc:
                status = exc.status_code
                if status not in self._RETRIABLE_STATUSES:
                    raise  # non-retriable (401, 404, etc.) — fail immediately

                last_exc = exc
                wait = 2 ** (attempt + 1)          # 2 s, 4 s, 8 s …
                logger.warning(
                    "HTTP %d on attempt %d/%d — retrying in %ds  [%s]",
                    status, attempt + 1, self.max_retries, wait,
                    getattr(exc, "message", str(exc))[:120],
                )
                print(
                    f"  ⚠️  HTTP {status} (attempt {attempt+1}/{self.max_retries})"
                    f" — retrying in {wait}s…"
                )
                time.sleep(wait)

            except Exception as exc:        # network timeout, SSL, etc.
                last_exc = exc
                wait = 2 ** (attempt + 1)
                logger.warning("Unexpected error on attempt %d: %s", attempt + 1, exc)
                print(f"  ⚠️  {type(exc).__name__} (attempt {attempt+1}/{self.max_retries}) — retrying in {wait}s…")
                time.sleep(wait)

        raise RuntimeError(
            f"All {self.max_retries} retry attempts failed."
        ) from last_exc

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
        # Distinguish native Azure from APIM-gateway Azure
        if self._provider == "azure" and "azure.com" not in (self.base_url or ""):
            provider_label = "azure_apim"
        else:
            provider_label = self._provider
        return (
            f"OpenAICompatibleTarget("
            f"provider={provider_label!r}, "
            f"model={self.model!r}, "
            f"endpoint={endpoint!r})"
        )
