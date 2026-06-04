"""
AzureOpenAITarget — Model connector for Azure OpenAI deployments.

Wraps the Azure OpenAI Python SDK into the standard target interface used
by all runners in this repo:  ``complete(user_prompt, system_prompt) -> str``

Credentials are read from environment variables (via .env):
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_API_VERSION
    AZURE_OPENAI_MODEL

Usage
-----
    from targets.azure_openai import AzureOpenAITarget

    target   = AzureOpenAITarget()
    response = target.complete("What is the capital of France?")
    print(response)  # "Paris"
"""

from __future__ import annotations

import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()


class AzureOpenAITarget:
    """
    Target connector for Azure-hosted OpenAI models (GPT-4o, GPT-4, etc.).

    Parameters
    ----------
    api_key : str | None
        Azure OpenAI API key.  Falls back to ``AZURE_OPENAI_API_KEY`` env var.
    endpoint : str | None
        Azure endpoint URL.  Falls back to ``AZURE_OPENAI_ENDPOINT``.
    api_version : str | None
        API version string.  Falls back to ``AZURE_OPENAI_API_VERSION``.
    model : str | None
        Deployment / model name.  Falls back to ``AZURE_OPENAI_MODEL``.
    temperature : float
        Sampling temperature (default 0.0 for deterministic output).
    max_tokens : int
        Maximum response length in tokens (default 512).
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
        self.api_key = api_key or os.environ["AZURE_OPENAI_API_KEY"]
        self.endpoint = endpoint or os.environ["AZURE_OPENAI_ENDPOINT"]
        self.api_version = api_version or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
        )
        self.model = model or os.getenv("AZURE_OPENAI_MODEL", "gpt-4o")
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._client = AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version,
        )

    def complete(
        self,
        user_prompt: str,
        system_prompt: str = "You are a helpful assistant.",
    ) -> str:
        """
        Send a chat completion request and return the response text.

        Parameters
        ----------
        user_prompt : str
            The user message / adversarial input.
        system_prompt : str
            The system message (default: generic helpful assistant).

        Returns
        -------
        str
            The model's response text, stripped of leading/trailing whitespace.
        """
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def get_sentiment(self, text: str) -> str:
        """
        Convenience wrapper: binary sentiment classification.

        Returns ``"positive"`` or ``"negative"`` (lowercased).
        """
        prompt = (
            "Determine the sentiment of the following sentence. "
            "Reply with only 'positive' or 'negative'.\n\n"
            f"{text}"
        )
        return self.complete(user_prompt=prompt).lower()

    def __repr__(self) -> str:
        return (
            f"AzureOpenAITarget(model={self.model!r}, "
            f"endpoint={self.endpoint!r})"
        )
