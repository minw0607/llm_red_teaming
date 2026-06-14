"""
targets/application.py — Target a deployed APPLICATION, not the bare model.

The other targets hit a raw model endpoint, so the red-team runners supply their
own (minimal) system prompt — a **model-level** test. A real application wraps the
model with its own system prompt, input/output guardrails, retrieval, and auth.
``ApplicationTarget`` points at the *application's* API and sends **user input
only**, so the app applies its own server-side configuration and guardrails — an
**application-level** test.

Configuration (env, falling back to the standard OPENAI_* vars):
    APP_BASE_URL   — the application's chat-completions-compatible endpoint
    APP_API_KEY    — its key
    APP_MODEL      — its model/deployment name

Usage
-----
    from targets import OpenAICompatibleTarget, ApplicationTarget

    model = OpenAICompatibleTarget()   # bare model (model-level)
    app   = ApplicationTarget()        # deployed app w/ guardrails (app-level)

    # Run the SAME probes against both, then measure what the guardrails caught:
    from evaluate import guardrail_comparison

Scope caveat — runners that *plant a secret in the system prompt* (NB06 Track A,
system-prompt disclosure) are model-level by construction: against a real app you
instead attack the app's own system prompt, so that track's canary metric does
not transfer. Injection, jailbreak, memorization, exfiltration, NLI and agent
probes all deliver via user input and transfer directly.
"""

from __future__ import annotations

import os

from .openai_compatible import OpenAICompatibleTarget


class ApplicationTarget(OpenAICompatibleTarget):
    """A target whose system prompt + guardrails are owned by the application."""

    def __init__(self, *, app_system_prompt: str | None = None, **kwargs):
        # Prefer APP_* env, else fall back to the standard OPENAI_* config.
        kwargs.setdefault("api_key", os.getenv("APP_API_KEY") or os.getenv("OPENAI_API_KEY", ""))
        kwargs.setdefault("base_url", os.getenv("APP_BASE_URL") or os.getenv("OPENAI_BASE_URL") or None)
        kwargs.setdefault("model", os.getenv("APP_MODEL") or os.getenv("TARGET_MODEL", "gpt-4o"))
        super().__init__(**kwargs)
        # What system message (if any) to send. Default '' so the runner's injected
        # system prompt is dropped and the application's own config governs behaviour.
        self.app_system_prompt = app_system_prompt if app_system_prompt is not None \
            else os.getenv("APP_SYSTEM_PROMPT", "")
        self.provider = "application"

    def complete(self, user_prompt: str, system_prompt: str | None = None,
                 model: str | None = None, max_completion_tokens: int | None = None) -> str:
        """Send USER input only — the runner-supplied ``system_prompt`` is ignored
        so the application applies its own system prompt and guardrails."""
        return super().complete(
            user_prompt=user_prompt,
            system_prompt=self.app_system_prompt,
            model=model,
            max_completion_tokens=max_completion_tokens,
        )

    def __repr__(self) -> str:
        return f"ApplicationTarget(model={self.model!r}, app_owns_system_prompt=True)"
