"""
targets/
Model connectors — all implement the standard interface:
    complete(user_prompt, system_prompt) -> str

Classes
-------
OpenAICompatibleTarget
    Generic connector for any OpenAI-compatible API:
    Azure OpenAI, OpenAI direct, Ollama, Groq, Together AI, LM Studio.
    Reads OPENAI_API_KEY / OPENAI_BASE_URL / TARGET_MODEL from .env.
    Recommended for all new usage.

AzureOpenAITarget
    Legacy connector — Azure OpenAI only.
    Kept for backward compatibility; wraps OpenAICompatibleTarget internally.
"""

from .openai_compatible import OpenAICompatibleTarget
from .azure_openai import AzureOpenAITarget
from .application import ApplicationTarget

__all__ = ["OpenAICompatibleTarget", "AzureOpenAITarget", "ApplicationTarget"]
