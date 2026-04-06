"""Google-specific provider adapter using OpenAI-compatible interface.

Shares the core mechanics of OpenAI Chat Completions but with Google-specific
defaults (large context window, vision support).  Uses the Gemini
``/v1beta/openai/`` endpoint which is wire-compatible with the OpenAI SDK.
"""

from __future__ import annotations

from turnstone.core.providers._openai_chat import OpenAIChatCompletionsProvider
from turnstone.core.providers._protocol import ModelCapabilities

# Dynamic Google capability baseline (assume large context, no temp limits)
_GOOGLE_DEFAULT = ModelCapabilities(
    context_window=2_000_000,
    max_output_tokens=8192,
    supports_temperature=True,
    supports_vision=True,
)


class GoogleProvider(OpenAIChatCompletionsProvider):
    """Provider for Google models using the OpenAI-compatible endpoint."""

    @property
    def provider_name(self) -> str:
        return "google"

    def get_capabilities(self, model: str) -> ModelCapabilities:
        # Since Google models update frequently, returning a robust default
        # safely handles dynamic versions.
        return _GOOGLE_DEFAULT
