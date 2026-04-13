"""Backend router for jie-code.

Auto-detects and creates the appropriate LLM client based on model name,
base URL, or explicit backend selection.
"""
from __future__ import annotations

from typing import Any, Iterator, Protocol

from .agent_types import (
    AssistantTurn,
    ModelConfig,
    OutputSchemaConfig,
    StreamEvent,
)
from .anthropic_client import AnthropicClient
from .openai_compat import OpenAICompatClient


class LLMClient(Protocol):
    """Unified interface for LLM clients."""

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ) -> AssistantTurn: ...

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        output_schema: OutputSchemaConfig | None = None,
    ) -> Iterator[StreamEvent]: ...


# Known Anthropic relay/proxy endpoints
ANTHROPIC_HOSTS = frozenset({
    'api.anthropic.com',
    '10.151.179.151',  # claude-relay-service
})

# Model name prefixes that indicate Anthropic backend
ANTHROPIC_MODEL_PREFIXES = (
    'claude-',
    'claude_',
    'anthropic/',
)


def detect_backend(config: ModelConfig, *, explicit: str = 'auto') -> str:
    """Detect which backend to use.

    Returns 'anthropic' or 'openai'.
    """
    if explicit != 'auto':
        return explicit

    # Check model name
    model_lower = config.model.lower()
    for prefix in ANTHROPIC_MODEL_PREFIXES:
        if model_lower.startswith(prefix):
            return 'anthropic'

    # Check base URL
    base_url_lower = config.base_url.lower()
    for host in ANTHROPIC_HOSTS:
        if host in base_url_lower:
            return 'anthropic'

    return 'openai'


def create_client(config: ModelConfig, *, backend: str = 'auto') -> LLMClient:
    """Create an LLM client based on backend detection.

    Args:
        config: Model configuration (model, base_url, api_key, etc.)
        backend: 'auto', 'anthropic', or 'openai'

    Returns:
        An LLMClient (either AnthropicClient or OpenAICompatClient)
    """
    detected = detect_backend(config, explicit=backend)
    if detected == 'anthropic':
        return AnthropicClient(config)
    return OpenAICompatClient(config)
