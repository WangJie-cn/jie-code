"""Model profiles for jie-code.

Predefined configurations for common LLM backends.
"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from .agent_types import ModelConfig, ModelPricing


# Built-in profiles
BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    'anthropic-proxy': {
        'model': 'claude-sonnet-4-6',
        'base_url': 'http://10.151.179.151:4500',
        'api_key_env': 'ANTHROPIC_API_KEY',
        'api_key_default': 'cr_default',
        'temperature': 0.0,
        'timeout_seconds': 300.0,
        'pricing': {
            'input': 3.0,
            'output': 15.0,
            'cache_creation': 3.75,
            'cache_read': 0.30,
        },
        'backend': 'anthropic',
    },
    'anthropic-opus': {
        'model': 'claude-opus-4-6',
        'base_url': 'http://10.151.179.151:4500',
        'api_key_env': 'ANTHROPIC_API_KEY',
        'api_key_default': 'cr_default',
        'temperature': 0.0,
        'timeout_seconds': 600.0,
        'pricing': {
            'input': 15.0,
            'output': 75.0,
            'cache_creation': 18.75,
            'cache_read': 1.50,
        },
        'backend': 'anthropic',
    },
    'local-qwen': {
        'model': 'Qwen/Qwen3-Coder-30B-A3B-Instruct',
        'base_url': 'http://127.0.0.1:8000/v1',
        'api_key': 'local-token',
        'temperature': 0.0,
        'timeout_seconds': 120.0,
        'pricing': {
            'input': 0.0,
            'output': 0.0,
        },
        'backend': 'openai',
    },
    'ollama': {
        'model': 'qwen3:30b-a3b',
        'base_url': 'http://127.0.0.1:11434/v1',
        'api_key': 'ollama',
        'temperature': 0.0,
        'timeout_seconds': 120.0,
        'pricing': {
            'input': 0.0,
            'output': 0.0,
        },
        'backend': 'openai',
    },
    'openrouter': {
        'model': 'anthropic/claude-sonnet-4',
        'base_url': 'https://openrouter.ai/api/v1',
        'api_key_env': 'OPENROUTER_API_KEY',
        'temperature': 0.0,
        'timeout_seconds': 300.0,
        'pricing': {
            'input': 3.0,
            'output': 15.0,
        },
        'backend': 'openai',
    },
}


def _get_user_profiles_path() -> Path:
    """Get path to user's custom profiles file."""
    config_home = os.environ.get('JIE_CONFIG_HOME', '')
    if config_home:
        return Path(config_home) / 'profiles.json'
    return Path.home() / '.jie' / 'profiles.json'


def _load_user_profiles() -> dict[str, dict[str, Any]]:
    """Load user-defined profiles from ~/.jie/profiles.json."""
    path = _get_user_profiles_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def list_profiles() -> dict[str, dict[str, Any]]:
    """List all available profiles (builtin + user)."""
    profiles = dict(BUILTIN_PROFILES)
    profiles.update(_load_user_profiles())
    return profiles


def resolve_profile(name: str) -> tuple[ModelConfig, str]:
    """Resolve a profile name to a ModelConfig and backend hint.

    Returns (ModelConfig, backend_hint).
    """
    profiles = list_profiles()
    if name not in profiles:
        available = ', '.join(sorted(profiles.keys()))
        raise KeyError(f"Unknown profile '{name}'. Available: {available}")

    p = profiles[name]

    # Resolve API key
    api_key = p.get('api_key', '')
    if not api_key:
        env_var = p.get('api_key_env', '')
        if env_var:
            api_key = os.environ.get(env_var, p.get('api_key_default', ''))
        if not api_key:
            api_key = 'local-token'

    pricing_data = p.get('pricing', {})
    pricing = ModelPricing(
        input_cost_per_million_tokens_usd=pricing_data.get('input', 0.0),
        output_cost_per_million_tokens_usd=pricing_data.get('output', 0.0),
        cache_creation_input_cost_per_million_tokens_usd=pricing_data.get('cache_creation', 0.0),
        cache_read_input_cost_per_million_tokens_usd=pricing_data.get('cache_read', 0.0),
    )

    config = ModelConfig(
        model=p.get('model', 'claude-sonnet-4-6'),
        base_url=p.get('base_url', 'http://127.0.0.1:8000/v1'),
        api_key=api_key,
        temperature=p.get('temperature', 0.0),
        timeout_seconds=p.get('timeout_seconds', 120.0),
        pricing=pricing,
    )

    backend = p.get('backend', 'auto')
    return config, backend
