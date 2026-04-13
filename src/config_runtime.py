from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_DIR = Path('.claude')
JIE_CONFIG_DIR = Path('.jie')
PROJECT_SETTINGS_PATH = DEFAULT_CONFIG_DIR / 'settings.json'
LOCAL_SETTINGS_PATH = DEFAULT_CONFIG_DIR / 'settings.local.json'
JIE_SETTINGS_PATH = JIE_CONFIG_DIR / 'settings.json'
JIE_LOCAL_SETTINGS_PATH = JIE_CONFIG_DIR / 'settings.local.json'
LEGACY_CONFIG_PATHS = (
    Path('.claw-config.json'),
    Path('.codex-config.json'),
)
# Codex YAML config (Phase 5)
CODEX_YAML_PATHS = (
    Path('.codex') / 'config.yaml',
    Path('.codex') / 'config.yml',
)


@dataclass(frozen=True)
class ConfigSource:
    name: str
    path: str
    settings: dict[str, Any]


@dataclass(frozen=True)
class ConfigMutation:
    source_name: str
    key_path: str
    store_path: str
    before_sha256: str | None
    after_sha256: str
    before_preview: str | None
    after_preview: str
    effective_key_count: int


@dataclass
class ConfigRuntime:
    cwd: Path
    sources: tuple[ConfigSource, ...] = field(default_factory=tuple)

    @classmethod
    def from_workspace(cls, cwd: Path) -> 'ConfigRuntime':
        root = cwd.resolve()
        sources: list[ConfigSource] = []
        for source_name, path in _discover_source_paths(root):
            payload = _load_json_object(path)
            if payload is None:
                continue
            sources.append(
                ConfigSource(
                    name=source_name,
                    path=str(path),
                    settings=payload,
                )
            )
        return cls(cwd=root, sources=tuple(sources))

    def has_config(self) -> bool:
        return bool(self.sources)

    def effective_settings(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for source in self.sources:
            merged = _deep_merge(merged, source.settings)
        return merged

    def list_keys(
        self,
        *,
        source: str | None = None,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> tuple[str, ...]:
        payload = self._payload_for_source(source)
        flattened = sorted(_flatten_keys(payload))
        if prefix:
            flattened = [key for key in flattened if key.startswith(prefix)]
        if limit is not None and limit >= 0:
            flattened = flattened[:limit]
        return tuple(flattened)

    def get_value(
        self,
        key_path: str,
        *,
        source: str | None = None,
    ) -> Any:
        payload = self._payload_for_source(source)
        return _get_nested_value(payload, key_path)

    def set_value(
        self,
        key_path: str,
        value: Any,
        *,
        source: str = 'local',
    ) -> ConfigMutation:
        resolved_source, path = self._resolve_writable_source(source)
        before_payload = _load_json_object(path) or {}
        before_text = path.read_text(encoding='utf-8') if path.exists() else None
        updated_payload = json.loads(json.dumps(before_payload))
        _set_nested_value(updated_payload, key_path, value)
        after_text = json.dumps(updated_payload, ensure_ascii=True, indent=2, sort_keys=True) + '\n'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(after_text, encoding='utf-8')
        refreshed = ConfigRuntime.from_workspace(self.cwd)
        self.sources = refreshed.sources
        return ConfigMutation(
            source_name=resolved_source,
            key_path=key_path,
            store_path=str(path),
            before_sha256=_sha256_or_none(before_text),
            after_sha256=_sha256(after_text),
            before_preview=_preview(before_text),
            after_preview=_preview(after_text),
            effective_key_count=len(_flatten_keys(self.effective_settings())),
        )

    def render_summary(self) -> str:
        lines = [
            f'Config sources: {len(self.sources)}',
            f'Effective keys: {len(_flatten_keys(self.effective_settings()))}',
        ]
        if not self.sources:
            lines.append('- No local config files discovered.')
            lines.append(f'- Project settings path: {(self.cwd / PROJECT_SETTINGS_PATH).resolve()}')
            lines.append(f'- Local settings path: {(self.cwd / LOCAL_SETTINGS_PATH).resolve()}')
            return '\n'.join(lines)
        for source in self.sources:
            lines.append(
                f'- {source.name}: {source.path} ({len(_flatten_keys(source.settings))} key(s))'
            )
        return '\n'.join(lines)

    def render_keys(
        self,
        *,
        source: str | None = None,
        prefix: str | None = None,
        limit: int | None = None,
    ) -> str:
        keys = self.list_keys(source=source, prefix=prefix, limit=limit)
        if not keys:
            return '(no config keys)'
        return '\n'.join(keys)

    def render_value(
        self,
        key_path: str,
        *,
        source: str | None = None,
    ) -> str:
        return json.dumps(
            self.get_value(key_path, source=source),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )

    def render_effective_config(self) -> str:
        return json.dumps(self.effective_settings(), ensure_ascii=True, indent=2, sort_keys=True)

    def render_source(self, source: str) -> str:
        resolved = self._find_source(source)
        if resolved is None:
            return f'# Config\n\nUnknown config source: {source}'
        return json.dumps(resolved.settings, ensure_ascii=True, indent=2, sort_keys=True)

    def _payload_for_source(self, source: str | None) -> dict[str, Any]:
        if source is None:
            return self.effective_settings()
        resolved = self._find_source(source)
        if resolved is None:
            raise KeyError(source)
        return resolved.settings

    def _find_source(self, source: str) -> ConfigSource | None:
        alias = _normalize_source_name(source)
        for config_source in self.sources:
            if _normalize_source_name(config_source.name) == alias:
                return config_source
        return None

    def _resolve_writable_source(self, source: str) -> tuple[str, Path]:
        alias = _normalize_source_name(source)
        if alias in {'project', 'project-settings', 'settings'}:
            return 'project', (self.cwd / PROJECT_SETTINGS_PATH).resolve()
        if alias in {'local', 'local-settings'}:
            return 'local', (self.cwd / LOCAL_SETTINGS_PATH).resolve()
        if alias in {'legacy', 'legacy-project'}:
            return 'legacy', (self.cwd / LEGACY_CONFIG_PATHS[0]).resolve()
        raise KeyError(source)


def _discover_source_paths(cwd: Path) -> tuple[tuple[str, Path], ...]:
    candidates = [
        ('legacy-claw', (cwd / LEGACY_CONFIG_PATHS[0]).resolve()),
        ('legacy-codex', (cwd / LEGACY_CONFIG_PATHS[1]).resolve()),
        ('project', (cwd / PROJECT_SETTINGS_PATH).resolve()),
        ('local', (cwd / LOCAL_SETTINGS_PATH).resolve()),
        ('jie', (cwd / JIE_SETTINGS_PATH).resolve()),
        ('jie-local', (cwd / JIE_LOCAL_SETTINGS_PATH).resolve()),
    ]
    # Codex YAML config (Phase 5 — Codex compatibility)
    codex_home = os.environ.get('CODEX_HOME', '')
    if codex_home:
        candidates.append(('codex-home', (Path(codex_home) / 'config.json').resolve()))
    for yaml_path in CODEX_YAML_PATHS:
        resolved = (cwd / yaml_path).resolve()
        if resolved.exists() and resolved.is_file():
            candidates.append(('codex-yaml', resolved))
    discovered: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for source_name, path in candidates:
        if path in seen or not path.exists() or not path.is_file():
            continue
        seen.add(path)
        discovered.append((source_name, path))
    return tuple(discovered)


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return None
    # YAML files (Codex config.yaml) — minimal key:value parser, no external deps
    if path.suffix in ('.yaml', '.yml'):
        return _parse_simple_yaml(text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _parse_simple_yaml(text: str) -> dict[str, Any] | None:
    """Minimal YAML parser for flat key: value configs (no nested structures)."""
    result: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if ':' not in stripped:
            continue
        key, _, value = stripped.partition(':')
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        # Try to coerce to int/float/bool
        if value.lower() in ('true', 'yes'):
            result[key] = True
        elif value.lower() in ('false', 'no'):
            result[key] = False
        else:
            try:
                result[key] = int(value)
            except ValueError:
                try:
                    result[key] = float(value)
                except ValueError:
                    result[key] = value
    return result if result else None


def _flatten_keys(payload: dict[str, Any], *, prefix: str = '') -> tuple[str, ...]:
    keys: list[str] = []
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        combined = f'{prefix}.{key}' if prefix else key
        keys.append(combined)
        if isinstance(value, dict):
            keys.extend(_flatten_keys(value, prefix=combined))
    return tuple(keys)


def _get_nested_value(payload: dict[str, Any], key_path: str) -> Any:
    current: Any = payload
    for segment in _split_key_path(key_path):
        if not isinstance(current, dict) or segment not in current:
            raise KeyError(key_path)
        current = current[segment]
    return current


def _set_nested_value(payload: dict[str, Any], key_path: str, value: Any) -> None:
    current: dict[str, Any] = payload
    segments = _split_key_path(key_path)
    for segment in segments[:-1]:
        child = current.get(segment)
        if not isinstance(child, dict):
            child = {}
            current[segment] = child
        current = child
    current[segments[-1]] = value


def _split_key_path(key_path: str) -> tuple[str, ...]:
    segments = tuple(
        segment.strip()
        for segment in key_path.split('.')
        if segment.strip()
    )
    if not segments:
        raise KeyError(key_path)
    return segments


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_source_name(source: str) -> str:
    return source.strip().lower().replace('_', '-')


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _sha256_or_none(text: str | None) -> str | None:
    if text is None:
        return None
    return _sha256(text)


def _preview(text: str | None, limit: int = 220) -> str | None:
    if text is None:
        return None
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit] + '...'
