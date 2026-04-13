"""Command discovery for jie-code.

Discovers custom slash commands from .claude/commands/, .jie/commands/, and plugins.
Commands are .md files where the filename becomes the command name.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommandDefinition:
    """A discovered custom command."""
    name: str
    description: str
    path: Path
    source: str  # 'project', 'user', 'plugin:<name>'

    @property
    def content(self) -> str:
        """Read full command content (prompt to inject)."""
        return self.path.read_text(encoding='utf-8')


@dataclass
class CommandRegistry:
    """Manages custom command discovery and lookup."""
    cwd: Path
    additional_working_directories: tuple[str, ...] = ()
    _commands: dict[str, CommandDefinition] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_workspace(
        cls,
        cwd: Path,
        additional_working_directories: tuple[str, ...] = (),
    ) -> 'CommandRegistry':
        return cls(cwd=cwd.resolve(), additional_working_directories=additional_working_directories)

    @property
    def commands(self) -> dict[str, CommandDefinition]:
        if self._commands is None:
            self._commands = _discover_all_commands(self.cwd, self.additional_working_directories)
        return self._commands

    def list_commands(self) -> list[CommandDefinition]:
        return sorted(self.commands.values(), key=lambda c: c.name)

    def get_command(self, name: str) -> CommandDefinition | None:
        """Look up a command by name (case-insensitive, with or without /)."""
        key = name.lower().strip().lstrip('/')
        return self.commands.get(key)

    def refresh(self) -> None:
        self._commands = None


# --- Discovery ---

_CMD_DIRS = ('.jie', '.claude', '.claw', '.codex')


def _discover_all_commands(
    cwd: Path,
    additional_dirs: tuple[str, ...],
) -> dict[str, CommandDefinition]:
    commands: dict[str, CommandDefinition] = {}

    # 1. Plugin commands (lowest priority)
    _discover_plugin_commands(commands)

    # 2. User home commands
    home = Path.home()
    for dirname in _CMD_DIRS:
        cmds_dir = home / dirname / 'commands'
        if cmds_dir.is_dir():
            _scan_commands_dir(cmds_dir, source=f'user:{dirname}', into=commands)

    # Env var config homes
    for env_var in ('JIE_CONFIG_HOME', 'CLAW_CONFIG_HOME', 'CLAUDE_CONFIG_DIR'):
        config_home = os.environ.get(env_var, '')
        if config_home:
            cmds_dir = Path(config_home) / 'commands'
            if cmds_dir.is_dir():
                _scan_commands_dir(cmds_dir, source=f'env:{env_var}', into=commands)

    # 3. Project ancestor commands (higher priority)
    roots = _walk_upwards(cwd)
    roots.extend(Path(d).resolve() for d in additional_dirs)
    for root in roots:
        for dirname in _CMD_DIRS:
            cmds_dir = root / dirname / 'commands'
            if cmds_dir.is_dir():
                _scan_commands_dir(cmds_dir, source=f'project:{dirname}', into=commands)

    return commands


def _scan_commands_dir(
    cmds_dir: Path,
    *,
    source: str,
    into: dict[str, CommandDefinition],
) -> None:
    if not cmds_dir.is_dir():
        return
    for item in sorted(cmds_dir.iterdir()):
        if item.is_file() and item.suffix == '.md':
            name = item.stem.lower()
            # First line is the description
            try:
                text = item.read_text(encoding='utf-8')
            except OSError:
                continue
            first_line = ''
            for line in text.splitlines():
                line = line.strip()
                if line:
                    first_line = line[:200]
                    break
            into[name] = CommandDefinition(
                name=name,
                description=first_line,
                path=item,
                source=source,
            )


def _discover_plugin_commands(into: dict[str, CommandDefinition]) -> None:
    """Discover commands from installed Claude Code marketplace plugins."""
    import json
    home = Path.home()
    registry_path = home / '.claude' / 'plugins' / 'installed_plugins.json'
    if not registry_path.is_file():
        return
    try:
        registry = json.loads(registry_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(registry, dict):
        return

    plugins = registry.get('plugins', {})
    if not isinstance(plugins, dict):
        return

    for plugin_key, entries in plugins.items():
        if not isinstance(entries, list) or not entries:
            continue
        entry = entries[-1]
        if not isinstance(entry, dict):
            continue
        install_path = entry.get('installPath', '')
        if not install_path:
            continue
        cmds_dir = Path(install_path) / 'commands'
        if cmds_dir.is_dir():
            plugin_name = plugin_key.split('@')[0] if '@' in plugin_key else plugin_key
            _scan_commands_dir(cmds_dir, source=f'plugin:{plugin_name}', into=into)


def _walk_upwards(path: Path) -> list[Path]:
    walked: list[Path] = []
    current = path
    while True:
        walked.append(current)
        if current.parent == current:
            break
        current = current.parent
    return walked
