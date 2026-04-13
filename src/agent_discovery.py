"""Agent discovery for jie-code.

Discovers agent definitions from .claude/agents/, .jie/agents/, and plugins.
Agent definitions are .md files with YAML frontmatter.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AgentDefinition:
    """A discovered agent definition."""
    name: str
    description: str
    model: str | None
    tools: tuple[str, ...]
    path: Path
    source: str  # 'project', 'user', 'plugin:<name>'
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        """Read full agent definition (system prompt for the agent)."""
        return self.path.read_text(encoding='utf-8')

    @property
    def system_prompt(self) -> str:
        """Extract the body (after frontmatter) as the agent's system prompt."""
        text = self.content
        match = re.match(r'^---\s*\n.*?\n---\s*\n', text, re.DOTALL)
        return text[match.end():] if match else text


@dataclass
class AgentRegistry:
    """Manages agent definition discovery and lookup."""
    cwd: Path
    additional_working_directories: tuple[str, ...] = ()
    _agents: dict[str, AgentDefinition] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_workspace(
        cls,
        cwd: Path,
        additional_working_directories: tuple[str, ...] = (),
    ) -> 'AgentRegistry':
        return cls(cwd=cwd.resolve(), additional_working_directories=additional_working_directories)

    @property
    def agents(self) -> dict[str, AgentDefinition]:
        if self._agents is None:
            self._agents = _discover_all_agents(self.cwd, self.additional_working_directories)
        return self._agents

    def list_agents(self) -> list[AgentDefinition]:
        return sorted(self.agents.values(), key=lambda a: a.name)

    def get_agent(self, name: str) -> AgentDefinition | None:
        """Look up an agent by name (case-insensitive)."""
        return self.agents.get(name.lower().strip())

    def refresh(self) -> None:
        self._agents = None


# --- Discovery ---

_AGENT_DIRS = ('.jie', '.claude', '.claw', '.codex')


def _discover_all_agents(
    cwd: Path,
    additional_dirs: tuple[str, ...],
) -> dict[str, AgentDefinition]:
    agents: dict[str, AgentDefinition] = {}

    # 1. Plugin agents (lowest priority)
    _discover_plugin_agents(agents)

    # 2. User home agents
    home = Path.home()
    for dirname in _AGENT_DIRS:
        agents_dir = home / dirname / 'agents'
        if agents_dir.is_dir():
            _scan_agents_dir(agents_dir, source=f'user:{dirname}', into=agents)

    # Env var config homes
    for env_var in ('JIE_CONFIG_HOME', 'CLAW_CONFIG_HOME', 'CLAUDE_CONFIG_DIR'):
        config_home = os.environ.get(env_var, '')
        if config_home:
            agents_dir = Path(config_home) / 'agents'
            if agents_dir.is_dir():
                _scan_agents_dir(agents_dir, source=f'env:{env_var}', into=agents)

    # 3. Project ancestor agents (higher priority)
    roots = _walk_upwards(cwd)
    roots.extend(Path(d).resolve() for d in additional_dirs)
    for root in roots:
        for dirname in _AGENT_DIRS:
            agents_dir = root / dirname / 'agents'
            if agents_dir.is_dir():
                _scan_agents_dir(agents_dir, source=f'project:{dirname}', into=agents)

    return agents


def _scan_agents_dir(
    agents_dir: Path,
    *,
    source: str,
    into: dict[str, AgentDefinition],
) -> None:
    if not agents_dir.is_dir():
        return
    for item in sorted(agents_dir.iterdir()):
        if item.is_file() and item.suffix == '.md':
            agent = _parse_agent_file(item, source=source)
            if agent:
                into[agent.name.lower()] = agent


def _parse_agent_file(path: Path, *, source: str) -> AgentDefinition | None:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return None

    name = ''
    description = ''
    model = None
    tools: list[str] = []
    metadata: dict[str, Any] = {}

    frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
    if frontmatter_match:
        fm_text = frontmatter_match.group(1)
        in_tools = False
        for line in fm_text.splitlines():
            stripped = line.strip()
            if stripped.startswith('- ') and in_tools:
                tools.append(stripped[2:].strip())
                continue
            in_tools = False
            if ':' not in stripped:
                continue
            key, _, value = stripped.partition(':')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == 'name':
                name = value
            elif key == 'description':
                description = value
            elif key == 'model':
                model = value if value else None
            elif key == 'tools':
                in_tools = True
                # tools might be inline: tools: [Read, Write]
                if value.startswith('['):
                    tools = [t.strip().strip('"').strip("'") for t in value.strip('[]').split(',') if t.strip()]
                    in_tools = False
            else:
                metadata[key] = value

    if not name:
        name = path.stem

    if not description:
        body = text[frontmatter_match.end():] if frontmatter_match else text
        for line in body.splitlines():
            line = line.strip().lstrip('#').strip()
            if line:
                description = line[:200]
                break

    return AgentDefinition(
        name=name,
        description=description,
        model=model,
        tools=tuple(tools),
        path=path,
        source=source,
        metadata=metadata,
    )


def _discover_plugin_agents(into: dict[str, AgentDefinition]) -> None:
    """Discover agents from installed Claude Code marketplace plugins."""
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
        agents_dir = Path(install_path) / 'agents'
        if agents_dir.is_dir():
            plugin_name = plugin_key.split('@')[0] if '@' in plugin_key else plugin_key
            _scan_agents_dir(agents_dir, source=f'plugin:{plugin_name}', into=into)


def _walk_upwards(path: Path) -> list[Path]:
    walked: list[Path] = []
    current = path
    while True:
        walked.append(current)
        if current.parent == current:
            break
        current = current.parent
    return walked
