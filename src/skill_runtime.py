"""Skill discovery and execution for jie-code.

Discovers skills from multiple roots (project, user home, plugins) and
makes them available as slash commands and as a Skill tool for the LLM.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SkillDefinition:
    """A discovered skill."""
    name: str
    description: str
    path: Path
    source: str  # 'project', 'user', 'plugin:<name>'
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        """Read full skill content (sent as prompt to LLM)."""
        return self.path.read_text(encoding='utf-8')


@dataclass
class SkillRuntime:
    """Manages skill discovery and lookup."""
    cwd: Path
    additional_working_directories: tuple[str, ...] = ()
    _skills: dict[str, SkillDefinition] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_workspace(
        cls,
        cwd: Path,
        additional_working_directories: tuple[str, ...] = (),
    ) -> 'SkillRuntime':
        return cls(cwd=cwd.resolve(), additional_working_directories=additional_working_directories)

    @property
    def skills(self) -> dict[str, SkillDefinition]:
        if self._skills is None:
            self._skills = _discover_all_skills(self.cwd, self.additional_working_directories)
        return self._skills

    def list_skills(self) -> list[SkillDefinition]:
        return sorted(self.skills.values(), key=lambda s: s.name)

    def get_skill(self, name: str) -> SkillDefinition | None:
        """Look up a skill by name (case-insensitive)."""
        lower = name.lower().strip().lstrip('/')
        return self.skills.get(lower)

    def resolve_skill(self, name: str) -> str | None:
        """Resolve a skill name to its full content for prompt injection."""
        skill = self.get_skill(name)
        if skill is None:
            return None
        return skill.content

    def refresh(self) -> None:
        """Force re-discovery of skills."""
        self._skills = None


# --- Discovery ---

# Config dir names to search for skills
_SKILL_DIRS = ('.jie', '.claude', '.claw', '.codex', '.omc')


def _discover_all_skills(
    cwd: Path,
    additional_dirs: tuple[str, ...],
) -> dict[str, SkillDefinition]:
    """Discover skills from all roots, with priority ordering."""
    skills: dict[str, SkillDefinition] = {}

    # 1. Plugin skills (lowest priority — overridden by user/project)
    _discover_plugin_skills(skills)

    # 2. User home skills
    home = Path.home()
    for dirname in _SKILL_DIRS:
        skills_dir = home / dirname / 'skills'
        if skills_dir.is_dir():
            _scan_skills_dir(skills_dir, source=f'user:{dirname}', into=skills)

    # Check env var config homes
    for env_var in ('JIE_CONFIG_HOME', 'CLAW_CONFIG_HOME', 'CLAUDE_CONFIG_DIR'):
        config_home = os.environ.get(env_var, '')
        if config_home:
            skills_dir = Path(config_home) / 'skills'
            if skills_dir.is_dir():
                _scan_skills_dir(skills_dir, source=f'env:{env_var}', into=skills)

    # 3. Project ancestor skills (higher priority)
    roots = _walk_upwards(cwd)
    roots.extend(Path(d).resolve() for d in additional_dirs)
    for root in roots:
        for dirname in _SKILL_DIRS:
            skills_dir = root / dirname / 'skills'
            if skills_dir.is_dir():
                _scan_skills_dir(skills_dir, source=f'project:{dirname}', into=skills)

    return skills


def _scan_skills_dir(
    skills_dir: Path,
    *,
    source: str,
    into: dict[str, SkillDefinition],
) -> None:
    """Scan a skills directory for SKILL.md files."""
    if not skills_dir.is_dir():
        return

    for item in sorted(skills_dir.iterdir()):
        if item.is_dir():
            # Directory-based skill: look for SKILL.md
            skill_file = item / 'SKILL.md'
            if skill_file.is_file():
                skill = _parse_skill_file(skill_file, source=source)
                if skill:
                    into[skill.name.lower()] = skill
        elif item.is_file() and item.suffix == '.md':
            # File-based skill (plain .md file)
            skill = _parse_skill_file(item, source=source)
            if skill:
                into[skill.name.lower()] = skill


def _parse_skill_file(path: Path, *, source: str) -> SkillDefinition | None:
    """Parse a skill file with optional YAML frontmatter."""
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return None

    name = ''
    description = ''
    metadata: dict[str, Any] = {}

    # Parse YAML frontmatter (between --- markers)
    frontmatter_match = re.match(r'^---\s*\n(.*?)\n---\s*\n', text, re.DOTALL)
    if frontmatter_match:
        fm_text = frontmatter_match.group(1)
        # Simple YAML-like parsing (no dependency on pyyaml)
        for line in fm_text.splitlines():
            line = line.strip()
            if ':' not in line:
                continue
            key, _, value = line.partition(':')
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key == 'name':
                name = value
            elif key == 'description':
                description = value
            elif key == 'metadata':
                continue  # Skip nested metadata header
            else:
                metadata[key] = value

    # Fallback: use filename/dirname as name
    if not name:
        if path.name == 'SKILL.md':
            name = path.parent.name
        else:
            name = path.stem

    # Fallback description: first non-empty line after frontmatter
    if not description:
        body = text[frontmatter_match.end():] if frontmatter_match else text
        for line in body.splitlines():
            line = line.strip().lstrip('#').strip()
            if line:
                description = line[:200]
                break

    return SkillDefinition(
        name=name,
        description=description,
        path=path,
        source=source,
        metadata=metadata,
    )


def _discover_plugin_skills(into: dict[str, SkillDefinition]) -> None:
    """Discover skills from installed Claude Code marketplace plugins."""
    home = Path.home()
    registry_path = home / '.claude' / 'plugins' / 'installed_plugins.json'
    if not registry_path.is_file():
        return

    import json
    try:
        registry = json.loads(registry_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return

    if not isinstance(registry, dict):
        return

    # Check enabled plugins from settings
    enabled = _load_enabled_plugins()

    plugins = registry.get('plugins', {})
    if not isinstance(plugins, dict):
        return

    for plugin_key, entries in plugins.items():
        # Only load enabled plugins
        if enabled is not None and plugin_key not in enabled:
            continue

        if not isinstance(entries, list) or not entries:
            continue

        # Use the most recent entry (last in list)
        entry = entries[-1]
        if not isinstance(entry, dict):
            continue

        install_path = entry.get('installPath', '')
        if not install_path:
            continue

        skills_dir = Path(install_path) / 'skills'
        if skills_dir.is_dir():
            plugin_name = plugin_key.split('@')[0] if '@' in plugin_key else plugin_key
            _scan_skills_dir(skills_dir, source=f'plugin:{plugin_name}', into=into)


def _load_enabled_plugins() -> set[str] | None:
    """Load enabled plugins from settings.json."""
    import json
    settings_path = Path.home() / '.claude' / 'settings.json'
    if not settings_path.is_file():
        return None
    try:
        settings = json.loads(settings_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    enabled = settings.get('enabledPlugins', {})
    if not isinstance(enabled, dict):
        return None
    return {k for k, v in enabled.items() if v}


def _walk_upwards(path: Path) -> list[Path]:
    walked: list[Path] = []
    current = path
    while True:
        walked.append(current)
        if current.parent == current:
            break
        current = current.parent
    return walked
