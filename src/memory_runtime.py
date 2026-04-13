"""Viking memory integration for jie-code.

Provides semantic memory recall and persistence using OpenViking.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


OV_CLI_PATH = '/data/miniconda3/envs/openviking/bin/ov'


@dataclass(frozen=True)
class MemoryResult:
    """A single memory search result."""
    uri: str
    score: float
    abstract: str
    context_type: str  # 'memory' or 'resource'


@dataclass
class MemoryRuntime:
    """Viking memory integration runtime."""
    enabled: bool = True
    ov_path: str = OV_CLI_PATH
    _available: bool | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_workspace(cls, cwd: Path) -> 'MemoryRuntime':
        # Check env var or default
        ov_path = os.environ.get('OV_CLI_PATH', OV_CLI_PATH)
        return cls(ov_path=ov_path)

    @property
    def available(self) -> bool:
        """Check if OpenViking CLI is available."""
        if self._available is None:
            self._available = Path(self.ov_path).is_file()
        return self._available

    def search(self, query: str, *, limit: int = 5, threshold: float = 0.55) -> list[MemoryResult]:
        """Search Viking for relevant memories."""
        if not self.available or not self.enabled:
            return []
        try:
            result = subprocess.run(
                [self.ov_path, 'find', query, '-n', str(limit)],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return []
            return _parse_search_results(result.stdout, threshold=threshold)
        except (subprocess.TimeoutExpired, OSError):
            return []

    def save_memory(self, content: str) -> bool:
        """Save a memory to Viking."""
        if not self.available or not self.enabled:
            return False
        try:
            result = subprocess.run(
                [self.ov_path, 'add-memory', content],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def build_context_injection(self, query: str) -> str | None:
        """Search Viking and format results as context for system prompt."""
        results = self.search(query)
        if not results:
            return None
        lines = ['[Viking Memory Context]']
        for r in results:
            lines.append(f'- ({r.score:.2f}) {r.abstract[:200]}')
        return '\n'.join(lines)

    def render_summary(self) -> str:
        if not self.available:
            return f'Viking: unavailable (ov not found at {self.ov_path})'
        return f'Viking: {"enabled" if self.enabled else "disabled"} (ov at {self.ov_path})'


def _parse_search_results(output: str, *, threshold: float) -> list[MemoryResult]:
    """Parse ov find output. The CLI outputs tab-separated table."""
    results: list[MemoryResult] = []
    lines = output.strip().splitlines()
    # Skip header lines (cmd: and column headers)
    data_lines = [l for l in lines if not l.startswith('cmd:') and not l.startswith('context_type')]
    for line in data_lines:
        parts = line.split('\t')
        if len(parts) < 4:
            # Try space-separated
            parts = line.split()
            if len(parts) < 4:
                continue
        try:
            context_type = parts[0].strip()
            uri = parts[1].strip()
            # level is parts[2]
            score = float(parts[3].strip())
            abstract = parts[4].strip() if len(parts) > 4 else ''
        except (ValueError, IndexError):
            continue
        if score >= threshold:
            results.append(MemoryResult(
                uri=uri,
                score=score,
                abstract=abstract,
                context_type=context_type,
            ))
    return sorted(results, key=lambda r: r.score, reverse=True)
