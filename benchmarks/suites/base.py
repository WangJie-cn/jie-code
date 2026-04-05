"""
Base class for all benchmark suites.

Every suite implements:
  - load_dataset()   -> download/load the evaluation dataset
  - run_single()     -> run the agent on one problem
  - evaluate()       -> score agent output for one problem
  - run_all()        -> orchestrate the full benchmark
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkResult:
    """Result for a single problem in a benchmark suite."""

    problem_id: str
    passed: bool
    expected: str = ""
    actual: str = ""
    duration_sec: float = 0.0
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SuiteReport:
    """Aggregated report for a full benchmark suite run."""

    suite_name: str
    total: int
    passed: int
    failed: int
    score_pct: float
    duration_sec: float
    model: str
    results: list[BenchmarkResult]
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "score_pct": self.score_pct,
            "duration_sec": round(self.duration_sec, 2),
            "model": self.model,
            "timestamp": self.timestamp,
            "results": [asdict(r) for r in self.results],
        }


class BenchmarkSuite(ABC):
    """Abstract base for a benchmark suite."""

    # Subclasses set these
    name: str = "base"
    description: str = ""
    category: str = "general"  # coding | math | instruction-following

    def __init__(
        self,
        *,
        data_dir: str | None = None,
        limit: int | None = None,
        agent_timeout: float = 300.0,
        verbose: bool = False,
    ) -> None:
        self.data_dir = data_dir or str(
            Path(__file__).resolve().parent.parent / "data"
        )
        self.limit = limit
        self.agent_timeout = agent_timeout
        self.verbose = verbose
        self.project_root = str(Path(__file__).resolve().parent.parent.parent)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_dataset(self) -> list[dict[str, Any]]:
        """Return a list of problem dicts. Each must have at least an 'id' key."""
        ...

    @abstractmethod
    def build_prompt(self, problem: dict[str, Any]) -> str:
        """Convert a problem dict into the instruction string sent to the agent."""
        ...

    @abstractmethod
    def evaluate(
        self, problem: dict[str, Any], workspace: str
    ) -> BenchmarkResult:
        """Score the agent's output for one problem.  Return a BenchmarkResult."""
        ...

    # ------------------------------------------------------------------
    # Agent execution helpers
    # ------------------------------------------------------------------

    def _run_shell(
        self, cmd: str, cwd: str, timeout: float = 30.0
    ) -> tuple[int, str]:
        """Run a shell command, return (exit_code, combined_output)."""
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return proc.returncode, (proc.stdout + proc.stderr).strip()
        except subprocess.TimeoutExpired:
            return 1, f"[TIMEOUT after {timeout}s]"
        except Exception as exc:
            return 1, str(exc)

    def run_agent(self, instruction: str, workspace: str) -> tuple[int, str, float]:
        """Run the claw-code-agent on *instruction* inside *workspace*.

        Returns (exit_code, output, duration_sec).
        """
        import shlex

        agent_cmd = (
            f"{sys.executable} -m src.main agent "
            f"{shlex.quote(instruction)} "
            f"--cwd {shlex.quote(workspace)} "
            f"--allow-write "
            f"--allow-shell"
        )
        if self.verbose:
            print(f"  agent cmd: {agent_cmd[:120]}...")

        start = time.time()
        code, output = self._run_shell(
            agent_cmd, cwd=self.project_root, timeout=self.agent_timeout
        )
        duration = time.time() - start

        if self.verbose:
            print(f"  agent exit={code}  duration={duration:.1f}s")

        return code, output, duration

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_all(self) -> SuiteReport:
        """Run the full benchmark suite and return a SuiteReport."""
        problems = self.load_dataset()
        if self.limit is not None:
            problems = problems[: self.limit]

        print()
        print("=" * 72)
        print(f"  {self.name} BENCHMARK")
        print(f"  {self.description}")
        print("=" * 72)
        model = os.environ.get("OPENAI_MODEL", "unknown")
        print(f"  Model:    {model}")
        print(f"  Problems: {len(problems)}")
        print(f"  Timeout:  {self.agent_timeout}s per problem")
        print("=" * 72)
        print()

        all_results: list[BenchmarkResult] = []
        suite_start = time.time()

        for i, problem in enumerate(problems, 1):
            pid = problem.get("id", problem.get("task_id", f"problem-{i}"))
            print(f"[{i}/{len(problems)}] {pid}")

            workspace = tempfile.mkdtemp(prefix=f"claw_{self.name}_{pid}_")
            try:
                # Prepare workspace
                self.setup_workspace(problem, workspace)

                # Build prompt and run agent
                prompt = self.build_prompt(problem)
                _code, _output, duration = self.run_agent(prompt, workspace)

                # Evaluate
                result = self.evaluate(problem, workspace)
                result.duration_sec = duration

                status = "PASS ✅" if result.passed else "FAIL ❌"
                print(f"  -> {status}  ({duration:.1f}s)")
            except Exception as exc:
                result = BenchmarkResult(
                    problem_id=pid,
                    passed=False,
                    error=str(exc),
                )
                print(f"  -> ERROR ❌  {exc}")
            finally:
                shutil.rmtree(workspace, ignore_errors=True)

            all_results.append(result)
            print()

        suite_duration = time.time() - suite_start
        passed = sum(1 for r in all_results if r.passed)
        total = len(all_results)

        report = SuiteReport(
            suite_name=self.name,
            total=total,
            passed=passed,
            failed=total - passed,
            score_pct=round(100.0 * passed / total, 1) if total else 0.0,
            duration_sec=suite_duration,
            model=model,
            results=all_results,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        self._print_report(report)
        return report

    def setup_workspace(
        self, problem: dict[str, Any], workspace: str
    ) -> None:
        """Optional: prepare files in workspace before the agent runs."""

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _print_report(report: SuiteReport) -> None:
        print()
        print("=" * 72)
        print(f"  {report.suite_name} — RESULTS")
        print("=" * 72)
        print()
        for r in report.results:
            icon = "✅" if r.passed else "❌"
            print(f"  {icon} {r.problem_id:<40} {r.duration_sec:.1f}s")
        print()
        print("─" * 72)
        print(
            f"  Total: {report.total}  |  Passed: {report.passed}  "
            f"|  Failed: {report.failed}  |  Score: {report.score_pct:.1f}%"
        )
        print(f"  Total time: {report.duration_sec:.1f}s")
        print("─" * 72)
        print()

    @staticmethod
    def save_report(report: SuiteReport, path: str) -> None:
        """Persist a SuiteReport to JSON."""
        with open(path, "w") as fh:
            json.dump(report.to_dict(), fh, indent=2)
        print(f"  Report saved to {path}")
