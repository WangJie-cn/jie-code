from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.run_terminal_bench_local import (
    TerminalBenchTask,
    build_host_agent_command,
    discover_tasks,
    filter_tasks,
    parse_dockerfile_workdir,
    strip_canary,
)


class TerminalBenchLocalTests(unittest.TestCase):
    def test_strip_canary_removes_leading_markers(self) -> None:
        raw = "<!-- canary -->\n# canary line\n\nactual instruction\n"
        self.assertEqual(strip_canary(raw), "actual instruction")

    def test_parse_dockerfile_workdir_uses_last_workdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dockerfile = Path(tmp_dir) / "Dockerfile"
            dockerfile.write_text(
                "FROM python:3.11\nWORKDIR /repo\nRUN echo hi\nWORKDIR /repo/app\n",
                encoding="utf-8",
            )
            self.assertEqual(parse_dockerfile_workdir(dockerfile), "/repo/app")

    def test_parse_dockerfile_workdir_defaults_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dockerfile = Path(tmp_dir) / "Dockerfile"
            dockerfile.write_text("FROM ubuntu:22.04\n", encoding="utf-8")
            self.assertEqual(parse_dockerfile_workdir(dockerfile), "/workspace")

    def test_discover_tasks_and_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_a = root / "task-a"
            task_a.mkdir()
            (task_a / "instruction.md").write_text("solve a\n", encoding="utf-8")
            (task_a / "task.toml").write_text(
                """
schema_version = "1.1"
[task]
name = "terminal-bench/headless-terminal"
description = "demo"
[environment]
docker_image = "example/demo:latest"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            env_a = task_a / "environment"
            env_a.mkdir()
            (env_a / "Dockerfile").write_text("FROM ubuntu\nWORKDIR /work\n", encoding="utf-8")

            task_b = root / "task-b"
            task_b.mkdir()
            (task_b / "instruction.md").write_text("solve b\n", encoding="utf-8")
            (task_b / "task.toml").write_text(
                """
schema_version = "1.1"
[task]
name = "terminal-bench/other-task"
description = "demo"
[environment]
docker_image = "example/other:latest"
""".strip()
                + "\n",
                encoding="utf-8",
            )
            env_b = task_b / "environment"
            env_b.mkdir()
            (env_b / "docker-compose.yaml").write_text("services: {}\n", encoding="utf-8")

            tasks = discover_tasks(root)
            self.assertEqual(len(tasks), 2)

            selected = filter_tasks(
                tasks,
                include_patterns=["headless-*"],
                exclude_patterns=[],
                limit=None,
            )
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].short_name, "headless-terminal")
            self.assertFalse(selected[0].has_docker_compose)

            selected = filter_tasks(
                tasks,
                include_patterns=[],
                exclude_patterns=["other-*"],
                limit=None,
            )
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].short_name, "headless-terminal")

    def test_build_host_agent_command_uses_current_interpreter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            workspace_dir = root / "workspace"
            repo_dir = root / "repo"
            agent_logs_dir = root / "agent"
            workspace_dir.mkdir()
            repo_dir.mkdir()
            agent_logs_dir.mkdir()
            task = TerminalBenchTask(
                task_dir=root,
                name="terminal-bench/demo",
                short_name="demo",
                instruction="solve it",
                docker_image="example/demo:latest",
                agent_timeout_sec=30.0,
                verifier_timeout_sec=30.0,
                workdir="/workspace",
                has_docker_compose=False,
            )

            cmd = build_host_agent_command(
                task=task,
                workspace_dir=workspace_dir,
                repo_dir=repo_dir,
                agent_logs_dir=agent_logs_dir,
            )

            self.assertIn(" -m src.main agent ", cmd)
            self.assertIn("instruction=$(cat", cmd)
            self.assertNotIn("claw-code-agent agent", cmd)


if __name__ == "__main__":
    unittest.main()
