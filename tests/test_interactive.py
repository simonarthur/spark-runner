"""Tests for the REPL-style interactive mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from spark_runner.interactive import (
    COMMANDS,
    SparkCompleter,
    dispatch,
    parse_command,
    _list_goal_names,
    _list_run_paths,
)
from spark_runner.models import GoalInfo, SparkConfig


# ── Helpers ──────────────────────────────────────────────────────────


def _write_goal(
    goal_summaries_dir: Path,
    name: str,
    main_task: str = "Do something",
    subtasks: list[dict[str, Any]] | None = None,
    observations: list[Any] | None = None,
    safety: dict[str, Any] | None = None,
) -> Path:
    goal_path = goal_summaries_dir / f"{name}-task.json"
    data: dict[str, Any] = {
        "main_task": main_task,
        "subtasks": subtasks or [],
        "key_observations": observations or [],
    }
    if safety is not None:
        data["safety"] = safety
    goal_path.write_text(json.dumps(data))
    return goal_path


def _identity(text: str) -> str:
    return text


def _make_config(tmp_path: Path) -> SparkConfig:
    gs_dir = tmp_path / "goal_summaries"
    gs_dir.mkdir()
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    return SparkConfig(
        data_dir=tmp_path,
        goal_summaries_dir=gs_dir,
        tasks_dir=tasks_dir,
        runs_dir=runs_dir,
    )


# ── parse_command ────────────────────────────────────────────────────


class TestParseCommand:
    def test_empty_string(self) -> None:
        assert parse_command("") == ("", [])

    def test_whitespace_only(self) -> None:
        assert parse_command("   ") == ("", [])

    def test_command_only(self) -> None:
        assert parse_command("goals") == ("goals", [])

    def test_command_with_args(self) -> None:
        assert parse_command("run login --unrun") == ("run", ["login", "--unrun"])

    def test_command_with_quoted_arg(self) -> None:
        assert parse_command('show "my goal"') == ("show", ["my goal"])

    def test_extra_whitespace(self) -> None:
        assert parse_command("  goals   --failed  ") == ("goals", ["--failed"])


# ── _list_goal_names ─────────────────────────────────────────────────


class TestListGoalNames:
    def test_empty_dir(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        assert _list_goal_names(config) == []

    def test_lists_names(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "beta")
        _write_goal(config.goal_summaries_dir, "alpha")
        names = _list_goal_names(config)
        assert names == ["alpha", "beta"]

    def test_none_dir(self) -> None:
        config = SparkConfig(goal_summaries_dir=None)
        assert _list_goal_names(config) == []


# ── _list_run_paths ──────────────────────────────────────────────────


class TestListRunPaths:
    def test_empty_dir(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        assert _list_run_paths(config) == []

    def test_lists_paths(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        assert config.runs_dir is not None
        (config.runs_dir / "login" / "2024-01-01").mkdir(parents=True)
        (config.runs_dir / "login" / "2024-01-02").mkdir(parents=True)
        paths = _list_run_paths(config)
        # Reverse sorted within task
        assert paths == ["login/2024-01-02", "login/2024-01-01"]


# ── SparkCompleter ───────────────────────────────────────────────────


class TestSparkCompleter:
    def _complete(self, completer: SparkCompleter, text: str) -> list[str]:
        from prompt_toolkit.document import Document

        doc = Document(text, len(text))
        return [c.text for c in completer.get_completions(doc, None)]

    def test_completes_commands(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        completer = SparkCompleter(config)
        results = self._complete(completer, "")
        assert set(COMMANDS.keys()).issubset(set(results))

    def test_completes_partial_command(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        completer = SparkCompleter(config)
        results = self._complete(completer, "go")
        assert "goals" in results
        assert "run" not in results

    def test_completes_goal_names(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        _write_goal(config.goal_summaries_dir, "logout")
        completer = SparkCompleter(config)
        results = self._complete(completer, "show ")
        assert "login" in results
        assert "logout" in results

    def test_completes_partial_goal_name(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        _write_goal(config.goal_summaries_dir, "logout")
        completer = SparkCompleter(config)
        results = self._complete(completer, "show log")
        assert "login" in results
        assert "logout" in results

    def test_completes_flags(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        completer = SparkCompleter(config)
        results = self._complete(completer, "goals --")
        assert "--unrun" in results
        assert "--failed" in results

    def test_completes_run_paths(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        assert config.runs_dir is not None
        (config.runs_dir / "login" / "2024-01-01").mkdir(parents=True)
        completer = SparkCompleter(config)
        results = self._complete(completer, "results ")
        assert "login/2024-01-01" in results

    def test_skips_already_used_goals(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        _write_goal(config.goal_summaries_dir, "logout")
        completer = SparkCompleter(config)
        results = self._complete(completer, "run login ")
        assert "login" not in results
        assert "logout" in results


# ── dispatch ─────────────────────────────────────────────────────────


class TestDispatch:
    def test_quit_returns_false(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        assert dispatch("quit", [], config, _identity) is False

    def test_exit_returns_false(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        assert dispatch("exit", [], config, _identity) is False

    def test_help_returns_true(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert dispatch("help", [], config, _identity) is True
        output = capsys.readouterr().out
        assert "goals" in output
        assert "run" in output

    def test_unknown_command(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert dispatch("banana", [], config, _identity) is True
        output = capsys.readouterr().out
        assert "Unknown command" in output

    def test_goals_command(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login", main_task="Log in")
        dispatch("goals", [], config, _identity)
        output = capsys.readouterr().out
        assert "1 goal(s)" in output
        assert "Log in" in output

    def test_goals_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("goals", [], config, _identity)
        output = capsys.readouterr().out
        assert "No goals found" in output

    def test_show_missing_arg(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("show", [], config, _identity)
        output = capsys.readouterr().out
        assert "Usage" in output

    def test_show_goal(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login", main_task="Log in")
        dispatch("show", ["login"], config, _identity)
        output = capsys.readouterr().out
        assert "Log in" in output

    def test_delete_missing_arg(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("delete", [], config, _identity)
        output = capsys.readouterr().out
        assert "Usage" in output

    def test_run_missing_arg(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("run", [], config, _identity)
        output = capsys.readouterr().out
        assert "No goals specified" in output

    def test_run_not_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("run", ["nonexistent"], config, _identity)
        output = capsys.readouterr().out
        assert "Goal not found" in output

    @patch("spark_runner.orchestrator.run_single")
    @patch("spark_runner.interactive.asyncio")
    def test_run_single_goal(
        self, mock_asyncio: MagicMock, mock_run: MagicMock,
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("run", ["login"], config, _identity)
        output = capsys.readouterr().out
        assert "Running 1 goal" in output
        mock_asyncio.run.assert_called_once()

    def test_results_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("results", [], config, _identity)
        output = capsys.readouterr().out
        assert "No runs found" in output

    def test_errors_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("errors", [], config, _identity)
        output = capsys.readouterr().out
        assert "No runs with errors" in output


# ── get_goal_summaries ───────────────────────────────────────────────


class TestGetGoalSummaries:
    def test_empty_dir(self, tmp_path: Path) -> None:
        from spark_runner.goals import get_goal_summaries

        result = get_goal_summaries(tmp_path, _identity)
        assert result == []

    def test_returns_goal_info(self, tmp_path: Path) -> None:
        from spark_runner.goals import get_goal_summaries

        _write_goal(tmp_path, "login", main_task="Log in")
        result = get_goal_summaries(tmp_path, _identity)
        assert len(result) == 1
        assert result[0].name == "login"
        assert result[0].main_task == "Log in"

    def test_counts_observations(self, tmp_path: Path) -> None:
        from spark_runner.goals import get_goal_summaries

        _write_goal(
            tmp_path, "obs",
            observations=[
                {"text": "err", "severity": "error"},
                {"text": "warn", "severity": "warning"},
                {"text": "warn2", "severity": "warning"},
            ],
        )
        result = get_goal_summaries(tmp_path, _identity)
        assert result[0].num_errors == 1
        assert result[0].num_warnings == 2
        assert result[0].num_observations == 3

    def test_unreadable_file(self, tmp_path: Path) -> None:
        from spark_runner.goals import get_goal_summaries

        bad = tmp_path / "broken-task.json"
        bad.write_text("NOT JSON{{{")
        result = get_goal_summaries(tmp_path, _identity)
        assert len(result) == 1
        assert result[0].main_task == "(unreadable)"

    def test_filter_unrun(self, tmp_path: Path) -> None:
        from spark_runner.goals import get_goal_summaries

        gs_dir = tmp_path / "goals"
        gs_dir.mkdir()
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        _write_goal(gs_dir, "never-run")
        _write_goal(gs_dir, "has-run")

        run_dir = runs_dir / "has-run" / "2024-01-01"
        run_dir.mkdir(parents=True)
        (run_dir / "run_metadata.json").write_text(
            json.dumps({"phases": [{"outcome": "SUCCESS"}]})
        )

        result = get_goal_summaries(gs_dir, _identity, runs_dir, filter_unrun=True)
        assert len(result) == 1
        assert result[0].name == "never-run"


# ── CLI flag ─────────────────────────────────────────────────────────


class TestInteractiveCLIFlag:
    @patch("spark_runner.interactive.interactive_loop")
    @patch("spark_runner.cli.build_config")
    def test_dash_i_invokes_interactive(
        self, mock_build: MagicMock, mock_loop: MagicMock,
    ) -> None:
        from spark_runner.cli import cli

        mock_config = MagicMock()
        mock_build.return_value = mock_config
        runner = CliRunner()
        result = runner.invoke(cli, ["-i"])
        mock_loop.assert_called_once_with(mock_config)
        assert result.exit_code == 0

    @patch("spark_runner.interactive.interactive_loop")
    @patch("spark_runner.cli.build_config")
    def test_interactive_subcommand(
        self, mock_build: MagicMock, mock_loop: MagicMock,
    ) -> None:
        from spark_runner.cli import cli

        mock_config = MagicMock()
        mock_build.return_value = mock_config
        runner = CliRunner()
        result = runner.invoke(cli, ["interactive"])
        mock_loop.assert_called_once_with(mock_config)
        assert result.exit_code == 0
