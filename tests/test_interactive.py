"""Tests for the interactive mode module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest
from click.testing import CliRunner

from spark_runner.goals import get_goal_summaries
from spark_runner.interactive import (
    _format_goal_entry,
    _show_action_menu,
    _show_goal_selector,
    interactive_loop,
)
from spark_runner.models import GoalInfo, SparkConfig


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_goal(
    goal_summaries_dir: Path,
    name: str,
    main_task: str = "Do something",
    subtasks: list[dict[str, Any]] | None = None,
    observations: list[Any] | None = None,
    safety: dict[str, Any] | None = None,
) -> Path:
    """Write a goal summary JSON file and return its path."""
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


def _make_goal(
    name: str = "test-goal",
    main_task: str = "Do a thing",
    last_run_status: str | None = None,
    last_run_timestamp: str | None = None,
    num_errors: int = 0,
) -> GoalInfo:
    return GoalInfo(
        name=name,
        file_path=Path(f"/tmp/{name}-task.json"),
        main_task=main_task,
        last_run_status=last_run_status,
        last_run_timestamp=last_run_timestamp,
        num_errors=num_errors,
    )


# ── TestGetGoalSummaries ─────────────────────────────────────────────────


class TestGetGoalSummaries:
    def test_no_goals(self, tmp_path: Path) -> None:
        result = get_goal_summaries(tmp_path, _identity)
        assert result == []

    def test_one_goal(self, tmp_path: Path) -> None:
        _write_goal(tmp_path, "login", main_task="Log into the app")
        result = get_goal_summaries(tmp_path, _identity)
        assert len(result) == 1
        assert result[0].name == "login"
        assert result[0].main_task == "Log into the app"

    def test_multiple_goals_sorted(self, tmp_path: Path) -> None:
        _write_goal(tmp_path, "beta", main_task="Beta task")
        _write_goal(tmp_path, "alpha", main_task="Alpha task")
        result = get_goal_summaries(tmp_path, _identity)
        assert len(result) == 2
        assert result[0].name == "alpha"
        assert result[1].name == "beta"

    def test_counts_subtasks_and_observations(self, tmp_path: Path) -> None:
        _write_goal(
            tmp_path, "counted",
            subtasks=[{"filename": "a.txt"}, {"filename": "b.txt"}],
            observations=[
                {"text": "err", "severity": "error"},
                {"text": "warn", "severity": "warning"},
                "plain obs",
            ],
        )
        result = get_goal_summaries(tmp_path, _identity)
        info = result[0]
        assert info.num_subtasks == 2
        assert info.num_observations == 3
        assert info.num_errors == 1
        assert info.num_warnings == 1

    def test_filter_unrun(self, tmp_path: Path) -> None:
        gs_dir = tmp_path / "goals"
        gs_dir.mkdir()
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        _write_goal(gs_dir, "never-run")
        _write_goal(gs_dir, "has-run")

        # Create a run directory for "has-run"
        run_task_dir = runs_dir / "has-run" / "2024-01-01_00-00-00"
        run_task_dir.mkdir(parents=True)
        meta = {"phases": [{"outcome": "SUCCESS"}]}
        (run_task_dir / "run_metadata.json").write_text(json.dumps(meta))

        result = get_goal_summaries(gs_dir, _identity, runs_dir, filter_unrun=True)
        assert len(result) == 1
        assert result[0].name == "never-run"

    def test_filter_failed(self, tmp_path: Path) -> None:
        gs_dir = tmp_path / "goals"
        gs_dir.mkdir()
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        _write_goal(gs_dir, "ok-goal")
        _write_goal(gs_dir, "bad-goal")

        # ok-goal run
        ok_dir = runs_dir / "ok-goal" / "2024-01-01_00-00-00"
        ok_dir.mkdir(parents=True)
        (ok_dir / "run_metadata.json").write_text(
            json.dumps({"phases": [{"outcome": "SUCCESS"}]})
        )

        # bad-goal run
        bad_dir = runs_dir / "bad-goal" / "2024-01-01_00-00-00"
        bad_dir.mkdir(parents=True)
        (bad_dir / "run_metadata.json").write_text(
            json.dumps({"phases": [{"outcome": "FAILED"}]})
        )

        result = get_goal_summaries(gs_dir, _identity, runs_dir, filter_failed=True)
        assert len(result) == 1
        assert result[0].name == "bad-goal"

    def test_unreadable_file(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "broken-task.json"
        bad_file.write_text("NOT JSON{{{")
        result = get_goal_summaries(tmp_path, _identity)
        assert len(result) == 1
        assert result[0].main_task == "(unreadable)"
        assert result[0].name == "broken"


# ── TestFormatGoalEntry ──────────────────────────────────────────────────


class TestFormatGoalEntry:
    def test_ok_status(self) -> None:
        goal = _make_goal(last_run_status="ok", last_run_timestamp="2024-01-01")
        entry = _format_goal_entry(goal)
        assert "[ok]" in entry
        assert "test-goal" in entry

    def test_errors_status(self) -> None:
        goal = _make_goal(last_run_status="errors", last_run_timestamp="2024-01-01")
        entry = _format_goal_entry(goal)
        assert "[errors]" in entry

    def test_never_run(self) -> None:
        goal = _make_goal()
        entry = _format_goal_entry(goal)
        assert "[never run]" in entry

    def test_description_truncation(self) -> None:
        long_desc = "A" * 100
        goal = _make_goal(main_task=long_desc)
        entry = _format_goal_entry(goal)
        assert "..." in entry
        assert len(entry) < len(long_desc) + 50  # name + status + truncated desc

    def test_short_description_not_truncated(self) -> None:
        goal = _make_goal(main_task="Short task")
        entry = _format_goal_entry(goal)
        assert "Short task" in entry
        assert "..." not in entry


# ── TestActionMenu ───────────────────────────────────────────────────────


class TestActionMenu:
    @patch("spark_runner.interactive.radiolist_dialog")
    def test_returns_run(self, mock_dialog: MagicMock) -> None:
        mock_dialog.return_value.run.return_value = "run"
        result = _show_action_menu([_make_goal()])
        assert result == "run"

    @patch("spark_runner.interactive.radiolist_dialog")
    def test_returns_show(self, mock_dialog: MagicMock) -> None:
        mock_dialog.return_value.run.return_value = "show"
        result = _show_action_menu([_make_goal()])
        assert result == "show"

    @patch("spark_runner.interactive.radiolist_dialog")
    def test_returns_delete(self, mock_dialog: MagicMock) -> None:
        mock_dialog.return_value.run.return_value = "delete"
        result = _show_action_menu([_make_goal()])
        assert result == "delete"

    @patch("spark_runner.interactive.radiolist_dialog")
    def test_returns_refresh(self, mock_dialog: MagicMock) -> None:
        mock_dialog.return_value.run.return_value = "refresh"
        result = _show_action_menu([_make_goal()])
        assert result == "refresh"

    @patch("spark_runner.interactive.radiolist_dialog")
    def test_quit_returns_none(self, mock_dialog: MagicMock) -> None:
        mock_dialog.return_value.run.return_value = "quit"
        result = _show_action_menu([_make_goal()])
        assert result is None

    @patch("spark_runner.interactive.radiolist_dialog")
    def test_escape_returns_none(self, mock_dialog: MagicMock) -> None:
        mock_dialog.return_value.run.return_value = None
        result = _show_action_menu([_make_goal()])
        assert result is None


# ── TestGoalSelector ─────────────────────────────────────────────────────


class TestGoalSelector:
    @patch("spark_runner.interactive.radiolist_dialog")
    def test_returns_selected_goal(self, mock_dialog: MagicMock) -> None:
        goals = [_make_goal("a"), _make_goal("b")]
        mock_dialog.return_value.run.return_value = 1
        result = _show_goal_selector(goals)
        assert result is not None
        assert result.name == "b"

    @patch("spark_runner.interactive.radiolist_dialog")
    def test_escape_returns_none(self, mock_dialog: MagicMock) -> None:
        goals = [_make_goal()]
        mock_dialog.return_value.run.return_value = None
        result = _show_goal_selector(goals)
        assert result is None

    def test_empty_list_returns_none(self, capsys: pytest.CaptureFixture[str]) -> None:
        result = _show_goal_selector([])
        assert result is None
        output = capsys.readouterr().out
        assert "No goals available" in output


# ── TestInteractiveLoop ──────────────────────────────────────────────────


class TestInteractiveLoop:
    def _make_config(self, tmp_path: Path) -> SparkConfig:
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

    @patch("spark_runner.interactive._show_action_menu", return_value=None)
    @patch("spark_runner.interactive.get_goal_summaries", return_value=[])
    def test_quit_immediately(
        self, mock_summaries: MagicMock, mock_menu: MagicMock,
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = self._make_config(tmp_path)
        interactive_loop(config)
        output = capsys.readouterr().out
        assert "Goodbye" in output

    @patch("spark_runner.interactive._handle_show")
    @patch("spark_runner.interactive._show_action_menu", side_effect=["show", None])
    @patch("spark_runner.interactive.get_goal_summaries")
    def test_dispatches_show(
        self, mock_summaries: MagicMock, mock_menu: MagicMock,
        mock_show: MagicMock, tmp_path: Path,
    ) -> None:
        goals = [_make_goal()]
        mock_summaries.return_value = goals
        config = self._make_config(tmp_path)
        interactive_loop(config)
        mock_show.assert_called_once()

    @patch("spark_runner.interactive._handle_run")
    @patch("spark_runner.interactive._show_action_menu", side_effect=["run", None])
    @patch("spark_runner.interactive.get_goal_summaries")
    def test_dispatches_run(
        self, mock_summaries: MagicMock, mock_menu: MagicMock,
        mock_run: MagicMock, tmp_path: Path,
    ) -> None:
        goals = [_make_goal()]
        mock_summaries.return_value = goals
        config = self._make_config(tmp_path)
        interactive_loop(config)
        mock_run.assert_called_once()

    @patch("spark_runner.interactive._handle_delete")
    @patch("spark_runner.interactive._show_action_menu", side_effect=["delete", None])
    @patch("spark_runner.interactive.get_goal_summaries")
    def test_dispatches_delete(
        self, mock_summaries: MagicMock, mock_menu: MagicMock,
        mock_delete: MagicMock, tmp_path: Path,
    ) -> None:
        goals = [_make_goal()]
        mock_summaries.return_value = goals
        config = self._make_config(tmp_path)
        interactive_loop(config)
        mock_delete.assert_called_once()

    @patch("spark_runner.interactive._show_action_menu", side_effect=["run", None])
    @patch("spark_runner.interactive.get_goal_summaries", return_value=[])
    @patch("spark_runner.interactive._handle_run", side_effect=RuntimeError("boom"))
    def test_error_recovery(
        self, mock_run: MagicMock, mock_summaries: MagicMock,
        mock_menu: MagicMock, tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = self._make_config(tmp_path)
        interactive_loop(config)
        output = capsys.readouterr().out
        assert "Error: boom" in output
        assert "Goodbye" in output


# ── TestInteractiveCLIFlag ───────────────────────────────────────────────


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
