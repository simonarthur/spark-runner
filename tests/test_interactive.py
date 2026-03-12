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

    def test_completes_run_flags(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        completer = SparkCompleter(config)
        results = self._complete(completer, "run --")
        assert "--unrun" in results
        assert "--failed" in results
        assert "--no-update-summary" in results
        assert "--no-update-tasks" in results
        assert "--no-knowledge-reuse" in results
        assert "--regenerate-tasks" in results

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

    def test_show_goal_displays_hints(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        goal_path = _write_goal(config.goal_summaries_dir, "login")
        data = json.loads(goal_path.read_text())
        data["hints"] = [
            {"phase": "Fill Form", "text": "Use dropdown"},
            {"phase": "", "text": "Split into two phases"},
        ]
        goal_path.write_text(json.dumps(data))
        dispatch("show", ["login"], config, _identity)
        output = capsys.readouterr().out
        assert "Hints (2)" in output
        assert "[Fill Form]" in output
        assert "Use dropdown" in output
        assert "[Goal]" in output
        assert "Split into two phases" in output

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

    @patch("spark_runner.orchestrator.run_multiple")
    @patch("spark_runner.interactive.asyncio")
    def test_run_multiple_goals_auto_closes_browser(
        self, mock_asyncio: MagicMock, mock_run_multi: MagicMock,
        tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        _write_goal(config.goal_summaries_dir, "logout")
        dispatch("run", ["login", "logout"], config, _identity)
        output = capsys.readouterr().out
        assert "Running 2 goal" in output
        mock_run_multi.assert_called_once()
        passed_config = mock_run_multi.call_args[0][1]
        assert passed_config.auto_close is True
        # Original config unchanged
        assert config.auto_close is False

    @patch("spark_runner.orchestrator.run_single")
    @patch("spark_runner.interactive.asyncio")
    def test_run_single_goal_preserves_auto_close(
        self, mock_asyncio: MagicMock, mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("run", ["login"], config, _identity)
        mock_run.assert_called_once()
        passed_config = mock_run.call_args[0][1]
        assert passed_config.auto_close is False

    @patch("spark_runner.orchestrator.run_single")
    @patch("spark_runner.interactive.asyncio")
    def test_run_no_update_summary_flag(
        self, mock_asyncio: MagicMock, mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("run", ["login", "--no-update-summary"], config, _identity)
        call_args = mock_asyncio.run.call_args[0][0]
        # Coroutine wraps run_single(task, run_config) — get the config from the mock
        mock_run.assert_called_once()
        passed_config = mock_run.call_args[0][1]
        assert passed_config.update_summary is False
        assert passed_config.update_tasks is True
        assert passed_config.knowledge_reuse is True

    @patch("spark_runner.orchestrator.run_single")
    @patch("spark_runner.interactive.asyncio")
    def test_run_no_update_tasks_flag(
        self, mock_asyncio: MagicMock, mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("run", ["login", "--no-update-tasks"], config, _identity)
        mock_run.assert_called_once()
        passed_config = mock_run.call_args[0][1]
        assert passed_config.update_tasks is False
        assert passed_config.update_summary is True
        assert passed_config.knowledge_reuse is True

    @patch("spark_runner.orchestrator.run_single")
    @patch("spark_runner.interactive.asyncio")
    def test_run_no_knowledge_reuse_flag(
        self, mock_asyncio: MagicMock, mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("run", ["login", "--no-knowledge-reuse"], config, _identity)
        mock_run.assert_called_once()
        passed_config = mock_run.call_args[0][1]
        assert passed_config.knowledge_reuse is False
        assert passed_config.update_summary is True
        assert passed_config.update_tasks is True

    @patch("spark_runner.orchestrator.run_single")
    @patch("spark_runner.interactive.asyncio")
    def test_run_all_no_flags(
        self, mock_asyncio: MagicMock, mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch(
            "run",
            ["login", "--no-update-summary", "--no-update-tasks", "--no-knowledge-reuse"],
            config, _identity,
        )
        mock_run.assert_called_once()
        passed_config = mock_run.call_args[0][1]
        assert passed_config.update_summary is False
        assert passed_config.update_tasks is False
        assert passed_config.knowledge_reuse is False

    @patch("spark_runner.orchestrator.run_single")
    @patch("spark_runner.interactive.asyncio")
    def test_run_flags_do_not_mutate_original_config(
        self, mock_asyncio: MagicMock, mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch(
            "run",
            ["login", "--no-update-summary", "--no-update-tasks", "--no-knowledge-reuse"],
            config, _identity,
        )
        # Original config must remain unchanged
        assert config.update_summary is True
        assert config.update_tasks is True
        assert config.knowledge_reuse is True

    @patch("spark_runner.orchestrator.run_single")
    @patch("spark_runner.interactive.asyncio")
    def test_run_regenerate_tasks_flag(
        self, mock_asyncio: MagicMock, mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("run", ["login", "--regenerate-tasks"], config, _identity)
        mock_run.assert_called_once()
        passed_config = mock_run.call_args[0][1]
        assert passed_config.regenerate_tasks is True
        # Original config unchanged
        assert config.regenerate_tasks is False

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


# ── interactive_loop ─────────────────────────────────────────────────


class TestInteractiveLoop:
    @patch("spark_runner.interactive.PromptSession")
    @patch("spark_runner.orchestrator._make_restore_fn", return_value=_identity)
    def test_uses_file_history(
        self, mock_restore: MagicMock, mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        from prompt_toolkit.history import FileHistory

        from spark_runner.interactive import interactive_loop

        config = _make_config(tmp_path)
        # Make the prompt raise EOFError immediately to exit the loop
        mock_session = MagicMock()
        mock_session.prompt.side_effect = EOFError
        mock_session_cls.return_value = mock_session

        interactive_loop(config)

        # Verify FileHistory was passed with the correct path
        call_kwargs = mock_session_cls.call_args[1]
        history = call_kwargs["history"]
        assert isinstance(history, FileHistory)
        assert history.filename == str(config.data_dir / ".repl_history")

    @patch("spark_runner.interactive.PromptSession")
    @patch("spark_runner.orchestrator._make_restore_fn", return_value=_identity)
    def test_prompt_session_has_bottom_toolbar(
        self, mock_restore: MagicMock, mock_session_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        from spark_runner.interactive import _bottom_toolbar, interactive_loop

        config = _make_config(tmp_path)
        mock_session = MagicMock()
        mock_session.prompt.side_effect = EOFError
        mock_session_cls.return_value = mock_session

        interactive_loop(config)

        call_kwargs = mock_session_cls.call_args[1]
        assert call_kwargs["bottom_toolbar"] is _bottom_toolbar


# ── toolbar ──────────────────────────────────────────────────────────


class TestBottomToolbar:
    def test_toolbar_shows_ready_when_empty(self) -> None:
        from spark_runner.interactive import _ToolbarState, _bottom_toolbar, _toolbar_state

        # Reset state
        _toolbar_state.goal_count = 0
        _toolbar_state.last_run_name = ""
        _toolbar_state.last_run_status = ""

        result = _bottom_toolbar()
        assert "Ready" in result.value

    def test_toolbar_shows_goal_count(self) -> None:
        from spark_runner.interactive import _bottom_toolbar, _toolbar_state

        _toolbar_state.goal_count = 5
        _toolbar_state.last_run_name = ""
        _toolbar_state.last_run_status = ""

        result = _bottom_toolbar()
        assert "5 goal(s)" in result.value
        assert "Ready" not in result.value

    def test_toolbar_shows_last_run(self) -> None:
        from spark_runner.interactive import _bottom_toolbar, _toolbar_state

        _toolbar_state.goal_count = 3
        _toolbar_state.last_run_name = "login"
        _toolbar_state.last_run_status = "PASS"

        result = _bottom_toolbar()
        assert "login" in result.value
        assert "PASS" in result.value
        assert "3 goal(s)" in result.value

    def test_toolbar_shows_fail_status(self) -> None:
        from spark_runner.interactive import _bottom_toolbar, _toolbar_state

        _toolbar_state.goal_count = 0
        _toolbar_state.last_run_name = "signup"
        _toolbar_state.last_run_status = "FAIL"

        result = _bottom_toolbar()
        assert "signup" in result.value
        assert "FAIL" in result.value

    @patch("spark_runner.orchestrator.run_single")
    @patch("spark_runner.interactive.asyncio")
    def test_run_updates_toolbar_state(
        self, mock_asyncio: MagicMock, mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        from spark_runner.interactive import _toolbar_state
        from spark_runner.models import RunResult

        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")

        mock_result = RunResult(task_name="login", phases=[], screenshots=[])
        mock_asyncio.run.return_value = mock_result

        dispatch("run", ["login"], config, _identity)

        assert _toolbar_state.last_run_name == "login"
        assert _toolbar_state.last_run_status == "PASS"


# ── hint commands ────────────────────────────────────────────────────


class TestHintCommands:
    def test_hint_command_saves_hint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(
            config.goal_summaries_dir, "login",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        dispatch("hint", ["login", "Fill", "Form", "--", "Click", "More", "Options"], config, _identity)
        output = capsys.readouterr().out
        assert "Hint saved" in output
        # Verify it was actually saved
        data = json.loads((config.goal_summaries_dir / "login-task.json").read_text())
        assert len(data["hints"]) == 1
        assert data["hints"][0]["phase"] == "Fill Form"
        assert data["hints"][0]["text"] == "Click More Options"

    def test_hint_command_missing_separator(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("hint", ["login", "Fill", "Form", "no", "separator"], config, _identity)
        output = capsys.readouterr().out
        assert "Usage" in output

    def test_hint_command_goal_not_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("hint", ["nonexistent", "Phase", "--", "text"], config, _identity)
        output = capsys.readouterr().out
        assert "Goal not found" in output

    def test_hints_command_lists(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        # Add a hint first
        goal_path = config.goal_summaries_dir / "login-task.json"
        data = json.loads(goal_path.read_text())
        data["hints"] = [{"phase": "Login", "text": "Use SSO"}]
        goal_path.write_text(json.dumps(data))
        dispatch("hints", ["login"], config, _identity)
        output = capsys.readouterr().out
        assert "[Login]" in output
        assert "Use SSO" in output

    def test_hints_command_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("hints", ["login"], config, _identity)
        output = capsys.readouterr().out
        assert "No hints" in output

    def test_hints_command_missing_arg(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("hints", [], config, _identity)
        output = capsys.readouterr().out
        assert "Usage" in output

    def test_unhint_command_removes(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        goal_path = config.goal_summaries_dir / "login-task.json"
        data = json.loads(goal_path.read_text())
        data["hints"] = [{"phase": "Login", "text": "Hint A"}, {"phase": "Login", "text": "Hint B"}]
        goal_path.write_text(json.dumps(data))
        dispatch("unhint", ["login", "0"], config, _identity)
        output = capsys.readouterr().out
        assert "removed" in output
        data = json.loads(goal_path.read_text())
        assert len(data["hints"]) == 1
        assert data["hints"][0]["text"] == "Hint B"

    def test_unhint_command_invalid_index(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("unhint", ["login", "99"], config, _identity)
        output = capsys.readouterr().out
        assert "Invalid hint index" in output

    def test_unhint_command_missing_args(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("unhint", ["login"], config, _identity)
        output = capsys.readouterr().out
        assert "Usage" in output

    def test_hint_goal_level_saves(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """``hint login -- some text`` saves a goal-level hint with phase=""."""
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("hint", ["login", "--", "split", "form", "into", "two", "phases"], config, _identity)
        output = capsys.readouterr().out
        assert "Goal-level hint saved" in output
        data = json.loads((config.goal_summaries_dir / "login-task.json").read_text())
        assert len(data["hints"]) == 1
        assert data["hints"][0]["phase"] == ""
        assert data["hints"][0]["text"] == "split form into two phases"

    def test_hints_displays_goal_level(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Goal-level hints display ``[Goal]`` label instead of empty phase."""
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        goal_path = config.goal_summaries_dir / "login-task.json"
        data = json.loads(goal_path.read_text())
        data["hints"] = [
            {"phase": "", "text": "Add a logout step at the end"},
            {"phase": "Login", "text": "Use the SSO button"},
        ]
        goal_path.write_text(json.dumps(data))
        dispatch("hints", ["login"], config, _identity)
        output = capsys.readouterr().out
        assert "[Goal]" in output
        assert "Add a logout step at the end" in output
        assert "[Login]" in output
        assert "Use the SSO button" in output

    def test_hint_still_works_with_phase(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Existing ``hint login Fill Form -- text`` behavior is unchanged."""
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(
            config.goal_summaries_dir, "login",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        dispatch("hint", ["login", "Fill", "Form", "--", "Use", "tab", "key"], config, _identity)
        output = capsys.readouterr().out
        assert 'Hint saved for phase "Fill Form"' in output
        data = json.loads((config.goal_summaries_dir / "login-task.json").read_text())
        assert data["hints"][0]["phase"] == "Fill Form"
        assert data["hints"][0]["text"] == "Use tab key"


    def test_hint_rejects_unknown_phase(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """hint with a non-existent phase should print an error."""
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(
            config.goal_summaries_dir, "login",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        dispatch("hint", ["login", "Nonexistent", "Phase", "--", "text"], config, _identity)
        output = capsys.readouterr().out
        assert "Unknown phase" in output
        assert "Fill Form" in output  # shows available phases

    def test_hint_rejects_phase_when_no_subtasks(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """hint with a phase on a goal that has no subtasks should print an error."""
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("hint", ["login", "Some", "Phase", "--", "text"], config, _identity)
        output = capsys.readouterr().out
        assert "Unknown phase" in output
        assert "no subtask phases" in output

    def test_hint_phase_case_insensitive(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Phase name matching should be case-insensitive and use canonical case."""
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(
            config.goal_summaries_dir, "login",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        dispatch("hint", ["login", "fill", "form", "--", "try", "dropdown"], config, _identity)
        output = capsys.readouterr().out
        assert "Hint saved" in output
        data = json.loads((config.goal_summaries_dir / "login-task.json").read_text())
        assert data["hints"][0]["phase"] == "Fill Form"  # canonical case

    def test_completes_phase_names_for_hint(self, tmp_path: Path) -> None:
        """After goal name, tab completion should suggest phase names."""
        from prompt_toolkit.document import Document

        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(
            config.goal_summaries_dir, "login",
            subtasks=[
                {"filename": "fill-form.txt"},
                {"filename": "verify-result.txt"},
            ],
        )
        completer = SparkCompleter(config)
        doc = Document("hint login ", len("hint login "))
        results = [c.text for c in completer.get_completions(doc, None)]
        assert "Fill Form" in results
        assert "Verify Result" in results
        assert "--" in results

    def test_completes_phase_prefix(self, tmp_path: Path) -> None:
        """Typing a partial phase name should filter completions."""
        from prompt_toolkit.document import Document

        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(
            config.goal_summaries_dir, "login",
            subtasks=[
                {"filename": "fill-form.txt"},
                {"filename": "verify-result.txt"},
            ],
        )
        completer = SparkCompleter(config)
        doc = Document("hint login F", len("hint login F"))
        results = [c.text for c in completer.get_completions(doc, None)]
        assert "Fill Form" in results
        assert "Verify Result" not in results

    def test_completes_separator_after_full_phase(self, tmp_path: Path) -> None:
        """After typing a complete phase name, suggest '--'."""
        from prompt_toolkit.document import Document

        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(
            config.goal_summaries_dir, "login",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        completer = SparkCompleter(config)
        doc = Document("hint login Fill Form ", len("hint login Fill Form "))
        results = [c.text for c in completer.get_completions(doc, None)]
        assert results == ["--"]

    def test_no_completion_after_separator(self, tmp_path: Path) -> None:
        """After '--', no completions should be offered."""
        from prompt_toolkit.document import Document

        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        completer = SparkCompleter(config)
        doc = Document("hint login -- ", len("hint login -- "))
        results = [c.text for c in completer.get_completions(doc, None)]
        assert results == []

    def test_hint_goal_name_completion(self, tmp_path: Path) -> None:
        """First arg position should still complete goal names."""
        from prompt_toolkit.document import Document

        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        _write_goal(config.goal_summaries_dir, "logout")
        completer = SparkCompleter(config)
        doc = Document("hint log", len("hint log"))
        results = [c.text for c in completer.get_completions(doc, None)]
        assert "login" in results
        assert "logout" in results


class TestRunHintsFlag:
    @patch("spark_runner.orchestrator.run_single")
    @patch("spark_runner.interactive.asyncio")
    def test_run_hints_flag_passes_callback(
        self, mock_asyncio: MagicMock, mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("run", ["login", "--hints"], config, _identity)
        mock_run.assert_called_once()
        kwargs = mock_run.call_args[1]
        assert kwargs["on_phase_failure"] is not None

    @patch("spark_runner.orchestrator.run_single")
    @patch("spark_runner.interactive.asyncio")
    def test_run_without_hints_flag_no_callback(
        self, mock_asyncio: MagicMock, mock_run: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("run", ["login"], config, _identity)
        mock_run.assert_called_once()
        # on_phase_failure should not be in kwargs (or be None)
        kwargs = mock_run.call_args[1]
        assert kwargs.get("on_phase_failure") is None

    @patch("spark_runner.orchestrator.run_multiple")
    @patch("spark_runner.interactive.asyncio")
    def test_run_multiple_hints_flag_passes_callback(
        self, mock_asyncio: MagicMock, mock_run_multi: MagicMock,
        tmp_path: Path,
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        _write_goal(config.goal_summaries_dir, "logout")
        dispatch("run", ["login", "logout", "--hints"], config, _identity)
        mock_run_multi.assert_called_once()
        kwargs = mock_run_multi.call_args[1]
        assert kwargs["on_phase_failure"] is not None

    def test_completes_run_hints_flag(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        completer = SparkCompleter(config)
        from prompt_toolkit.document import Document
        doc = Document("run --", len("run --"))
        results = [c.text for c in completer.get_completions(doc, None)]
        assert "--hints" in results


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


# ── reset commands ────────────────────────────────────────────────


class TestResetCommands:
    def test_reset_command_marks_phase(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(
            config.goal_summaries_dir, "login",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        dispatch("reset", ["login", "Fill", "Form"], config, _identity)
        output = capsys.readouterr().out
        assert "marked for fresh decomposition" in output
        data = json.loads((config.goal_summaries_dir / "login-task.json").read_text())
        assert data["reset_phases"] == ["Fill Form"]

    def test_reset_command_unknown_phase(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(
            config.goal_summaries_dir, "login",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        dispatch("reset", ["login", "Nonexistent", "Phase"], config, _identity)
        output = capsys.readouterr().out
        assert "Unknown phase" in output
        assert "Fill Form" in output

    def test_reset_command_goal_not_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("reset", ["nonexistent", "Phase"], config, _identity)
        output = capsys.readouterr().out
        assert "Goal not found" in output

    def test_reset_command_missing_args(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("reset", ["login"], config, _identity)
        output = capsys.readouterr().out
        assert "Usage" in output

    def test_resets_command_lists_reset_phases(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(
            config.goal_summaries_dir, "login",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        # Mark a phase as reset
        goal_path = config.goal_summaries_dir / "login-task.json"
        data = json.loads(goal_path.read_text())
        data["reset_phases"] = ["Fill Form"]
        goal_path.write_text(json.dumps(data))
        dispatch("resets", ["login"], config, _identity)
        output = capsys.readouterr().out
        assert "Fill Form" in output

    def test_resets_command_empty(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("resets", ["login"], config, _identity)
        output = capsys.readouterr().out
        assert "No phases reset" in output

    def test_resets_command_missing_arg(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        dispatch("resets", [], config, _identity)
        output = capsys.readouterr().out
        assert "Usage" in output

    def test_unreset_command_removes_phase(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(
            config.goal_summaries_dir, "login",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        goal_path = config.goal_summaries_dir / "login-task.json"
        data = json.loads(goal_path.read_text())
        data["reset_phases"] = ["Fill Form"]
        goal_path.write_text(json.dumps(data))
        dispatch("unreset", ["login", "Fill", "Form"], config, _identity)
        output = capsys.readouterr().out
        assert "unreset" in output
        data = json.loads(goal_path.read_text())
        assert data["reset_phases"] == []

    def test_unreset_command_nonexistent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        dispatch("unreset", ["login", "Fill", "Form"], config, _identity)
        output = capsys.readouterr().out
        assert "not in the reset list" in output

    def test_completes_goal_names_for_reset(self, tmp_path: Path) -> None:
        from prompt_toolkit.document import Document

        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        _write_goal(config.goal_summaries_dir, "logout")
        completer = SparkCompleter(config)
        doc = Document("reset ", len("reset "))
        results = [c.text for c in completer.get_completions(doc, None)]
        assert "login" in results
        assert "logout" in results

    def test_completes_phase_names_for_reset(self, tmp_path: Path) -> None:
        from prompt_toolkit.document import Document

        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(
            config.goal_summaries_dir, "login",
            subtasks=[
                {"filename": "fill-form.txt"},
                {"filename": "verify-result.txt"},
            ],
        )
        completer = SparkCompleter(config)
        doc = Document("reset login ", len("reset login "))
        results = [c.text for c in completer.get_completions(doc, None)]
        assert "Fill Form" in results
        assert "Verify Result" in results

    def test_completes_goal_names_for_unreset(self, tmp_path: Path) -> None:
        from prompt_toolkit.document import Document

        config = _make_config(tmp_path)
        assert config.goal_summaries_dir is not None
        _write_goal(config.goal_summaries_dir, "login")
        completer = SparkCompleter(config)
        doc = Document("unreset ", len("unreset "))
        results = [c.text for c in completer.get_completions(doc, None)]
        assert "login" in results
