"""Tests for CLI option parsing and error handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import click.testing
import pytest

from spark_runner.cli import cli, _parse_model_overrides
from spark_runner.models import EnvironmentProfile


@pytest.fixture()
def runner() -> click.testing.CliRunner:
    return click.testing.CliRunner()


# ── Top-level CLI ────────────────────────────────────────────────────────


class TestTopLevelCLI:
    def test_no_args_shows_help_when_config_exists(
        self, runner: click.testing.CliRunner, tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("general:\n  base_url: https://example.com\n")
        with patch("spark_runner.cli.resolve_config_path", return_value=config_path):
            result = runner.invoke(cli, [])
        assert result.exit_code == 0
        assert "Spark Runner" in result.output

    def test_no_args_prompts_and_runs_init_when_no_config(
        self, runner: click.testing.CliRunner, tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "no_such_config.yaml"
        with patch("spark_runner.cli.resolve_config_path", return_value=config_path), \
             patch("spark_runner.cli.run_setup_wizard", return_value=config_path) as mock_wiz:
            result = runner.invoke(cli, [], input="y\n")
        assert result.exit_code == 0
        assert "Set up spark-runner now?" in result.output
        mock_wiz.assert_called_once_with(config_path)

    def test_no_args_declines_init_shows_help(
        self, runner: click.testing.CliRunner, tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "no_such_config.yaml"
        with patch("spark_runner.cli.resolve_config_path", return_value=config_path), \
             patch("spark_runner.cli.run_setup_wizard") as mock_wiz:
            result = runner.invoke(cli, [], input="n\n")
        assert result.exit_code == 0
        mock_wiz.assert_not_called()
        assert "Spark Runner" in result.output
        assert "--help" in result.output

    def test_help_flag(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "goals" in result.output
        assert "results" in result.output

    def test_help_shows_data_dir(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert "--data-dir" in result.output
        assert "Spark Runner home directory" in result.output

    def test_help_shows_config(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert "--config" in result.output

    def test_unknown_subcommand(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["nonexistent"])
        assert result.exit_code != 0


# ── --data-dir propagation ───────────────────────────────────────────────


class TestDataDirPropagation:
    """--data-dir on the top-level group propagates to all subcommands."""

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_data_dir_propagates_to_run(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        mock_config.return_value = MagicMock(base_url="https://x.com")
        result = runner.invoke(
            cli, ["--data-dir", str(tmp_path), "run", "-p", "test"]
        )
        assert result.exit_code == 0
        assert mock_config.call_args.kwargs["data_dir"] == tmp_path

    @patch("spark_runner.cli.build_config")
    def test_data_dir_propagates_to_goals_list(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        mock_cfg = MagicMock()
        mock_cfg.goal_summaries_dir = tmp_path
        mock_config.return_value = mock_cfg
        with patch("spark_runner.goals.list_goals"):
            result = runner.invoke(
                cli, ["--data-dir", str(tmp_path), "goals", "list"]
            )
        assert result.exit_code == 0
        assert mock_config.call_args.kwargs["data_dir"] == tmp_path

    @patch("spark_runner.cli.build_config")
    def test_data_dir_propagates_to_results_list(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        mock_cfg = MagicMock()
        mock_cfg.runs_dir = tmp_path
        mock_config.return_value = mock_cfg
        with patch("spark_runner.results.list_runs", return_value=[]):
            result = runner.invoke(
                cli, ["--data-dir", str(tmp_path), "results", "list"]
            )
        assert result.exit_code == 0
        assert mock_config.call_args.kwargs["data_dir"] == tmp_path

    @patch("spark_runner.cli.build_config")
    def test_data_dir_propagates_to_results_errors(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        mock_cfg = MagicMock()
        mock_cfg.runs_dir = tmp_path
        mock_config.return_value = mock_cfg
        with patch("spark_runner.results.list_runs", return_value=[]):
            result = runner.invoke(
                cli, ["--data-dir", str(tmp_path), "results", "errors"]
            )
        assert result.exit_code == 0
        assert mock_config.call_args.kwargs["data_dir"] == tmp_path

    @patch("spark_runner.cli.build_config")
    def test_data_dir_propagates_to_goals_classify(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        mock_cfg = MagicMock()
        mock_cfg.goal_summaries_dir = tmp_path
        mock_cfg.get_model.return_value = MagicMock()
        mock_config.return_value = mock_cfg
        with patch("anthropic.Anthropic"), \
             patch("spark_runner.goals.classify_existing_goals"):
            result = runner.invoke(
                cli, ["--data-dir", str(tmp_path), "goals", "classify"]
            )
        assert result.exit_code == 0
        assert mock_config.call_args.kwargs["data_dir"] == tmp_path

    @patch("spark_runner.cli.build_config")
    def test_data_dir_propagates_to_goals_orphans(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        mock_cfg = MagicMock()
        mock_cfg.tasks_dir = tmp_path
        mock_cfg.goal_summaries_dir = tmp_path
        mock_config.return_value = mock_cfg
        with patch("spark_runner.storage.find_orphan_tasks"):
            result = runner.invoke(
                cli, ["--data-dir", str(tmp_path), "goals", "orphans"]
            )
        assert result.exit_code == 0
        assert mock_config.call_args.kwargs["data_dir"] == tmp_path

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_no_data_dir_passes_none(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
    ) -> None:
        mock_config.return_value = MagicMock(base_url="https://x.com")
        result = runner.invoke(cli, ["run", "-p", "test"])
        assert result.exit_code == 0
        assert mock_config.call_args.kwargs["data_dir"] is None


# ── run: URL validation ─────────────────────────────────────────────────


class TestRunURLValidation:
    def test_url_missing_value_eats_next_flag(
        self, runner: click.testing.CliRunner
    ) -> None:
        """``-u -p`` should error because -p looks like a flag, not a URL."""
        result = runner.invoke(cli, ["run", "-u", "-p", "some prompt"])
        assert result.exit_code != 0
        assert "looks like a flag" in result.output
        assert "--url" in result.output

    def test_url_missing_value_eats_long_flag(
        self, runner: click.testing.CliRunner
    ) -> None:
        result = runner.invoke(cli, ["run", "-u", "--headless"])
        assert result.exit_code != 0
        assert "looks like a flag" in result.output

    def test_url_missing_value_eats_double_dash_flag(
        self, runner: click.testing.CliRunner
    ) -> None:
        result = runner.invoke(cli, ["run", "--url", "--auto-close"])
        assert result.exit_code != 0
        assert "looks like a flag" in result.output

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_url_with_valid_value_accepted(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
    ) -> None:
        mock_cfg = MagicMock()
        mock_cfg.base_url = "https://example.com"
        mock_config.return_value = mock_cfg
        result = runner.invoke(
            cli, ["run", "-u", "https://example.com", "-p", "test"]
        )
        assert result.exit_code == 0
        mock_config.assert_called_once()
        assert mock_config.call_args.kwargs["base_url"] == "https://example.com"


# ── run: goal file validation ────────────────────────────────────────────


class TestRunGoalFileValidation:
    @patch("spark_runner.cli.build_config")
    def test_nonexistent_goal_file(
        self, mock_config: MagicMock, runner: click.testing.CliRunner,
    ) -> None:
        mock_config.return_value = MagicMock(
            base_url="https://x.com", goal_summaries_dir=None, active_environment=None,
        )
        result = runner.invoke(cli, ["run", "/no/such/file.json"])
        assert result.exit_code != 0
        assert "Goal file not found" in result.output

    @patch("spark_runner.cli.build_config")
    def test_nonexistent_goal_file_with_prompt(
        self, mock_config: MagicMock, runner: click.testing.CliRunner,
    ) -> None:
        mock_config.return_value = MagicMock(
            base_url="https://x.com", goal_summaries_dir=None, active_environment=None,
        )
        result = runner.invoke(
            cli, ["run", "-p", "test", "/no/such/goal.json"]
        )
        assert result.exit_code != 0
        assert "Goal file not found" in result.output

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_existing_goal_file_accepted(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        goal = tmp_path / "goal.json"
        goal.write_text("{}")
        mock_config.return_value = MagicMock(base_url="https://x.com")
        result = runner.invoke(cli, ["run", str(goal)])
        assert result.exit_code == 0

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_goal_name_resolved_from_goal_summaries_dir(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        """A bare goal name resolves to a file in goal_summaries_dir."""
        gs_dir = tmp_path / "goal_summaries"
        gs_dir.mkdir()
        goal = gs_dir / "my-task.json"
        goal.write_text("{}")
        mock_cfg = MagicMock(base_url="https://x.com", goal_summaries_dir=gs_dir)
        mock_config.return_value = mock_cfg
        result = runner.invoke(cli, ["run", "my-task.json"])
        assert result.exit_code == 0
        task_spec = mock_run.call_args.args[0]
        assert task_spec.goal_path == goal

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_goal_name_without_extension_resolved(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        """A bare goal name without .json still resolves."""
        gs_dir = tmp_path / "goal_summaries"
        gs_dir.mkdir()
        goal = gs_dir / "login-test.json"
        goal.write_text("{}")
        mock_cfg = MagicMock(base_url="https://x.com", goal_summaries_dir=gs_dir)
        mock_config.return_value = mock_cfg
        result = runner.invoke(cli, ["run", "login-test"])
        assert result.exit_code == 0
        task_spec = mock_run.call_args.args[0]
        assert task_spec.goal_path == goal

    @patch("spark_runner.cli.build_config")
    def test_goal_name_not_found_in_goal_summaries_dir(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        """A name that doesn't exist anywhere still errors."""
        gs_dir = tmp_path / "goal_summaries"
        gs_dir.mkdir()
        mock_cfg = MagicMock(base_url="https://x.com", goal_summaries_dir=gs_dir)
        mock_config.return_value = mock_cfg
        result = runner.invoke(cli, ["run", "nonexistent-goal"])
        assert result.exit_code != 0
        assert "Goal file not found" in result.output

    @patch("spark_runner.cli.build_config")
    def test_close_match_shows_did_you_mean(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        """A misspelled goal name suggests close matches."""
        gs_dir = tmp_path / "goal_summaries"
        gs_dir.mkdir()
        (gs_dir / "login-task.json").write_text("{}")
        (gs_dir / "checkout-task.json").write_text("{}")
        mock_cfg = MagicMock(base_url="https://x.com", goal_summaries_dir=gs_dir)
        mock_config.return_value = mock_cfg
        result = runner.invoke(cli, ["run", "logn-task"])
        assert result.exit_code != 0
        assert "Did you mean" in result.output
        assert "login-task" in result.output

    @patch("spark_runner.cli.build_config")
    def test_no_close_match_shows_available_goals(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        """A completely unrelated name lists all available goals."""
        gs_dir = tmp_path / "goal_summaries"
        gs_dir.mkdir()
        (gs_dir / "checkout-task.json").write_text("{}")
        mock_cfg = MagicMock(base_url="https://x.com", goal_summaries_dir=gs_dir)
        mock_config.return_value = mock_cfg
        result = runner.invoke(cli, ["run", "zzzzz"])
        assert result.exit_code != 0
        assert "Available goals" in result.output
        assert "checkout-task.json" in result.output

    @patch("spark_runner.cli.build_config")
    def test_no_goals_at_all_no_suggestions(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        """An empty goal dir shows no suggestions section."""
        gs_dir = tmp_path / "goal_summaries"
        gs_dir.mkdir()
        mock_cfg = MagicMock(base_url="https://x.com", goal_summaries_dir=gs_dir)
        mock_config.return_value = mock_cfg
        result = runner.invoke(cli, ["run", "anything"])
        assert result.exit_code != 0
        assert "Goal file not found" in result.output
        assert "Did you mean" not in result.output
        assert "Available goals" not in result.output


# ── run: --model validation ──────────────────────────────────────────────


class TestRunModelOption:
    def test_model_missing_equals(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(
            cli, ["run", "-p", "test", "--model", "bad-value"]
        )
        assert result.exit_code != 0
        assert "PURPOSE=MODEL_ID" in result.output

    def test_model_valid_format(self) -> None:
        result = _parse_model_overrides(("browser_control=claude-sonnet-4-5-20250929",))
        assert result == {"browser_control": "claude-sonnet-4-5-20250929"}

    def test_model_multiple_overrides(self) -> None:
        result = _parse_model_overrides((
            "browser_control=claude-sonnet-4-5-20250929",
            "summarization=claude-haiku-35-20241022",
        ))
        assert result == {
            "browser_control": "claude-sonnet-4-5-20250929",
            "summarization": "claude-haiku-35-20241022",
        }

    def test_model_value_with_equals(self) -> None:
        """Model IDs with ``=`` in the value should be preserved."""
        result = _parse_model_overrides(("key=val=ue",))
        assert result == {"key": "val=ue"}

    def test_model_empty_purpose_raises(self) -> None:
        with pytest.raises(click.BadParameter, match="PURPOSE=MODEL_ID"):
            _parse_model_overrides(("no-equals-sign",))


# ── run: --parallel validation ───────────────────────────────────────────


class TestRunParallelOption:
    def test_parallel_non_integer(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(
            cli, ["run", "-p", "test", "--parallel", "abc"]
        )
        assert result.exit_code != 0
        assert "not a valid integer" in result.output.lower() or "invalid" in result.output.lower()

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_parallel_valid_integer(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
    ) -> None:
        mock_config.return_value = MagicMock(base_url="https://x.com")
        result = runner.invoke(cli, ["run", "-p", "test", "--parallel", "4"])
        assert result.exit_code == 0


# ── run: prompt handling ─────────────────────────────────────────────────


class TestRunPromptHandling:
    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_single_prompt(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
    ) -> None:
        mock_config.return_value = MagicMock(base_url="https://x.com")
        result = runner.invoke(cli, ["run", "-p", "do a thing"])
        assert result.exit_code == 0
        mock_run.assert_called_once()

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_multiple", new_callable=AsyncMock)
    def test_multiple_prompts(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
    ) -> None:
        mock_config.return_value = MagicMock(base_url="https://x.com")
        result = runner.invoke(
            cli, ["run", "-p", "first", "-p", "second"]
        )
        assert result.exit_code == 0
        mock_run.assert_called_once()
        tasks = mock_run.call_args.args[0]
        assert len(tasks) == 2
        assert tasks[0].prompt == "first"
        assert tasks[1].prompt == "second"

    @patch("spark_runner.cli.build_config")
    def test_no_prompt_no_goal_prompts_interactively(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
    ) -> None:
        mock_config.return_value = MagicMock(base_url="https://x.com")
        result = runner.invoke(cli, ["run"], input="my interactive prompt\n")
        assert result.exit_code == 0 or "Enter your task" in result.output


# ── run: flag combinations ───────────────────────────────────────────────


class TestRunFlagCombinations:
    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_boolean_flags_passed_to_config(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
    ) -> None:
        mock_config.return_value = MagicMock(base_url="https://x.com")
        result = runner.invoke(cli, [
            "run", "-p", "test",
            "--no-update-summary",
            "--no-update-tasks",
            "--no-knowledge-reuse",
            "--auto-close",
            "--headless",
        ])
        assert result.exit_code == 0
        kwargs = mock_config.call_args.kwargs
        assert kwargs["update_summary"] is False
        assert kwargs["update_tasks"] is False
        assert kwargs["knowledge_reuse"] is False
        assert kwargs["auto_close"] is True
        assert kwargs["headless"] is True

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_default_flags(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
    ) -> None:
        mock_config.return_value = MagicMock(base_url="https://x.com")
        result = runner.invoke(cli, ["run", "-p", "test"])
        assert result.exit_code == 0
        kwargs = mock_config.call_args.kwargs
        assert kwargs["update_summary"] is True
        assert kwargs["update_tasks"] is True
        assert kwargs["knowledge_reuse"] is True
        assert kwargs["auto_close"] is False
        assert kwargs["headless"] is False


# ── run: --unrun / --failed filters ──────────────────────────────────────


class TestRunFilterFlags:
    @patch("spark_runner.cli.build_config")
    def test_unrun_no_matches_exits_with_error(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        """--unrun with no unrun goals should error, not prompt interactively."""
        gs_dir = tmp_path / "goal_summaries"
        gs_dir.mkdir()
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        # Create a goal that has a run directory (so it's not "unrun")
        (gs_dir / "login-task.json").write_text("{}")
        task_run_dir = runs_dir / "login" / "2025-01-01_00-00-00"
        task_run_dir.mkdir(parents=True)
        mock_cfg = MagicMock(
            base_url="https://x.com",
            goal_summaries_dir=gs_dir,
            runs_dir=runs_dir,
        )
        mock_config.return_value = mock_cfg
        result = runner.invoke(cli, ["run", "--unrun"])
        assert result.exit_code != 0
        assert "No matching goals found" in result.output

    @patch("spark_runner.cli.build_config")
    def test_failed_no_matches_exits_with_error(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        """--failed with no failed goals should error, not prompt interactively."""
        gs_dir = tmp_path / "goal_summaries"
        gs_dir.mkdir()
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        mock_cfg = MagicMock(
            base_url="https://x.com",
            goal_summaries_dir=gs_dir,
            runs_dir=runs_dir,
        )
        mock_config.return_value = mock_cfg
        result = runner.invoke(cli, ["run", "--failed"])
        assert result.exit_code != 0
        assert "No matching goals found" in result.output

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_unrun_discovers_never_run_goals(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        """--unrun should select goals with no run directory."""
        gs_dir = tmp_path / "goal_summaries"
        gs_dir.mkdir()
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        (gs_dir / "new-goal-task.json").write_text("{}")
        mock_cfg = MagicMock(
            base_url="https://x.com",
            goal_summaries_dir=gs_dir,
            runs_dir=runs_dir,
        )
        mock_config.return_value = mock_cfg
        result = runner.invoke(cli, ["run", "--unrun"])
        assert result.exit_code == 0
        task_spec = mock_run.call_args.args[0]
        assert task_spec.goal_path == gs_dir / "new-goal-task.json"


# ── run: unknown options ─────────────────────────────────────────────────


class TestRunUnknownOptions:
    def test_unknown_flag(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--nonexistent-flag"])
        assert result.exit_code != 0
        assert "No such option" in result.output or "no such option" in result.output


# ── goals subcommand errors ──────────────────────────────────────────────


class TestGoalsSubcommandErrors:
    def test_goals_no_subcommand_shows_usage(
        self, runner: click.testing.CliRunner
    ) -> None:
        result = runner.invoke(cli, ["goals"])
        assert result.exit_code != 0
        assert "Usage" in result.output

    def test_goals_show_missing_name(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["goals", "show"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_goals_delete_missing_name(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["goals", "delete"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_goals_unknown_subcommand(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["goals", "nonexistent"])
        assert result.exit_code != 0


# ── results subcommand errors ────────────────────────────────────────────


class TestResultsSubcommandErrors:
    def test_results_no_subcommand_shows_usage(
        self, runner: click.testing.CliRunner
    ) -> None:
        result = runner.invoke(cli, ["results"])
        assert result.exit_code != 0
        assert "Usage" in result.output

    @patch("spark_runner.cli.build_config")
    def test_results_show_missing_path_lists_runs(
        self, mock_config: MagicMock, runner: click.testing.CliRunner, tmp_path: Path
    ) -> None:
        """Missing RUN_PATH lists available runs."""
        runs_dir = tmp_path / "runs"
        task_dir = runs_dir / "my-task"
        run_dir = task_dir / "2026-01-01_12-00-00"
        run_dir.mkdir(parents=True)
        mock_config.return_value = MagicMock(runs_dir=runs_dir)
        result = runner.invoke(cli, ["results", "show"])
        assert result.exit_code != 0
        assert "my-task/2026-01-01_12-00-00" in result.output

    @patch("spark_runner.cli.build_config")
    def test_results_show_missing_path_no_runs(
        self, mock_config: MagicMock, runner: click.testing.CliRunner, tmp_path: Path
    ) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        mock_config.return_value = MagicMock(runs_dir=runs_dir)
        result = runner.invoke(cli, ["results", "show"])
        assert result.exit_code != 0
        assert "No runs found" in result.output

    @patch("spark_runner.cli.build_config")
    def test_results_show_nonexistent_path_lists_runs(
        self, mock_config: MagicMock, runner: click.testing.CliRunner, tmp_path: Path
    ) -> None:
        """Bad RUN_PATH shows available runs in the error."""
        runs_dir = tmp_path / "runs"
        task_dir = runs_dir / "login-task"
        run_dir = task_dir / "2026-02-01_08-00-00"
        run_dir.mkdir(parents=True)
        mock_config.return_value = MagicMock(runs_dir=runs_dir)
        result = runner.invoke(cli, ["results", "show", "bad/path"])
        assert result.exit_code != 0
        assert "Run not found" in result.output
        assert "login-task/2026-02-01_08-00-00" in result.output

    @patch("spark_runner.cli.build_config")
    def test_results_show_resolves_relative_to_runs_dir(
        self, mock_config: MagicMock, runner: click.testing.CliRunner, tmp_path: Path
    ) -> None:
        """A relative run path like 'task/timestamp' resolves inside runs_dir."""
        runs_dir = tmp_path / "runs"
        run_dir = runs_dir / "my-task" / "2026-03-03_09-00-00"
        run_dir.mkdir(parents=True)
        mock_config.return_value = MagicMock(runs_dir=runs_dir)
        with patch("spark_runner.results.get_run_detail") as mock_detail, \
             patch("spark_runner.results.format_run_detail", return_value="detail"):
            mock_detail.return_value = MagicMock(
                screenshots=[], phases=[], task_name="my-task",
            )
            result = runner.invoke(
                cli, ["results", "show", "my-task/2026-03-03_09-00-00"]
            )
        assert result.exit_code == 0
        mock_detail.assert_called_once_with(run_dir)

    @patch("spark_runner.cli.build_config")
    def test_results_screenshots_missing_path_lists_runs(
        self, mock_config: MagicMock, runner: click.testing.CliRunner, tmp_path: Path
    ) -> None:
        runs_dir = tmp_path / "runs"
        task_dir = runs_dir / "search-task"
        run_dir = task_dir / "2026-01-15_10-30-00"
        run_dir.mkdir(parents=True)
        mock_config.return_value = MagicMock(runs_dir=runs_dir)
        result = runner.invoke(cli, ["results", "screenshots"])
        assert result.exit_code != 0
        assert "search-task/2026-01-15_10-30-00" in result.output

    @patch("spark_runner.cli.build_config")
    def test_results_screenshots_nonexistent_path(
        self, mock_config: MagicMock, runner: click.testing.CliRunner, tmp_path: Path
    ) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        mock_config.return_value = MagicMock(runs_dir=runs_dir)
        result = runner.invoke(cli, ["results", "screenshots", "/no/such/path"])
        assert result.exit_code != 0
        assert "Run not found" in result.output


# ── run help ─────────────────────────────────────────────────────────────


class TestRunHelp:
    def test_run_help(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--prompt" in result.output
        assert "--url" in result.output
        assert "--auto-close" in result.output
        assert "--headless" in result.output
        assert "--model" in result.output
        assert "--parallel" in result.output
        assert "--shared-session" in result.output
        assert "--credential-profile" in result.output


# ── legacy_main ──────────────────────────────────────────────────────────


class TestLegacyMain:
    @patch("spark_runner.cli.cli")
    def test_list_goals_flag(self, mock_cli: MagicMock) -> None:
        from spark_runner.cli import legacy_main

        import sys
        original = sys.argv[:]
        try:
            sys.argv = ["spark_runner.py", "--list-goals"]
            legacy_main()
            assert sys.argv == ["spark_runner.py", "goals", "list"]
            mock_cli.assert_called_once()
        finally:
            sys.argv = original

    @patch("spark_runner.cli.cli")
    def test_classify_existing_flag(self, mock_cli: MagicMock) -> None:
        from spark_runner.cli import legacy_main

        import sys
        original = sys.argv[:]
        try:
            sys.argv = ["spark_runner.py", "--classify-existing"]
            legacy_main()
            assert sys.argv == ["spark_runner.py", "goals", "classify"]
            mock_cli.assert_called_once()
        finally:
            sys.argv = original

    @patch("spark_runner.cli.cli")
    def test_find_orphans_flag(self, mock_cli: MagicMock) -> None:
        from spark_runner.cli import legacy_main

        import sys
        original = sys.argv[:]
        try:
            sys.argv = ["spark_runner.py", "--find-orphans"]
            legacy_main()
            assert sys.argv == ["spark_runner.py", "goals", "orphans"]
            mock_cli.assert_called_once()
        finally:
            sys.argv = original

    @patch("spark_runner.cli.cli")
    def test_clean_orphans_flag(self, mock_cli: MagicMock) -> None:
        from spark_runner.cli import legacy_main

        import sys
        original = sys.argv[:]
        try:
            sys.argv = ["spark_runner.py", "--clean-orphans"]
            legacy_main()
            assert sys.argv == ["spark_runner.py", "goals", "orphans", "--clean"]
            mock_cli.assert_called_once()
        finally:
            sys.argv = original

    @patch("spark_runner.cli.cli")
    def test_prompt_flag_maps_to_run(self, mock_cli: MagicMock) -> None:
        from spark_runner.cli import legacy_main

        import sys
        original = sys.argv[:]
        try:
            sys.argv = ["spark_runner.py", "-p", "do stuff", "--auto-close"]
            legacy_main()
            assert sys.argv == [
                "spark_runner.py", "run", "-p", "do stuff", "--auto-close"
            ]
            mock_cli.assert_called_once()
        finally:
            sys.argv = original


# ── results report --all ─────────────────────────────────────────────────


class TestResultsReportAll:
    @patch("spark_runner.cli.build_config")
    def test_results_report_all(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        """--all regenerates reports for every run."""
        runs_dir = tmp_path / "runs"
        for task, ts in [("task-a", "2026-01-01_00-00-00"), ("task-b", "2026-01-02_00-00-00")]:
            d = runs_dir / task / ts
            d.mkdir(parents=True)
            (d / "run_metadata.json").write_text("{}")

        mock_config.return_value = MagicMock(runs_dir=runs_dir)

        with patch("spark_runner.report.generate_report") as mock_gen:
            mock_gen.side_effect = lambda rd: rd / "report" / "index.html"
            result = runner.invoke(cli, ["results", "report", "--all"])

        assert result.exit_code == 0
        assert mock_gen.call_count == 2
        assert "2 run(s)" in result.output

    @patch("spark_runner.cli.build_config")
    def test_results_report_all_empty(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        """--all on empty runs dir prints 0."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        mock_config.return_value = MagicMock(runs_dir=runs_dir)

        with patch("spark_runner.report.generate_report") as mock_gen:
            result = runner.invoke(cli, ["results", "report", "--all"])

        assert result.exit_code == 0
        assert mock_gen.call_count == 0
        assert "0 run(s)" in result.output


# ── run: --env / --force-unsafe / safety gate ────────────────────────────


class TestRunEnvironmentAndSafety:
    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_env_flag_passed_to_config(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
    ) -> None:
        mock_config.return_value = MagicMock(
            base_url="https://staging.example.com",
            active_environment="staging",
        )
        result = runner.invoke(cli, ["run", "-p", "test", "--env", "staging"])
        assert result.exit_code == 0
        assert mock_config.call_args.kwargs["env"] == "staging"

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_force_unsafe_flag_passed_to_config(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
    ) -> None:
        mock_config.return_value = MagicMock(
            base_url="https://x.com",
            active_environment=None,
        )
        result = runner.invoke(cli, ["run", "-p", "test", "--force-unsafe"])
        assert result.exit_code == 0
        assert mock_config.call_args.kwargs["force_unsafe"] is True

    @patch("spark_runner.cli.build_config")
    def test_blocked_goal_prints_error(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        import json

        goal = tmp_path / "admin-settings-task.json"
        goal.write_text(json.dumps({
            "main_task": "Change global settings",
            "safety": {
                "blocked_in_production": True,
                "reason": "Modifies global settings that affect all users",
            },
        }))
        mock_cfg = MagicMock(
            base_url="https://app.example.com",
            active_environment="production",
            force_unsafe=False,
            goal_summaries_dir=tmp_path,
            environments={
                "production": EnvironmentProfile(
                    name="production", is_production=True,
                ),
            },
        )
        mock_config.return_value = mock_cfg
        result = runner.invoke(cli, ["run", str(goal)])
        assert result.exit_code != 0
        assert "BLOCKED" in result.output
        assert "admin-settings-task" in result.output
        assert "Modifies global settings" in result.output

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_force_unsafe_overrides_blocked_goal(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        import json

        goal = tmp_path / "admin-settings-task.json"
        goal.write_text(json.dumps({
            "main_task": "Change global settings",
            "safety": {"blocked_in_production": True, "reason": "dangerous"},
        }))
        mock_cfg = MagicMock(
            base_url="https://app.example.com",
            active_environment="production",
            force_unsafe=True,
            goal_summaries_dir=tmp_path,
            environments={
                "production": EnvironmentProfile(
                    name="production", is_production=True,
                ),
            },
        )
        mock_config.return_value = mock_cfg
        result = runner.invoke(cli, ["run", str(goal)])
        assert result.exit_code == 0
        assert "BLOCKED" not in result.output

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_partial_block_continues_with_allowed(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        import json

        blocked_goal = tmp_path / "blocked-task.json"
        blocked_goal.write_text(json.dumps({
            "main_task": "Blocked",
            "safety": {"blocked_in_production": True, "reason": "nope"},
        }))
        safe_goal = tmp_path / "safe-task.json"
        safe_goal.write_text(json.dumps({"main_task": "Safe goal"}))

        mock_cfg = MagicMock(
            base_url="https://app.example.com",
            active_environment="production",
            force_unsafe=False,
            goal_summaries_dir=tmp_path,
            environments={
                "production": EnvironmentProfile(
                    name="production", is_production=True,
                ),
            },
        )
        mock_config.return_value = mock_cfg
        result = runner.invoke(cli, ["run", str(blocked_goal), str(safe_goal)])
        assert result.exit_code == 0
        assert "BLOCKED" in result.output
        # The remaining task should still run (only 1 left, so run_single is used)
        mock_run.assert_called_once()

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_goal_without_safety_runs_in_production(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        import json

        goal = tmp_path / "normal-task.json"
        goal.write_text(json.dumps({"main_task": "Normal task, no safety block"}))
        mock_cfg = MagicMock(
            base_url="https://app.example.com",
            active_environment="production",
            force_unsafe=False,
            goal_summaries_dir=tmp_path,
            environments={
                "production": EnvironmentProfile(
                    name="production", is_production=True,
                ),
            },
        )
        mock_config.return_value = mock_cfg
        result = runner.invoke(cli, ["run", str(goal)])
        assert result.exit_code == 0
        assert "BLOCKED" not in result.output

    @patch("spark_runner.cli.build_config")
    def test_unknown_env_shows_error(
        self,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
    ) -> None:
        mock_config.side_effect = ValueError(
            "Unknown environment 'bad'. Available: dev, staging"
        )
        result = runner.invoke(cli, ["run", "-p", "test", "--env", "bad"])
        assert result.exit_code != 0
        assert "Unknown environment" in result.output

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_banner_shows_environment(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
    ) -> None:
        mock_config.return_value = MagicMock(
            base_url="https://staging.example.com",
            active_environment="staging",
        )
        result = runner.invoke(cli, ["run", "-p", "test", "--env", "staging"])
        assert result.exit_code == 0
        assert "Environment: staging" in result.output


# ── init subcommand ──────────────────────────────────────────────────────


class TestInitSubcommand:
    def test_init_creates_config(
        self, runner: click.testing.CliRunner, tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "config.yaml"
        with patch("spark_runner.cli.resolve_config_path", return_value=config_path), \
             patch("spark_runner.cli.run_setup_wizard", return_value=config_path) as mock_wiz:
            result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0
        mock_wiz.assert_called_once_with(config_path)

    def test_init_asks_before_overwriting(
        self, runner: click.testing.CliRunner, tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("existing: true\n")
        with patch("spark_runner.cli.resolve_config_path", return_value=config_path), \
             patch("spark_runner.cli.run_setup_wizard") as mock_wiz:
            # User declines overwrite
            result = runner.invoke(cli, ["init"], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output
        mock_wiz.assert_not_called()

    def test_init_overwrites_when_confirmed(
        self, runner: click.testing.CliRunner, tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("existing: true\n")
        with patch("spark_runner.cli.resolve_config_path", return_value=config_path), \
             patch("spark_runner.cli.run_setup_wizard", return_value=config_path) as mock_wiz:
            result = runner.invoke(cli, ["init"], input="y\n")
        assert result.exit_code == 0
        mock_wiz.assert_called_once_with(config_path)

    def test_init_force_overwrites_without_asking(
        self, runner: click.testing.CliRunner, tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("existing: true\n")
        with patch("spark_runner.cli.resolve_config_path", return_value=config_path), \
             patch("spark_runner.cli.run_setup_wizard", return_value=config_path) as mock_wiz:
            result = runner.invoke(cli, ["init", "--force"])
        assert result.exit_code == 0
        mock_wiz.assert_called_once_with(config_path)
        assert "Overwrite" not in result.output

    def test_init_shown_in_help(self, runner: click.testing.CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert "init" in result.output


# ── run: first-run auto-detection ────────────────────────────────────────


class TestRunFirstRunDetection:
    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_run_triggers_wizard_when_config_missing_tty(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "no_such_config.yaml"
        mock_config.return_value = MagicMock(base_url="https://x.com")
        with patch("spark_runner.cli.resolve_config_path", return_value=config_path), \
             patch("spark_runner.cli.run_setup_wizard") as mock_wiz, \
             patch("spark_runner.cli._is_interactive", return_value=True):
            result = runner.invoke(cli, ["run", "-p", "test"])
        assert result.exit_code == 0
        assert "No config found" in result.output
        mock_wiz.assert_called_once_with(config_path)

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_run_prints_hint_when_config_missing_non_tty(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "no_such_config.yaml"
        mock_config.return_value = MagicMock(base_url="https://x.com")
        with patch("spark_runner.cli.resolve_config_path", return_value=config_path), \
             patch("spark_runner.cli.run_setup_wizard") as mock_wiz, \
             patch("spark_runner.cli._is_interactive", return_value=False):
            result = runner.invoke(cli, ["run", "-p", "test"])
        assert result.exit_code == 0
        assert "spark-runner init" in result.output
        mock_wiz.assert_not_called()

    @patch("spark_runner.cli.build_config")
    @patch("spark_runner.orchestrator.run_single", new_callable=AsyncMock)
    def test_run_skips_wizard_when_config_exists(
        self,
        mock_run: AsyncMock,
        mock_config: MagicMock,
        runner: click.testing.CliRunner,
        tmp_path: Path,
    ) -> None:
        config_path = tmp_path / "config.yaml"
        config_path.write_text("general:\n  base_url: https://example.com\n")
        mock_config.return_value = MagicMock(base_url="https://example.com")
        with patch("spark_runner.cli.resolve_config_path", return_value=config_path), \
             patch("spark_runner.cli.run_setup_wizard") as mock_wiz:
            result = runner.invoke(cli, ["run", "-p", "test"])
        assert result.exit_code == 0
        assert "No config found" not in result.output
        mock_wiz.assert_not_called()
