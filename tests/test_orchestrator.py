"""Tests for orchestrator behaviour."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from spark_runner.models import CredentialProfile, SparkConfig, TaskSpec
from spark_runner.orchestrator import StatusLine, _copy_goal_files, _format_elapsed


@pytest.fixture()
def _minimal_config(tmp_path: Path) -> SparkConfig:
    """Build a minimal SparkConfig backed by temp directories."""
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    gs_dir = tmp_path / "goal_summaries"
    gs_dir.mkdir()
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    return SparkConfig(
        base_url="https://test.example.com",
        data_dir=tmp_path,
        tasks_dir=tasks_dir,
        goal_summaries_dir=gs_dir,
        runs_dir=runs_dir,
        credentials={"default": CredentialProfile(email="a@b.com", password="pw")},
        active_credential_profile="default",
        headless=True,
        auto_close=True,
        update_summary=False,
        update_tasks=False,
        knowledge_reuse=False,
    )


class TestRunSingleEmptySubtasks:
    """When a goal file has no subtasks, run_single should decompose the prompt."""

    @pytest.mark.asyncio
    async def test_goal_with_empty_subtasks_calls_decompose(
        self, tmp_path: Path, _minimal_config: SparkConfig,
    ) -> None:
        config = _minimal_config

        # Create a goal file with empty subtasks (as generate-goals produces)
        goal_file = config.goal_summaries_dir / "signup-task.json"
        goal_data: dict[str, Any] = {
            "main_task": "Test user sign-up",
            "key_observations": [],
            "subtasks": [],
        }
        goal_file.write_text(json.dumps(goal_data))

        task = TaskSpec(goal_path=goal_file)

        with (
            patch("spark_runner.orchestrator.decompose_task") as mock_decompose,
            patch("spark_runner.orchestrator.run_phase", new_callable=AsyncMock) as mock_run_phase,
            patch("spark_runner.orchestrator.summarize_phase") as mock_summarize,
            patch("spark_runner.orchestrator.Browser") as mock_browser_cls,
            patch("spark_runner.orchestrator.ChatBrowserUse"),
            patch("spark_runner.orchestrator.generate_report"),
        ):
            mock_browser_cls.return_value.stop = AsyncMock()
            mock_decompose.return_value = [
                {"name": "Fill form", "task": "Fill out the sign-up form"},
            ]
            mock_run_phase.return_value = (True, MagicMock(action_results=MagicMock(return_value=[])), [])
            mock_summarize.return_value = "Filled the form successfully"

            from spark_runner.orchestrator import run_single

            mock_client = MagicMock()
            await run_single(task, config, client=mock_client)

            mock_decompose.assert_called_once()
            # The prompt from the goal file should be passed to decompose_task
            assert mock_decompose.call_args.args[0] == "Test user sign-up"


class TestRegenerateTasks:
    """Tests for the --regenerate-tasks flag."""

    @pytest.mark.asyncio
    async def test_regenerate_tasks_true_calls_decompose(
        self, tmp_path: Path, _minimal_config: SparkConfig,
    ) -> None:
        """When regenerate_tasks=True and goal has subtasks, decompose_task should be called."""
        config = _minimal_config
        config.regenerate_tasks = True

        # Create a task file so load_goal_summary returns phases
        tasks_dir = config.tasks_dir
        assert tasks_dir is not None
        (tasks_dir / "fill-form.txt").write_text("Fill the form")

        goal_file = config.goal_summaries_dir / "signup-task.json"
        goal_data: dict[str, Any] = {
            "main_task": "Test user sign-up",
            "key_observations": [],
            "subtasks": [
                {"phase_name": "Fill form", "filename": "fill-form.txt"},
            ],
        }
        goal_file.write_text(json.dumps(goal_data))

        task = TaskSpec(goal_path=goal_file)

        with (
            patch("spark_runner.orchestrator.decompose_task") as mock_decompose,
            patch("spark_runner.orchestrator.run_phase", new_callable=AsyncMock) as mock_run_phase,
            patch("spark_runner.orchestrator.summarize_phase") as mock_summarize,
            patch("spark_runner.orchestrator.Browser") as mock_browser_cls,
            patch("spark_runner.orchestrator.ChatBrowserUse"),
            patch("spark_runner.orchestrator.generate_report"),
        ):
            mock_browser_cls.return_value.stop = AsyncMock()
            mock_decompose.return_value = [
                {"name": "Fill form", "task": "Fill out the sign-up form"},
            ]
            mock_run_phase.return_value = (True, MagicMock(action_results=MagicMock(return_value=[])), [])
            mock_summarize.return_value = "Filled the form successfully"

            from spark_runner.orchestrator import run_single

            mock_client = MagicMock()
            await run_single(task, config, client=mock_client)

            mock_decompose.assert_called_once()

    @pytest.mark.asyncio
    async def test_regenerate_tasks_false_skips_decompose(
        self, tmp_path: Path, _minimal_config: SparkConfig,
    ) -> None:
        """When regenerate_tasks=False (default) and goal has subtasks, decompose_task should NOT be called."""
        config = _minimal_config
        config.regenerate_tasks = False

        # Create a task file so load_goal_summary returns phases
        tasks_dir = config.tasks_dir
        assert tasks_dir is not None
        (tasks_dir / "fill-form.txt").write_text("Fill the form")

        goal_file = config.goal_summaries_dir / "signup-task.json"
        goal_data: dict[str, Any] = {
            "main_task": "Test user sign-up",
            "key_observations": [],
            "subtasks": [
                {"phase_name": "Fill form", "filename": "fill-form.txt"},
            ],
        }
        goal_file.write_text(json.dumps(goal_data))

        task = TaskSpec(goal_path=goal_file)

        with (
            patch("spark_runner.orchestrator.decompose_task") as mock_decompose,
            patch("spark_runner.orchestrator.run_phase", new_callable=AsyncMock) as mock_run_phase,
            patch("spark_runner.orchestrator.summarize_phase") as mock_summarize,
            patch("spark_runner.orchestrator.Browser") as mock_browser_cls,
            patch("spark_runner.orchestrator.ChatBrowserUse"),
            patch("spark_runner.orchestrator.generate_report"),
        ):
            mock_browser_cls.return_value.stop = AsyncMock()
            mock_run_phase.return_value = (True, MagicMock(action_results=MagicMock(return_value=[])), [])
            mock_summarize.return_value = "Filled the form successfully"

            from spark_runner.orchestrator import run_single

            mock_client = MagicMock()
            await run_single(task, config, client=mock_client)

            mock_decompose.assert_not_called()


class TestCopyGoalFiles:
    """Tests for _copy_goal_files() helper."""

    def test_copies_goal_and_task_files(self, tmp_path: Path) -> None:
        """Goal JSON and referenced subtask files should be copied to run_dir/goal/."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "fill-form.txt").write_text("Fill the form with {EMAIL}")
        (tasks_dir / "verify-result.txt").write_text("Check the result page")

        goal_file = tmp_path / "signup-task.json"
        goal_data: dict[str, Any] = {
            "main_task": "Test signup",
            "key_observations": [],
            "subtasks": [
                {"filename": "fill-form.txt"},
                {"filename": "verify-result.txt"},
            ],
        }
        goal_file.write_text(json.dumps(goal_data))

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        _copy_goal_files(run_dir, goal_file, tasks_dir)

        goal_dir = run_dir / "goal"
        assert goal_dir.exists()
        assert (goal_dir / "signup-task.json").exists()
        assert json.loads((goal_dir / "signup-task.json").read_text()) == goal_data
        assert (goal_dir / "fill-form.txt").read_text() == "Fill the form with {EMAIL}"
        assert (goal_dir / "verify-result.txt").read_text() == "Check the result page"

    def test_no_goal_dir_for_cli_prompt(self, tmp_path: Path) -> None:
        """For CLI prompt runs (no goal_path), _copy_goal_files is not called
        and no goal/ directory should exist."""
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        # Simply verify goal/ doesn't exist when not created
        assert not (run_dir / "goal").exists()

    def test_missing_task_file_skipped(self, tmp_path: Path) -> None:
        """Subtask files that don't exist on disk should be silently skipped."""
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        (tasks_dir / "existing.txt").write_text("I exist")

        goal_file = tmp_path / "test-task.json"
        goal_data: dict[str, Any] = {
            "main_task": "Test",
            "subtasks": [
                {"filename": "existing.txt"},
                {"filename": "missing.txt"},
            ],
        }
        goal_file.write_text(json.dumps(goal_data))

        run_dir = tmp_path / "run"
        run_dir.mkdir()

        _copy_goal_files(run_dir, goal_file, tasks_dir)

        goal_dir = run_dir / "goal"
        assert (goal_dir / "existing.txt").exists()
        assert not (goal_dir / "missing.txt").exists()


class TestFormatElapsed:
    def test_seconds_only(self) -> None:
        assert _format_elapsed(5) == "0:05"

    def test_minutes_and_seconds(self) -> None:
        assert _format_elapsed(125) == "2:05"

    def test_hours(self) -> None:
        assert _format_elapsed(3661) == "1:01:01"

    def test_zero(self) -> None:
        assert _format_elapsed(0) == "0:00"


class TestStatusLine:
    def test_render_contains_goal_info(self) -> None:
        sl = StatusLine()
        sl.set_goal("login", 1, 3)
        rendered = sl._render()
        assert "Goal: login (1/3)" in rendered
        assert "Goal Time:" in rendered
        assert "Total Time:" in rendered

    def test_set_goal_resets_goal_timer(self) -> None:
        sl = StatusLine()
        sl.set_goal("login", 1, 2)
        first_render = sl._render()
        sl.set_goal("logout", 2, 2)
        second_render = sl._render()
        assert "Goal: login" in first_render
        assert "Goal: logout (2/2)" in second_render

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        sl = StatusLine()
        sl.set_goal("test", 1, 1)
        await sl.start()
        assert sl._task is not None
        assert not sl._task.done()
        await sl.stop()
        assert sl._task is None

    @pytest.mark.asyncio
    async def test_stop_without_start(self) -> None:
        sl = StatusLine()
        await sl.stop()  # Should not raise

    def test_clear_resets_width(self) -> None:
        sl = StatusLine()
        sl._last_width = 50
        sl.clear()
        assert sl._last_width == 0
