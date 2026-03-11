"""Tests for orchestrator behaviour."""

from __future__ import annotations

import json
import sys
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

    def test_render_contains_phase_info(self) -> None:
        sl = StatusLine()
        sl.set_goal("login", 1, 3)
        sl.set_phase("Fill Form", 2, 5)
        rendered = sl._render()
        assert "Goal: login (1/3)" in rendered
        assert "Phase: Fill Form (2/5)" in rendered

    def test_render_contains_status(self) -> None:
        sl = StatusLine()
        sl.set_goal("login", 1, 1)
        sl.set_status("Decomposing task")
        rendered = sl._render()
        assert "Decomposing task" in rendered

    def test_render_phase_and_status(self) -> None:
        sl = StatusLine()
        sl.set_goal("login", 1, 1)
        sl.set_phase("Submit", 1, 3)
        sl.set_status("Summarizing")
        rendered = sl._render()
        assert "Phase: Submit (1/3)" in rendered
        assert "Summarizing" in rendered

    def test_set_phase_clears_status(self) -> None:
        sl = StatusLine()
        sl.set_goal("login", 1, 1)
        sl.set_status("Old status")
        sl.set_phase("New Phase", 1, 2)
        rendered = sl._render()
        assert "Old status" not in rendered
        assert "Phase: New Phase (1/2)" in rendered

    def test_set_goal_clears_phase_and_status(self) -> None:
        sl = StatusLine()
        sl.set_goal("login", 1, 2)
        sl.set_phase("Phase A", 1, 3)
        sl.set_status("Summarizing")
        sl.set_goal("logout", 2, 2)
        rendered = sl._render()
        assert "Phase A" not in rendered
        assert "Summarizing" not in rendered
        assert "Goal: logout (2/2)" in rendered

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

    def test_write_uses_scroll_region_on_tty(self) -> None:
        """When stderr is a TTY, _write should use ANSI cursor positioning with styling."""
        sl = StatusLine()
        sl._is_tty = True
        sl._scroll_region_active = True
        sl._height = 24
        sl.set_goal("login", 1, 1)
        with (
            patch.object(sys, "stderr") as mock_stderr,
            patch("shutil.get_terminal_size", return_value=MagicMock(lines=24, columns=80)),
        ):
            mock_stderr.isatty.return_value = True
            sl._write()
            written = "".join(
                call.args[0] for call in mock_stderr.write.call_args_list
            )
            # Should contain save cursor, move to bottom row, clear line, restore cursor
            assert "\033[s" in written
            assert "\033[24;1H" in written
            assert "\033[K" in written
            assert "\033[u" in written
            # Should have prompt_toolkit-rendered styling (256-color yellow bg, black fg)
            assert "48;5;226" in written   # yellow background
            assert "38;5;16" in written    # black foreground
            assert "\033[0m" in written

    def test_write_falls_back_on_non_tty(self) -> None:
        """When stderr is not a TTY, _write should use \\r fallback."""
        sl = StatusLine()
        sl._is_tty = False
        sl._scroll_region_active = False
        sl.set_goal("login", 1, 1)
        with patch.object(sys, "stderr") as mock_stderr:
            mock_stderr.isatty.return_value = False
            sl._write()
            written = "".join(
                call.args[0] for call in mock_stderr.write.call_args_list
            )
            assert written.startswith("\r")
            assert "\033[s" not in written

    def test_setup_teardown_scroll_region(self) -> None:
        """Verify exact ANSI codes for setup and teardown."""
        sl = StatusLine()
        sl._is_tty = True
        with (
            patch.object(sys, "stderr") as mock_stderr,
            patch("shutil.get_terminal_size", return_value=MagicMock(lines=24, columns=80)),
        ):
            mock_stderr.isatty.return_value = True
            sl._setup_scroll_region()
            assert sl._scroll_region_active is True
            assert sl._height == 24
            setup_written = "".join(
                call.args[0] for call in mock_stderr.write.call_args_list
            )
            # Set scroll region to rows 1..23
            assert "\033[1;23r" in setup_written
            # Move cursor into scrollable area
            assert "\033[23;1H" in setup_written

            mock_stderr.reset_mock()
            sl._teardown_scroll_region()
            assert sl._scroll_region_active is False
            teardown_written = "".join(
                call.args[0] for call in mock_stderr.write.call_args_list
            )
            # Clear bottom row
            assert "\033[24;1H" in teardown_written
            assert "\033[K" in teardown_written
            # Reset scroll region
            assert "\033[r" in teardown_written

    def test_teardown_is_idempotent(self) -> None:
        """Calling teardown twice should not error."""
        sl = StatusLine()
        sl._is_tty = True
        sl._scroll_region_active = False
        # Should be a no-op when not active
        sl._teardown_scroll_region()
        sl._teardown_scroll_region()
        assert sl._scroll_region_active is False

    def test_clear_on_tty_clears_bottom_row(self) -> None:
        """clear() should position cursor at bottom row and clear when scroll region is active."""
        sl = StatusLine()
        sl._is_tty = True
        sl._scroll_region_active = True
        sl._height = 30
        with patch.object(sys, "stderr") as mock_stderr:
            sl.clear()
            written = "".join(
                call.args[0] for call in mock_stderr.write.call_args_list
            )
            assert "\033[s" in written
            assert "\033[30;1H" in written
            assert "\033[K" in written
            assert "\033[u" in written

    def test_resize_updates_scroll_region(self) -> None:
        """_handle_resize should teardown old region and setup new one."""
        sl = StatusLine()
        sl._is_tty = True
        sl._scroll_region_active = True
        sl._height = 24
        sl.set_goal("login", 1, 1)
        with (
            patch.object(sys, "stderr") as mock_stderr,
            patch("shutil.get_terminal_size", return_value=MagicMock(lines=30, columns=80)),
        ):
            mock_stderr.isatty.return_value = True
            sl._handle_resize()
            assert sl._height == 30
            assert sl._scroll_region_active is True
            written = "".join(
                call.args[0] for call in mock_stderr.write.call_args_list
            )
            # Teardown old: clear row 24 and reset
            assert "\033[24;1H" in written
            assert "\033[r" in written
            # Setup new: set scroll region 1..29
            assert "\033[1;29r" in written


class TestPhaseFailureCallback:
    """Tests for the on_phase_failure retry mechanism in run_single."""

    @pytest.mark.asyncio
    async def test_phase_failure_calls_callback(
        self, tmp_path: Path, _minimal_config: SparkConfig,
    ) -> None:
        """When a phase fails and callback returns a hint, retry should happen."""
        config = _minimal_config

        tasks_dir = config.tasks_dir
        assert tasks_dir is not None
        (tasks_dir / "fill-form.txt").write_text("Fill the form")

        goal_file = config.goal_summaries_dir / "signup-task.json"
        goal_data = {
            "main_task": "Test user sign-up",
            "key_observations": [],
            "subtasks": [
                {"phase_name": "Fill form", "filename": "fill-form.txt"},
            ],
        }
        goal_file.write_text(json.dumps(goal_data))

        task = TaskSpec(goal_path=goal_file)

        callback = AsyncMock(return_value="Use the dropdown menu")

        with (
            patch("spark_runner.orchestrator.decompose_task") as mock_decompose,
            patch("spark_runner.orchestrator.run_phase", new_callable=AsyncMock) as mock_run_phase,
            patch("spark_runner.orchestrator.summarize_phase") as mock_summarize,
            patch("spark_runner.orchestrator.Browser") as mock_browser_cls,
            patch("spark_runner.orchestrator.ChatBrowserUse"),
            patch("spark_runner.orchestrator.generate_report"),
        ):
            mock_browser_cls.return_value.stop = AsyncMock()
            # First call fails, retry succeeds
            mock_result_fail = MagicMock()
            mock_result_fail.final_result.return_value = "Button not found"
            mock_result_success = MagicMock(action_results=MagicMock(return_value=[]))
            mock_run_phase.side_effect = [
                (False, mock_result_fail, []),
                (True, mock_result_success, []),
            ]
            mock_summarize.return_value = "Phase summary"

            from spark_runner.orchestrator import run_single

            result = await run_single(task, config, client=MagicMock(), on_phase_failure=callback)

            callback.assert_called_once_with("Fill Form", "Button not found")
            assert mock_run_phase.call_count == 2

    @pytest.mark.asyncio
    async def test_phase_failure_callback_none_stops(
        self, tmp_path: Path, _minimal_config: SparkConfig,
    ) -> None:
        """When callback returns None, the run should stop without retry."""
        config = _minimal_config

        tasks_dir = config.tasks_dir
        assert tasks_dir is not None
        (tasks_dir / "fill-form.txt").write_text("Fill the form")

        goal_file = config.goal_summaries_dir / "signup-task.json"
        goal_data = {
            "main_task": "Test user sign-up",
            "key_observations": [],
            "subtasks": [
                {"phase_name": "Fill form", "filename": "fill-form.txt"},
            ],
        }
        goal_file.write_text(json.dumps(goal_data))

        task = TaskSpec(goal_path=goal_file)

        callback = AsyncMock(return_value=None)

        with (
            patch("spark_runner.orchestrator.decompose_task"),
            patch("spark_runner.orchestrator.run_phase", new_callable=AsyncMock) as mock_run_phase,
            patch("spark_runner.orchestrator.summarize_phase") as mock_summarize,
            patch("spark_runner.orchestrator.Browser") as mock_browser_cls,
            patch("spark_runner.orchestrator.ChatBrowserUse"),
            patch("spark_runner.orchestrator.generate_report"),
        ):
            mock_browser_cls.return_value.stop = AsyncMock()
            mock_result = MagicMock()
            mock_result.final_result.return_value = "Failed"
            mock_run_phase.return_value = (False, mock_result, [])
            mock_summarize.return_value = "Phase summary"

            from spark_runner.orchestrator import run_single

            await run_single(task, config, client=MagicMock(), on_phase_failure=callback)

            callback.assert_called_once()
            # Only one phase execution — no retry
            assert mock_run_phase.call_count == 1

    @pytest.mark.asyncio
    async def test_hint_persisted_to_goal_file(
        self, tmp_path: Path, _minimal_config: SparkConfig,
    ) -> None:
        """After callback returns a hint, it should be saved to the goal JSON."""
        config = _minimal_config

        tasks_dir = config.tasks_dir
        assert tasks_dir is not None
        (tasks_dir / "fill-form.txt").write_text("Fill the form")

        goal_file = config.goal_summaries_dir / "signup-task.json"
        goal_data = {
            "main_task": "Test user sign-up",
            "key_observations": [],
            "subtasks": [
                {"phase_name": "Fill form", "filename": "fill-form.txt"},
            ],
        }
        goal_file.write_text(json.dumps(goal_data))

        task = TaskSpec(goal_path=goal_file)

        callback = AsyncMock(return_value="Try the other button")

        with (
            patch("spark_runner.orchestrator.decompose_task"),
            patch("spark_runner.orchestrator.run_phase", new_callable=AsyncMock) as mock_run_phase,
            patch("spark_runner.orchestrator.summarize_phase") as mock_summarize,
            patch("spark_runner.orchestrator.Browser") as mock_browser_cls,
            patch("spark_runner.orchestrator.ChatBrowserUse"),
            patch("spark_runner.orchestrator.generate_report"),
        ):
            mock_browser_cls.return_value.stop = AsyncMock()
            mock_result = MagicMock()
            mock_result.final_result.return_value = "Failed"
            mock_result_success = MagicMock(action_results=MagicMock(return_value=[]))
            mock_run_phase.side_effect = [
                (False, mock_result, []),
                (True, mock_result_success, []),
            ]
            mock_summarize.return_value = "Phase summary"

            from spark_runner.orchestrator import run_single

            await run_single(task, config, client=MagicMock(), on_phase_failure=callback)

            # Verify hint was saved to goal file
            saved = json.loads(goal_file.read_text())
            assert "hints" in saved
            assert len(saved["hints"]) == 1
            assert saved["hints"][0]["phase"] == "Fill Form"
            assert saved["hints"][0]["text"] == "Try the other button"

    @pytest.mark.asyncio
    async def test_retry_success_continues_to_next_phase(
        self, tmp_path: Path, _minimal_config: SparkConfig,
    ) -> None:
        """After a successful retry, remaining phases should execute."""
        config = _minimal_config

        tasks_dir = config.tasks_dir
        assert tasks_dir is not None
        (tasks_dir / "fill-form.txt").write_text("Fill the form")
        (tasks_dir / "verify-result.txt").write_text("Verify result")

        goal_file = config.goal_summaries_dir / "signup-task.json"
        goal_data = {
            "main_task": "Test user sign-up",
            "key_observations": [],
            "subtasks": [
                {"phase_name": "Fill form", "filename": "fill-form.txt"},
                {"phase_name": "Verify result", "filename": "verify-result.txt"},
            ],
        }
        goal_file.write_text(json.dumps(goal_data))

        task = TaskSpec(goal_path=goal_file)

        callback = AsyncMock(return_value="Use dropdown")

        with (
            patch("spark_runner.orchestrator.decompose_task"),
            patch("spark_runner.orchestrator.run_phase", new_callable=AsyncMock) as mock_run_phase,
            patch("spark_runner.orchestrator.summarize_phase") as mock_summarize,
            patch("spark_runner.orchestrator.Browser") as mock_browser_cls,
            patch("spark_runner.orchestrator.ChatBrowserUse"),
            patch("spark_runner.orchestrator.generate_report"),
        ):
            mock_browser_cls.return_value.stop = AsyncMock()
            mock_result_fail = MagicMock()
            mock_result_fail.final_result.return_value = "Failed"
            mock_result_success = MagicMock(action_results=MagicMock(return_value=[]))
            # Phase 1 fails, retry succeeds, Phase 2 succeeds
            mock_run_phase.side_effect = [
                (False, mock_result_fail, []),
                (True, mock_result_success, []),
                (True, mock_result_success, []),
            ]
            mock_summarize.return_value = "Phase summary"

            from spark_runner.orchestrator import run_single

            result = await run_single(task, config, client=MagicMock(), on_phase_failure=callback)

            # 3 run_phase calls: fail + retry + phase 2
            assert mock_run_phase.call_count == 3
