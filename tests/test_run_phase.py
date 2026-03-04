"""Tests for the async run_phase function."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import spark_runner

# The actual run_phase lives in spark_runner.execution, so patches
# must target that module for Agent and extract_phase_history.
_AGENT_PATCH = "spark_runner.execution.Agent"
_HISTORY_PATCH = "spark_runner.execution.extract_phase_history"


def _make_history_item(error: str | None = None) -> MagicMock:
    """Build a minimal mock of one history step."""
    result_entry = MagicMock()
    result_entry.error = error
    result_entry.extracted_content = None

    item = MagicMock()
    item.model_output = None
    item.result = [result_entry]
    item.state = None
    return item


def _make_agent_result(
    *, done: bool = True, successful: bool = True, history: list[MagicMock] | None = None
) -> MagicMock:
    """Build a mock ``AgentHistoryList``."""
    result = MagicMock()
    result.is_done.return_value = done
    result.is_successful.return_value = successful
    result.final_result.return_value = "All good"
    result.errors.return_value = []
    result.history = history or []
    # screenshot_paths returns None for each step (no temp files in tests)
    result.screenshot_paths.return_value = [None] * len(result.history)
    return result


class TestRunPhase:
    @pytest.mark.asyncio
    async def test_successful_phase_logs_success(self, tmp_path: Path) -> None:
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        conversation_log = tmp_path / "conv.json"

        agent_result = _make_agent_result(done=True, successful=True)

        with patch(_AGENT_PATCH) as MockAgent, \
             patch(_HISTORY_PATCH, return_value="step log"):
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.run = AsyncMock(return_value=agent_result)

            success, result, screenshots = await spark_runner.run_phase(
                "Login", "Do login", MagicMock(), MagicMock(),
                conversation_log, event_log, problem_log, tmp_path,
            )

        assert success is True
        log_text = event_log.read_text()
        assert "PHASE SUCCESS: Login" in log_text

    @pytest.mark.asyncio
    async def test_failed_phase_logs_failure(self, tmp_path: Path) -> None:
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        conversation_log = tmp_path / "conv.json"

        agent_result = _make_agent_result(done=True, successful=False)

        mock_page = AsyncMock()
        mock_browser = MagicMock()
        mock_browser.get_current_page = AsyncMock(return_value=mock_page)

        with patch(_AGENT_PATCH) as MockAgent, \
             patch(_HISTORY_PATCH, return_value="step log"):
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.run = AsyncMock(return_value=agent_result)

            success, result, screenshots = await spark_runner.run_phase(
                "Search", "Search stuff", MagicMock(), mock_browser,
                conversation_log, event_log, problem_log, tmp_path,
            )

        assert success is False
        assert "PHASE FAILED: Search" in event_log.read_text()
        assert "PHASE FAILED: Search" in problem_log.read_text()

    @pytest.mark.asyncio
    async def test_action_errors_logged_to_problem_log(self, tmp_path: Path) -> None:
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        conversation_log = tmp_path / "conv.json"

        history_item = _make_history_item(error="Element not found")
        agent_result = _make_agent_result(done=True, successful=True, history=[history_item])

        with patch(_AGENT_PATCH) as MockAgent, \
             patch(_HISTORY_PATCH, return_value="log"):
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.run = AsyncMock(return_value=agent_result)

            await spark_runner.run_phase(
                "Fill", "Fill form", MagicMock(), MagicMock(),
                conversation_log, event_log, problem_log, tmp_path,
            )

        assert "ACTION ERROR (Fill): Element not found" in problem_log.read_text()

    @pytest.mark.asyncio
    async def test_failure_screenshot_saved_to_run_dir(self, tmp_path: Path) -> None:
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        conversation_log = tmp_path / "conv.json"

        agent_result = _make_agent_result(done=True, successful=False)

        mock_page = AsyncMock()
        mock_browser = MagicMock()
        mock_browser.get_current_page = AsyncMock(return_value=mock_page)

        with patch(_AGENT_PATCH) as MockAgent, \
             patch(_HISTORY_PATCH, return_value="log"):
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.run = AsyncMock(return_value=agent_result)

            await spark_runner.run_phase(
                "Login Step", "Do login", MagicMock(), mock_browser,
                conversation_log, event_log, problem_log, tmp_path,
            )

        expected_path = str(tmp_path / "screenshots" / "failure_Login_Step.png")
        mock_page.screenshot.assert_called_once_with(expected_path)

    @pytest.mark.asyncio
    async def test_screenshot_exception_does_not_crash(self, tmp_path: Path) -> None:
        """If taking the failure screenshot raises, run_phase still returns."""
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        conversation_log = tmp_path / "conv.json"

        agent_result = _make_agent_result(done=True, successful=False)

        mock_browser = MagicMock()
        mock_browser.get_current_page = AsyncMock(side_effect=RuntimeError("browser gone"))

        with patch(_AGENT_PATCH) as MockAgent, \
             patch(_HISTORY_PATCH, return_value="log"):
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.run = AsyncMock(return_value=agent_result)

            success, result, screenshots = await spark_runner.run_phase(
                "Crash", "Do stuff", MagicMock(), mock_browser,
                conversation_log, event_log, problem_log, tmp_path,
            )

        assert success is False
        assert "Could not save failure screenshot" in event_log.read_text()
        assert "Could not save failure screenshot" in problem_log.read_text()

    @pytest.mark.asyncio
    async def test_not_done_phase_reports_failure(self, tmp_path: Path) -> None:
        """is_done()=False, is_successful()=True → success=False."""
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        conversation_log = tmp_path / "conv.json"

        agent_result = _make_agent_result(done=False, successful=True)

        mock_page = AsyncMock()
        mock_browser = MagicMock()
        mock_browser.get_current_page = AsyncMock(return_value=mock_page)

        with patch(_AGENT_PATCH) as MockAgent, \
             patch(_HISTORY_PATCH, return_value="log"):
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.run = AsyncMock(return_value=agent_result)

            success, result, screenshots = await spark_runner.run_phase(
                "Stuck", "Do stuff", MagicMock(), mock_browser,
                conversation_log, event_log, problem_log, tmp_path,
            )

        assert success is False

    @pytest.mark.asyncio
    async def test_step_limit_logged_as_warning(self, tmp_path: Path) -> None:
        """When the agent exhausts all 50 steps without finishing, log a warning."""
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        conversation_log = tmp_path / "conv.json"

        history = [_make_history_item() for _ in range(50)]
        agent_result = _make_agent_result(done=False, successful=False, history=history)

        mock_page = AsyncMock()
        mock_browser = MagicMock()
        mock_browser.get_current_page = AsyncMock(return_value=mock_page)

        with patch(_AGENT_PATCH) as MockAgent, \
             patch(_HISTORY_PATCH, return_value="log"):
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.run = AsyncMock(return_value=agent_result)

            await spark_runner.run_phase(
                "Search", "Find stuff", MagicMock(), mock_browser,
                conversation_log, event_log, problem_log, tmp_path,
            )

        assert "exhausted step limit" in event_log.read_text()
        assert "STEP LIMIT" in problem_log.read_text()

    @pytest.mark.asyncio
    async def test_step_screenshots_collected(self, tmp_path: Path) -> None:
        """Step screenshots from browser-use temp dir are copied into run_dir/screenshots/."""
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        conversation_log = tmp_path / "conv.json"

        # Create fake temp screenshot files that browser-use would produce
        tmp_screenshots = tmp_path / "browser_use_tmp"
        tmp_screenshots.mkdir()
        step0 = tmp_screenshots / "step_0.png"
        step1 = tmp_screenshots / "step_1.png"
        step0.write_bytes(b"PNG-step0")
        step1.write_bytes(b"PNG-step1")

        history = [_make_history_item(), _make_history_item()]
        agent_result = _make_agent_result(done=True, successful=True, history=history)
        agent_result.screenshot_paths.return_value = [str(step0), str(step1)]

        with patch(_AGENT_PATCH) as MockAgent, \
             patch(_HISTORY_PATCH, return_value="log"):
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.run = AsyncMock(return_value=agent_result)

            success, result, screenshots = await spark_runner.run_phase(
                "Login", "Do login", MagicMock(), MagicMock(),
                conversation_log, event_log, problem_log, tmp_path,
            )

        assert success is True
        assert len(screenshots) == 2
        # Check files were copied into screenshots/ dir
        screenshots_dir = tmp_path / "screenshots"
        assert screenshots_dir.is_dir()
        copied = sorted(screenshots_dir.iterdir())
        assert len(copied) == 2
        assert copied[0].name == "login_step_000.png"
        assert copied[1].name == "login_step_001.png"
        assert copied[0].read_bytes() == b"PNG-step0"

    @pytest.mark.asyncio
    async def test_error_step_screenshots_tagged(self, tmp_path: Path) -> None:
        """Screenshots at error steps have event_type='error' and error_message set."""
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        conversation_log = tmp_path / "conv.json"

        tmp_screenshots = tmp_path / "browser_use_tmp"
        tmp_screenshots.mkdir()
        step0 = tmp_screenshots / "step_0.png"
        step0.write_bytes(b"PNG-error")

        history = [_make_history_item(error="Element not found")]
        agent_result = _make_agent_result(done=True, successful=True, history=history)
        agent_result.screenshot_paths.return_value = [str(step0)]

        with patch(_AGENT_PATCH) as MockAgent, \
             patch(_HISTORY_PATCH, return_value="log"):
            mock_agent_instance = MockAgent.return_value
            mock_agent_instance.run = AsyncMock(return_value=agent_result)

            success, result, screenshots = await spark_runner.run_phase(
                "Fill", "Fill form", MagicMock(), MagicMock(),
                conversation_log, event_log, problem_log, tmp_path,
            )

        assert len(screenshots) == 1
        assert screenshots[0].event_type == "error"
        assert screenshots[0].error_message == "Element not found"
        assert screenshots[0].phase_name == "Fill"
        assert screenshots[0].step_number == 0
