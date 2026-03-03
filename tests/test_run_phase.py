"""Tests for the async run_phase function."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sparky_runner

# The actual run_phase lives in sparky_runner.execution, so patches
# must target that module for Agent and extract_phase_history.
_AGENT_PATCH = "sparky_runner.execution.Agent"
_HISTORY_PATCH = "sparky_runner.execution.extract_phase_history"


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

            success, result = await sparky_runner.run_phase(
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

            success, result = await sparky_runner.run_phase(
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

            await sparky_runner.run_phase(
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

            await sparky_runner.run_phase(
                "Login Step", "Do login", MagicMock(), mock_browser,
                conversation_log, event_log, problem_log, tmp_path,
            )

        expected_path = str(tmp_path / "failure_Login_Step.png")
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

            success, result = await sparky_runner.run_phase(
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

            success, result = await sparky_runner.run_phase(
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

            await sparky_runner.run_phase(
                "Search", "Find stuff", MagicMock(), mock_browser,
                conversation_log, event_log, problem_log, tmp_path,
            )

        assert "exhausted step limit" in event_log.read_text()
        assert "STEP LIMIT" in problem_log.read_text()
