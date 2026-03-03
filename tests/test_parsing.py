"""Tests for parsing and extraction functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import sparky_runner


class TestExtractAndLogObservations:
    def test_observations_standalone_go_to_event_log(self, tmp_path: Path) -> None:
        """Without sub-phase failures, observations go to the event log (not problem log)."""
        summary = "<OBSERVATIONS>The search bar was missing</OBSERVATIONS>"
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Test", event_log, problem_log)
        # Observations are informational — they go to event log
        event_content = event_log.read_text()
        assert "OBSERVATIONS" in event_content
        assert "search bar was missing" in event_content
        # Problem log should NOT have observations
        assert not problem_log.exists()

    def test_case_insensitive_tags(self, tmp_path: Path) -> None:
        summary = "<observations>Something odd</observations>"
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Phase", event_log, problem_log)
        assert "Something odd" in event_log.read_text()
        assert not problem_log.exists()

    def test_trivial_observations_skipped(self, tmp_path: Path) -> None:
        for trivial in ("None", "N/A", "none.", "n/a."):
            event_log = tmp_path / f"e_{trivial}.log"
            problem_log = tmp_path / f"p_{trivial}.log"
            summary = f"<OBSERVATIONS>{trivial}</OBSERVATIONS>"
            sparky_runner._extract_and_log_observations(summary, "P", event_log, problem_log)
            assert not problem_log.exists(), f"Trivial observation '{trivial}' should not be in problem log"
            assert not event_log.exists(), f"Trivial observation '{trivial}' should not be in event log"

    def test_observations_logged_to_event_log_when_success(self, tmp_path: Path) -> None:
        """Explicit success=True routes observations to event log."""
        summary = (
            "### Sub-phase 1: Login\n"
            "**Status**: SUCCESS\n"
            "\n"
            "<OBSERVATIONS>Element indices shifted</OBSERVATIONS>"
        )
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(
            summary, "Login", event_log, problem_log, success=True,
        )
        event_content = event_log.read_text()
        assert "OBSERVATIONS" in event_content
        assert "Element indices shifted" in event_content
        assert not problem_log.exists()

    def test_failure_routes_observations_to_problem_log(self, tmp_path: Path) -> None:
        """success=False routes observations to problem_log."""
        summary = "<OBSERVATIONS>Button was disabled and unresponsive</OBSERVATIONS>"
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(
            summary, "Click", event_log, problem_log, success=False,
        )
        content = problem_log.read_text()
        assert "OBSERVATIONS" in content
        assert "Button was disabled and unresponsive" in content
        # Event log should NOT have observations when phase failed
        assert not event_log.exists()

    def test_failure_observations_only_in_problem_log(self, tmp_path: Path) -> None:
        """success=False sends observations only to problem_log, not event_log."""
        summary = "<OBSERVATIONS>Search broken</OBSERVATIONS>"
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(
            summary, "Phase", event_log, problem_log, success=False,
        )
        # Problem log has exactly one timestamped entry
        content = problem_log.read_text()
        lines = [line for line in content.splitlines() if line.startswith("[")]
        assert len(lines) == 1
        assert "OBSERVATIONS" in lines[0]
        # Event log should not exist
        assert not event_log.exists()

    def test_no_observations_no_logging(self, tmp_path: Path) -> None:
        """No <OBSERVATIONS> block means neither log is written."""
        summary = "### Sub-phase 1: Login\n**Status**: SUCCESS\n"
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Login", event_log, problem_log)
        assert not event_log.exists()
        assert not problem_log.exists()

    def test_failure_without_observations(self, tmp_path: Path) -> None:
        """success=False but no observations — problem_log not written by this function."""
        summary = "### Sub-phase 1: Login\n**Status**: FAILED\n"
        event_log = tmp_path / "event.log"
        problem_log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(
            summary, "Login", event_log, problem_log, success=False,
        )
        assert not problem_log.exists()
        assert not event_log.exists()


# ── extract_phase_history ───────────────────────────────────────────────

def _make_history_step(
    *,
    model_output: MagicMock | None = None,
    result_error: str | None = None,
    result_extracted: str | None = None,
    url: str | None = None,
) -> MagicMock:
    """Build a minimal mock of one history step for extract_phase_history."""
    result_entry = MagicMock()
    result_entry.error = result_error
    result_entry.extracted_content = result_extracted

    step = MagicMock()
    step.model_output = model_output
    step.result = [result_entry]
    if url:
        step.state = MagicMock()
        step.state.url = url
    else:
        step.state = None
    return step


class TestExtractPhaseHistory:
    def test_empty_history(self) -> None:
        result = MagicMock()
        result.history = []
        assert sparky_runner.extract_phase_history(result) == ""

    def test_full_model_output(self) -> None:
        mo = MagicMock()
        mo.evaluation_previous_goal = "goal met"
        mo.memory = "remember this"
        mo.next_goal = "click button"
        action = MagicMock()
        action.model_dump.return_value = {"click": 5}
        mo.action = [action]

        step = _make_history_step(model_output=mo, url="https://example.com")
        result = MagicMock()
        result.history = [step]

        text = sparky_runner.extract_phase_history(result)
        assert "--- Step 1 ---" in text
        assert "Eval: goal met" in text
        assert "Memory: remember this" in text
        assert "Next goal: click button" in text
        assert "Action:" in text
        assert "URL: https://example.com" in text

    def test_error_in_result(self) -> None:
        step = _make_history_step(result_error="Element not found")
        result = MagicMock()
        result.history = [step]

        text = sparky_runner.extract_phase_history(result)
        assert "ERROR: Element not found" in text

    def test_no_model_output(self) -> None:
        step = _make_history_step()
        result = MagicMock()
        result.history = [step]

        text = sparky_runner.extract_phase_history(result)
        assert "--- Step 1 ---" in text
        # No Eval/Memory/Action lines
        assert "Eval:" not in text
        assert "Memory:" not in text
