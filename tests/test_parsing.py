"""Tests for parsing and extraction functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import sparky_runner


class TestExtractAndLogObservations:
    def test_success_sub_phases_not_logged(self, tmp_path: Path) -> None:
        summary = (
            "### Sub-phase 1: Login\n"
            "**Status**: SUCCESS\n"
        )
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Login", log)
        assert not log.exists()

    def test_failed_sub_phase_logged_as_error(self, tmp_path: Path) -> None:
        summary = (
            "### Sub-phase 2: Search\n"
            "**Status**: FAILED\n"
        )
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Search Phase", log)
        content = log.read_text()
        assert "ERROR" in content
        assert "Search" in content
        assert "[diagnostic:" in content

    def test_partial_failure_logged(self, tmp_path: Path) -> None:
        summary = (
            "### Sub-phase A: Form Fill\n"
            "**Status**: PARTIAL FAILURE\n"
        )
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Form", log)
        assert "ERROR" in log.read_text()

    def test_result_line_variant(self, tmp_path: Path) -> None:
        summary = (
            "### 3.1 Navigate to Landing Page\n"
            "**Result**: Failed - page not found\n"
        )
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Nav", log)
        assert "ERROR" in log.read_text()

    def test_partial_success_treated_as_success(self, tmp_path: Path) -> None:
        """'Partial Success' contains 'SUCCESS' so it is NOT logged as an error."""
        summary = (
            "### 3.1 Navigate to Landing Page\n"
            "**Result**: Partial Success\n"
        )
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Nav", log)
        assert not log.exists()

    def test_succeeded_treated_as_success(self, tmp_path: Path) -> None:
        """'Succeeded' (as opposed to 'SUCCESS') should not be logged as error."""
        summary = (
            "### Sub-phase 1: Pre-completion Verification\n"
            "**Status**: ✅ Succeeded - All categories confirmed as tested\n"
        )
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Verify", log)
        assert not log.exists()

    def test_checkmark_emoji_treated_as_success(self, tmp_path: Path) -> None:
        """A ✅ emoji in the status indicates success."""
        summary = (
            "### Sub-phase 1: Wrap Up\n"
            "**Status**: ✅ Done\n"
        )
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Phase", log)
        assert not log.exists()

    def test_passed_treated_as_success(self, tmp_path: Path) -> None:
        """'PASSED' in the status indicates success."""
        summary = (
            "### Sub-phase 1: Validation\n"
            "**Status**: PASSED\n"
        )
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Phase", log)
        assert not log.exists()

    def test_observations_standalone_when_no_errors(self, tmp_path: Path) -> None:
        """Without sub-phase failures, observations are logged standalone."""
        summary = "<OBSERVATIONS>The search bar was missing</OBSERVATIONS>"
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Test", log)
        content = log.read_text()
        assert "OBSERVATIONS" in content
        assert "search bar was missing" in content

    def test_case_insensitive_tags(self, tmp_path: Path) -> None:
        summary = "<observations>Something odd</observations>"
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Phase", log)
        assert "Something odd" in log.read_text()

    def test_trivial_observations_skipped(self, tmp_path: Path) -> None:
        for trivial in ("None", "N/A", "none.", "n/a."):
            log = tmp_path / f"p_{trivial}.log"
            summary = f"<OBSERVATIONS>{trivial}</OBSERVATIONS>"
            sparky_runner._extract_and_log_observations(summary, "P", log)
            assert not log.exists(), f"Trivial observation '{trivial}' should be skipped"

    def test_status_with_intervening_lines(self, tmp_path: Path) -> None:
        """Status line several lines below the heading should still be detected."""
        summary = (
            "### Sub-phase 2: Search Bar Interaction\n"
            "Typed 'Email Body' into the search field.\n"
            "Waited for results to appear.\n"
            "**Status**: FAILED\n"
        )
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Search", log)
        content = log.read_text()
        assert "ERROR" in content
        assert "Search Bar Interaction" in content

    def test_error_includes_observations_detail(self, tmp_path: Path) -> None:
        """When a sub-phase fails, observations are folded into the ERROR entry."""
        summary = (
            "### Sub-phase 1: Click Button\n"
            "**Status**: FAILED\n"
            "\n"
            "<OBSERVATIONS>Button was disabled and unresponsive</OBSERVATIONS>"
        )
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Click", log)
        content = log.read_text()
        assert "ERROR" in content
        assert "Button was disabled and unresponsive" in content

    def test_observations_not_duplicated_when_error_present(self, tmp_path: Path) -> None:
        """When observations are folded into ERROR, no standalone OBSERVATIONS line."""
        summary = (
            "### Sub-phase 1: Search\n"
            "**Status**: FAILED\n"
            "\n"
            "<OBSERVATIONS>Search broken</OBSERVATIONS>"
        )
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Phase", log)
        content = log.read_text()
        # Should have exactly one timestamped entry (the ERROR), not a second OBSERVATIONS
        lines = [l for l in content.splitlines() if l.startswith("[")]
        assert len(lines) == 1
        assert "ERROR" in lines[0]

    def test_observations_logged_standalone_when_success(self, tmp_path: Path) -> None:
        """Sub-phase SUCCESS + observations still logs observations standalone."""
        summary = (
            "### Sub-phase 1: Login\n"
            "**Status**: SUCCESS\n"
            "\n"
            "<OBSERVATIONS>Element indices shifted</OBSERVATIONS>"
        )
        log = tmp_path / "problem.log"
        sparky_runner._extract_and_log_observations(summary, "Login", log)
        content = log.read_text()
        assert "OBSERVATIONS" in content
        assert "Element indices shifted" in content
        assert "ERROR" not in content


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
