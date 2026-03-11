"""Tests for execution helpers (build_augmented_task, etc.)."""

from __future__ import annotations

from spark_runner.execution import build_augmented_task


def _identity(text: str) -> str:
    return text


class TestBuildAugmentedTask:
    def test_hints_rendered_in_augmented_task(self) -> None:
        result = build_augmented_task(
            "Do the thing",
            prior_summaries=[],
            restore_fn=_identity,
            hints=["Click More Options first"],
        )
        assert "OPERATOR HINTS" in result
        assert "Click More Options first" in result

    def test_hints_none_omits_section(self) -> None:
        result = build_augmented_task(
            "Do the thing",
            prior_summaries=[],
            restore_fn=_identity,
            hints=None,
        )
        assert "OPERATOR HINTS" not in result

    def test_hints_empty_list_omits_section(self) -> None:
        result = build_augmented_task(
            "Do the thing",
            prior_summaries=[],
            restore_fn=_identity,
            hints=[],
        )
        assert "OPERATOR HINTS" not in result

    def test_hints_with_prior_summaries(self) -> None:
        """Hints should appear alongside other context sections."""
        result = build_augmented_task(
            "Do the thing",
            prior_summaries=[{"name": "Login", "outcome": "SUCCESS", "summary": "Logged in"}],
            restore_fn=_identity,
            hints=["Use the dropdown"],
        )
        assert "OPERATOR HINTS" in result
        assert "Use the dropdown" in result
        assert "CONTEXT FROM PRIOR PHASES" in result

    def test_multiple_hints(self) -> None:
        result = build_augmented_task(
            "Do the thing",
            prior_summaries=[],
            restore_fn=_identity,
            hints=["Hint one", "Hint two"],
        )
        assert "- Hint one" in result
        assert "- Hint two" in result

    def test_task_text_still_present_with_hints(self) -> None:
        result = build_augmented_task(
            "Do the thing",
            prior_summaries=[],
            restore_fn=_identity,
            hints=["A hint"],
        )
        assert "Do the thing" in result

    def test_no_context_returns_rules_and_task(self) -> None:
        result = build_augmented_task(
            "Do the thing",
            prior_summaries=[],
            restore_fn=_identity,
        )
        assert "Do the thing" in result
        assert "OPERATOR HINTS" not in result
