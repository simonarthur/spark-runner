"""Tests for build_augmented_task."""

from __future__ import annotations

import spark_runner


class TestBuildAugmentedTask:
    def test_no_context_passthrough(self) -> None:
        result = spark_runner.build_augmented_task("Do something at {BASE_URL}", [])
        assert "Do something at https://test.example.com" in result

    def test_no_context_restores_credentials(self) -> None:
        result = spark_runner.build_augmented_task(
            "Login as {USER_EMAIL} with {USER_PASSWORD}", []
        )
        assert "test@example.com" in result
        assert "test-password-123" in result

    def test_prior_summaries_included(self) -> None:
        summaries = [
            {"name": "Login", "outcome": "SUCCESS", "summary": "Logged in OK"},
        ]
        result = spark_runner.build_augmented_task("Next step", summaries)
        assert "CONTEXT FROM PRIOR PHASES" in result
        assert "Login" in result
        assert "Logged in OK" in result

    def test_cross_goal_observations_str(self) -> None:
        result = spark_runner.build_augmented_task(
            "task", [], cross_goal_observations=["obs A", "obs B"]
        )
        assert "KNOWLEDGE FROM PRIOR SUCCESSFUL GOALS" in result
        assert "obs A" in result
        assert "obs B" in result

    def test_cross_goal_observations_dict(self) -> None:
        obs = [{"text": "dict obs", "severity": "error"}]
        result = spark_runner.build_augmented_task("task", [], cross_goal_observations=obs)
        assert "dict obs" in result

    def test_both_contexts(self) -> None:
        summaries = [{"name": "P1", "outcome": "SUCCESS", "summary": "done"}]
        obs = ["cross-goal note"]
        result = spark_runner.build_augmented_task("task", summaries, cross_goal_observations=obs)
        assert "KNOWLEDGE FROM PRIOR" in result
        assert "CONTEXT FROM PRIOR PHASES" in result
        assert "YOUR TASK" in result

    def test_empty_observations_not_included(self) -> None:
        result = spark_runner.build_augmented_task("task", [], cross_goal_observations=[])
        assert "KNOWLEDGE FROM PRIOR" not in result

    def test_none_observations_not_included(self) -> None:
        result = spark_runner.build_augmented_task("task", [], cross_goal_observations=None)
        assert "KNOWLEDGE FROM PRIOR" not in result

    def test_ui_instructions_injected_no_context(self) -> None:
        result = spark_runner.build_augmented_task(
            "Do something", [], ui_instructions=["Save is blue", "Toast top-right"],
        )
        assert "SITE-SPECIFIC UI INSTRUCTIONS" in result
        assert "- Save is blue" in result
        assert "- Toast top-right" in result

    def test_ui_instructions_injected_with_context(self) -> None:
        summaries = [{"name": "Login", "outcome": "SUCCESS", "summary": "OK"}]
        result = spark_runner.build_augmented_task(
            "Next step", summaries, ui_instructions=["Hint A"],
        )
        assert "SITE-SPECIFIC UI INSTRUCTIONS" in result
        assert "- Hint A" in result
        assert "CONTEXT FROM PRIOR PHASES" in result

    def test_ui_instructions_empty_list_not_injected(self) -> None:
        result = spark_runner.build_augmented_task("task", [], ui_instructions=[])
        assert "SITE-SPECIFIC UI INSTRUCTIONS" not in result

    def test_ui_instructions_none_not_injected(self) -> None:
        result = spark_runner.build_augmented_task("task", [], ui_instructions=None)
        assert "SITE-SPECIFIC UI INSTRUCTIONS" not in result
