"""Tests for build_augmented_task."""

from __future__ import annotations

import sparky_runner


class TestBuildAugmentedTask:
    def test_no_context_passthrough(self) -> None:
        result = sparky_runner.build_augmented_task("Do something at {BASE_URL}", [])
        assert "Do something at https://test.example.com" in result

    def test_no_context_restores_credentials(self) -> None:
        result = sparky_runner.build_augmented_task(
            "Login as {USER_EMAIL} with {USER_PASSWORD}", []
        )
        assert "test@example.com" in result
        assert "test-password-123" in result

    def test_prior_summaries_included(self) -> None:
        summaries = [
            {"name": "Login", "outcome": "SUCCESS", "summary": "Logged in OK"},
        ]
        result = sparky_runner.build_augmented_task("Next step", summaries)
        assert "CONTEXT FROM PRIOR PHASES" in result
        assert "Login" in result
        assert "Logged in OK" in result

    def test_cross_goal_observations_str(self) -> None:
        result = sparky_runner.build_augmented_task(
            "task", [], cross_goal_observations=["obs A", "obs B"]
        )
        assert "KNOWLEDGE FROM PRIOR SUCCESSFUL GOALS" in result
        assert "obs A" in result
        assert "obs B" in result

    def test_cross_goal_observations_dict(self) -> None:
        obs = [{"text": "dict obs", "severity": "error"}]
        result = sparky_runner.build_augmented_task("task", [], cross_goal_observations=obs)
        assert "dict obs" in result

    def test_both_contexts(self) -> None:
        summaries = [{"name": "P1", "outcome": "SUCCESS", "summary": "done"}]
        obs = ["cross-goal note"]
        result = sparky_runner.build_augmented_task("task", summaries, cross_goal_observations=obs)
        assert "KNOWLEDGE FROM PRIOR" in result
        assert "CONTEXT FROM PRIOR PHASES" in result
        assert "YOUR TASK" in result

    def test_empty_observations_not_included(self) -> None:
        result = sparky_runner.build_augmented_task("task", [], cross_goal_observations=[])
        assert "KNOWLEDGE FROM PRIOR" not in result

    def test_none_observations_not_included(self) -> None:
        result = sparky_runner.build_augmented_task("task", [], cross_goal_observations=None)
        assert "KNOWLEDGE FROM PRIOR" not in result
