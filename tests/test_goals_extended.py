"""Tests for show_goal_detail and delete_goal in sparky_runner.goals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from sparky_runner.goals import delete_goal, show_goal_detail


# ── Helpers ──────────────────────────────────────────────────────────────


def _write_goal(
    goal_summaries_dir: Path,
    name: str,
    main_task: str = "Do something",
    subtasks: list[dict[str, Any]] | None = None,
    observations: list[Any] | None = None,
) -> Path:
    """Write a goal summary JSON file and return its path."""
    goal_path = goal_summaries_dir / f"{name}-task.json"
    data: dict[str, Any] = {
        "main_task": main_task,
        "subtasks": subtasks or [],
        "key_observations": observations or [],
    }
    goal_path.write_text(json.dumps(data))
    return goal_path


def _identity(text: str) -> str:
    """No-op restore function for use as restore_fn."""
    return text


# ── show_goal_detail ─────────────────────────────────────────────────────


class TestShowGoalDetail:
    def test_prints_goal_name(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _write_goal(tmp_path, "my-goal", main_task="Buy milk")
        show_goal_detail(tmp_path, "my-goal", _identity)
        output = capsys.readouterr().out
        assert "my-goal-task.json" in output

    def test_prints_main_task(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        _write_goal(tmp_path, "shopping", main_task="Buy groceries from the store")
        show_goal_detail(tmp_path, "shopping", _identity)
        output = capsys.readouterr().out
        assert "Buy groceries from the store" in output

    def test_prints_subtasks(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        subtasks: list[dict[str, Any]] = [
            {"filename": "login.txt", "subtask": 1},
            {"filename": "search.txt", "subtask": 2},
        ]
        _write_goal(tmp_path, "flow", subtasks=subtasks)
        show_goal_detail(tmp_path, "flow", _identity)
        output = capsys.readouterr().out
        assert "login.txt" in output
        assert "search.txt" in output

    def test_prints_observations(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        observations: list[Any] = [
            {"text": "Login was slow", "severity": "warning"},
            {"text": "Search broke", "severity": "error"},
        ]
        _write_goal(tmp_path, "obs-goal", observations=observations)
        show_goal_detail(tmp_path, "obs-goal", _identity)
        output = capsys.readouterr().out
        assert "Login was slow" in output
        assert "Search broke" in output

    def test_prints_observation_severity(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        observations: list[Any] = [{"text": "Button broken", "severity": "error"}]
        _write_goal(tmp_path, "sev-goal", observations=observations)
        show_goal_detail(tmp_path, "sev-goal", _identity)
        output = capsys.readouterr().out
        assert "error" in output

    def test_missing_goal_prints_not_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        show_goal_detail(tmp_path, "nonexistent-goal", _identity)
        output = capsys.readouterr().out
        assert "not found" in output.lower() or "Goal not found" in output

    def test_accepts_name_with_suffix(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_goal(tmp_path, "my-goal", main_task="Test task")
        show_goal_detail(tmp_path, "my-goal-task.json", _identity)
        output = capsys.readouterr().out
        assert "Test task" in output

    def test_accepts_name_without_suffix(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_goal(tmp_path, "my-goal", main_task="Test task")
        show_goal_detail(tmp_path, "my-goal", _identity)
        output = capsys.readouterr().out
        assert "Test task" in output

    def test_restore_fn_applied_to_content(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The restore_fn transforms the file content before parsing."""
        goal_path = tmp_path / "encoded-task.json"
        data: dict[str, Any] = {
            "main_task": "{BASE_URL}/checkout",
            "subtasks": [],
            "key_observations": [],
        }
        goal_path.write_text(json.dumps(data))

        def restore(text: str) -> str:
            return text.replace("{BASE_URL}", "https://example.com")

        show_goal_detail(tmp_path, "encoded", restore)
        output = capsys.readouterr().out
        assert "https://example.com/checkout" in output


# ── delete_goal ──────────────────────────────────────────────────────────


class TestDeleteGoal:
    def test_deletes_goal_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        goal_summaries_dir = tmp_path / "goals"
        goal_summaries_dir.mkdir()
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        goal_path = _write_goal(goal_summaries_dir, "old-goal")
        delete_goal(goal_summaries_dir, tasks_dir, "old-goal", force=True)

        assert not goal_path.exists()

    def test_deletes_unreferenced_task_files(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        goal_summaries_dir = tmp_path / "goals"
        goal_summaries_dir.mkdir()
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        task_file = tasks_dir / "login.txt"
        task_file.write_text("Login steps")

        subtasks: list[dict[str, Any]] = [{"filename": "login.txt", "subtask": 1}]
        _write_goal(goal_summaries_dir, "solo-goal", subtasks=subtasks)

        delete_goal(goal_summaries_dir, tasks_dir, "solo-goal", force=True)

        assert not task_file.exists()

    def test_keeps_tasks_referenced_by_other_goals(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        goal_summaries_dir = tmp_path / "goals"
        goal_summaries_dir.mkdir()
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        shared_task = tasks_dir / "shared.txt"
        shared_task.write_text("Shared steps")

        subtasks: list[dict[str, Any]] = [{"filename": "shared.txt", "subtask": 1}]
        _write_goal(goal_summaries_dir, "goal-a", subtasks=subtasks)
        _write_goal(goal_summaries_dir, "goal-b", subtasks=subtasks)

        delete_goal(goal_summaries_dir, tasks_dir, "goal-a", force=True)

        # goal-a should be gone
        assert not (goal_summaries_dir / "goal-a-task.json").exists()
        # shared task must still exist because goal-b references it
        assert shared_task.exists()

    def test_missing_goal_prints_not_found(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        goal_summaries_dir = tmp_path / "goals"
        goal_summaries_dir.mkdir()
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        delete_goal(goal_summaries_dir, tasks_dir, "nonexistent", force=True)
        output = capsys.readouterr().out
        assert "not found" in output.lower() or "Goal not found" in output

    def test_accepts_name_without_suffix(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        goal_summaries_dir = tmp_path / "goals"
        goal_summaries_dir.mkdir()
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        goal_path = _write_goal(goal_summaries_dir, "my-goal")
        delete_goal(goal_summaries_dir, tasks_dir, "my-goal", force=True)
        assert not goal_path.exists()

    def test_accepts_name_with_suffix(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        goal_summaries_dir = tmp_path / "goals"
        goal_summaries_dir.mkdir()
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        goal_path = _write_goal(goal_summaries_dir, "my-goal")
        delete_goal(goal_summaries_dir, tasks_dir, "my-goal-task.json", force=True)
        assert not goal_path.exists()

    def test_prints_done_on_success(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        goal_summaries_dir = tmp_path / "goals"
        goal_summaries_dir.mkdir()
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        _write_goal(goal_summaries_dir, "done-goal")
        delete_goal(goal_summaries_dir, tasks_dir, "done-goal", force=True)
        output = capsys.readouterr().out
        assert "Done" in output or "done" in output.lower()

    def test_without_force_prompts_and_aborts_on_n(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        goal_summaries_dir = tmp_path / "goals"
        goal_summaries_dir.mkdir()
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        goal_path = _write_goal(goal_summaries_dir, "keep-goal")

        with patch("builtins.input", return_value="n"):
            delete_goal(goal_summaries_dir, tasks_dir, "keep-goal", force=False)

        assert goal_path.exists()
        output = capsys.readouterr().out
        assert "Aborted" in output

    def test_without_force_proceeds_on_y(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        goal_summaries_dir = tmp_path / "goals"
        goal_summaries_dir.mkdir()
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()

        goal_path = _write_goal(goal_summaries_dir, "delete-me")

        with patch("builtins.input", return_value="y"):
            delete_goal(goal_summaries_dir, tasks_dir, "delete-me", force=False)

        assert not goal_path.exists()
