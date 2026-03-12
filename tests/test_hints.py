"""Tests for hint I/O helpers in goals.py."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spark_runner.goals import (
    clear_reset_phases,
    get_phase_names,
    get_reset_phases,
    load_hints,
    remove_hint,
    reset_phase,
    save_hint,
    unreset_phase,
)


def _write_goal(
    path: Path,
    hints: list[dict[str, str]] | None = None,
    subtasks: list[dict[str, str]] | None = None,
) -> Path:
    """Write a minimal goal JSON, optionally with hints and subtasks."""
    data: dict[str, Any] = {
        "main_task": "Test goal",
        "subtasks": subtasks or [],
        "key_observations": [],
    }
    if hints is not None:
        data["hints"] = hints
    path.write_text(json.dumps(data))
    return path


class TestLoadHints:
    def test_load_hints_empty(self, tmp_path: Path) -> None:
        goal_path = _write_goal(tmp_path / "test-task.json")
        assert load_hints(goal_path) == []

    def test_load_hints_populated(self, tmp_path: Path) -> None:
        hints = [
            {"phase": "Login", "text": "Use the dropdown"},
            {"phase": "Fill Form", "text": "Click More Options first"},
        ]
        goal_path = _write_goal(tmp_path / "test-task.json", hints=hints)
        result = load_hints(goal_path)
        assert len(result) == 2
        assert result[0]["phase"] == "Login"
        assert result[1]["text"] == "Click More Options first"


class TestSaveHint:
    def test_save_hint_appends(self, tmp_path: Path) -> None:
        existing_hints = [{"phase": "Login", "text": "Existing hint"}]
        goal_path = _write_goal(tmp_path / "test-task.json", hints=existing_hints)
        save_hint(goal_path, "Fill Form", "New hint")
        data = json.loads(goal_path.read_text())
        assert len(data["hints"]) == 2
        assert data["hints"][1] == {"phase": "Fill Form", "text": "New hint"}

    def test_save_hint_creates_field(self, tmp_path: Path) -> None:
        goal_path = _write_goal(tmp_path / "test-task.json")
        save_hint(goal_path, "Login", "Use SSO")
        data = json.loads(goal_path.read_text())
        assert data["hints"] == [{"phase": "Login", "text": "Use SSO"}]

    def test_save_hint_preserves_other_fields(self, tmp_path: Path) -> None:
        goal_path = _write_goal(tmp_path / "test-task.json")
        save_hint(goal_path, "Login", "A hint")
        data = json.loads(goal_path.read_text())
        assert data["main_task"] == "Test goal"
        assert data["subtasks"] == []


class TestRemoveHint:
    def test_remove_hint_valid_index(self, tmp_path: Path) -> None:
        hints = [
            {"phase": "A", "text": "First"},
            {"phase": "B", "text": "Second"},
            {"phase": "C", "text": "Third"},
        ]
        goal_path = _write_goal(tmp_path / "test-task.json", hints=hints)
        assert remove_hint(goal_path, 1) is True
        data = json.loads(goal_path.read_text())
        assert len(data["hints"]) == 2
        assert data["hints"][0]["text"] == "First"
        assert data["hints"][1]["text"] == "Third"

    def test_remove_hint_invalid_index(self, tmp_path: Path) -> None:
        hints = [{"phase": "A", "text": "Only"}]
        goal_path = _write_goal(tmp_path / "test-task.json", hints=hints)
        assert remove_hint(goal_path, 5) is False
        data = json.loads(goal_path.read_text())
        assert len(data["hints"]) == 1

    def test_remove_hint_negative_index(self, tmp_path: Path) -> None:
        hints = [{"phase": "A", "text": "Only"}]
        goal_path = _write_goal(tmp_path / "test-task.json", hints=hints)
        assert remove_hint(goal_path, -1) is False

    def test_remove_hint_empty_list(self, tmp_path: Path) -> None:
        goal_path = _write_goal(tmp_path / "test-task.json")
        assert remove_hint(goal_path, 0) is False


class TestGetPhaseNames:
    def test_returns_phase_names_from_subtasks(self, tmp_path: Path) -> None:
        goal_path = tmp_path / "test-task.json"
        data: dict[str, Any] = {
            "main_task": "Test goal",
            "subtasks": [
                {"filename": "fill-form.txt"},
                {"filename": "verify-result.txt"},
            ],
            "key_observations": [],
        }
        goal_path.write_text(json.dumps(data))
        assert get_phase_names(goal_path) == ["Fill Form", "Verify Result"]

    def test_empty_subtasks(self, tmp_path: Path) -> None:
        goal_path = _write_goal(tmp_path / "test-task.json")
        assert get_phase_names(goal_path) == []

    def test_missing_file(self, tmp_path: Path) -> None:
        assert get_phase_names(tmp_path / "missing.json") == []

    def test_single_word_filename(self, tmp_path: Path) -> None:
        goal_path = tmp_path / "test-task.json"
        data: dict[str, Any] = {
            "main_task": "Test",
            "subtasks": [{"filename": "login.txt"}],
            "key_observations": [],
        }
        goal_path.write_text(json.dumps(data))
        assert get_phase_names(goal_path) == ["Login"]


class TestResetPhase:
    def test_reset_phase_adds_to_json(self, tmp_path: Path) -> None:
        goal_path = _write_goal(
            tmp_path / "test-task.json",
            subtasks=[{"filename": "fill-form.txt"}, {"filename": "verify-result.txt"}],
        )
        assert reset_phase(goal_path, "Fill Form") is True
        data = json.loads(goal_path.read_text())
        assert data["reset_phases"] == ["Fill Form"]

    def test_reset_phase_deduplicates(self, tmp_path: Path) -> None:
        goal_path = _write_goal(
            tmp_path / "test-task.json",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        reset_phase(goal_path, "Fill Form")
        reset_phase(goal_path, "Fill Form")
        data = json.loads(goal_path.read_text())
        assert data["reset_phases"] == ["Fill Form"]

    def test_reset_phase_unknown_returns_false(self, tmp_path: Path) -> None:
        goal_path = _write_goal(
            tmp_path / "test-task.json",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        assert reset_phase(goal_path, "Nonexistent Phase") is False
        data = json.loads(goal_path.read_text())
        assert "reset_phases" not in data

    def test_reset_phase_case_insensitive(self, tmp_path: Path) -> None:
        goal_path = _write_goal(
            tmp_path / "test-task.json",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        assert reset_phase(goal_path, "fill form") is True
        data = json.loads(goal_path.read_text())
        assert data["reset_phases"] == ["Fill Form"]

    def test_unreset_phase_removes(self, tmp_path: Path) -> None:
        goal_path = _write_goal(
            tmp_path / "test-task.json",
            subtasks=[{"filename": "fill-form.txt"}, {"filename": "verify-result.txt"}],
        )
        reset_phase(goal_path, "Fill Form")
        reset_phase(goal_path, "Verify Result")
        assert unreset_phase(goal_path, "Fill Form") is True
        data = json.loads(goal_path.read_text())
        assert data["reset_phases"] == ["Verify Result"]

    def test_unreset_phase_nonexistent_returns_false(self, tmp_path: Path) -> None:
        goal_path = _write_goal(
            tmp_path / "test-task.json",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        assert unreset_phase(goal_path, "Fill Form") is False

    def test_get_reset_phases_empty_by_default(self, tmp_path: Path) -> None:
        goal_path = _write_goal(tmp_path / "test-task.json")
        assert get_reset_phases(goal_path) == []

    def test_clear_reset_phases(self, tmp_path: Path) -> None:
        goal_path = _write_goal(
            tmp_path / "test-task.json",
            subtasks=[{"filename": "fill-form.txt"}],
        )
        reset_phase(goal_path, "Fill Form")
        clear_reset_phases(goal_path)
        data = json.loads(goal_path.read_text())
        assert data["reset_phases"] == []
