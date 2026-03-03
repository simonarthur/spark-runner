"""Tests for filesystem-related functions in sparky_runner."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

import sparky_runner


# ── safe_write_path ──────────────────────────────────────────────────────

class TestSafeWritePath:
    def test_no_conflict(self, tmp_path: Path) -> None:
        p = tmp_path / "report.json"
        assert sparky_runner.safe_write_path(p) == p

    def test_single_conflict(self, tmp_path: Path) -> None:
        p = tmp_path / "report.json"
        p.write_text("existing")
        result = sparky_runner.safe_write_path(p)
        assert result == tmp_path / "report-2.json"

    def test_multiple_conflicts(self, tmp_path: Path) -> None:
        for suffix in ("", "-2", "-3"):
            (tmp_path / f"report{suffix}.json").write_text("x")
        result = sparky_runner.safe_write_path(tmp_path / "report.json")
        assert result == tmp_path / "report-4.json"


# ── log_event ────────────────────────────────────────────────────────────

class TestLogEvent:
    def test_appends_timestamped_line(self, tmp_path: Path) -> None:
        log = tmp_path / "event.log"
        sparky_runner.log_event(log, "hello")
        sparky_runner.log_event(log, "world")
        lines = log.read_text().splitlines()
        assert len(lines) == 2
        assert "hello" in lines[0]
        assert "world" in lines[1]

    def test_prints_to_stdout(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        sparky_runner.log_event(tmp_path / "e.log", "visible")
        assert "visible" in capsys.readouterr().out


# ── log_problem ──────────────────────────────────────────────────────────

class TestLogProblem:
    def test_appends_to_file(self, tmp_path: Path) -> None:
        log = tmp_path / "problem.log"
        sparky_runner.log_problem(log, "issue 1")
        sparky_runner.log_problem(log, "issue 2")
        assert log.read_text().count("\n") == 2

    def test_does_not_print(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        sparky_runner.log_problem(tmp_path / "p.log", "silent")
        assert capsys.readouterr().out == ""


# ── load_knowledge_index ─────────────────────────────────────────────────

class TestLoadKnowledgeIndex:
    def test_empty_dir(self, fake_goal_summaries_dir: Path) -> None:
        assert sparky_runner.load_knowledge_index() == []

    def test_loads_goal_with_subtasks(
        self, fake_tasks_dir: Path, fake_goal_summaries_dir: Path
    ) -> None:
        # Create a subtask file
        (fake_tasks_dir / "login.txt").write_text("Logged in OK")
        # Create a goal summary referencing it
        goal: dict[str, Any] = {
            "main_task": "Test login",
            "key_observations": ["obs1"],
            "subtasks": [{"filename": "login.txt"}],
        }
        (fake_goal_summaries_dir / "login-test-task.json").write_text(json.dumps(goal))

        index = sparky_runner.load_knowledge_index()
        assert len(index) == 1
        assert index[0]["main_task"] == "Test login"
        assert len(index[0]["subtasks"]) == 1
        assert index[0]["subtasks"][0]["content"] == "Logged in OK"
        assert index[0]["subtasks"][0]["name"] == "Login"

    def test_skips_malformed_json(self, fake_goal_summaries_dir: Path) -> None:
        (fake_goal_summaries_dir / "bad-task.json").write_text("not json{{{")
        index = sparky_runner.load_knowledge_index()
        assert index == []

    def test_skips_missing_subtask_file(
        self, fake_tasks_dir: Path, fake_goal_summaries_dir: Path
    ) -> None:
        goal: dict[str, Any] = {
            "main_task": "test",
            "key_observations": [],
            "subtasks": [{"filename": "missing.txt"}],
        }
        (fake_goal_summaries_dir / "test-task.json").write_text(json.dumps(goal))
        index = sparky_runner.load_knowledge_index()
        assert len(index) == 1
        assert index[0]["subtasks"] == []

    def test_restores_placeholders(
        self, fake_tasks_dir: Path, fake_goal_summaries_dir: Path
    ) -> None:
        (fake_tasks_dir / "step.txt").write_text("Visit {BASE_URL}/page")
        goal: dict[str, Any] = {
            "main_task": "go to {BASE_URL}",
            "key_observations": [],
            "subtasks": [{"filename": "step.txt"}],
        }
        (fake_goal_summaries_dir / "step-task.json").write_text(json.dumps(goal))
        index = sparky_runner.load_knowledge_index()
        assert "https://test.example.com/page" in index[0]["subtasks"][0]["content"]


# ── load_goal_summary ────────────────────────────────────────────────────

class TestLoadGoalSummary:
    def test_loads_phases(self, fake_tasks_dir: Path) -> None:
        (fake_tasks_dir / "login.txt").write_text("Did the login")
        goal: dict[str, Any] = {
            "main_task": "Login test",
            "subtasks": [{"filename": "login.txt"}],
        }
        goal_path = fake_tasks_dir.parent / "goal.json"
        goal_path.write_text(json.dumps(goal))

        prompt, task_name, phases = sparky_runner.load_goal_summary(goal_path)
        assert prompt == "Login test"
        assert task_name == "goal"
        assert len(phases) == 1
        assert phases[0]["name"] == "Login"
        assert sparky_runner._REPLAY_PREFIX in phases[0]["task"]

    def test_raises_on_missing_subtask(self, fake_tasks_dir: Path) -> None:
        goal: dict[str, Any] = {
            "main_task": "test",
            "subtasks": [{"filename": "nope.txt"}],
        }
        goal_path = fake_tasks_dir.parent / "goal.json"
        goal_path.write_text(json.dumps(goal))
        with pytest.raises(FileNotFoundError):
            sparky_runner.load_goal_summary(goal_path)

    def test_replay_prefix_injected(self, fake_tasks_dir: Path) -> None:
        (fake_tasks_dir / "nav.txt").write_text("Navigated to page")
        goal: dict[str, Any] = {
            "main_task": "nav test",
            "subtasks": [{"filename": "nav.txt"}],
        }
        goal_path = fake_tasks_dir.parent / "g.json"
        goal_path.write_text(json.dumps(goal))
        _, _, phases = sparky_runner.load_goal_summary(goal_path)
        assert phases[0]["task"].startswith(sparky_runner._REPLAY_PREFIX)


# ── _get_orphan_tasks / clean_orphan_tasks ───────────────────────────────

class TestOrphans:
    def test_no_orphans(
        self, fake_tasks_dir: Path, fake_goal_summaries_dir: Path
    ) -> None:
        (fake_tasks_dir / "used.txt").write_text("x")
        goal: dict[str, Any] = {"subtasks": [{"filename": "used.txt"}]}
        (fake_goal_summaries_dir / "g-task.json").write_text(json.dumps(goal))
        assert sparky_runner._get_orphan_tasks() == []

    def test_detects_orphan(
        self, fake_tasks_dir: Path, fake_goal_summaries_dir: Path
    ) -> None:
        (fake_tasks_dir / "used.txt").write_text("x")
        (fake_tasks_dir / "orphan.txt").write_text("x")
        goal: dict[str, Any] = {"subtasks": [{"filename": "used.txt"}]}
        (fake_goal_summaries_dir / "g-task.json").write_text(json.dumps(goal))
        orphans = sparky_runner._get_orphan_tasks()
        assert orphans == ["orphan.txt"]

    def test_clean_orphan_confirm(
        self,
        fake_tasks_dir: Path,
        fake_goal_summaries_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (fake_tasks_dir / "orphan.txt").write_text("x")
        monkeypatch.setattr("builtins.input", lambda _: "y")
        sparky_runner.clean_orphan_tasks()
        assert not (fake_tasks_dir / "orphan.txt").exists()

    def test_clean_orphan_abort(
        self,
        fake_tasks_dir: Path,
        fake_goal_summaries_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (fake_tasks_dir / "orphan.txt").write_text("x")
        monkeypatch.setattr("builtins.input", lambda _: "n")
        sparky_runner.clean_orphan_tasks()
        assert (fake_tasks_dir / "orphan.txt").exists()

    def test_empty_dirs(
        self, fake_tasks_dir: Path, fake_goal_summaries_dir: Path
    ) -> None:
        assert sparky_runner._get_orphan_tasks() == []


# ── phase_name_to_slug ──────────────────────────────────────────────────

class TestPhaseNameToSlug:
    def test_basic(self) -> None:
        assert sparky_runner.phase_name_to_slug("Login") == "login"

    def test_spaces(self) -> None:
        assert sparky_runner.phase_name_to_slug("Fill Form") == "fill-form"

    def test_special_chars(self) -> None:
        assert sparky_runner.phase_name_to_slug("Step #1: Login!") == "step-1-login"

    def test_strip_edges(self) -> None:
        assert sparky_runner.phase_name_to_slug("---Login---") == "login"

    def test_all_special(self) -> None:
        assert sparky_runner.phase_name_to_slug("!!!") == ""


# ── make_run_dir ────────────────────────────────────────────────────────

class TestMakeRunDir:
    def test_creates_nested_dir(self, fake_runs_dir: Path) -> None:
        run_dir = sparky_runner.make_run_dir(fake_runs_dir, "login-test")
        assert run_dir.exists()
        assert run_dir.parent.name == "login-test"
        assert run_dir.parent.parent == fake_runs_dir

    def test_dir_name_matches_timestamp_format(self, fake_runs_dir: Path) -> None:
        run_dir = sparky_runner.make_run_dir(fake_runs_dir, "t")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", run_dir.name)

    def test_two_calls_same_task_no_crash(self, fake_runs_dir: Path) -> None:
        d1 = sparky_runner.make_run_dir(fake_runs_dir, "t")
        d2 = sparky_runner.make_run_dir(fake_runs_dir, "t")
        # Even if same second, exist_ok=True prevents a crash
        assert d1.exists() and d2.exists()


# ── format_phase_plan ───────────────────────────────────────────────────

class TestFormatPhasePlan:
    def test_two_phases(self) -> None:
        phases = [
            {"name": "Login", "task": "Do login"},
            {"name": "Search", "task": "Search items"},
        ]
        lines = sparky_runner.format_phase_plan(phases)
        assert lines[0] == "PHASE PLAN (2 phases):"
        assert "Phase 1: Login" in lines[1]
        assert "Instructions: Do login" in lines[2]
        assert "Phase 2: Search" in lines[3]
        assert "Instructions: Search items" in lines[4]

    def test_long_task_truncated(self) -> None:
        phases = [{"name": "Big", "task": "x" * 3000}]
        lines = sparky_runner.format_phase_plan(phases)
        instructions_line = lines[2]
        # The task value should be truncated to 2000 chars
        assert len(instructions_line) <= len("    Instructions: ") + 2000

    def test_missing_task_key(self) -> None:
        phases = [{"name": "Reuse"}]
        lines = sparky_runner.format_phase_plan(phases)
        assert "(reuse — see above)" in lines[2]


# ── format_knowledge_match ──────────────────────────────────────────────

class TestFormatKnowledgeMatch:
    def test_full_match(self) -> None:
        km: dict[str, Any] = {
            "reusable_subtasks": [
                {"filename": "login.txt", "phase_name": "Login", "reason": "exact"}
            ],
            "relevant_observations": ["obs1"],
            "coverage_notes": "All covered",
        }
        lines = sparky_runner.format_knowledge_match(km)
        assert any("REUSABLE SUBTASKS" in l for l in lines)
        assert any("RELEVANT OBSERVATIONS" in l for l in lines)
        assert any("COVERAGE NOTES" in l for l in lines)

    def test_empty_match(self) -> None:
        km: dict[str, Any] = {
            "reusable_subtasks": [],
            "relevant_observations": [],
            "coverage_notes": "",
        }
        assert sparky_runner.format_knowledge_match(km) == []

    def test_partial_match_only_observations(self) -> None:
        km: dict[str, Any] = {
            "reusable_subtasks": [],
            "relevant_observations": ["slow load"],
            "coverage_notes": "",
        }
        lines = sparky_runner.format_knowledge_match(km)
        assert any("RELEVANT OBSERVATIONS" in l for l in lines)
        assert not any("REUSABLE SUBTASKS" in l for l in lines)
        assert not any("COVERAGE NOTES" in l for l in lines)
