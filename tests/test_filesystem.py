"""Tests for filesystem-related functions in spark_runner."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pytest

import spark_runner
from spark_runner.storage import write_with_history, _has_history


# ── safe_write_path ──────────────────────────────────────────────────────

class TestSafeWritePath:
    def test_no_conflict(self, tmp_path: Path) -> None:
        p = tmp_path / "report.json"
        assert spark_runner.safe_write_path(p) == p

    def test_single_conflict(self, tmp_path: Path) -> None:
        p = tmp_path / "report.json"
        p.write_text("existing")
        result = spark_runner.safe_write_path(p)
        assert result == tmp_path / "report-2.json"

    def test_multiple_conflicts(self, tmp_path: Path) -> None:
        for suffix in ("", "-2", "-3"):
            (tmp_path / f"report{suffix}.json").write_text("x")
        result = spark_runner.safe_write_path(tmp_path / "report.json")
        assert result == tmp_path / "report-4.json"


# ── write_with_history ───────────────────────────────────────────────────


class TestWriteWithHistory:
    def test_new_file_creates_file_and_history(self, tmp_path: Path) -> None:
        p = tmp_path / "login-task.json"
        write_with_history(p, '{"v": 1}')
        assert p.read_text() == '{"v": 1}'
        history = list(tmp_path.glob("login-task-[0-9]*-[0-9]*.json"))
        assert len(history) == 1
        assert history[0].read_text() == '{"v": 1}'

    def test_existing_file_bootstraps_old_version(self, tmp_path: Path) -> None:
        p = tmp_path / "login-task.json"
        p.write_text('{"v": 0}')
        # Set a known mtime so we can verify the bootstrap stamp
        old_time = 1700000000.0  # 2023-11-14
        os.utime(p, (old_time, old_time))

        write_with_history(p, '{"v": 1}')

        assert p.read_text() == '{"v": 1}'
        history = sorted(tmp_path.glob("login-task-[0-9]*-[0-9]*.json"))
        # Should have bootstrap copy + new copy = 2 history files
        assert len(history) == 2
        # Bootstrap copy has old content
        assert history[0].read_text() == '{"v": 0}'
        # New copy has new content
        assert history[1].read_text() == '{"v": 1}'

    def test_no_double_bootstrap(self, tmp_path: Path) -> None:
        p = tmp_path / "login-task.json"
        p.write_text('{"v": 0}')
        old_time = 1700000000.0
        os.utime(p, (old_time, old_time))
        # First write bootstraps
        write_with_history(p, '{"v": 1}')
        # Second write should NOT bootstrap again
        write_with_history(p, '{"v": 2}')

        assert p.read_text() == '{"v": 2}'
        history = sorted(tmp_path.glob("login-task-[0-9]*.json"))
        # bootstrap (old mtime) + first write + second write = 3
        assert len(history) == 3
        # Bootstrap copy has the original content
        assert history[0].read_text() == '{"v": 0}'

    def test_returns_history_path(self, tmp_path: Path) -> None:
        p = tmp_path / "task.txt"
        result = write_with_history(p, "content")
        assert result.exists()
        assert result.read_text() == "content"
        assert result != p

    def test_txt_file_history(self, tmp_path: Path) -> None:
        p = tmp_path / "fill-form.txt"
        write_with_history(p, "step 1")
        history = list(tmp_path.glob("fill-form-[0-9]*-[0-9]*.txt"))
        assert len(history) == 1
        assert history[0].read_text() == "step 1"


class TestHasHistory:
    def test_no_history(self, tmp_path: Path) -> None:
        p = tmp_path / "login-task.json"
        p.write_text("{}")
        assert _has_history(p) is False

    def test_with_history(self, tmp_path: Path) -> None:
        p = tmp_path / "login-task.json"
        p.write_text("{}")
        (tmp_path / "login-task-20240101-120000.json").write_text("{}")
        assert _has_history(p) is True

    def test_unrelated_file_not_counted(self, tmp_path: Path) -> None:
        p = tmp_path / "login-task.json"
        p.write_text("{}")
        (tmp_path / "logout-task-20240101-120000.json").write_text("{}")
        assert _has_history(p) is False


# ── log_event ────────────────────────────────────────────────────────────

class TestLogEvent:
    def test_appends_timestamped_line(self, tmp_path: Path) -> None:
        log = tmp_path / "event.log"
        spark_runner.log_event(log, "hello")
        spark_runner.log_event(log, "world")
        lines = log.read_text().splitlines()
        assert len(lines) == 2
        assert "hello" in lines[0]
        assert "world" in lines[1]

    def test_prints_to_stdout(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        spark_runner.log_event(tmp_path / "e.log", "visible")
        assert "visible" in capsys.readouterr().out


# ── log_problem ──────────────────────────────────────────────────────────

class TestLogProblem:
    def test_appends_to_file(self, tmp_path: Path) -> None:
        log = tmp_path / "problem.log"
        spark_runner.log_problem(log, "issue 1")
        spark_runner.log_problem(log, "issue 2")
        assert log.read_text().count("\n") == 2

    def test_does_not_print(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        spark_runner.log_problem(tmp_path / "p.log", "silent")
        assert capsys.readouterr().out == ""


# ── load_knowledge_index ─────────────────────────────────────────────────

class TestLoadKnowledgeIndex:
    def test_empty_dir(self, fake_tasks_dir: Path) -> None:
        assert spark_runner.load_knowledge_index() == []

    def test_loads_multiple_txt_files(self, fake_tasks_dir: Path) -> None:
        (fake_tasks_dir / "login.txt").write_text("Logged in OK")
        (fake_tasks_dir / "search.txt").write_text("Searched items")

        index = spark_runner.load_knowledge_index()
        assert len(index) == 2
        filenames = [entry["filename"] for entry in index]
        assert "login.txt" in filenames
        assert "search.txt" in filenames

    def test_ignores_non_txt_files(self, fake_tasks_dir: Path) -> None:
        (fake_tasks_dir / "login.txt").write_text("Logged in OK")
        (fake_tasks_dir / "notes.json").write_text('{"key": "value"}')
        (fake_tasks_dir / "readme.md").write_text("# Readme")

        index = spark_runner.load_knowledge_index()
        assert len(index) == 1
        assert index[0]["filename"] == "login.txt"

    def test_name_derived_from_filename(self, fake_tasks_dir: Path) -> None:
        (fake_tasks_dir / "fill-login-form.txt").write_text("content")

        index = spark_runner.load_knowledge_index()
        assert index[0]["name"] == "Fill Login Form"

    def test_content_loaded(self, fake_tasks_dir: Path) -> None:
        (fake_tasks_dir / "login.txt").write_text("Step 1: Navigate to login page")

        index = spark_runner.load_knowledge_index()
        assert index[0]["content"] == "Step 1: Navigate to login page"

    def test_restore_fn_applied_to_content(self, fake_tasks_dir: Path) -> None:
        (fake_tasks_dir / "step.txt").write_text("Visit {BASE_URL}/page")

        index = spark_runner.load_knowledge_index()
        assert "https://test.example.com/page" in index[0]["content"]

    def test_preserves_credential_placeholders(self, fake_tasks_dir: Path) -> None:
        """Credential placeholders like {USER_EMAIL} must survive into the knowledge index."""
        (fake_tasks_dir / "step.txt").write_text("Login as {USER_EMAIL} with {USER_PASSWORD}")

        index = spark_runner.load_knowledge_index()
        content: str = index[0]["content"]
        assert "{USER_EMAIL}" in content
        assert "{USER_PASSWORD}" in content
        assert "test@example.com" not in content
        assert "test-password-123" not in content

    def test_skips_unreadable_files(
        self, fake_tasks_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        task_file = fake_tasks_dir / "broken.txt"
        task_file.write_text("content")
        task_file.chmod(0o000)

        index = spark_runner.load_knowledge_index()
        assert index == []

        captured = capsys.readouterr()
        assert "Warning: could not read" in captured.out

        # Restore permissions for cleanup
        task_file.chmod(0o644)


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

        prompt, task_name, phases = spark_runner.load_goal_summary(goal_path)
        assert prompt == "Login test"
        assert task_name == "goal"
        assert len(phases) == 1
        assert phases[0]["name"] == "Login"
        assert spark_runner._REPLAY_PREFIX in phases[0]["task"]

    def test_skips_missing_subtask_with_warning(
        self, fake_tasks_dir: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        goal: dict[str, Any] = {
            "main_task": "test",
            "subtasks": [{"filename": "nope.txt"}],
        }
        goal_path = fake_tasks_dir.parent / "goal.json"
        goal_path.write_text(json.dumps(goal))
        prompt, task_name, phases = spark_runner.load_goal_summary(goal_path)
        assert prompt == "test"
        assert phases == []
        assert "Warning: subtask file not found" in capsys.readouterr().out

    def test_skips_non_dict_subtask_entries(self, fake_tasks_dir: Path) -> None:
        (fake_tasks_dir / "step.txt").write_text("Do the thing")
        goal: dict[str, Any] = {
            "main_task": "test",
            "subtasks": ["step.txt", {"filename": "step.txt"}],
        }
        goal_path = fake_tasks_dir.parent / "goal.json"
        goal_path.write_text(json.dumps(goal))
        prompt, task_name, phases = spark_runner.load_goal_summary(goal_path)
        assert prompt == "test"
        # The plain string entry is skipped; only the dict entry produces a phase
        assert len(phases) == 1

    def test_preserves_credential_placeholders(self, fake_tasks_dir: Path) -> None:
        """Credential placeholders in task text must not be resolved for LLM consumption."""
        (fake_tasks_dir / "login.txt").write_text("Log in as {USER_EMAIL} with {USER_PASSWORD}")
        goal: dict[str, Any] = {
            "main_task": "Login test",
            "subtasks": [{"filename": "login.txt"}],
        }
        goal_path = fake_tasks_dir.parent / "goal.json"
        goal_path.write_text(json.dumps(goal))

        _, _, phases = spark_runner.load_goal_summary(goal_path)
        task_text: str = phases[0]["task"]
        assert "{USER_EMAIL}" in task_text
        assert "{USER_PASSWORD}" in task_text
        assert "test@example.com" not in task_text
        assert "test-password-123" not in task_text

    def test_replay_prefix_injected(self, fake_tasks_dir: Path) -> None:
        (fake_tasks_dir / "nav.txt").write_text("Navigated to page")
        goal: dict[str, Any] = {
            "main_task": "nav test",
            "subtasks": [{"filename": "nav.txt"}],
        }
        goal_path = fake_tasks_dir.parent / "g.json"
        goal_path.write_text(json.dumps(goal))
        _, _, phases = spark_runner.load_goal_summary(goal_path)
        assert phases[0]["task"].startswith(spark_runner._REPLAY_PREFIX)


# ── _get_orphan_tasks / clean_orphan_tasks ───────────────────────────────

class TestOrphans:
    def test_no_orphans(
        self, fake_tasks_dir: Path, fake_goal_summaries_dir: Path
    ) -> None:
        (fake_tasks_dir / "used.txt").write_text("x")
        goal: dict[str, Any] = {"subtasks": [{"filename": "used.txt"}]}
        (fake_goal_summaries_dir / "g-task.json").write_text(json.dumps(goal))
        assert spark_runner._get_orphan_tasks() == []

    def test_detects_orphan(
        self, fake_tasks_dir: Path, fake_goal_summaries_dir: Path
    ) -> None:
        (fake_tasks_dir / "used.txt").write_text("x")
        (fake_tasks_dir / "orphan.txt").write_text("x")
        goal: dict[str, Any] = {"subtasks": [{"filename": "used.txt"}]}
        (fake_goal_summaries_dir / "g-task.json").write_text(json.dumps(goal))
        orphans = spark_runner._get_orphan_tasks()
        assert orphans == ["orphan.txt"]

    def test_clean_orphan_confirm(
        self,
        fake_tasks_dir: Path,
        fake_goal_summaries_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (fake_tasks_dir / "orphan.txt").write_text("x")
        monkeypatch.setattr("builtins.input", lambda _: "y")
        spark_runner.clean_orphan_tasks()
        assert not (fake_tasks_dir / "orphan.txt").exists()

    def test_clean_orphan_abort(
        self,
        fake_tasks_dir: Path,
        fake_goal_summaries_dir: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        (fake_tasks_dir / "orphan.txt").write_text("x")
        monkeypatch.setattr("builtins.input", lambda _: "n")
        spark_runner.clean_orphan_tasks()
        assert (fake_tasks_dir / "orphan.txt").exists()

    def test_empty_dirs(
        self, fake_tasks_dir: Path, fake_goal_summaries_dir: Path
    ) -> None:
        assert spark_runner._get_orphan_tasks() == []


# ── phase_name_to_slug ──────────────────────────────────────────────────

class TestPhaseNameToSlug:
    def test_basic(self) -> None:
        assert spark_runner.phase_name_to_slug("Login") == "login"

    def test_spaces(self) -> None:
        assert spark_runner.phase_name_to_slug("Fill Form") == "fill-form"

    def test_special_chars(self) -> None:
        assert spark_runner.phase_name_to_slug("Step #1: Login!") == "step-1-login"

    def test_strip_edges(self) -> None:
        assert spark_runner.phase_name_to_slug("---Login---") == "login"

    def test_all_special(self) -> None:
        assert spark_runner.phase_name_to_slug("!!!") == ""


# ── make_run_dir ────────────────────────────────────────────────────────

class TestMakeRunDir:
    def test_creates_nested_dir(self, fake_runs_dir: Path) -> None:
        run_dir = spark_runner.make_run_dir(fake_runs_dir, "login-test")
        assert run_dir.exists()
        assert run_dir.parent.name == "login-test"
        assert run_dir.parent.parent == fake_runs_dir

    def test_dir_name_matches_timestamp_format(self, fake_runs_dir: Path) -> None:
        run_dir = spark_runner.make_run_dir(fake_runs_dir, "t")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", run_dir.name)

    def test_two_calls_same_task_no_crash(self, fake_runs_dir: Path) -> None:
        d1 = spark_runner.make_run_dir(fake_runs_dir, "t")
        d2 = spark_runner.make_run_dir(fake_runs_dir, "t")
        # Even if same second, exist_ok=True prevents a crash
        assert d1.exists() and d2.exists()


# ── format_phase_plan ───────────────────────────────────────────────────

class TestFormatPhasePlan:
    def test_two_phases(self) -> None:
        phases = [
            {"name": "Login", "task": "Do login"},
            {"name": "Search", "task": "Search items"},
        ]
        lines = spark_runner.format_phase_plan(phases)
        assert lines[0] == "PHASE PLAN (2 phases):"
        assert "Phase 1: Login" in lines[1]
        assert "Instructions: Do login" in lines[2]
        assert "Phase 2: Search" in lines[3]
        assert "Instructions: Search items" in lines[4]

    def test_long_task_truncated(self) -> None:
        phases = [{"name": "Big", "task": "x" * 3000}]
        lines = spark_runner.format_phase_plan(phases)
        instructions_line = lines[2]
        # The task value should be truncated to 2000 chars
        assert len(instructions_line) <= len("    Instructions: ") + 2000

    def test_missing_task_key(self) -> None:
        phases = [{"name": "Reuse"}]
        lines = spark_runner.format_phase_plan(phases)
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
        lines = spark_runner.format_knowledge_match(km)
        assert any("REUSABLE SUBTASKS" in l for l in lines)
        assert any("RELEVANT OBSERVATIONS" in l for l in lines)
        assert any("COVERAGE NOTES" in l for l in lines)

    def test_empty_match(self) -> None:
        km: dict[str, Any] = {
            "reusable_subtasks": [],
            "relevant_observations": [],
            "coverage_notes": "",
        }
        assert spark_runner.format_knowledge_match(km) == []

    def test_partial_match_only_observations(self) -> None:
        km: dict[str, Any] = {
            "reusable_subtasks": [],
            "relevant_observations": ["slow load"],
            "coverage_notes": "",
        }
        lines = spark_runner.format_knowledge_match(km)
        assert any("RELEVANT OBSERVATIONS" in l for l in lines)
        assert not any("REUSABLE SUBTASKS" in l for l in lines)
        assert not any("COVERAGE NOTES" in l for l in lines)
