"""Tests for spark_runner.results: run listing, filtering, detail formatting, metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from spark_runner.models import ScreenshotRecord
from spark_runner.results import (
    RunDetail,
    RunSummary,
    format_run_summary,
    get_run_detail,
    list_runs,
    write_run_metadata,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_run_dir(
    runs_dir: Path,
    task_name: str,
    timestamp: str,
    metadata: dict[str, Any] | None = None,
    problem_log_content: str = "",
) -> Path:
    """Create a minimal run directory with optional metadata and problem log."""
    run_dir = runs_dir / task_name / timestamp
    run_dir.mkdir(parents=True)

    if metadata is not None:
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata))

    if problem_log_content:
        (run_dir / "problem_log.txt").write_text(problem_log_content)

    return run_dir


# ── list_runs ────────────────────────────────────────────────────────────


class TestListRuns:
    def test_empty_runs_dir_returns_empty_list(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        result = list_runs(runs_dir)
        assert result == []

    def test_nonexistent_runs_dir_returns_empty_list(self, tmp_path: Path) -> None:
        result = list_runs(tmp_path / "no_such_dir")
        assert result == []

    def test_finds_runs_in_task_directory(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        _make_run_dir(runs_dir, "my-task", "20250101_120000")
        result = list_runs(runs_dir)
        assert len(result) == 1
        assert result[0].task_name == "my-task"
        assert result[0].timestamp == "20250101_120000"

    def test_sorts_newest_run_first(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        _make_run_dir(runs_dir, "task-a", "20250101_120000")
        _make_run_dir(runs_dir, "task-a", "20250102_120000")
        _make_run_dir(runs_dir, "task-a", "20250103_120000")
        result = list_runs(runs_dir)
        timestamps = [r.timestamp for r in result if r.task_name == "task-a"]
        assert timestamps == sorted(timestamps, reverse=True)

    def test_filters_by_task_name(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        _make_run_dir(runs_dir, "task-a", "20250101_120000")
        _make_run_dir(runs_dir, "task-b", "20250102_120000")
        result = list_runs(runs_dir, task_name="task-a")
        assert len(result) == 1
        assert result[0].task_name == "task-a"

    def test_filter_nonexistent_task_returns_empty(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        _make_run_dir(runs_dir, "task-a", "20250101_120000")
        result = list_runs(runs_dir, task_name="ghost-task")
        assert result == []

    def test_loads_phase_count_from_metadata(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        metadata: dict[str, Any] = {
            "phases": [
                {"name": "Login", "outcome": "SUCCESS"},
                {"name": "Search", "outcome": "SUCCESS"},
            ]
        }
        _make_run_dir(runs_dir, "task-a", "20250101_120000", metadata=metadata)
        result = list_runs(runs_dir)
        assert result[0].num_phases == 2

    def test_has_errors_when_phase_failed(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        metadata: dict[str, Any] = {
            "phases": [{"name": "Login", "outcome": "FAILED"}]
        }
        _make_run_dir(runs_dir, "task-a", "20250101_120000", metadata=metadata)
        result = list_runs(runs_dir)
        assert result[0].has_errors is True

    def test_has_errors_when_problem_log_nonempty(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        _make_run_dir(
            runs_dir,
            "task-a",
            "20250101_120000",
            problem_log_content="Something went wrong",
        )
        result = list_runs(runs_dir)
        assert result[0].has_errors is True

    def test_no_errors_when_all_phases_succeed(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        metadata: dict[str, Any] = {
            "phases": [{"name": "Login", "outcome": "SUCCESS"}]
        }
        _make_run_dir(runs_dir, "task-a", "20250101_120000", metadata=metadata)
        result = list_runs(runs_dir)
        assert result[0].has_errors is False

    def test_loads_prompt_from_metadata(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        metadata: dict[str, Any] = {"prompt": "Do the thing", "phases": []}
        _make_run_dir(runs_dir, "task-a", "20250101_120000", metadata=metadata)
        result = list_runs(runs_dir)
        assert result[0].prompt == "Do the thing"

    def test_run_summary_has_run_dir(self, tmp_path: Path) -> None:
        runs_dir = tmp_path / "runs"
        expected_dir = _make_run_dir(runs_dir, "task-a", "20250101_120000")
        result = list_runs(runs_dir)
        assert result[0].run_dir == expected_dir


# ── get_run_detail ───────────────────────────────────────────────────────


class TestGetRunDetail:
    def test_loads_metadata_fields(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "my-run"
        run_dir.mkdir()
        metadata: dict[str, Any] = {
            "task_name": "my-task",
            "prompt": "Do something",
            "timestamp": "2025-01-01T12:00:00",
            "base_url": "https://example.com",
            "credential_profile": "admin",
            "phases": [],
            "screenshots": [],
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata))

        detail = get_run_detail(run_dir)

        assert detail.task_name == "my-task"
        assert detail.prompt == "Do something"
        assert detail.base_url == "https://example.com"
        assert detail.credential_profile == "admin"

    def test_infers_task_name_from_dir_when_no_metadata(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "inferred-task"
        run_dir = task_dir / "20250101_120000"
        run_dir.mkdir(parents=True)

        detail = get_run_detail(run_dir)

        assert detail.task_name == "inferred-task"

    def test_infers_timestamp_from_dir_name(self, tmp_path: Path) -> None:
        task_dir = tmp_path / "task-x"
        run_dir = task_dir / "20250201_090000"
        run_dir.mkdir(parents=True)

        detail = get_run_detail(run_dir)

        assert detail.timestamp == "20250201_090000"

    def test_loads_phases_from_metadata(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        metadata: dict[str, Any] = {
            "task_name": "t",
            "phases": [
                {"name": "Login", "outcome": "SUCCESS", "screenshots": []},
                {"name": "Search", "outcome": "FAILED", "screenshots": []},
            ],
        }
        (run_dir / "run_metadata.json").write_text(json.dumps(metadata))

        detail = get_run_detail(run_dir)

        assert len(detail.phases) == 2
        assert detail.phases[0].name == "Login"
        assert detail.phases[0].outcome == "SUCCESS"
        assert detail.phases[1].outcome == "FAILED"

    def test_handles_missing_metadata_gracefully(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "empty-run"
        run_dir.mkdir()
        detail = get_run_detail(run_dir)
        assert isinstance(detail, RunDetail)

    def test_handles_malformed_metadata_gracefully(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "bad-run"
        run_dir.mkdir()
        (run_dir / "run_metadata.json").write_text("not valid json {{{")
        detail = get_run_detail(run_dir)
        assert isinstance(detail, RunDetail)


# ── format_run_summary ───────────────────────────────────────────────────


class TestFormatRunSummary:
    def test_includes_task_name_and_timestamp(self) -> None:
        run = RunSummary(
            task_name="my-task",
            timestamp="20250101_120000",
            run_dir=Path("/runs/my-task/20250101_120000"),
        )
        output = format_run_summary(run)
        assert "my-task" in output
        assert "20250101_120000" in output

    def test_ok_status_when_no_errors(self) -> None:
        run = RunSummary(
            task_name="t",
            timestamp="ts",
            run_dir=Path("/runs/t/ts"),
            has_errors=False,
        )
        output = format_run_summary(run)
        assert "OK" in output

    def test_errors_status_when_has_errors(self) -> None:
        run = RunSummary(
            task_name="t",
            timestamp="ts",
            run_dir=Path("/runs/t/ts"),
            has_errors=True,
        )
        output = format_run_summary(run)
        assert "ERRORS" in output

    def test_includes_phase_count(self) -> None:
        run = RunSummary(
            task_name="t",
            timestamp="ts",
            run_dir=Path("/runs/t/ts"),
            num_phases=3,
        )
        output = format_run_summary(run)
        assert "3" in output

    def test_includes_prompt_preview(self) -> None:
        run = RunSummary(
            task_name="t",
            timestamp="ts",
            run_dir=Path("/runs/t/ts"),
            prompt="Search for products and add to cart",
        )
        output = format_run_summary(run)
        assert "Search for products" in output

    def test_long_prompt_is_truncated(self) -> None:
        run = RunSummary(
            task_name="t",
            timestamp="ts",
            run_dir=Path("/runs/t/ts"),
            prompt="A" * 100,
        )
        output = format_run_summary(run)
        assert "..." in output

    def test_no_phase_count_when_zero(self) -> None:
        run = RunSummary(
            task_name="t",
            timestamp="ts",
            run_dir=Path("/runs/t/ts"),
            num_phases=0,
        )
        output = format_run_summary(run)
        # Zero phases should not clutter the output with "(0 phases)"
        assert "0 phases" not in output

    def test_returns_string(self) -> None:
        run = RunSummary(task_name="t", timestamp="ts", run_dir=Path("/r"))
        assert isinstance(format_run_summary(run), str)


# ── write_run_metadata ───────────────────────────────────────────────────


class TestWriteRunMetadata:
    def test_creates_json_file(self, tmp_path: Path) -> None:
        write_run_metadata(
            run_dir=tmp_path,
            task_name="test-task",
            prompt="Do a thing",
            base_url="https://example.com",
            credential_profile="default",
            phases=[],
        )
        metadata_path = tmp_path / "run_metadata.json"
        assert metadata_path.exists()

    def test_written_file_is_valid_json(self, tmp_path: Path) -> None:
        write_run_metadata(
            run_dir=tmp_path,
            task_name="test-task",
            prompt="Do a thing",
            base_url="https://example.com",
            credential_profile="default",
            phases=[],
        )
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert isinstance(data, dict)

    def test_metadata_contains_expected_fields(self, tmp_path: Path) -> None:
        write_run_metadata(
            run_dir=tmp_path,
            task_name="my-task",
            prompt="My prompt",
            base_url="https://example.com",
            credential_profile="staging",
            phases=[],
        )
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert data["task_name"] == "my-task"
        assert data["prompt"] == "My prompt"
        assert data["base_url"] == "https://example.com"
        assert data["credential_profile"] == "staging"
        assert "timestamp" in data

    def test_phases_written_to_metadata(self, tmp_path: Path) -> None:
        phases: list[dict[str, Any]] = [
            {"name": "Login", "outcome": "SUCCESS", "screenshots": []},
            {"name": "Search", "outcome": "FAILED", "screenshots": []},
        ]
        write_run_metadata(
            run_dir=tmp_path,
            task_name="t",
            prompt="p",
            base_url="https://example.com",
            credential_profile="default",
            phases=phases,
        )
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert len(data["phases"]) == 2
        assert data["phases"][0]["name"] == "Login"
        assert data["phases"][1]["outcome"] == "FAILED"

    def test_screenshots_written_to_metadata(self, tmp_path: Path) -> None:
        screenshots_dir = tmp_path / "screenshots"
        screenshots_dir.mkdir()
        ss = ScreenshotRecord(
            path=screenshots_dir / "task-end.png",
            event_type="task_end",
            timestamp="2025-01-01T12:00:00",
        )
        write_run_metadata(
            run_dir=tmp_path,
            task_name="t",
            prompt="p",
            base_url="https://example.com",
            credential_profile="default",
            phases=[],
            screenshots=[ss],
        )
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert len(data["screenshots"]) == 1
        assert data["screenshots"][0]["event_type"] == "task_end"

    def test_timestamp_is_iso_format(self, tmp_path: Path) -> None:
        write_run_metadata(
            run_dir=tmp_path,
            task_name="t",
            prompt="p",
            base_url="https://example.com",
            credential_profile="default",
            phases=[],
        )
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        # ISO format contains 'T' separator
        assert "T" in data["timestamp"]

    def test_environment_recorded_when_provided(self, tmp_path: Path) -> None:
        write_run_metadata(
            run_dir=tmp_path,
            task_name="t",
            prompt="p",
            base_url="https://example.com",
            credential_profile="default",
            phases=[],
            environment="staging",
        )
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert data["environment"] == "staging"

    def test_environment_omitted_when_none(self, tmp_path: Path) -> None:
        write_run_metadata(
            run_dir=tmp_path,
            task_name="t",
            prompt="p",
            base_url="https://example.com",
            credential_profile="default",
            phases=[],
        )
        data = json.loads((tmp_path / "run_metadata.json").read_text())
        assert "environment" not in data
