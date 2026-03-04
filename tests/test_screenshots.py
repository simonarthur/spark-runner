"""Tests for spark_runner.screenshots: directory creation and capture helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from spark_runner.models import ScreenshotRecord
from spark_runner.screenshots import (
    capture_screenshot,
    make_screenshots_dir,
)


# ── make_screenshots_dir ─────────────────────────────────────────────────


class TestMakeScreenshotsDir:
    def test_creates_screenshots_subdirectory(self, tmp_path: Path) -> None:
        result = make_screenshots_dir(tmp_path)
        assert result.exists()
        assert result.is_dir()

    def test_returns_correct_path(self, tmp_path: Path) -> None:
        result = make_screenshots_dir(tmp_path)
        assert result == tmp_path / "screenshots"

    def test_idempotent_when_already_exists(self, tmp_path: Path) -> None:
        existing = tmp_path / "screenshots"
        existing.mkdir()
        result = make_screenshots_dir(tmp_path)
        assert result == existing

    def test_creates_intermediate_run_dir(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs" / "my-task" / "20250101_120000"
        run_dir.mkdir(parents=True)
        result = make_screenshots_dir(run_dir)
        assert result.name == "screenshots"
        assert result.parent == run_dir


# ── capture_screenshot ───────────────────────────────────────────────────


class TestCaptureScreenshot:
    @pytest.mark.asyncio
    async def test_success_returns_screenshot_record(self, tmp_path: Path) -> None:
        page = AsyncMock()
        page.screenshot = AsyncMock(return_value=None)

        result = await capture_screenshot(
            page=page,
            run_dir=tmp_path,
            filename="test.png",
            event_type="phase_end",
            phase_name="Login",
        )

        assert isinstance(result, ScreenshotRecord)
        assert result.event_type == "phase_end"
        assert result.phase_name == "Login"
        assert result.path == tmp_path / "screenshots" / "test.png"

    @pytest.mark.asyncio
    async def test_success_sets_timestamp(self, tmp_path: Path) -> None:
        page = AsyncMock()
        page.screenshot = AsyncMock(return_value=None)

        result = await capture_screenshot(
            page=page,
            run_dir=tmp_path,
            filename="snap.png",
            event_type="task_end",
        )

        assert result is not None
        assert result.timestamp != ""

    @pytest.mark.asyncio
    async def test_calls_page_screenshot_with_correct_path(self, tmp_path: Path) -> None:
        page = AsyncMock()
        page.screenshot = AsyncMock(return_value=None)

        await capture_screenshot(
            page=page,
            run_dir=tmp_path,
            filename="output.png",
            event_type="error",
        )

        expected_path = str(tmp_path / "screenshots" / "output.png")
        page.screenshot.assert_called_once_with(expected_path)

    @pytest.mark.asyncio
    async def test_failure_returns_none(self, tmp_path: Path) -> None:
        page = AsyncMock()
        page.screenshot = AsyncMock(side_effect=RuntimeError("browser crashed"))

        result = await capture_screenshot(
            page=page,
            run_dir=tmp_path,
            filename="fail.png",
            event_type="error",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_includes_step_number(self, tmp_path: Path) -> None:
        page = AsyncMock()
        page.screenshot = AsyncMock(return_value=None)

        result = await capture_screenshot(
            page=page,
            run_dir=tmp_path,
            filename="step.png",
            event_type="step",
            phase_name="Search",
            step_number=3,
        )

        assert result is not None
        assert result.step_number == 3

    @pytest.mark.asyncio
    async def test_includes_error_message(self, tmp_path: Path) -> None:
        page = AsyncMock()
        page.screenshot = AsyncMock(return_value=None)

        result = await capture_screenshot(
            page=page,
            run_dir=tmp_path,
            filename="err.png",
            event_type="error",
            error_message="Element not found",
        )

        assert result is not None
        assert result.error_message == "Element not found"

    @pytest.mark.asyncio
    async def test_creates_screenshots_dir_if_missing(self, tmp_path: Path) -> None:
        page = AsyncMock()
        page.screenshot = AsyncMock(return_value=None)

        screenshots_dir = tmp_path / "screenshots"
        assert not screenshots_dir.exists()

        await capture_screenshot(
            page=page,
            run_dir=tmp_path,
            filename="auto_dir.png",
            event_type="task_end",
        )

        assert screenshots_dir.exists()

    @pytest.mark.asyncio
    async def test_page_exception_type_does_not_matter(self, tmp_path: Path) -> None:
        """Any exception from page.screenshot should result in None, not a re-raise."""
        page = AsyncMock()
        page.screenshot = AsyncMock(side_effect=TimeoutError("timed out"))

        result = await capture_screenshot(
            page=page,
            run_dir=tmp_path,
            filename="timeout.png",
            event_type="error",
        )

        assert result is None
