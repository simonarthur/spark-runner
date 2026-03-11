"""Screenshot capture and linking with errors/steps."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from spark_runner.models import ScreenshotRecord

_FALLBACK_SCREENSHOT: Path = Path(__file__).parent / "assets" / "fallback_screenshot.png"


def make_screenshots_dir(run_dir: Path) -> Path:
    """Create and return the screenshots subdirectory in a run directory."""
    screenshots_dir: Path = run_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    return screenshots_dir


async def capture_screenshot(
    page: Any,
    run_dir: Path,
    filename: str,
    event_type: str,
    phase_name: str = "",
    step_number: int | None = None,
    error_message: str | None = None,
) -> ScreenshotRecord | None:
    """Capture a screenshot and return a record.

    Args:
        page: Playwright page object.
        run_dir: Run directory for artifacts.
        filename: Name for the screenshot file.
        event_type: Type of event triggering the screenshot.
        phase_name: Name of the current phase.
        step_number: Optional step number.
        error_message: Optional error message associated with the screenshot.

    Returns:
        A ``ScreenshotRecord`` on success, or ``None`` if capture failed.
    """
    screenshots_dir = make_screenshots_dir(run_dir)
    screenshot_path = screenshots_dir / filename
    try:
        await page.screenshot(str(screenshot_path))
    except Exception:
        shutil.copy2(str(_FALLBACK_SCREENSHOT), str(screenshot_path))
    return ScreenshotRecord(
        path=screenshot_path,
        event_type=event_type,
        phase_name=phase_name,
        step_number=step_number,
        error_message=error_message,
        timestamp=datetime.now().isoformat(),
    )


async def capture_phase_end_screenshot(
    page: Any,
    run_dir: Path,
    phase_name: str,
    success: bool,
) -> ScreenshotRecord | None:
    """Capture a screenshot at phase end.

    Args:
        page: Playwright page object.
        run_dir: Run directory for artifacts.
        phase_name: Name of the phase that just completed.
        success: Whether the phase succeeded.
    """
    slug = phase_name.lower().replace(" ", "-")
    suffix = "end" if success else "error"
    filename = f"phase-{slug}-{suffix}.png"
    return await capture_screenshot(
        page, run_dir, filename,
        event_type="phase_end",
        phase_name=phase_name,
    )


async def capture_error_screenshot(
    page: Any,
    run_dir: Path,
    phase_name: str,
    step_number: int,
    error_message: str,
) -> ScreenshotRecord | None:
    """Capture a screenshot when an error occurs during a phase.

    Args:
        page: Playwright page object.
        run_dir: Run directory for artifacts.
        phase_name: Name of the current phase.
        step_number: The step number where the error occurred.
        error_message: The error message.
    """
    slug = phase_name.lower().replace(" ", "-")
    filename = f"phase-{slug}-error-step{step_number}.png"
    return await capture_screenshot(
        page, run_dir, filename,
        event_type="error",
        phase_name=phase_name,
        step_number=step_number,
        error_message=error_message,
    )


async def capture_task_end_screenshot(
    page: Any,
    run_dir: Path,
) -> ScreenshotRecord | None:
    """Capture a screenshot at the end of the entire task.

    Args:
        page: Playwright page object.
        run_dir: Run directory for artifacts.
    """
    return await capture_screenshot(
        page, run_dir, "task-end.png",
        event_type="task_end",
    )
