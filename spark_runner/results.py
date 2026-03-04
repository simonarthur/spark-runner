"""Query and display run results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from spark_runner.models import ScreenshotRecord


@dataclass
class RunSummary:
    """Lightweight summary of a single run."""

    task_name: str
    timestamp: str
    run_dir: Path
    num_phases: int = 0
    has_errors: bool = False
    prompt: str = ""


@dataclass
class PhaseDetail:
    """Detailed info about one phase in a run."""

    name: str = ""
    outcome: str = ""
    screenshots: list[ScreenshotRecord] = field(default_factory=list)


@dataclass
class RunDetail:
    """Full detail of a single run."""

    task_name: str = ""
    prompt: str = ""
    timestamp: str = ""
    base_url: str = ""
    credential_profile: str = ""
    run_dir: Path = field(default_factory=lambda: Path())
    phases: list[PhaseDetail] = field(default_factory=list)
    screenshots: list[ScreenshotRecord] = field(default_factory=list)


def list_runs(
    runs_dir: Path, task_name: str | None = None
) -> list[RunSummary]:
    """List all runs, optionally filtered by task name.

    Args:
        runs_dir: The top-level runs directory.
        task_name: Optional task name to filter by.

    Returns:
        A list of ``RunSummary`` sorted by timestamp (newest first).
    """
    summaries: list[RunSummary] = []

    if not runs_dir.exists():
        return summaries

    task_dirs: list[Path]
    if task_name:
        task_dir = runs_dir / task_name
        task_dirs = [task_dir] if task_dir.exists() else []
    else:
        task_dirs = [d for d in sorted(runs_dir.iterdir()) if d.is_dir()]

    for task_dir in task_dirs:
        for run_dir in sorted(task_dir.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue

            # Try to load metadata
            metadata_path = run_dir / "run_metadata.json"
            problem_log = run_dir / "problem_log.txt"

            num_phases = 0
            has_errors = False
            prompt = ""

            if metadata_path.exists():
                try:
                    meta: dict[str, Any] = json.loads(metadata_path.read_text())
                    num_phases = len(meta.get("phases", []))
                    prompt = meta.get("prompt", "")
                    has_errors = any(
                        p.get("outcome") != "SUCCESS" for p in meta.get("phases", [])
                    )
                except (json.JSONDecodeError, OSError):
                    pass

            # Fall back to problem log only when no metadata is available
            if not metadata_path.exists():
                has_errors = problem_log.exists() and problem_log.stat().st_size > 0

            summaries.append(RunSummary(
                task_name=task_dir.name,
                timestamp=run_dir.name,
                run_dir=run_dir,
                num_phases=num_phases,
                has_errors=has_errors,
                prompt=prompt,
            ))

    return summaries


def get_run_detail(run_dir: Path) -> RunDetail:
    """Load full detail for a single run.

    Args:
        run_dir: Path to the run directory.

    Returns:
        A ``RunDetail`` with all available information.
    """
    detail = RunDetail(run_dir=run_dir)

    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            meta: dict[str, Any] = json.loads(metadata_path.read_text())
            detail.task_name = meta.get("task_name", "")
            detail.prompt = meta.get("prompt", "")
            detail.timestamp = meta.get("timestamp", "")
            detail.base_url = meta.get("base_url", "")
            detail.credential_profile = meta.get("credential_profile", "")

            for phase_data in meta.get("phases", []):
                screenshots: list[ScreenshotRecord] = []
                for ss_data in phase_data.get("screenshots", []):
                    screenshots.append(ScreenshotRecord(
                        path=run_dir / ss_data.get("path", ""),
                        event_type=ss_data.get("event_type", ""),
                        phase_name=phase_data.get("name", ""),
                        timestamp=ss_data.get("timestamp", ""),
                    ))
                detail.phases.append(PhaseDetail(
                    name=phase_data.get("name", ""),
                    outcome=phase_data.get("outcome", ""),
                    screenshots=screenshots,
                ))

            for ss_data in meta.get("screenshots", []):
                detail.screenshots.append(ScreenshotRecord(
                    path=run_dir / ss_data.get("path", ""),
                    event_type=ss_data.get("event_type", ""),
                    timestamp=ss_data.get("timestamp", ""),
                ))
        except (json.JSONDecodeError, OSError):
            pass

    # Infer from directory structure if no metadata
    if not detail.task_name:
        detail.task_name = run_dir.parent.name
    if not detail.timestamp:
        detail.timestamp = run_dir.name

    return detail


def format_run_summary(run: RunSummary) -> str:
    """Format a run summary for CLI display."""
    status = "ERRORS" if run.has_errors else "OK"
    parts = [f"  {run.task_name}/{run.timestamp}  [{status}]"]
    if run.num_phases:
        parts.append(f"  ({run.num_phases} phases)")
    if run.prompt:
        preview = run.prompt[:60] + "..." if len(run.prompt) > 60 else run.prompt
        parts.append(f'  "{preview}"')
    return " ".join(parts)


def format_run_detail(detail: RunDetail) -> str:
    """Format full run detail for CLI display."""
    lines: list[str] = [
        f"Run: {detail.task_name}/{detail.timestamp}",
        f"  Prompt: {detail.prompt or '(unknown)'}",
        f"  Base URL: {detail.base_url or '(unknown)'}",
        f"  Credential profile: {detail.credential_profile or 'default'}",
        f"  Run dir: {detail.run_dir}",
    ]

    if detail.phases:
        lines.append(f"\n  Phases ({len(detail.phases)}):")
        for i, phase in enumerate(detail.phases, 1):
            lines.append(f"    {i}. {phase.name} [{phase.outcome}]")
            for ss in phase.screenshots:
                lines.append(f"       Screenshot: {ss.path.name} ({ss.event_type})")

    if detail.screenshots:
        lines.append(f"\n  Task-level screenshots ({len(detail.screenshots)}):")
        for ss in detail.screenshots:
            lines.append(f"    {ss.path.name} ({ss.event_type})")

    # Show problem log (actual errors) if it exists
    problem_log = detail.run_dir / "problem_log.txt"
    if problem_log.exists():
        content = problem_log.read_text().strip()
        if content:
            lines.append("\n  Errors:")
            for line in content.splitlines()[:20]:
                lines.append(f"    {line}")
            total_lines = content.count("\n") + 1
            if total_lines > 20:
                lines.append(f"    ... ({total_lines - 20} more lines)")

    return "\n".join(lines)


def write_run_metadata(
    run_dir: Path,
    task_name: str,
    prompt: str,
    base_url: str,
    credential_profile: str,
    phases: list[dict[str, Any]],
    screenshots: list[ScreenshotRecord] | None = None,
) -> None:
    """Write run_metadata.json to the run directory.

    Args:
        run_dir: Path to the run directory.
        task_name: Name of the task.
        prompt: The task prompt.
        base_url: Base URL used.
        credential_profile: Credential profile used.
        phases: List of phase dicts with name, outcome, and screenshots.
        screenshots: Optional task-level screenshots.
    """
    metadata: dict[str, Any] = {
        "task_name": task_name,
        "prompt": prompt,
        "timestamp": datetime.now().isoformat(),
        "base_url": base_url,
        "credential_profile": credential_profile,
        "phases": [],
        "screenshots": [],
    }

    for phase in phases:
        phase_entry: dict[str, Any] = {
            "name": phase.get("name", ""),
            "outcome": phase.get("outcome", ""),
            "screenshots": [],
        }
        for ss in phase.get("screenshots", []):
            if isinstance(ss, ScreenshotRecord):
                phase_entry["screenshots"].append({
                    "path": str(ss.path.relative_to(run_dir)) if ss.path.is_relative_to(run_dir) else str(ss.path),
                    "event_type": ss.event_type,
                    "timestamp": ss.timestamp,
                })
        metadata["phases"].append(phase_entry)

    if screenshots:
        for ss in screenshots:
            metadata["screenshots"].append({
                "path": str(ss.path.relative_to(run_dir)) if ss.path.is_relative_to(run_dir) else str(ss.path),
                "event_type": ss.event_type,
                "timestamp": ss.timestamp,
            })

    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
