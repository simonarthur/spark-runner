"""File I/O helpers: safe writes, slugification, run directories, orphan management."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def _history_stamp(dt: datetime) -> str:
    """Format a datetime as a compact timestamp for history filenames."""
    return dt.strftime("%Y%m%d-%H%M%S")


def _has_history(path: Path) -> bool:
    """Return True if *path* already has at least one timestamped history sibling."""
    stem = path.stem
    suffix = path.suffix
    # History files look like: stem-YYYYMMDD-HHMMSS.ext (possibly with -N suffix from safe_write_path)
    pattern = f"{stem}-[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9][0-9][0-9]*{suffix}"
    return any(path.parent.glob(pattern))


def write_with_history(path: Path, content: str) -> Path:
    """Write *content* to *path* and create a timestamped history copy.

    If the file already exists and has no prior history backup, the current
    file is first copied to ``<stem>-<mtime>.<ext>`` to preserve it.

    Then the new content is written to *path* **and** to a timestamped copy
    using the current time.

    Returns:
        The history copy path that was created for the new content.
    """
    if path.exists() and not _has_history(path):
        # Bootstrap: preserve the existing file using its last-modified time
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        bootstrap_name = f"{path.stem}-{_history_stamp(mtime)}{path.suffix}"
        bootstrap_path = path.parent / bootstrap_name
        bootstrap_path.write_text(path.read_text())

    path.write_text(content)

    now_stamp = _history_stamp(datetime.now())
    history_name = f"{path.stem}-{now_stamp}{path.suffix}"
    history_path = safe_write_path(path.parent / history_name)
    history_path.write_text(content)
    return history_path


def safe_write_path(path: Path) -> Path:
    """Return a non-conflicting file path, appending ``-2``, ``-3``, etc. if needed.

    Args:
        path: The desired file path.

    Returns:
        The original path if it doesn't exist, otherwise a suffixed variant.
    """
    if not path.exists():
        return path
    stem: str = path.stem
    suffix: str = path.suffix
    parent: Path = path.parent
    i: int = 2
    while True:
        candidate: Path = parent / f"{stem}-{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def phase_name_to_slug(name: str) -> str:
    """Convert a human-readable phase name to a filename-safe slug.

    >>> phase_name_to_slug("Fill Form")
    'fill-form'
    """
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def make_run_dir(runs_dir: Path, task_name: str) -> Path:
    """Create and return a timestamped run directory under *runs_dir*/*task_name*."""
    run_timestamp: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir: Path = runs_dir / task_name / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_orphan_tasks(
    tasks_dir: Path, goal_summaries_dir: Path
) -> list[str]:
    """Return sorted list of task filenames not referenced by any goal summary."""
    task_files: set[str] = {f.name for f in tasks_dir.iterdir() if f.is_file()}

    referenced: set[str] = set()
    for goal_file in goal_summaries_dir.glob("*-task.json"):
        try:
            data: dict[str, Any] = json.loads(goal_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for subtask in data.get("subtasks", []):
            filename: str = subtask.get("filename", "")
            if filename:
                referenced.add(filename)

    return sorted(task_files - referenced)


def find_orphan_tasks(
    tasks_dir: Path, goal_summaries_dir: Path
) -> None:
    """Print task files in *tasks_dir* not referenced by any goal summary."""
    orphans: list[str] = get_orphan_tasks(tasks_dir, goal_summaries_dir)
    if not orphans:
        print("No orphan task files found.")
        return

    print(f"Found {len(orphans)} orphan task file(s) not referenced in any goal summary:\n")
    for name in orphans:
        print(f"  {name}")


def clean_orphan_tasks(
    tasks_dir: Path, goal_summaries_dir: Path
) -> None:
    """Delete task files not referenced by any goal summary (with confirmation)."""
    orphans: list[str] = get_orphan_tasks(tasks_dir, goal_summaries_dir)
    if not orphans:
        print("No orphan task files to clean.")
        return

    print(f"Found {len(orphans)} orphan task file(s) to delete:\n")
    for name in orphans:
        print(f"  {name}")

    answer: str = input(f"\nDelete {len(orphans)} file(s)? [y/N] ").strip().lower()
    if answer != "y":
        print("Aborted.")
        return

    for name in orphans:
        (tasks_dir / name).unlink()
        print(f"  Deleted {name}")
    print(f"\nDeleted {len(orphans)} orphan task file(s).")
