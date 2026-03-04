"""Timestamped logging to event and problem log files."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path


def log_event(event_log: Path, msg: str) -> None:
    """Append a timestamped message to the event log file and print it.

    Each log entry is prefixed with an ISO-style timestamp in brackets.
    Messages are both printed to stdout and appended to the log file.

    Args:
        event_log: Path to the event log text file.
        msg: The message to log.
    """
    ts: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line: str = f"[{ts}] {msg}"
    print(line)
    with open(event_log, "a") as f:
        f.write(line + "\n")


def log_problem(problem_log: Path, msg: str) -> None:
    """Append a timestamped problem entry to the problem log file.

    Unlike ``log_event``, this does **not** print to stdout.

    Args:
        problem_log: Path to the problem log text file.
        msg: The problem description to log.
    """
    ts: str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line: str = f"[{ts}] {msg}"
    with open(problem_log, "a") as f:
        f.write(line + "\n")
