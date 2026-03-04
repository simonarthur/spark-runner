"""Timestamped logging to event and problem log files."""

from __future__ import annotations

import logging
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


_AGENT_LOG_LOGGERS: list[str] = ["browser_use", "bubus"]


def attach_agent_log_handler(run_dir: Path) -> logging.FileHandler:
    """Create a file handler for browser-use agent logs and attach it.

    Attaches a :class:`logging.FileHandler` writing to ``agent_log.txt``
    inside *run_dir* to the ``browser_use`` and ``bubus`` loggers so that
    per-run agent output is captured to disk.

    Args:
        run_dir: The run directory where ``agent_log.txt`` will be created.

    Returns:
        The handler instance (pass it to :func:`detach_agent_log_handler`
        when the run is finished).
    """
    agent_log_path: Path = run_dir / "agent_log.txt"
    handler = logging.FileHandler(agent_log_path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s")
    )
    for name in _AGENT_LOG_LOGGERS:
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        # Ensure the logger itself passes INFO records to our handler.
        if logger.level == logging.NOTSET or logger.level > logging.INFO:
            logger.setLevel(logging.INFO)
    return handler


def detach_agent_log_handler(handler: logging.FileHandler) -> None:
    """Remove and close a previously attached agent log handler.

    Args:
        handler: The handler returned by :func:`attach_agent_log_handler`.
    """
    for name in _AGENT_LOG_LOGGERS:
        logging.getLogger(name).removeHandler(handler)
    handler.close()
