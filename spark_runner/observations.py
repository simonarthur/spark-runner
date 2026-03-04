"""Observation extraction, merging, and logging."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import anthropic

from spark_runner.classification import _observation_text
from spark_runner.log import log_event, log_problem
from spark_runner.models import ModelConfig


def _extract_and_log_observations(
    summary: str, phase_name: str, event_log: Path, problem_log: Path,
    *, success: bool = True,
) -> None:
    """Extract observations from a phase summary and route them to the appropriate log.

    Extracts ``<OBSERVATIONS>...</OBSERVATIONS>`` content from the summary.
    Successful phases log observations to the event log; failed phases log
    them to the problem log.  Real failure detection is handled upstream by
    ``run_phase()`` in ``execution.py``.

    Args:
        summary: The full structured summary text returned by ``summarize_phase()``.
        phase_name: Human-readable phase name for the log entry prefix.
        event_log: Path to the event log file (for observations on success).
        problem_log: Path to the problem log file (for observations on failure).
        success: Whether the phase succeeded (from ``run_phase()``).
    """
    obs_match: re.Match[str] | None = re.search(
        r"<OBSERVATIONS>(.*?)</OBSERVATIONS>",
        summary,
        re.DOTALL | re.IGNORECASE,
    )
    if not obs_match:
        return

    observations: str = obs_match.group(1).strip()
    if observations.lower() in ("none", "n/a", "none.", "n/a."):
        return

    if success:
        log_event(event_log, f"OBSERVATIONS ({phase_name}): {observations}")
    else:
        log_problem(problem_log, f"OBSERVATIONS ({phase_name}): {observations}")


def merge_observations(
    existing: list[str | dict[str, str]],
    new: list[str | dict[str, str]],
    client: anthropic.Anthropic,
    model_config: ModelConfig | None = None,
) -> list[dict[str, str]]:
    """Use an LLM to merge and de-duplicate two lists of observations.

    Args:
        existing: Observations from the previously saved goal summary.
        new: Observations from the current run.
        client: Anthropic client for LLM calls.
        model_config: Model configuration.

    Returns:
        A merged, de-duplicated list of observation dicts.
    """
    if model_config is None:
        model_config = ModelConfig()

    severity_map: dict[str, str] = {}
    for obs in list(existing) + list(new):
        if isinstance(obs, dict):
            severity_map[obs.get("text", "")] = obs.get("severity", "warning")

    existing_texts: list[str] = [_observation_text(o) for o in existing]
    new_texts: list[str] = [_observation_text(o) for o in new]

    response: anthropic.types.Message = client.messages.create(
        model=model_config.model,
        max_tokens=model_config.max_tokens,
        messages=[{"role": "user", "content": f"""You have two lists of observations from browser automation runs.
Merge them into a single de-duplicated list. If two observations say the same thing in different words, keep the more detailed or recent one. Preserve all unique information.

Existing observations:
{json.dumps(existing_texts, indent=2)}

New observations:
{json.dumps(new_texts, indent=2)}

Return ONLY a valid JSON array of strings — the merged, de-duplicated observations."""}],
    )
    text: str = response.content[0].text.strip()
    match: re.Match[str] | None = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    merged_texts: list[str] = json.loads(text)

    return [
        {"text": t, "severity": severity_map.get(t, "warning")}
        for t in merged_texts
    ]
