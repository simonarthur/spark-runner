"""Observation extraction, merging, and logging."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import anthropic

from sparky_runner.classification import _observation_text
from sparky_runner.log import log_problem
from sparky_runner.models import ModelConfig


def _extract_and_log_observations(
    summary: str, phase_name: str, problem_log: Path
) -> None:
    """Extract observations and sub-phase failures from a phase summary and log them.

    Performs two passes over the LLM-generated phase summary:

    1. **Observations**: Extracts ``<OBSERVATIONS>...</OBSERVATIONS>`` content.
    2. **Sub-phase failures**: Parses ``### Sub-phase N: ...`` sections and logs
       any with a non-SUCCESS status as ``ERROR`` entries in the problem log.

    Args:
        summary: The full structured summary text returned by ``summarize_phase()``.
        phase_name: Human-readable phase name for the log entry prefix.
        problem_log: Path to the problem log file.
    """
    # --- Extract <OBSERVATIONS> block first ---
    observations: str = ""
    obs_match: re.Match[str] | None = re.search(
        r"<OBSERVATIONS>(.*?)</OBSERVATIONS>",
        summary,
        re.DOTALL | re.IGNORECASE,
    )
    if obs_match:
        observations = obs_match.group(1).strip()
        if observations.lower() in ("none", "n/a", "none.", "n/a."):
            observations = ""

    # --- Detect sub-phase failures ---
    _SUCCESS_INDICATORS: tuple[str, ...] = ("SUCCESS", "SUCCEED", "PASSED", "✅", "IN PROGRESS")
    has_errors: bool = False
    for sp_match in re.finditer(
        r"###\s*(?:Sub-phase\s+)?[\dA-Za-z.]+[:\s]+([^\n]+)\n"
        r".*?\*\*(?:Status|Result)\*\*:\s*([^\n]+)",
        summary,
        re.DOTALL,
    ):
        sp_name: str = sp_match.group(1).strip()
        sp_status: str = sp_match.group(2).strip()
        sp_status_upper: str = sp_status.upper()
        is_success: bool = any(ind in sp_status_upper for ind in _SUCCESS_INDICATORS)
        if not is_success:
            error_msg: str = (
                f"ERROR ({phase_name} / {sp_name}): Sub-phase status: {sp_status}"
                f'\n  [diagnostic: status "{sp_status}" matched none of {_SUCCESS_INDICATORS}]'
            )
            if observations:
                error_msg += f"\n  Details:\n  {observations}"
            log_problem(problem_log, error_msg)
            has_errors = True

    # Log observations standalone only when no sub-phase errors consumed them
    if observations and not has_errors:
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
