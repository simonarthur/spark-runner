"""Route cross-goal observations to relevant phases via LLM."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import anthropic

from spark_runner.classification import _observation_text
from spark_runner.llm_trace import save_llm_conversation
from spark_runner.models import ModelConfig


def route_observations_to_phases(
    observations: list[str | dict[str, str]],
    phases: list[dict[str, str]],
    client: anthropic.Anthropic,
    model_config: ModelConfig | None = None,
    run_dir: Path | None = None,
) -> dict[str, list[str | dict[str, str]]]:
    """Route each observation to only the phase(s) where it is relevant.

    Makes one LLM call to map observations to phases, preventing irrelevant
    observations from leaking into phases where they cause the agent to
    overrun its scope.

    Args:
        observations: Cross-goal observations (strings or dicts with "text"/"severity").
        phases: The ordered execution phases (dicts with "name" and "task" keys).
        client: Anthropic client for LLM calls.
        model_config: Model configuration.

    Returns:
        A dict mapping phase name → list of observations relevant to that phase.
        On parse error, falls back to returning all observations for every phase.
    """
    if not observations or not phases:
        return {}

    if model_config is None:
        model_config = ModelConfig()

    # Build phase listing (name + truncated task)
    phase_lines: list[str] = []
    for i, phase in enumerate(phases):
        task_preview: str = phase.get("task", "")[:200]
        phase_lines.append(f"  Phase {i}: {phase['name']} — {task_preview}")

    # Build observation listing
    obs_lines: list[str] = []
    for i, obs in enumerate(observations):
        obs_lines.append(f"  Observation {i}: {_observation_text(obs)}")

    prompt: str = (
        "You are routing observations from prior browser automation runs to the "
        "specific phases where each observation is relevant.\n\n"
        "PHASES:\n" + "\n".join(phase_lines) + "\n\n"
        "OBSERVATIONS:\n" + "\n".join(obs_lines) + "\n\n"
        "An observation is relevant to a phase if knowing it would help the agent "
        "execute THAT SPECIFIC phase correctly. Do NOT assign an observation to a "
        "phase just because it mentions a later step.\n\n"
        "Return ONLY valid JSON: an object mapping observation index (as string) to "
        "a list of phase indices where that observation is relevant.\n"
        'Example: {"0": [1, 2], "2": [3]}\n'
        "If an observation is not relevant to any phase, omit it."
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    response: anthropic.types.Message = client.messages.create(
        model=model_config.model,
        max_tokens=model_config.max_tokens,
        messages=messages,
    )
    if run_dir is not None:
        save_llm_conversation(run_dir, "observation_routing", messages, response)
    text: str = response.content[0].text.strip()

    try:
        match: re.Match[str] | None = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        routing: dict[str, list[int]] = json.loads(text)

        # Convert indices back to phase names, preserving original observation objects
        result: dict[str, list[str | dict[str, str]]] = {}
        for obs_idx_str, phase_indices in routing.items():
            obs_idx: int = int(obs_idx_str)
            if obs_idx < 0 or obs_idx >= len(observations):
                continue
            obs: str | dict[str, str] = observations[obs_idx]
            for phase_idx in phase_indices:
                if phase_idx < 0 or phase_idx >= len(phases):
                    continue
                phase_name: str = phases[phase_idx]["name"]
                if phase_name not in result:
                    result[phase_name] = []
                result[phase_name].append(obs)

        return result
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
        # Fallback: return all observations for all phases (never worse than before)
        return {phase["name"]: list(observations) for phase in phases}
