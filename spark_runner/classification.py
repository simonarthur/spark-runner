"""Observation classification rules: loading, prompt building, and LLM classification."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import anthropic

from spark_runner.models import ClassificationRules, ModelConfig


def load_classification_rules(path: Path) -> ClassificationRules:
    """Parse a classification rules file into a ``ClassificationRules`` object.

    The file uses ``[ERRORS]`` and ``[WARNINGS]`` section headers (case-insensitive).
    Blank lines and lines starting with ``#`` are ignored.

    Returns an empty ``ClassificationRules`` if the file is missing or unreadable.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ClassificationRules()

    rules = ClassificationRules()
    current_section: list[str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lower = line.lower()
        if lower == "[errors]":
            current_section = rules.error_rules
        elif lower == "[warnings]":
            current_section = rules.warning_rules
        elif current_section is not None:
            current_section.append(line)

    return rules


def _build_rules_prompt_section(rules: ClassificationRules) -> str:
    """Format classification rules into a prompt block for the LLM.

    Returns an empty string when no rules exist.
    """
    if not rules.error_rules and not rules.warning_rules:
        return ""

    parts: list[str] = [
        "\n**PRIORITY CLASSIFICATION RULES** (these override the general criteria above):\n",
    ]

    if rules.error_rules:
        parts.append('The following should ALWAYS be classified as "error":')
        for i, rule in enumerate(rules.error_rules, 1):
            parts.append(f"  {i}. {rule}")
        parts.append("")

    if rules.warning_rules:
        parts.append('The following should ALWAYS be classified as "warning":')
        for i, rule in enumerate(rules.warning_rules, 1):
            parts.append(f"  {i}. {rule}")
        parts.append("")

    return "\n".join(parts)


def _observation_text(obs: str | dict[str, str]) -> str:
    """Extract text from an observation in either string or dict format."""
    if isinstance(obs, dict):
        return obs.get("text", "")
    return obs


def classify_observations(
    prompt: str,
    observations: list[str | dict[str, str]],
    client: anthropic.Anthropic,
    model_config: ModelConfig | None = None,
    rules: ClassificationRules | None = None,
) -> list[dict[str, str]]:
    """Use an LLM to classify each observation as an error or a warning.

    Args:
        prompt: The original task description that produced these observations.
        observations: List of observation strings (or dicts) to classify.
        client: Anthropic client for LLM calls.
        model_config: Model configuration. Defaults to sonnet with 4096 tokens.
        rules: Optional classification rules to guide the LLM.

    Returns:
        A list of dicts with ``text`` and ``severity`` keys.
    """
    if model_config is None:
        model_config = ModelConfig()
    if rules is None:
        rules = ClassificationRules()

    obs_texts: list[str] = [_observation_text(o) for o in observations]

    if not obs_texts:
        return []

    rules_section: str = _build_rules_prompt_section(rules)

    response: anthropic.types.Message = client.messages.create(
        model=model_config.model,
        max_tokens=model_config.max_tokens,
        messages=[{"role": "user", "content": f"""You are a QA analyst classifying observations from a browser automation test run.

Classify each observation as "error" or "warning" using these criteria:

**error**: Application functionality is broken. The feature under test did not
work as specified, even if a workaround achieved the broader goal. If a
workaround was used, consider whether it tests the SAME feature as the original
step — if the workaround uses a DIFFERENT mechanism (e.g. browsing categories
instead of using search), the original feature is still broken → "error".

**warning**: The environment or UI changed from the original instructions, but
the feature under test worked correctly. Examples: element indices shifted,
domain/URL changed, layout rearranged, extra fields present, cosmetic
differences.

For each observation, reason through:
1. Was the underlying feature/functionality working correctly?
2. Was a workaround needed that bypasses the feature under test?
3. Is this just an environmental or documentation difference?
{rules_section}
Original task instructions:
{prompt}

Observations:
{json.dumps(obs_texts, indent=2)}

Return ONLY a valid JSON array with one entry per observation, in the same order:
[
  {{"text": "observation text here", "severity": "error"}},
  {{"text": "observation text here", "severity": "warning"}}
]"""}],
    )
    text: str = response.content[0].text.strip()
    match: re.Match[str] | None = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        classified: list[dict[str, str]] = json.loads(text)
        result: list[dict[str, str]] = []
        for i, entry in enumerate(classified):
            obs_text: str = entry.get("text", obs_texts[i] if i < len(obs_texts) else "")
            severity: str = entry.get("severity", "warning")
            if severity not in ("error", "warning"):
                severity = "warning"
            result.append({"text": obs_text, "severity": severity})
        return result
    except (json.JSONDecodeError, AttributeError, IndexError) as e:
        print(f"  Warning: could not parse classification response: {e}")
        return [{"text": t, "severity": "warning"} for t in obs_texts]
