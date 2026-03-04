"""Knowledge index loading and LLM-based knowledge matching."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import anthropic

from spark_runner.classification import _observation_text
from spark_runner.models import ModelConfig


def load_knowledge_index(
    goal_summaries_dir: Path,
    tasks_dir: Path,
    restore_fn: Callable[[str], str],
) -> list[dict[str, Any]]:
    """Read all goal summaries and their referenced task files into a knowledge index.

    Args:
        goal_summaries_dir: Directory containing ``*-task.json`` goal summary files.
        tasks_dir: Directory containing subtask ``.txt`` files.
        restore_fn: Function to restore placeholders in stored text.

    Returns:
        A list of dicts representing prior goals with subtask contents loaded.
    """
    index: list[dict[str, Any]] = []

    for goal_file in sorted(goal_summaries_dir.glob("*-task.json")):
        try:
            data: dict[str, Any] = json.loads(restore_fn(goal_file.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: skipping malformed goal file {goal_file.name}: {e}")
            continue

        entry: dict[str, Any] = {
            "goal_file": goal_file.name,
            "main_task": data.get("main_task", ""),
            "key_observations": data.get("key_observations", []),
            "subtasks": [],
        }

        for subtask_entry in data.get("subtasks", []):
            if not isinstance(subtask_entry, dict):
                continue
            filename: str = subtask_entry.get("filename", "")
            subtask_path: Path = tasks_dir / filename
            if not subtask_path.exists():
                print(f"  Warning: subtask file not found {subtask_path}, skipping")
                continue
            try:
                content: str = restore_fn(subtask_path.read_text())
            except OSError as e:
                print(f"  Warning: could not read {subtask_path}: {e}")
                continue
            name: str = subtask_path.stem.replace("-", " ").title()
            entry["subtasks"].append({
                "filename": filename,
                "name": name,
                "content": content,
            })

        index.append(entry)
    return index


def find_relevant_knowledge(
    prompt: str,
    knowledge_index: list[dict[str, Any]],
    client: anthropic.Anthropic,
    model_config: ModelConfig | None = None,
) -> dict[str, Any]:
    """Use an LLM to find reusable subtasks and observations from prior goals.

    Args:
        prompt: The user's new task description.
        knowledge_index: The full knowledge index from ``load_knowledge_index()``.
        client: Anthropic client for LLM calls.
        model_config: Model configuration.

    Returns:
        A dict with ``reusable_subtasks``, ``relevant_observations``, and ``coverage_notes``.
    """
    if model_config is None:
        model_config = ModelConfig()

    if not knowledge_index:
        return {"reusable_subtasks": [], "relevant_observations": [], "coverage_notes": ""}

    total_subtasks: int = sum(len(g["subtasks"]) for g in knowledge_index)
    total_observations: int = sum(len(g["key_observations"]) for g in knowledge_index)
    print(f"  Searching {len(knowledge_index)} goal(s) with {total_subtasks} subtask(s) and {total_observations} observation(s)")
    for i, goal in enumerate(knowledge_index, 1):
        subtask_names: str = ", ".join(st["name"] for st in goal["subtasks"]) or "(none)"
        print(f"    Goal {i}: {goal['goal_file']}")
        print(f"      Task: {goal['main_task']}")
        print(f"      Subtasks: {subtask_names}")
        print(f"      Observations: {len(goal['key_observations'])}")

    index_text_parts: list[str] = []
    for i, goal in enumerate(knowledge_index):
        parts: list[str] = [f"Goal {i + 1}: {goal['goal_file']}"]
        parts.append(f"  Main task: {goal['main_task']}")
        if goal["key_observations"]:
            parts.append("  Observations:")
            for obs in goal["key_observations"]:
                parts.append(f"    - {_observation_text(obs)}")
        if goal["subtasks"]:
            parts.append("  Subtasks:")
            for st in goal["subtasks"]:
                parts.append(f"    [{st['filename']}] {st['name']}:")
                parts.append(f"      {st['content']}")
        index_text_parts.append("\n".join(parts))

    index_text: str = "\n\n".join(index_text_parts)
    print(f"  Sending {len(index_text)} chars to LLM for knowledge matching...")

    response: anthropic.types.Message = client.messages.create(
        model=model_config.model,
        max_tokens=model_config.max_tokens,
        messages=[{"role": "user", "content": f"""You are analyzing a knowledge base of prior browser automation runs to find reusable components for a new task.

New task prompt:
{prompt}

Existing goals and their subtasks:
{index_text}

Analyze each existing subtask's CONTENT (not just its name) and determine:
1. Which subtasks can be directly reused for the new task (the steps described in the content must actually match what the new task needs)
2. Which observations from prior goals are relevant to the new task
3. What the new task needs that isn't covered by existing knowledge

Return ONLY valid JSON:
{{
    "reusable_subtasks": [
        {{"filename": "exact-filename.txt", "phase_name": "Phase Name", "reason": "Why this subtask is reusable"}}
    ],
    "relevant_observations": ["observation string 1", "observation string 2"],
    "coverage_notes": "What the new task needs that isn't covered by existing subtasks"
}}

IMPORTANT:
- Only include subtasks whose content genuinely matches what the new task needs — don't match on name alone.
- For observations, include any that would help the new task succeed (UI patterns, timing, form quirks, etc).
- If nothing is reusable, return empty arrays."""}],
    )

    text: str = response.content[0].text.strip()
    try:
        match: re.Match[str] | None = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        result: dict[str, Any] = json.loads(text)
        result.setdefault("reusable_subtasks", [])
        result.setdefault("relevant_observations", [])
        result.setdefault("coverage_notes", "")

        print("  Knowledge matching complete:")
        if result["reusable_subtasks"]:
            print(f"  Reusable subtasks ({len(result['reusable_subtasks'])}):")
            for st in result["reusable_subtasks"]:
                print(f"    - [{st.get('filename', '?')}] {st.get('phase_name', '?')}")
                print(f"      Reason: {st.get('reason', '(none)')}")
        else:
            print("  Reusable subtasks: none found")
        if result["relevant_observations"]:
            print(f"  Relevant observations ({len(result['relevant_observations'])}):")
            for obs in result["relevant_observations"]:
                print(f"    - {obs}")
        else:
            print("  Relevant observations: none found")
        if result["coverage_notes"]:
            print(f"  Coverage gaps: {result['coverage_notes']}")

        return result
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"  Warning: could not parse knowledge match response: {e}")
        return {"reusable_subtasks": [], "relevant_observations": [], "coverage_notes": ""}
