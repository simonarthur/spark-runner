"""Knowledge index loading and LLM-based knowledge matching."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import anthropic

from spark_runner.llm_trace import save_llm_conversation
from spark_runner.models import ModelConfig

# Budget for the knowledge index text sent to the LLM (~140k tokens at ~3.5
# chars/token, leaving room for the prompt template + response within 200k).
_MAX_KNOWLEDGE_CHARS: int = 500_000


def load_knowledge_index(
    tasks_dir: Path,
    restore_fn: Callable[[str], str],
) -> list[dict[str, Any]]:
    """Read all ``.txt`` task files into a flat knowledge index.

    Args:
        tasks_dir: Directory containing task ``.txt`` files.
        restore_fn: Function to restore placeholders in stored text.

    Returns:
        A flat list of dicts with ``filename``, ``name``, and ``content`` keys.
    """
    index: list[dict[str, Any]] = []

    for task_file in sorted(tasks_dir.glob("*.txt")):
        try:
            content: str = restore_fn(task_file.read_text())
        except OSError as e:
            print(f"  Warning: could not read {task_file}: {e}")
            continue
        name: str = task_file.stem.replace("-", " ").title()
        index.append({
            "filename": task_file.name,
            "name": name,
            "content": content,
        })

    return index


def find_relevant_knowledge(
    prompt: str,
    knowledge_index: list[dict[str, Any]],
    client: anthropic.Anthropic,
    model_config: ModelConfig | None = None,
    run_dir: Path | None = None,
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

    print(f"  Searching {len(knowledge_index)} task file(s)")
    for i, task_entry in enumerate(knowledge_index, 1):
        print(f"    {i}. {task_entry['filename']} ({task_entry['name']})")

    index_text_parts: list[str] = []
    for i, task_entry in enumerate(knowledge_index):
        parts: list[str] = [f"Task file {i + 1}: {task_entry['filename']}"]
        parts.append(f"  Name: {task_entry['name']}")
        parts.append(f"  Content:\n    {task_entry['content']}")
        index_text_parts.append("\n".join(parts))

    # Truncate if the combined index would exceed the character budget.
    total_files: int = len(index_text_parts)
    kept_parts: list[str] = []
    char_count: int = 0
    for part in index_text_parts:
        # Account for the "\n\n" separator between parts.
        added: int = len(part) + (2 if kept_parts else 0)
        if char_count + added > _MAX_KNOWLEDGE_CHARS and kept_parts:
            break
        kept_parts.append(part)
        char_count += added

    if len(kept_parts) < total_files:
        print(
            f"  Warning: knowledge index truncated from {total_files} to"
            f" {len(kept_parts)} task file(s) to fit token limit"
        )

    index_text: str = "\n\n".join(kept_parts)
    print(f"  Sending {len(index_text)} chars to LLM for knowledge matching...")

    messages: list[dict[str, Any]] = [{"role": "user", "content": f"""You are analyzing a collection of prior browser automation task files to find reusable components for a new task.

New task prompt:
{prompt}

Existing task files:
{index_text}

Analyze each task file's CONTENT and determine:
1. Which task files can be directly reused for the new task (the steps described in the content must actually match what the new task needs)
2. What relevant observations can be extracted from the task file content (UI patterns, workarounds, timing notes, form quirks)
3. What the new task needs that isn't covered by existing task files

Return ONLY valid JSON:
{{
    "reusable_subtasks": [
        {{"filename": "exact-filename.txt", "phase_name": "Phase Name", "reason": "Why this task file is reusable"}}
    ],
    "relevant_observations": ["observation string 1", "observation string 2"],
    "coverage_notes": "What the new task needs that isn't covered by existing task files"
}}

IMPORTANT:
- Only include task files whose content genuinely matches what the new task needs — don't match on name alone.
- For observations, extract any useful insights from the task file content that would help the new task succeed.
- If nothing is reusable, return empty arrays."""}]
    response: anthropic.types.Message = client.messages.create(
        model=model_config.model,
        max_tokens=model_config.max_tokens,
        messages=messages,
    )
    if run_dir is not None:
        save_llm_conversation(run_dir, "knowledge_matching", messages, response)

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
