"""Conversation saving helper for LLM calls."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import anthropic


def save_llm_conversation(
    run_dir: Path,
    step_name: str,
    messages: list[dict[str, Any]],
    response: anthropic.types.Message,
) -> str:
    """Save an LLM conversation to a JSON file in the run directory.

    Args:
        run_dir: Path to the run directory.
        step_name: Identifier for this pipeline step (e.g. ``"task_decomposition"``).
        messages: The prompt messages sent to the API.
        response: The Anthropic API response object.

    Returns:
        The filename of the saved file (e.g. ``"llm_task_decomposition.json"``).
    """
    response_text = ""
    if response.content:
        response_text = response.content[0].text

    # Truncate very large prompt content to avoid bloating the trace files.
    truncated_messages: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str) and len(content) > 100_000:
            content = content[:100_000] + "\n... (truncated)"
        truncated_messages.append({**msg, "content": content})

    data: dict[str, Any] = {
        "step": step_name,
        "model": response.model,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "messages": truncated_messages,
        "response_text": response_text,
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }

    filename = f"llm_{step_name}.json"
    (run_dir / filename).write_text(json.dumps(data, indent=2))
    return filename
