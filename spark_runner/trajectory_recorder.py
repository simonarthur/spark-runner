"""Record user demonstrations in the browser and generate goals from them."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import anthropic

from spark_runner.models import ModelConfig, SparkConfig


@dataclass
class RecordedAction:
    """A single recorded user action."""

    action_type: str = ""  # "click", "type", "navigate", "select", "scroll"
    selector: str = ""
    value: str = ""
    url: str = ""
    timestamp: str = ""
    description: str = ""


async def record_user_trajectory(
    base_url: str,
    config: SparkConfig,
) -> list[RecordedAction]:
    """Open a browser and record user actions as structured events.

    The user interacts with the browser normally. Actions are captured via
    Playwright's event listeners. Recording stops when the user presses
    Ctrl+C or closes the browser.

    Args:
        base_url: Starting URL for the browser.
        config: Configuration.

    Returns:
        A list of recorded actions in order.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("Error: playwright is required for trajectory recording.")
        print("Install it with: pip install playwright && playwright install")
        return []

    actions: list[RecordedAction] = []
    print(f"Opening browser at {base_url}")
    print("Perform your actions in the browser. Press Ctrl+C when done.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        # Track navigation
        def on_navigation(frame: Any) -> None:
            if frame == page.main_frame:
                actions.append(RecordedAction(
                    action_type="navigate",
                    url=page.url,
                    timestamp=datetime.now().isoformat(),
                    description=f"Navigated to {page.url}",
                ))

        page.on("framenavigated", on_navigation)

        # Navigate to starting URL
        await page.goto(base_url)

        # Set up console listener for action reports
        def on_console(msg: Any) -> None:
            text = msg.text
            if text.startswith("SPARK_ACTION:"):
                try:
                    data = json.loads(text[len("SPARK_ACTION:"):])
                    actions.append(RecordedAction(
                        action_type=data.get("type", "unknown"),
                        selector=data.get("selector", ""),
                        value=data.get("value", ""),
                        url=page.url,
                        timestamp=datetime.now().isoformat(),
                        description=data.get("description", ""),
                    ))
                except json.JSONDecodeError:
                    pass

        page.on("console", on_console)

        # Inject action tracking script
        await page.evaluate("""() => {
            document.addEventListener('click', (e) => {
                const el = e.target;
                const selector = el.tagName.toLowerCase() +
                    (el.id ? '#' + el.id : '') +
                    (el.className ? '.' + el.className.split(' ').join('.') : '');
                const text = el.textContent?.trim().substring(0, 50) || '';
                console.log('SPARK_ACTION:' + JSON.stringify({
                    type: 'click',
                    selector: selector,
                    description: 'Clicked ' + (text || selector)
                }));
            }, true);

            document.addEventListener('input', (e) => {
                const el = e.target;
                const selector = el.tagName.toLowerCase() +
                    (el.id ? '#' + el.id : '') +
                    (el.name ? '[name=' + el.name + ']' : '');
                console.log('SPARK_ACTION:' + JSON.stringify({
                    type: 'type',
                    selector: selector,
                    value: el.value,
                    description: 'Typed into ' + selector
                }));
            }, true);

            document.addEventListener('change', (e) => {
                const el = e.target;
                if (el.tagName === 'SELECT') {
                    console.log('SPARK_ACTION:' + JSON.stringify({
                        type: 'select',
                        selector: el.tagName.toLowerCase() + (el.id ? '#' + el.id : ''),
                        value: el.value,
                        description: 'Selected ' + el.value
                    }));
                }
            }, true);
        }""")

        # Wait for user to finish (Ctrl+C)
        try:
            while True:
                await page.wait_for_timeout(1000)
        except KeyboardInterrupt:
            print("\nRecording stopped.")
        finally:
            await browser.close()

    print(f"Recorded {len(actions)} action(s)")
    return actions


def actions_to_goal(
    actions: list[RecordedAction],
    client: anthropic.Anthropic,
    model_config: ModelConfig | None = None,
) -> tuple[str, list[dict[str, str]]]:
    """Use an LLM to convert recorded actions into a goal structure.

    Args:
        actions: List of recorded user actions.
        client: Anthropic client for LLM calls.
        model_config: Model configuration.

    Returns:
        A tuple of ``(prompt, phases)`` where prompt is a one-line task
        description and phases is a list of phase dicts.
    """
    if model_config is None:
        model_config = ModelConfig()

    actions_text = "\n".join(
        f"  {i}. [{a.action_type}] {a.description}"
        + (f" (value: {a.value})" if a.value else "")
        + (f" at {a.url}" if a.url else "")
        for i, a in enumerate(actions, 1)
    )

    response: anthropic.types.Message = client.messages.create(
        model=model_config.model,
        max_tokens=model_config.max_tokens,
        messages=[{"role": "user", "content": f"""Analyze these recorded user actions from a browser session and create a structured test goal.

Recorded actions:
{actions_text}

Return ONLY valid JSON with this structure:
{{
    "prompt": "One-line description of what the user was doing",
    "phases": [
        {{"name": "Phase Name", "task": "Detailed step-by-step instructions to reproduce this phase"}}
    ]
}}

Guidelines:
- Group related actions into logical phases
- The first phase should always be "Login" if the session includes login actions
- Write clear, reproducible instructions for each phase
- Include specific values that were entered"""}],
    )

    text: str = response.content[0].text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    data: dict[str, Any] = json.loads(text)
    return data.get("prompt", ""), data.get("phases", [])


async def record_and_generate_goal(
    base_url: str,
    config: SparkConfig,
) -> Path | None:
    """Record a user demonstration and generate a goal file.

    Args:
        base_url: Starting URL.
        config: Configuration.

    Returns:
        Path to the generated goal file, or None if no actions were recorded.
    """
    actions = await record_user_trajectory(base_url, config)
    if not actions:
        print("No actions recorded.")
        return None

    client: anthropic.Anthropic = anthropic.Anthropic()
    model_config = config.get_model("task_decomposition")

    print("Converting actions to goal...")
    prompt, phases = actions_to_goal(actions, client, model_config)

    if not prompt:
        print("Could not generate goal from recorded actions.")
        return None

    # Generate a task name
    from spark_runner.decomposition import generate_task_name

    task_name = generate_task_name(prompt, client, config.get_model("task_naming"))

    # Build goal data
    goal_data: dict[str, Any] = {
        "main_task": prompt,
        "key_observations": [],
        "subtasks": [],
    }

    # Save phase content as task files
    assert config.tasks_dir is not None
    assert config.goal_summaries_dir is not None

    config.tasks_dir.mkdir(parents=True, exist_ok=True)
    config.goal_summaries_dir.mkdir(parents=True, exist_ok=True)

    from spark_runner.storage import phase_name_to_slug, safe_write_path, write_with_history

    for i, phase in enumerate(phases, 1):
        slug = phase_name_to_slug(phase.get("name", f"phase-{i}"))
        task_path = safe_write_path(config.tasks_dir / f"{slug}.txt")
        write_with_history(task_path, phase.get("task", ""))
        goal_data["subtasks"].append({
            "subtask": i,
            "filename": task_path.name,
        })
        print(f"  Saved phase: {task_path.name}")

    goal_path = config.goal_summaries_dir / f"{task_name}-task.json"
    goal_path = safe_write_path(goal_path)
    write_with_history(goal_path, json.dumps(goal_data, indent=2))
    print(f"\nGoal saved to: {goal_path}")

    return goal_path
