"""Goal listing, loading, management, and classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from spark_runner.classification import _observation_text
from spark_runner.execution import _REPLAY_PREFIX


def get_last_run_info(
    runs_dir: Path | None, task_name: str,
) -> tuple[str, str] | None:
    """Return ``(timestamp, status)`` for the most recent run, or None."""
    if runs_dir is None:
        return None
    task_dir = runs_dir / task_name
    if not task_dir.is_dir():
        return None
    run_dirs = sorted(
        (d for d in task_dir.iterdir() if d.is_dir()),
        reverse=True,
    )
    if not run_dirs:
        return None

    last_dir = run_dirs[0]
    status = "unknown"
    metadata_path = last_dir / "run_metadata.json"
    if metadata_path.exists():
        try:
            meta: dict[str, Any] = json.loads(metadata_path.read_text())
            phases = meta.get("phases", [])
            if phases:
                has_errors = any(
                    p.get("outcome") != "SUCCESS" for p in phases
                )
                status = "errors" if has_errors else "ok"
        except (json.JSONDecodeError, OSError):
            pass
    return last_dir.name, status


def list_goals(
    goal_summaries_dir: Path,
    restore_fn: Callable[[str], str],
    runs_dir: Path | None = None,
    *,
    filter_unrun: bool = False,
    filter_failed: bool = False,
) -> None:
    """Print a summary of all existing goals from the goal summaries directory."""
    goal_files: list[Path] = sorted(goal_summaries_dir.glob("*-task.json"))
    if not goal_files:
        print("No goals found.")
        return

    # Apply --unrun / --failed filters when requested
    if filter_unrun or filter_failed:
        filtered: list[Path] = []
        for gf in goal_files:
            task_name = gf.stem.removesuffix("-task")
            run_info = get_last_run_info(runs_dir, task_name)
            if filter_unrun and run_info is None:
                filtered.append(gf)
            elif filter_failed and run_info is not None and run_info[1] == "errors":
                filtered.append(gf)
        goal_files = filtered
        if not goal_files:
            print("No matching goals found.")
            return

    print(f"Found {len(goal_files)} goal(s):\n")
    for goal_file in goal_files:
        try:
            data: dict[str, Any] = json.loads(restore_fn(goal_file.read_text()))
        except (json.JSONDecodeError, OSError):
            print(f" {goal_file.stem}  (unreadable)")
            continue
        main_task: str = data.get("main_task", "(no description)")
        num_subtasks: int = len(data.get("subtasks", []))
        observations: list[str | dict[str, str]] = data.get("key_observations", [])
        num_observations: int = len(observations)
        num_errors: int = sum(
            1 for o in observations
            if isinstance(o, dict) and o.get("severity") == "error"
        )
        num_warnings: int = sum(
            1 for o in observations
            if isinstance(o, dict) and o.get("severity") == "warning"
        )
        num_unclassified: int = num_observations - num_errors - num_warnings
        print(f"{goal_file.stem}")
        print(f"  {main_task}")
        severity_parts: list[str] = []
        if num_errors:
            severity_parts.append(f"{num_errors} errors")
        if num_warnings:
            severity_parts.append(f"{num_warnings} warnings")
        if num_unclassified:
            severity_parts.append(f"{num_unclassified} unclassified")
        severity_str: str = f" ({', '.join(severity_parts)})" if severity_parts else ""
        safety_raw: dict[str, Any] = data.get("safety", {})
        safety_label: str = ""
        if safety_raw:
            if safety_raw.get("blocked_in_production"):
                safety_label = " [restricted: production-blocked]"
            elif safety_raw.get("allowed_environments"):
                envs = ", ".join(safety_raw["allowed_environments"])
                safety_label = f" [restricted: {envs} only]"
        task_name = goal_file.stem.removesuffix("-task")
        last_run_info = get_last_run_info(runs_dir, task_name)
        if last_run_info:
            timestamp, status = last_run_info
            last_run_str = f"Last run: {timestamp} [{status}]"
        else:
            last_run_str = "  Last run: never"
        print(f"  Subtasks: {num_subtasks}  Observations: {num_observations}{severity_str}{safety_label}  {last_run_str}")
        print()


def load_goal_summary(
    goal_path: Path,
    tasks_dir: Path,
    restore_fn: Callable[[str], str],
) -> tuple[str, str, list[dict[str, str]]]:
    """Load a goal summary JSON and reconstruct the prompt, task name, and phases.

    Args:
        goal_path: Path to the goal summary JSON file.
        tasks_dir: Directory containing subtask ``.txt`` files.
        restore_fn: Function to restore placeholders.

    Returns:
        A tuple of ``(prompt, task_name, phases)``.
    """
    goal_data: dict[str, Any] = json.loads(restore_fn(goal_path.read_text()))
    prompt: str = goal_data["main_task"]

    phases: list[dict[str, str]] = []
    for entry in goal_data["subtasks"]:
        if not isinstance(entry, dict):
            continue
        subtask_path: Path = tasks_dir / entry["filename"]
        if not subtask_path.exists():
            print(f"  Warning: subtask file not found {subtask_path}, skipping")
            continue
        task_content: str = restore_fn(subtask_path.read_text())
        name: str = subtask_path.stem.replace("-", " ").title()
        phases.append({"name": name, "task": _REPLAY_PREFIX + task_content})

    task_name: str = goal_path.stem.removesuffix("-task")
    return prompt, task_name, phases


def classify_existing_goals(
    goal_summaries_dir: Path,
    classify_fn: Callable[[str, list[str | dict[str, str]]], list[dict[str, str]]],
) -> None:
    """Retroactively classify observations in all existing goal summaries.

    Args:
        goal_summaries_dir: Directory containing goal summary JSON files.
        classify_fn: The observation classification callable (e.g. classify_observations).
    """
    goal_files: list[Path] = sorted(goal_summaries_dir.glob("*-task.json"))
    if not goal_files:
        print("No goals found.")
        return

    print(f"Classifying observations in {len(goal_files)} goal(s)...\n")
    for goal_file in goal_files:
        try:
            data: dict[str, Any] = json.loads(goal_file.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Skipping {goal_file.name}: {e}")
            continue

        observations: list[str | dict[str, str]] = data.get("key_observations", [])
        if not observations:
            print(f"  {goal_file.name}: no observations, skipping")
            continue

        if all(isinstance(o, dict) and "severity" in o for o in observations):
            print(f"  {goal_file.name}: already classified, skipping")
            continue

        prompt: str = data.get("main_task", "")
        print(f"  {goal_file.name}: classifying {len(observations)} observations...")
        classified: list[dict[str, str]] = classify_fn(prompt, observations)
        data["key_observations"] = classified
        goal_file.write_text(json.dumps(data, indent=2))

        num_errors: int = sum(1 for o in classified if o["severity"] == "error")
        num_warnings: int = len(classified) - num_errors
        print(f"    -> {num_errors} errors, {num_warnings} warnings")

    print("\nDone.")


def show_goal_detail(
    goal_summaries_dir: Path,
    goal_name: str,
    restore_fn: Callable[[str], str],
) -> None:
    """Print detailed information about a specific goal.

    Args:
        goal_summaries_dir: Directory containing goal summary JSON files.
        goal_name: Name of the goal (with or without ``-task.json`` suffix).
        restore_fn: Function to restore placeholders in stored text.
    """
    if goal_name.endswith("-task.json"):
        pass
    elif goal_name.endswith("-task"):
        goal_name = f"{goal_name}.json"
    else:
        goal_name = f"{goal_name}-task.json"
    goal_path: Path = goal_summaries_dir / goal_name
    if not goal_path.exists():
        print(f"Goal not found: {goal_path}")
        return

    try:
        data: dict[str, Any] = json.loads(restore_fn(goal_path.read_text()))
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading goal: {e}")
        return

    print(f"Goal: {goal_path.name}")
    print(f"  Task: {data.get('main_task', '(no description)')}")

    safety_raw: dict[str, Any] = data.get("safety", {})
    if safety_raw:
        print("\n  Safety:")
        if safety_raw.get("blocked_in_production"):
            print("    Blocked in production: yes")
        if safety_raw.get("allowed_environments"):
            print(f"    Allowed environments: {', '.join(safety_raw['allowed_environments'])}")
        if safety_raw.get("risk_level"):
            print(f"    Risk level: {safety_raw['risk_level']}")
        if safety_raw.get("reason"):
            print(f"    Reason: {safety_raw['reason']}")

    subtasks: list[dict[str, Any]] = data.get("subtasks", [])
    if subtasks:
        print(f"\n  Subtasks ({len(subtasks)}):")
        for i, st in enumerate(subtasks, 1):
            print(f"    {i}. {st.get('filename', '?')}")

    observations: list[str | dict[str, str]] = data.get("key_observations", [])
    if observations:
        print(f"\n  Observations ({len(observations)}):")
        for obs in observations:
            text = _observation_text(obs)
            severity = obs.get("severity", "unclassified") if isinstance(obs, dict) else "unclassified"
            print(f"    [{severity}] {text}")


def delete_goal(
    goal_summaries_dir: Path,
    tasks_dir: Path,
    goal_name: str,
    force: bool = False,
) -> None:
    """Delete a goal and its unreferenced task files.

    Args:
        goal_summaries_dir: Directory containing goal summary JSON files.
        tasks_dir: Directory containing subtask ``.txt`` files.
        goal_name: Name of the goal (with or without ``-task.json`` suffix).
        force: If True, skip confirmation prompt.
    """
    if goal_name.endswith("-task.json"):
        pass
    elif goal_name.endswith("-task"):
        goal_name = f"{goal_name}.json"
    else:
        goal_name = f"{goal_name}-task.json"
    goal_path: Path = goal_summaries_dir / goal_name
    if not goal_path.exists():
        print(f"Goal not found: {goal_path}")
        return

    try:
        data: dict[str, Any] = json.loads(goal_path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading goal: {e}")
        return

    # Find task files referenced by this goal
    task_files: list[str] = [
        st.get("filename", "") for st in data.get("subtasks", []) if st.get("filename")
    ]

    # Check which task files are also referenced by other goals
    other_refs: set[str] = set()
    for other_goal in goal_summaries_dir.glob("*-task.json"):
        if other_goal.name == goal_name:
            continue
        try:
            other_data: dict[str, Any] = json.loads(other_goal.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for st in other_data.get("subtasks", []):
            fname = st.get("filename", "")
            if fname:
                other_refs.add(fname)

    safe_to_delete: list[str] = [f for f in task_files if f not in other_refs]

    print(f"Will delete goal: {goal_path.name}")
    if safe_to_delete:
        print(f"  Will also delete {len(safe_to_delete)} unreferenced task file(s):")
        for f in safe_to_delete:
            print(f"    {f}")
    shared: list[str] = [f for f in task_files if f in other_refs]
    if shared:
        print(f"  Keeping {len(shared)} task file(s) referenced by other goals:")
        for f in shared:
            print(f"    {f}")

    if not force:
        answer: str = input("\nProceed? [y/N] ").strip().lower()
        if answer != "y":
            print("Aborted.")
            return

    goal_path.unlink()
    print(f"  Deleted {goal_path.name}")
    for f in safe_to_delete:
        p = tasks_dir / f
        if p.exists():
            p.unlink()
            print(f"  Deleted {f}")

    print("Done.")
