"""Goal listing, loading, management, and classification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from spark_runner.classification import _observation_text
from spark_runner.execution import _REPLAY_PREFIX
from spark_runner.models import GoalInfo
from spark_runner.storage import write_with_history


def load_hints(goal_path: Path) -> list[dict[str, str]]:
    """Load hints from a goal summary JSON.

    Args:
        goal_path: Path to the goal summary JSON file.

    Returns:
        A list of ``{"phase": ..., "text": ...}`` dicts.
    """
    data: dict[str, Any] = json.loads(goal_path.read_text())
    return data.get("hints", [])


def save_hint(goal_path: Path, phase: str, text: str) -> None:
    """Append a hint to the goal summary JSON.

    Args:
        goal_path: Path to the goal summary JSON file.
        phase: The phase name this hint applies to.
        text: The hint text.
    """
    data: dict[str, Any] = json.loads(goal_path.read_text())
    hints: list[dict[str, str]] = data.setdefault("hints", [])
    hints.append({"phase": phase, "text": text})
    goal_path.write_text(json.dumps(data, indent=2))


def get_phase_names(goal_path: Path) -> list[str]:
    """Return phase names for a goal, derived from subtask filenames.

    Phase names are computed the same way as :func:`load_goal_summary`:
    ``stem.replace("-", " ").title()``.

    Args:
        goal_path: Path to the goal summary JSON file.

    Returns:
        A list of phase name strings (e.g. ``["Fill Form", "Verify Result"]``).
    """
    try:
        data: dict[str, Any] = json.loads(goal_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    names: list[str] = []
    for entry in data.get("subtasks", []):
        if isinstance(entry, dict) and "filename" in entry:
            stem = Path(entry["filename"]).stem
            names.append(stem.replace("-", " ").title())
    return names


def remove_hint(goal_path: Path, index: int) -> bool:
    """Remove a hint by index.

    Args:
        goal_path: Path to the goal summary JSON file.
        index: Zero-based index of the hint to remove.

    Returns:
        True if the hint was removed, False if the index was out of range.
    """
    data: dict[str, Any] = json.loads(goal_path.read_text())
    hints: list[dict[str, str]] = data.get("hints", [])
    if 0 <= index < len(hints):
        hints.pop(index)
        data["hints"] = hints
        goal_path.write_text(json.dumps(data, indent=2))
        return True
    return False


def reset_phase(goal_path: Path, phase_name: str) -> bool:
    """Mark a phase for fresh decomposition on the next run.

    Args:
        goal_path: Path to the goal summary JSON file.
        phase_name: The phase name to reset (case-insensitive match).

    Returns:
        True if the phase was found and marked, False if not found.
    """
    valid_phases: list[str] = get_phase_names(goal_path)
    matched: list[str] = [
        p for p in valid_phases if p.lower() == phase_name.lower()
    ]
    if not matched:
        return False
    canonical: str = matched[0]
    data: dict[str, Any] = json.loads(goal_path.read_text())
    reset_phases: list[str] = data.get("reset_phases", [])
    if canonical not in reset_phases:
        reset_phases.append(canonical)
    data["reset_phases"] = reset_phases
    goal_path.write_text(json.dumps(data, indent=2))
    return True


def unreset_phase(goal_path: Path, phase_name: str) -> bool:
    """Remove a phase from the reset list.

    Args:
        goal_path: Path to the goal summary JSON file.
        phase_name: The phase name to unreset (case-insensitive match).

    Returns:
        True if the phase was found and removed, False if not found.
    """
    data: dict[str, Any] = json.loads(goal_path.read_text())
    reset_phases: list[str] = data.get("reset_phases", [])
    original_len: int = len(reset_phases)
    reset_phases = [p for p in reset_phases if p.lower() != phase_name.lower()]
    if len(reset_phases) == original_len:
        return False
    data["reset_phases"] = reset_phases
    goal_path.write_text(json.dumps(data, indent=2))
    return True


def get_reset_phases(goal_path: Path) -> list[str]:
    """Return the list of phases marked for reset.

    Args:
        goal_path: Path to the goal summary JSON file.

    Returns:
        A list of phase name strings marked for fresh decomposition.
    """
    data: dict[str, Any] = json.loads(goal_path.read_text())
    return data.get("reset_phases", [])


def clear_reset_phases(goal_path: Path) -> None:
    """Clear all reset phase markers from a goal.

    Args:
        goal_path: Path to the goal summary JSON file.
    """
    data: dict[str, Any] = json.loads(goal_path.read_text())
    data["reset_phases"] = []
    goal_path.write_text(json.dumps(data, indent=2))


def reset_errored_phases(goal_path: Path, runs_dir: Path) -> list[str]:
    """Reset phases that had errors in the last run.

    Reads the last run's ``run_metadata.json``, finds phases with
    ``outcome != "SUCCESS"``, and marks each for fresh decomposition
    via :func:`reset_phase`.

    Args:
        goal_path: Path to the goal summary JSON file.
        runs_dir: Directory containing run output directories.

    Returns:
        A list of phase names that were reset.
    """
    task_name: str = goal_path.stem.removesuffix("-task")
    run_info = get_last_run_info(runs_dir, task_name)
    if run_info is None:
        return []

    timestamp, _status = run_info
    metadata_path: Path = runs_dir / task_name / timestamp / "run_metadata.json"
    if not metadata_path.exists():
        return []

    try:
        meta: dict[str, Any] = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []

    reset_names: list[str] = []
    for phase in meta.get("phases", []):
        if phase.get("outcome") != "SUCCESS":
            name: str = phase.get("name", "")
            if name and reset_phase(goal_path, name):
                reset_names.append(name)

    if reset_names:
        names_str: str = ", ".join(reset_names)
        print(
            f"Auto-resetting {len(reset_names)} errored phase(s) "
            f"from last run: {names_str}"
        )

    return reset_names


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


def get_goal_summaries(
    goal_summaries_dir: Path,
    restore_fn: Callable[[str], str],
    runs_dir: Path | None = None,
    *,
    filter_unrun: bool = False,
    filter_failed: bool = False,
) -> list[GoalInfo]:
    """Return structured summaries for all goals in *goal_summaries_dir*.

    Returns:
        A list of :class:`GoalInfo` objects, sorted by file name.
    """
    goal_files: list[Path] = sorted(goal_summaries_dir.glob("*-task.json"))
    if not goal_files:
        return []

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

    results: list[GoalInfo] = []
    for goal_file in goal_files:
        try:
            data: dict[str, Any] = json.loads(restore_fn(goal_file.read_text()))
        except (json.JSONDecodeError, OSError):
            results.append(GoalInfo(
                name=goal_file.stem.removesuffix("-task"),
                file_path=goal_file,
                main_task="(unreadable)",
            ))
            continue

        observations: list[str | dict[str, str]] = data.get("key_observations", [])
        num_errors = sum(
            1 for o in observations
            if isinstance(o, dict) and o.get("severity") == "error"
        )
        num_warnings = sum(
            1 for o in observations
            if isinstance(o, dict) and o.get("severity") == "warning"
        )

        task_name = goal_file.stem.removesuffix("-task")
        run_info = get_last_run_info(runs_dir, task_name)
        last_ts: str | None = None
        last_status: str | None = None
        if run_info is not None:
            last_ts, last_status = run_info

        results.append(GoalInfo(
            name=task_name,
            file_path=goal_file,
            main_task=data.get("main_task", "(no description)"),
            num_subtasks=len(data.get("subtasks", [])),
            num_observations=len(observations),
            num_errors=num_errors,
            num_warnings=num_warnings,
            last_run_timestamp=last_ts,
            last_run_status=last_status,
            safety=data.get("safety", {}),
        ))

    return results


def list_goals(
    goal_summaries_dir: Path,
    restore_fn: Callable[[str], str],
    runs_dir: Path | None = None,
    *,
    filter_unrun: bool = False,
    filter_failed: bool = False,
) -> None:
    """Print a summary of all existing goals from the goal summaries directory."""
    summaries = get_goal_summaries(
        goal_summaries_dir, restore_fn, runs_dir,
        filter_unrun=filter_unrun, filter_failed=filter_failed,
    )
    if not summaries:
        label = "No matching goals found." if (filter_unrun or filter_failed) else "No goals found."
        print(label)
        return

    print(f"Found {len(summaries)} goal(s):\n")
    for info in summaries:
        if info.main_task == "(unreadable)":
            print(f" {info.name}-task  (unreadable)")
            continue
        print(f"{info.name}-task")
        print(f"  {info.main_task}")
        num_unclassified = info.num_observations - info.num_errors - info.num_warnings
        severity_parts: list[str] = []
        if info.num_errors:
            severity_parts.append(f"{info.num_errors} errors")
        if info.num_warnings:
            severity_parts.append(f"{info.num_warnings} warnings")
        if num_unclassified:
            severity_parts.append(f"{num_unclassified} unclassified")
        severity_str = f" ({', '.join(severity_parts)})" if severity_parts else ""
        safety_label = ""
        if info.safety:
            if info.safety.get("blocked_in_production"):
                safety_label = " [restricted: production-blocked]"
            elif info.safety.get("allowed_environments"):
                envs = ", ".join(info.safety["allowed_environments"])
                safety_label = f" [restricted: {envs} only]"
        if info.last_run_timestamp:
            last_run_str = f"Last run: {info.last_run_timestamp} [{info.last_run_status}]"
        else:
            last_run_str = "  Last run: never"
        print(f"  Subtasks: {info.num_subtasks}  Observations: {info.num_observations}{severity_str}{safety_label}  {last_run_str}")
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

    reset_set: set[str] = {
        p.lower() for p in goal_data.get("reset_phases", [])
    }

    phases: list[dict[str, str]] = []
    for entry in goal_data["subtasks"]:
        if not isinstance(entry, dict):
            continue
        subtask_path: Path = tasks_dir / entry["filename"]
        name: str = subtask_path.stem.replace("-", " ").title()
        if name.lower() in reset_set:
            phases.append({"name": name, "task": ""})
            print(f"  Phase '{name}' marked for fresh decomposition")
            continue
        if not subtask_path.exists():
            print(f"  Warning: subtask file not found {subtask_path}, skipping")
            continue
        task_content: str = restore_fn(subtask_path.read_text())
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
        write_with_history(goal_file, json.dumps(data, indent=2))

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

    reset_phases: list[str] = data.get("reset_phases", [])
    if reset_phases:
        print(f"\n  Reset phases ({len(reset_phases)}):")
        for rp in reset_phases:
            print(f"    - {rp}")

    hints: list[dict[str, str]] = data.get("hints", [])
    if hints:
        print(f"\n  Hints ({len(hints)}):")
        for i, h in enumerate(hints):
            label: str = h["phase"] if h["phase"] else "Goal"
            print(f"    {i}. [{label}] {h['text']}")


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
