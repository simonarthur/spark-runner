#!/usr/bin/env python3
"""Find orphan task files that are not referenced in any goal summary."""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(BASE_DIR, "tasks")
GOAL_SUMMARIES_DIR = os.path.join(BASE_DIR, "goal_summaries")


def get_task_files():
    """Return set of filenames in the tasks directory."""
    return set(
        f for f in os.listdir(TASKS_DIR)
        if os.path.isfile(os.path.join(TASKS_DIR, f))
    )


def get_referenced_tasks():
    """Return set of task filenames referenced in goal summaries."""
    referenced = set()
    for filename in os.listdir(GOAL_SUMMARIES_DIR):
        if not filename.endswith(".json"):
            continue
        filepath = os.path.join(GOAL_SUMMARIES_DIR, filename)
        with open(filepath) as f:
            data = json.load(f)
        for subtask in data.get("subtasks", []):
            if "filename" in subtask:
                referenced.add(subtask["filename"])
    return referenced


def main():
    task_files = get_task_files()
    referenced = get_referenced_tasks()
    orphans = sorted(task_files - referenced)

    if not orphans:
        print("No orphan task files found.")
        return

    print(f"Found {len(orphans)} orphan task file(s) not referenced in any goal summary:\n")
    for f in orphans:
        print(f"  {f}")


if __name__ == "__main__":
    main()
