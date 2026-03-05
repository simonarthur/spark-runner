"""Goal safety checks: parse safety metadata and enforce environment restrictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spark_runner.models import GoalSafety, SparkConfig


def parse_goal_safety(goal_data: dict[str, Any]) -> GoalSafety:
    """Parse the ``safety`` block from a goal JSON dict.

    Returns a default (unrestricted) ``GoalSafety`` if the block is absent.
    """
    raw: dict[str, Any] = goal_data.get("safety", {})
    if not raw:
        return GoalSafety()
    return GoalSafety(
        blocked_in_production=bool(raw.get("blocked_in_production", False)),
        allowed_environments=list(raw.get("allowed_environments", [])),
        risk_level=str(raw.get("risk_level", "")),
        reason=str(raw.get("reason", "")),
    )


def load_goal_safety(goal_path: Path) -> GoalSafety:
    """Load and parse safety metadata from a goal JSON file.

    Returns a default (unrestricted) ``GoalSafety`` if the file cannot be read.
    """
    try:
        goal_data: dict[str, Any] = json.loads(goal_path.read_text())
    except (json.JSONDecodeError, OSError):
        return GoalSafety()
    return parse_goal_safety(goal_data)


def check_goal_allowed(safety: GoalSafety, config: SparkConfig) -> tuple[bool, str]:
    """Check whether a goal is allowed to run given the current config.

    Returns ``(allowed, reason)`` where *reason* is non-empty when blocked.
    """
    # No environment selected -> no blocking (backward compat)
    if config.active_environment is None:
        return True, ""

    # Force-unsafe overrides all checks
    if config.force_unsafe:
        return True, ""

    # No safety block -> allowed everywhere
    if not safety.blocked_in_production and not safety.allowed_environments:
        return True, ""

    # Look up the environment profile
    env_profile = config.environments.get(config.active_environment)
    is_production = env_profile.is_production if env_profile else False

    # Only block in production environments
    if not is_production:
        return True, ""

    # Check blocked_in_production
    if safety.blocked_in_production:
        reason = safety.reason or "Goal is blocked in production environments"
        return False, reason

    # Check allowed_environments whitelist
    if safety.allowed_environments and config.active_environment not in safety.allowed_environments:
        reason = safety.reason or (
            f"Goal is only allowed in: {', '.join(safety.allowed_environments)}"
        )
        return False, reason

    return True, ""
