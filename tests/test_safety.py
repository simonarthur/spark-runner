"""Tests for spark_runner.safety: goal safety parsing and environment enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spark_runner.models import EnvironmentProfile, GoalSafety, SparkConfig
from spark_runner.safety import check_goal_allowed, load_goal_safety, parse_goal_safety


# ── parse_goal_safety ────────────────────────────────────────────────────


class TestParseGoalSafety:
    def test_no_safety_block_returns_default(self) -> None:
        safety = parse_goal_safety({"main_task": "do stuff"})
        assert safety == GoalSafety()

    def test_blocked_in_production_parsed(self) -> None:
        safety = parse_goal_safety({
            "safety": {
                "blocked_in_production": True,
                "risk_level": "high",
                "reason": "Modifies global settings",
            }
        })
        assert safety.blocked_in_production is True
        assert safety.risk_level == "high"
        assert safety.reason == "Modifies global settings"

    def test_allowed_environments_parsed(self) -> None:
        safety = parse_goal_safety({
            "safety": {
                "allowed_environments": ["dev", "staging"],
                "risk_level": "medium",
                "reason": "Changes user permissions",
            }
        })
        assert safety.allowed_environments == ["dev", "staging"]
        assert safety.blocked_in_production is False

    def test_empty_safety_block_returns_default(self) -> None:
        safety = parse_goal_safety({"safety": {}})
        assert safety == GoalSafety()


# ── load_goal_safety ─────────────────────────────────────────────────────


class TestLoadGoalSafety:
    def test_loads_from_file(self, tmp_path: Path) -> None:
        goal_file = tmp_path / "goal.json"
        goal_file.write_text(json.dumps({
            "main_task": "test",
            "safety": {"blocked_in_production": True, "reason": "dangerous"},
        }))
        safety = load_goal_safety(goal_file)
        assert safety.blocked_in_production is True
        assert safety.reason == "dangerous"

    def test_missing_file_returns_default(self, tmp_path: Path) -> None:
        safety = load_goal_safety(tmp_path / "nonexistent.json")
        assert safety == GoalSafety()

    def test_invalid_json_returns_default(self, tmp_path: Path) -> None:
        goal_file = tmp_path / "bad.json"
        goal_file.write_text("not json!")
        safety = load_goal_safety(goal_file)
        assert safety == GoalSafety()


# ── check_goal_allowed ───────────────────────────────────────────────────


def _make_config(
    *,
    active_env: str | None = None,
    envs: dict[str, EnvironmentProfile] | None = None,
    force_unsafe: bool = False,
) -> SparkConfig:
    """Helper to build a SparkConfig for safety tests."""
    return SparkConfig(
        environments=envs or {},
        active_environment=active_env,
        force_unsafe=force_unsafe,
    )


class TestCheckGoalAllowed:
    def test_no_restrictions_always_allowed(self) -> None:
        """A goal without safety metadata runs everywhere."""
        safety = GoalSafety()
        config = _make_config(
            active_env="production",
            envs={"production": EnvironmentProfile(name="production", is_production=True)},
        )
        allowed, reason = check_goal_allowed(safety, config)
        assert allowed is True
        assert reason == ""

    def test_no_env_selected_always_allowed(self) -> None:
        """Without --env, no blocking occurs (backward compat)."""
        safety = GoalSafety(blocked_in_production=True, reason="dangerous")
        config = _make_config(active_env=None)
        allowed, reason = check_goal_allowed(safety, config)
        assert allowed is True

    def test_blocked_in_production_blocks_production(self) -> None:
        safety = GoalSafety(blocked_in_production=True, reason="Modifies global settings")
        config = _make_config(
            active_env="production",
            envs={"production": EnvironmentProfile(name="production", is_production=True)},
        )
        allowed, reason = check_goal_allowed(safety, config)
        assert allowed is False
        assert "Modifies global settings" in reason

    def test_blocked_in_production_allows_non_production(self) -> None:
        safety = GoalSafety(blocked_in_production=True, reason="dangerous")
        config = _make_config(
            active_env="staging",
            envs={"staging": EnvironmentProfile(name="staging", is_production=False)},
        )
        allowed, reason = check_goal_allowed(safety, config)
        assert allowed is True

    def test_allowed_environments_whitelist_blocks_production(self) -> None:
        safety = GoalSafety(allowed_environments=["dev", "staging"], reason="only dev/staging")
        config = _make_config(
            active_env="production",
            envs={"production": EnvironmentProfile(name="production", is_production=True)},
        )
        allowed, reason = check_goal_allowed(safety, config)
        assert allowed is False
        assert "only dev/staging" in reason

    def test_allowed_environments_whitelist_allows_listed_production(self) -> None:
        """If production is in the whitelist, it should be allowed."""
        safety = GoalSafety(allowed_environments=["dev", "production"])
        config = _make_config(
            active_env="production",
            envs={"production": EnvironmentProfile(name="production", is_production=True)},
        )
        allowed, reason = check_goal_allowed(safety, config)
        assert allowed is True

    def test_force_unsafe_overrides_block(self) -> None:
        safety = GoalSafety(blocked_in_production=True, reason="dangerous")
        config = _make_config(
            active_env="production",
            envs={"production": EnvironmentProfile(name="production", is_production=True)},
            force_unsafe=True,
        )
        allowed, reason = check_goal_allowed(safety, config)
        assert allowed is True

    def test_non_production_env_never_blocked(self) -> None:
        """Non-production environments are never blocked, even if not in allowed_environments."""
        safety = GoalSafety(allowed_environments=["dev"], reason="only dev")
        config = _make_config(
            active_env="staging",
            envs={"staging": EnvironmentProfile(name="staging", is_production=False)},
        )
        allowed, reason = check_goal_allowed(safety, config)
        assert allowed is True

    def test_blocked_default_reason_when_no_reason_given(self) -> None:
        safety = GoalSafety(blocked_in_production=True)
        config = _make_config(
            active_env="production",
            envs={"production": EnvironmentProfile(name="production", is_production=True)},
        )
        allowed, reason = check_goal_allowed(safety, config)
        assert allowed is False
        assert "blocked in production" in reason.lower()
