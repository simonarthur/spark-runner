"""Tests for spark_runner.config: YAML loading, defaults, env var and CLI overrides."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from spark_runner.config import build_config, load_config_from_yaml
from spark_runner.models import SparkConfig


# ── load_config_from_yaml ────────────────────────────────────────────────


class TestLoadConfigFromYaml:
    def test_missing_file_returns_empty_dict(self, tmp_path: Path) -> None:
        result = load_config_from_yaml(tmp_path / "does_not_exist.yaml")
        assert result == {}

    def test_valid_yaml_is_loaded(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("general:\n  base_url: https://example.com\n")
        result = load_config_from_yaml(config_file)
        assert result["general"]["base_url"] == "https://example.com"

    def test_empty_yaml_file_returns_empty_dict(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        result = load_config_from_yaml(config_file)
        assert result == {}

    def test_malformed_yaml_returns_empty_dict(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("key: [unclosed bracket\n")
        result = load_config_from_yaml(config_file)
        assert result == {}

    def test_credentials_section_loaded(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "credentials:\n"
            "  admin:\n"
            "    email: admin@example.com\n"
            "    password: secret\n"
        )
        result = load_config_from_yaml(config_file)
        assert "admin" in result["credentials"]
        assert result["credentials"]["admin"]["email"] == "admin@example.com"


# ── build_config: defaults ───────────────────────────────────────────────


class TestBuildConfigDefaults:
    def test_returns_spark_config_instance(self, tmp_path: Path) -> None:
        config = build_config(data_dir=tmp_path)
        assert isinstance(config, SparkConfig)

    def test_default_base_url_is_set(self, tmp_path: Path) -> None:
        config = build_config(data_dir=tmp_path)
        assert config.base_url == "https://sparky-web-dev.vercel.app"

    def test_default_credential_profile_is_default(self, tmp_path: Path) -> None:
        config = build_config(data_dir=tmp_path)
        assert config.active_credential_profile == "default"

    def test_data_dir_subdirectories_set(self, tmp_path: Path) -> None:
        config = build_config(data_dir=tmp_path)
        assert config.tasks_dir == tmp_path / "tasks"
        assert config.goal_summaries_dir == tmp_path / "goal_summaries"
        assert config.runs_dir == tmp_path / "runs"

    def test_default_models_populated(self, tmp_path: Path) -> None:
        config = build_config(data_dir=tmp_path)
        assert "browser_control" in config.models
        assert "summarization" in config.models

    def test_headless_defaults_to_false(self, tmp_path: Path) -> None:
        config = build_config(data_dir=tmp_path)
        assert config.headless is False

    def test_auto_close_defaults_to_false(self, tmp_path: Path) -> None:
        config = build_config(data_dir=tmp_path)
        assert config.auto_close is False


# ── build_config: YAML file loading ─────────────────────────────────────


class TestBuildConfigFromYaml:
    def test_loads_base_url_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "general:\n  base_url: https://staging.example.com\n"
        )
        config = build_config(config_path=config_file, data_dir=tmp_path)
        assert config.base_url == "https://staging.example.com"

    def test_loads_data_dir_from_yaml(self, tmp_path: Path) -> None:
        alt_data_dir = tmp_path / "alt_data"
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"general:\n  data_dir: {alt_data_dir}\n")
        config = build_config(config_path=config_file)
        assert config.data_dir == alt_data_dir

    def test_loads_credentials_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "credentials:\n"
            "  admin:\n"
            "    email: admin@example.com\n"
            "    password: adminpass\n"
        )
        config = build_config(config_path=config_file, data_dir=tmp_path)
        assert "admin" in config.credentials
        assert config.credentials["admin"].email == "admin@example.com"
        assert config.credentials["admin"].password == "adminpass"

    def test_loads_models_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "models:\n"
            "  summarization:\n"
            "    model: claude-haiku-3-20241022\n"
            "    max_tokens: 512\n"
        )
        config = build_config(config_path=config_file, data_dir=tmp_path)
        assert config.models["summarization"].model == "claude-haiku-3-20241022"
        assert config.models["summarization"].max_tokens == 512

    def test_extra_credential_fields_go_to_extra(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "credentials:\n"
            "  user:\n"
            "    email: u@example.com\n"
            "    password: pw\n"
            "    api_key: abc123\n"
        )
        config = build_config(config_path=config_file, data_dir=tmp_path)
        assert config.credentials["user"].extra.get("api_key") == "abc123"


# ── build_config: environment variable overrides ─────────────────────────


class TestBuildConfigEnvVarOverrides:
    def test_spark_base_url_env_overrides_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("general:\n  base_url: https://yaml.example.com\n")
        monkeypatch.setenv("SPARK_BASE_URL", "https://env.example.com")
        config = build_config(config_path=config_file, data_dir=tmp_path)
        assert config.base_url == "https://env.example.com"

    def test_spark_data_dir_env_sets_data_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_data_dir = tmp_path / "env_data"
        monkeypatch.setenv("SPARK_DATA_DIR", str(env_data_dir))
        config = build_config()
        assert config.data_dir == env_data_dir

    def test_user_email_env_populates_default_profile(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("USER_EMAIL", "env@example.com")
        monkeypatch.setenv("USER_PASSWORD", "envpass")
        config = build_config(data_dir=tmp_path)
        assert config.credentials["default"].email == "env@example.com"
        assert config.credentials["default"].password == "envpass"

    def test_spark_config_env_sets_config_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        config_file = tmp_path / "env_config.yaml"
        config_file.write_text("general:\n  base_url: https://fromenv.example.com\n")
        monkeypatch.setenv("SPARK_CONFIG", str(config_file))
        config = build_config(data_dir=tmp_path)
        assert config.base_url == "https://fromenv.example.com"


# ── build_config: CLI arg overrides ─────────────────────────────────────


class TestBuildConfigCliOverrides:
    def test_base_url_arg_overrides_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPARK_BASE_URL", "https://env.example.com")
        config = build_config(data_dir=tmp_path, base_url="https://cli.example.com")
        assert config.base_url == "https://cli.example.com"

    def test_base_url_arg_overrides_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("general:\n  base_url: https://yaml.example.com\n")
        config = build_config(
            config_path=config_file,
            data_dir=tmp_path,
            base_url="https://cli.example.com",
        )
        assert config.base_url == "https://cli.example.com"

    def test_headless_arg_is_respected(self, tmp_path: Path) -> None:
        config = build_config(data_dir=tmp_path, headless=True)
        assert config.headless is True

    def test_auto_close_arg_is_respected(self, tmp_path: Path) -> None:
        config = build_config(data_dir=tmp_path, auto_close=True)
        assert config.auto_close is True

    def test_credential_profile_arg_sets_active_profile(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "credentials:\n"
            "  staging:\n"
            "    email: s@example.com\n"
            "    password: stagingpw\n"
        )
        config = build_config(
            config_path=config_file,
            data_dir=tmp_path,
            credential_profile="staging",
        )
        assert config.active_credential_profile == "staging"

    def test_model_overrides_arg_applied(self, tmp_path: Path) -> None:
        config = build_config(
            data_dir=tmp_path,
            model_overrides={"summarization": "claude-haiku-3-20241022"},
        )
        assert config.models["summarization"].model == "claude-haiku-3-20241022"

    def test_model_overrides_creates_new_entry_if_needed(self, tmp_path: Path) -> None:
        config = build_config(
            data_dir=tmp_path,
            model_overrides={"custom_purpose": "claude-opus-4-6"},
        )
        assert config.models["custom_purpose"].model == "claude-opus-4-6"

    def test_update_summary_flag(self, tmp_path: Path) -> None:
        config = build_config(data_dir=tmp_path, update_summary=False)
        assert config.update_summary is False

    def test_update_tasks_flag(self, tmp_path: Path) -> None:
        config = build_config(data_dir=tmp_path, update_tasks=False)
        assert config.update_tasks is False

    def test_knowledge_reuse_flag(self, tmp_path: Path) -> None:
        config = build_config(data_dir=tmp_path, knowledge_reuse=False)
        assert config.knowledge_reuse is False
