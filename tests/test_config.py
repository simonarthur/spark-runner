"""Tests for spark_runner.config: YAML loading, defaults, env var and CLI overrides."""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

from spark_runner.config import (
    build_config,
    load_config_from_yaml,
    resolve_config_path,
    run_setup_wizard,
    _parse_credentials,
    _parse_environments,
    _resolve_value,
)
from spark_runner.models import EnvironmentProfile, SparkConfig


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
        monkeypatch.setenv("SPARK_RUNNER_BASE_URL", "https://env.example.com")
        config = build_config(config_path=config_file, data_dir=tmp_path)
        assert config.base_url == "https://env.example.com"

    def test_spark_data_dir_env_sets_data_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_data_dir = tmp_path / "env_data"
        monkeypatch.setenv("SPARK_RUNNER_DATA_DIR", str(env_data_dir))
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
        monkeypatch.setenv("SPARK_RUNNER_CONFIG", str(config_file))
        config = build_config(data_dir=tmp_path)
        assert config.base_url == "https://fromenv.example.com"


# ── build_config: CLI arg overrides ─────────────────────────────────────


class TestBuildConfigCliOverrides:
    def test_base_url_arg_overrides_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SPARK_RUNNER_BASE_URL", "https://env.example.com")
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


# ── _parse_environments ──────────────────────────────────────────────────


class TestParseEnvironments:
    def test_parses_environment_profiles(self) -> None:
        raw = {
            "dev": {
                "base_url": "https://dev.example.com",
                "credentials": {
                    "default": {"email": "dev@example.com", "password": "devpass"},
                },
            },
            "production": {
                "base_url": "https://app.example.com",
                "is_production": True,
                "credentials": {
                    "default": {"email": "prod@example.com", "password": "prodpass"},
                },
            },
        }
        result = _parse_environments(raw)
        assert "dev" in result
        assert "production" in result
        assert result["dev"].base_url == "https://dev.example.com"
        assert result["dev"].is_production is False
        assert result["production"].is_production is True
        assert result["production"].credentials["default"].email == "prod@example.com"

    def test_empty_dict_returns_empty(self) -> None:
        assert _parse_environments({}) == {}

    def test_non_dict_entries_skipped(self) -> None:
        result = _parse_environments({"bad": "string_value"})
        assert result == {}


# ── build_config: environment selection ──────────────────────────────────


class TestBuildConfigEnvironments:
    def _write_env_config(self, tmp_path: Path) -> Path:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "environments:\n"
            "  staging:\n"
            "    base_url: https://staging.example.com\n"
            "    credentials:\n"
            "      default:\n"
            "        email: stage@example.com\n"
            "        password: stagepass\n"
            "      admin:\n"
            "        email: admin@example.com\n"
            "        password: adminpass\n"
            "  production:\n"
            "    base_url: https://app.example.com\n"
            "    is_production: true\n"
            "    credentials:\n"
            "      default:\n"
            "        email: prod@example.com\n"
            "        password: prodpass\n"
        )
        return config_file

    def test_env_sets_base_url(self, tmp_path: Path) -> None:
        config_file = self._write_env_config(tmp_path)
        config = build_config(config_path=config_file, data_dir=tmp_path, env="staging")
        assert config.base_url == "https://staging.example.com"

    def test_env_sets_credentials(self, tmp_path: Path) -> None:
        config_file = self._write_env_config(tmp_path)
        config = build_config(config_path=config_file, data_dir=tmp_path, env="staging")
        assert "admin" in config.credentials
        assert config.credentials["default"].email == "stage@example.com"

    def test_cli_base_url_overrides_env(self, tmp_path: Path) -> None:
        config_file = self._write_env_config(tmp_path)
        config = build_config(
            config_path=config_file, data_dir=tmp_path,
            env="staging", base_url="https://override.example.com",
        )
        assert config.base_url == "https://override.example.com"

    def test_unknown_env_raises_value_error(self, tmp_path: Path) -> None:
        config_file = self._write_env_config(tmp_path)
        with pytest.raises(ValueError, match="Unknown environment 'nonexistent'"):
            build_config(config_path=config_file, data_dir=tmp_path, env="nonexistent")

    def test_no_env_uses_global_credentials(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("USER_EMAIL", raising=False)
        monkeypatch.delenv("USER_PASSWORD", raising=False)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "credentials:\n"
            "  default:\n"
            "    email: global@example.com\n"
            "    password: globalpass\n"
            "environments:\n"
            "  staging:\n"
            "    base_url: https://staging.example.com\n"
            "    credentials:\n"
            "      default:\n"
            "        email: stage@example.com\n"
            "        password: stagepass\n"
        )
        config = build_config(config_path=config_file, data_dir=tmp_path)
        assert config.credentials["default"].email == "global@example.com"
        assert config.active_environment is None

    def test_env_sets_active_environment(self, tmp_path: Path) -> None:
        config_file = self._write_env_config(tmp_path)
        config = build_config(config_path=config_file, data_dir=tmp_path, env="production")
        assert config.active_environment == "production"
        assert config.environments["production"].is_production is True

    def test_force_unsafe_flag(self, tmp_path: Path) -> None:
        config_file = self._write_env_config(tmp_path)
        config = build_config(
            config_path=config_file, data_dir=tmp_path,
            env="production", force_unsafe=True,
        )
        assert config.force_unsafe is True

    def test_environments_parsed_into_config(self, tmp_path: Path) -> None:
        config_file = self._write_env_config(tmp_path)
        config = build_config(config_path=config_file, data_dir=tmp_path)
        assert "staging" in config.environments
        assert "production" in config.environments

    def test_no_environments_section_backward_compat(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("general:\n  base_url: https://example.com\n")
        config = build_config(config_path=config_file, data_dir=tmp_path)
        assert config.environments == {}
        assert config.active_environment is None


# ── resolve_config_path ──────────────────────────────────────────────────


class TestResolveConfigPath:
    def test_explicit_path_returned(self, tmp_path: Path) -> None:
        p = tmp_path / "my_config.yaml"
        result = resolve_config_path(config_path=p)
        assert result == p.resolve()

    def test_env_var_overrides_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("SPARK_RUNNER_CONFIG", str(tmp_path / "env.yaml"))
        result = resolve_config_path()
        assert result == (tmp_path / "env.yaml").resolve()

    def test_default_uses_data_dir(self, tmp_path: Path) -> None:
        result = resolve_config_path(data_dir=tmp_path)
        assert result == tmp_path / "config.yaml"

    def test_default_uses_home_spark_runner(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("SPARK_RUNNER_CONFIG", raising=False)
        monkeypatch.delenv("SPARK_RUNNER_DATA_DIR", raising=False)
        result = resolve_config_path()
        assert result == Path("~/spark_runner").expanduser().resolve() / "config.yaml"


# ── run_setup_wizard ─────────────────────────────────────────────────────


class TestRunSetupWizard:
    # IMPORTANT: Always use tmp_path for data directories in these tests.
    # Never pass "~/spark_runner" or any real home-relative path to
    # run_setup_wizard — it creates directories on disk and will pollute
    # the user's home directory.
    def _run_wizard(
        self, tmp_path: Path, prompts: list[str],
        confirms: list[bool] | None = None,
    ) -> Path:
        """Helper: run the wizard with canned prompt/confirm responses.

        ``prompts`` maps to click.prompt calls:
        - data_dir, base_url, email, password
        - (if additional logins: profile_name, username, password per each)
        - anthropic_api_key
        - (if use_browseruse_llm: browseruse_api_key)
        - (if environments: env prompts)

        ``confirms`` maps to click.confirm calls:
        - "set up additional logins?"
        - ... (if yes: "add another login?" after each)
        - "Use BrowserUse cloud LLM?"
        - "store secrets as env var references?" (only when secrets present)
        - "set up more environments?"
        - ... (if yes: "is production?", "add another environment?" after each)

        Defaults to [False, False, False] (no logins, no BrowserUse, no envs —
        only valid when no secrets are provided).
        """
        if confirms is None:
            confirms = [False, False, False]
        with patch("click.prompt", side_effect=prompts), \
             patch("click.confirm", side_effect=confirms):
            return run_setup_wizard(tmp_path / "ignored.yaml")

    def test_writes_valid_yaml(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [str(data_dir), "https://example.com", "me@test.com", "secret", ""],
            # no logins, no browseruse, decline env-vars, no envs
            confirms=[False, False, False, False],
        )
        expected = data_dir / "config.yaml"
        assert result == expected
        assert expected.exists()
        data = yaml.safe_load(expected.read_text())
        assert data["general"]["data_dir"] == str(data_dir)
        assert data["general"]["base_url"] == "https://example.com"
        assert data["credentials"]["default"]["email"] == "me@test.com"
        assert data["credentials"]["default"]["password"] == "secret"

    def test_creates_data_directories(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "mydata"
        self._run_wizard(tmp_path, [
            str(data_dir), "https://example.com", "", "", "",
        ])
        assert (data_dir / "tasks").is_dir()
        assert (data_dir / "goal_summaries").is_dir()
        assert (data_dir / "runs").is_dir()

    def test_sets_permissions_when_credentials_provided(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [str(data_dir), "https://example.com", "user@test.com", "pw", ""],
            # no logins, no browseruse, decline env-vars, no envs
            confirms=[False, False, False, False],
        )
        mode = result.stat().st_mode & 0o777
        assert mode == 0o600

    def test_skips_permissions_when_no_credentials(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(tmp_path, [
            str(data_dir), "https://example.com", "", "", "",
        ])
        mode = result.stat().st_mode & 0o777
        assert mode != 0o600

    def test_uses_defaults_when_accepted(self, tmp_path: Path) -> None:
        # NOTE: Never use ~/spark_runner here — tests must use tmp_path to
        # avoid creating real directories in the user's home.
        default_dir = str(tmp_path / "spark_runner")
        result = self._run_wizard(tmp_path, [
            default_dir, "https://sparky-web-dev.vercel.app", "", "", "",
        ])
        data = yaml.safe_load(result.read_text())
        assert data["general"]["data_dir"] == default_dir
        assert data["general"]["base_url"] == "https://sparky-web-dev.vercel.app"

    def test_returns_config_inside_data_dir(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(tmp_path, [
            str(data_dir), "https://example.com", "", "", "",
        ])
        assert result == data_dir / "config.yaml"

    def test_shows_completion_message(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        data_dir = tmp_path / "data"
        self._run_wizard(tmp_path, [
            str(data_dir), "https://example.com", "", "", "",
        ])
        output = capsys.readouterr().out
        assert "Setup complete!" in output
        assert "spark-runner run" in output

    def test_custom_data_dir_shows_env_var_hint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        data_dir = tmp_path / "custom_data"
        self._run_wizard(tmp_path, [
            str(data_dir), "https://example.com", "", "", "",
        ])
        output = capsys.readouterr().out
        assert "SPARK_RUNNER_DATA_DIR" in output
        assert str(data_dir) in output

    def test_default_data_dir_no_env_var_hint(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        # NOTE: We must redirect _resolve_path so "~/spark_runner" doesn't
        # create real directories in the user's home.
        fake_home = tmp_path / "fake_spark_runner"
        with patch("spark_runner.config._resolve_path", return_value=fake_home):
            self._run_wizard(tmp_path, [
                "~/spark_runner", "https://example.com", "", "", "",
            ])
        output = capsys.readouterr().out
        assert "SPARK_RUNNER_DATA_DIR" not in output

    def test_single_environment(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [
                str(data_dir), "https://example.com", "", "", "",
                "staging", "https://staging.example.com", "stage@test.com", "stagepass",
            ],
            # no logins, no browseruse, yes envs, no is_prod, no add another env
            confirms=[False, False, True, False, False],
        )
        data = yaml.safe_load(result.read_text())
        assert "environments" in data
        assert "staging" in data["environments"]
        env = data["environments"]["staging"]
        assert env["base_url"] == "https://staging.example.com"
        assert env["credentials"]["default"]["email"] == "stage@test.com"

    def test_multiple_environments(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [
                str(data_dir), "https://example.com", "", "", "",
                "staging", "https://staging.example.com", "", "",
                "production", "https://app.example.com", "prod@test.com", "prodpass",
            ],
            # no logins, no browseruse, yes envs, no is_prod, yes add another, yes is_prod, no add another
            confirms=[False, False, True, False, True, True, False],
        )
        data = yaml.safe_load(result.read_text())
        assert "staging" in data["environments"]
        assert "production" in data["environments"]
        assert data["environments"]["production"].get("is_production") is True

    def test_no_environments_produces_comments(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(tmp_path, [
            str(data_dir), "https://example.com", "", "", "",
        ])
        raw = result.read_text()
        assert "# environments:" in raw

    def test_environment_credentials_trigger_permissions(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [
                str(data_dir), "https://example.com", "", "", "",
                "staging", "https://staging.example.com", "s@test.com", "pw",
            ],
            # no logins, no browseruse, yes envs, no is_prod, no add another env
            confirms=[False, False, True, False, False],
        )
        mode = result.stat().st_mode & 0o777
        assert mode == 0o600

    def test_additional_logins(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [
                str(data_dir), "https://example.com", "default@test.com", "defpw",
                "admin", "admin@test.com", "adminpw", "",
            ],
            # yes logins, no add another, no browseruse, decline env-vars, no envs
            confirms=[True, False, False, False, False],
        )
        data = yaml.safe_load(result.read_text())
        assert "admin" in data["credentials"]
        assert data["credentials"]["admin"]["email"] == "admin@test.com"
        assert data["credentials"]["admin"]["password"] == "adminpw"
        assert data["credentials"]["default"]["email"] == "default@test.com"

    def test_multiple_additional_logins(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [
                str(data_dir), "https://example.com", "", "",
                "admin", "admin@test.com", "adminpw",
                "viewer", "viewer@test.com", "viewpw", "",
            ],
            # yes logins, yes add another, no add another, no browseruse, decline env-vars, no envs
            confirms=[True, True, False, False, False, False],
        )
        data = yaml.safe_load(result.read_text())
        assert "admin" in data["credentials"]
        assert "viewer" in data["credentials"]

    def test_additional_login_credentials_trigger_permissions(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [
                str(data_dir), "https://example.com", "", "",
                "admin", "admin@test.com", "pw", "",
            ],
            # yes logins, no add another, no browseruse, decline env-vars, no envs
            confirms=[True, False, False, False, False],
        )
        mode = result.stat().st_mode & 0o777
        assert mode == 0o600


# ── _resolve_value ───────────────────────────────────────────────────────


class TestResolveValue:
    def test_plain_string_unchanged(self) -> None:
        assert _resolve_value("hello") == "hello"

    def test_dollar_syntax(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_VAR", "resolved")
        assert _resolve_value("$MY_VAR") == "resolved"

    def test_braced_syntax(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_VAR", "resolved")
        assert _resolve_value("${MY_VAR}") == "resolved"

    def test_unset_var_returns_original(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        assert _resolve_value("$NONEXISTENT_VAR") == "$NONEXISTENT_VAR"

    def test_non_string_passes_through(self) -> None:
        assert _resolve_value(42) == 42  # type: ignore[arg-type]

    def test_embedded_dollar_not_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FOO", "bar")
        assert _resolve_value("prefix$FOO") == "prefix$FOO"

    def test_whitespace_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_VAR", "resolved")
        assert _resolve_value("  $MY_VAR  ") == "resolved"


# ── _parse_credentials with env vars ─────────────────────────────────────


class TestParseCredentialsEnvVars:
    def test_resolves_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEST_EMAIL", "env@example.com")
        monkeypatch.setenv("TEST_PASSWORD", "envpass")
        raw: dict[str, Any] = {
            "default": {
                "email": "$TEST_EMAIL",
                "password": "${TEST_PASSWORD}",
            },
        }
        result = _parse_credentials(raw)
        assert result["default"].email == "env@example.com"
        assert result["default"].password == "envpass"

    def test_unset_env_var_keeps_reference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MISSING_VAR", raising=False)
        raw: dict[str, Any] = {
            "default": {
                "email": "$MISSING_VAR",
                "password": "literal",
            },
        }
        result = _parse_credentials(raw)
        assert result["default"].email == "$MISSING_VAR"
        assert result["default"].password == "literal"

    def test_extra_fields_resolved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_API_KEY", "secret123")
        raw: dict[str, Any] = {
            "default": {
                "email": "user@example.com",
                "password": "pw",
                "api_key": "$MY_API_KEY",
            },
        }
        result = _parse_credentials(raw)
        assert result["default"].extra["api_key"] == "secret123"


# ── Wizard env-var mode ──────────────────────────────────────────────────


class TestWizardEnvVarMode:
    def _run_wizard(
        self, tmp_path: Path, prompts: list[str],
        confirms: list[bool],
    ) -> Path:
        with patch("click.prompt", side_effect=prompts), \
             patch("click.confirm", side_effect=confirms):
            return run_setup_wizard(tmp_path / "ignored.yaml")

    def test_env_var_mode_writes_references(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [str(data_dir), "https://example.com", "me@test.com", "secret", ""],
            # no logins, no browseruse, accept env-vars, no envs
            confirms=[False, False, True, False],
        )
        data = yaml.safe_load(result.read_text())
        assert data["credentials"]["default"]["email"] == "$SPARK_RUNNER_DEFAULT_EMAIL"
        assert data["credentials"]["default"]["password"] == "$SPARK_RUNNER_DEFAULT_PASSWORD"

    def test_env_var_mode_shows_export_hints(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        data_dir = tmp_path / "data"
        self._run_wizard(
            tmp_path,
            [str(data_dir), "https://example.com", "me@test.com", "secret", ""],
            # no logins, no browseruse, accept env-vars, no envs
            confirms=[False, False, True, False],
        )
        output = capsys.readouterr().out
        assert 'export SPARK_RUNNER_DEFAULT_PASSWORD="secret"' in output
        assert 'export SPARK_RUNNER_DEFAULT_EMAIL="me@test.com"' in output

    def test_env_var_mode_skips_permissions(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [str(data_dir), "https://example.com", "me@test.com", "secret", ""],
            # no logins, no browseruse, accept env-vars, no envs
            confirms=[False, False, True, False],
        )
        mode = result.stat().st_mode & 0o777
        assert mode != 0o600

    def test_plaintext_mode_unchanged(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [str(data_dir), "https://example.com", "me@test.com", "secret", ""],
            # no logins, no browseruse, decline env-vars, no envs
            confirms=[False, False, False, False],
        )
        data = yaml.safe_load(result.read_text())
        assert data["credentials"]["default"]["password"] == "secret"

    def test_env_var_mode_extra_credentials(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [
                str(data_dir), "https://example.com", "me@test.com", "secret",
                "admin", "admin@test.com", "adminpw", "",
            ],
            # yes logins, no add another, no browseruse, accept env-vars, no envs
            confirms=[True, False, False, True, False],
        )
        data = yaml.safe_load(result.read_text())
        assert data["credentials"]["admin"]["password"] == "$SPARK_RUNNER_ADMIN_PASSWORD"
        output = capsys.readouterr().out
        assert 'export SPARK_RUNNER_ADMIN_PASSWORD="adminpw"' in output

    def test_no_password_skips_env_var_prompt(self, tmp_path: Path) -> None:
        """When no password or API keys provided, env-var prompt is not shown."""
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [str(data_dir), "https://example.com", "me@test.com", "", ""],
            # no logins, no browseruse, no envs
            confirms=[False, False, False],
        )
        data = yaml.safe_load(result.read_text())
        assert data["credentials"]["default"]["email"] == "me@test.com"


# ── API key configuration ────────────────────────────────────────────────


class TestBuildConfigApiKeys:
    def test_api_keys_loaded_from_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "api_keys:\n"
            "  anthropic: sk-ant-test123\n"
            "  browseruse: bu-test456\n"
            "general:\n"
            "  use_browseruse_llm: true\n"
        )
        config = build_config(config_path=config_file, data_dir=tmp_path)
        assert config.anthropic_api_key == "sk-ant-test123"
        assert config.browseruse_api_key == "bu-test456"
        assert config.use_browseruse_llm is True

    def test_api_keys_resolve_env_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "api_keys:\n"
            "  anthropic: $ANTHROPIC_API_KEY\n"
        )
        config = build_config(config_path=config_file, data_dir=tmp_path)
        assert config.anthropic_api_key == "sk-ant-from-env"

    def test_missing_api_keys_default_to_empty(self, tmp_path: Path) -> None:
        config = build_config(data_dir=tmp_path)
        assert config.anthropic_api_key == ""
        assert config.browseruse_api_key == ""
        assert config.use_browseruse_llm is False

    def test_use_browseruse_llm_defaults_to_false(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("general:\n  base_url: https://example.com\n")
        config = build_config(config_path=config_file, data_dir=tmp_path)
        assert config.use_browseruse_llm is False


class TestWizardApiKeys:
    def _run_wizard(
        self, tmp_path: Path, prompts: list[str],
        confirms: list[bool],
    ) -> Path:
        with patch("click.prompt", side_effect=prompts), \
             patch("click.confirm", side_effect=confirms):
            return run_setup_wizard(tmp_path / "ignored.yaml")

    def test_anthropic_key_written_to_config(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [str(data_dir), "https://example.com", "", "", "sk-ant-test"],
            # no logins, no browseruse, decline env-vars, no envs
            confirms=[False, False, False, False],
        )
        data = yaml.safe_load(result.read_text())
        assert data["api_keys"]["anthropic"] == "sk-ant-test"

    def test_browseruse_key_written_when_enabled(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [str(data_dir), "https://example.com", "", "", "sk-ant-test", "bu-key123"],
            # no logins, yes browseruse, decline env-vars, no envs
            confirms=[False, True, False, False],
        )
        data = yaml.safe_load(result.read_text())
        assert data["general"]["use_browseruse_llm"] is True
        assert data["api_keys"]["browseruse"] == "bu-key123"

    def test_api_key_triggers_permissions(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [str(data_dir), "https://example.com", "", "", "sk-ant-test"],
            # no logins, no browseruse, decline env-vars, no envs
            confirms=[False, False, False, False],
        )
        mode = result.stat().st_mode & 0o777
        assert mode == 0o600

    def test_api_key_triggers_env_var_prompt(self, tmp_path: Path) -> None:
        """An API key alone (no password) should trigger the env-var prompt."""
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            [str(data_dir), "https://example.com", "", "", "sk-ant-test"],
            # no logins, no browseruse, accept env-vars, no envs
            confirms=[False, False, True, False],
        )
        data = yaml.safe_load(result.read_text())
        assert data["api_keys"]["anthropic"] == "$ANTHROPIC_API_KEY"

    def test_api_key_env_var_export_hints(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        data_dir = tmp_path / "data"
        self._run_wizard(
            tmp_path,
            [str(data_dir), "https://example.com", "", "", "sk-ant-test", "bu-key123"],
            # no logins, yes browseruse, accept env-vars, no envs
            confirms=[False, True, True, False],
        )
        output = capsys.readouterr().out
        assert 'export ANTHROPIC_API_KEY="sk-ant-test"' in output
        assert 'export BROWSER_USE_API_KEY="bu-key123"' in output

    def test_no_browseruse_key_prompt_when_declined(self, tmp_path: Path) -> None:
        """Declining BrowserUse should not prompt for its API key."""
        data_dir = tmp_path / "data"
        result = self._run_wizard(
            tmp_path,
            # Only 5 prompts: data_dir, url, email, pw, anthropic_key
            [str(data_dir), "https://example.com", "", "", "sk-ant-test"],
            # no logins, no browseruse, decline env-vars, no envs
            confirms=[False, False, False, False],
        )
        data = yaml.safe_load(result.read_text())
        assert data["api_keys"]["browseruse"] == ""
        assert data["general"]["use_browseruse_llm"] is False
