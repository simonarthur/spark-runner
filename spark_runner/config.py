"""Configuration system: YAML loading, defaults, and environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from spark_runner.models import CredentialProfile, EnvironmentProfile, ModelConfig, SparkConfig


def _resolve_path(p: str | Path) -> Path:
    """Expand ``~`` and return an absolute Path."""
    return Path(p).expanduser().resolve()


def load_config_from_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file and return its contents as a dict.

    Returns an empty dict if the file is missing or ``pyyaml`` is not installed.
    """
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError:
        print("Warning: pyyaml not installed, skipping YAML config")
        return {}
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return data
    except Exception as e:
        print(f"Warning: could not parse config file {path}: {e}")
        return {}


def _parse_credentials(raw: dict[str, Any]) -> dict[str, CredentialProfile]:
    """Parse the ``credentials:`` section of a YAML config."""
    result: dict[str, CredentialProfile] = {}
    for name, profile_data in raw.items():
        if isinstance(profile_data, dict):
            result[name] = CredentialProfile(
                email=profile_data.get("email", ""),
                password=profile_data.get("password", ""),
                extra={
                    k: v
                    for k, v in profile_data.items()
                    if k not in ("email", "password")
                },
            )
    return result


def _parse_models(raw: dict[str, Any]) -> dict[str, ModelConfig]:
    """Parse the ``models:`` section of a YAML config."""
    result: dict[str, ModelConfig] = {}
    for purpose, model_data in raw.items():
        if isinstance(model_data, dict):
            result[purpose] = ModelConfig(
                model=model_data.get("model", "claude-sonnet-4-5-20250929"),
                max_tokens=model_data.get("max_tokens", 4096),
                temperature=model_data.get("temperature", 0.0),
            )
    return result


def _parse_environments(raw: dict[str, Any]) -> dict[str, EnvironmentProfile]:
    """Parse the ``environments:`` section of a YAML config."""
    result: dict[str, EnvironmentProfile] = {}
    for name, env_data in raw.items():
        if isinstance(env_data, dict):
            creds: dict[str, CredentialProfile] = {}
            raw_creds = env_data.get("credentials", {})
            if raw_creds:
                creds = _parse_credentials(raw_creds)
            result[name] = EnvironmentProfile(
                name=name,
                base_url=env_data.get("base_url", ""),
                is_production=bool(env_data.get("is_production", False)),
                credentials=creds,
            )
    return result


def build_config(
    config_path: Path | None = None,
    data_dir: Path | None = None,
    base_url: str | None = None,
    credential_profile: str | None = None,
    model_overrides: dict[str, str] | None = None,
    headless: bool = False,
    auto_close: bool = False,
    update_summary: bool = True,
    update_tasks: bool = True,
    knowledge_reuse: bool = True,
    regenerate_tasks: bool = False,
    env: str | None = None,
    force_unsafe: bool = False,
) -> SparkConfig:
    """Build a ``SparkConfig`` by merging YAML file, env vars, and CLI args.

    Priority: CLI args > YAML file > environment variables > defaults.
    """
    # Determine config file location
    if config_path is None:
        env_config = os.environ.get("SPARK_CONFIG")
        if env_config:
            config_path = _resolve_path(env_config)
        else:
            default_data_dir = _resolve_path(
                data_dir or os.environ.get("SPARK_RUNNER_DATA_DIR", "~/spark_runner")
            )
            config_path = default_data_dir / "config.yaml"

    yaml_data: dict[str, Any] = load_config_from_yaml(config_path)
    general: dict[str, Any] = yaml_data.get("general", {})

    # Data directory
    resolved_data_dir: Path
    if data_dir is not None:
        resolved_data_dir = _resolve_path(data_dir)
    elif os.environ.get("SPARK_RUNNER_DATA_DIR"):
        resolved_data_dir = _resolve_path(os.environ["SPARK_RUNNER_DATA_DIR"])
    elif general.get("data_dir"):
        resolved_data_dir = _resolve_path(general["data_dir"])
    else:
        resolved_data_dir = _resolve_path("~/spark_runner")

    # Base URL
    resolved_base_url: str
    if base_url is not None:
        resolved_base_url = base_url
    elif os.environ.get("SPARK_BASE_URL"):
        resolved_base_url = os.environ["SPARK_BASE_URL"]
    elif general.get("base_url"):
        resolved_base_url = general["base_url"]
    else:
        resolved_base_url = "https://sparky-web-dev.vercel.app"

    # Credentials
    credentials: dict[str, CredentialProfile] = {}
    yaml_creds: dict[str, Any] = yaml_data.get("credentials", {})
    if yaml_creds:
        credentials = _parse_credentials(yaml_creds)

    # Override/supplement with environment variables (backward compat)
    env_email = os.environ.get("USER_EMAIL", "")
    env_password = os.environ.get("USER_PASSWORD", "")
    if env_email or env_password:
        if "default" not in credentials:
            credentials["default"] = CredentialProfile()
        if env_email:
            credentials["default"].email = env_email
        if env_password:
            credentials["default"].password = env_password

    # Models
    models: dict[str, ModelConfig] = {}
    yaml_models: dict[str, Any] = yaml_data.get("models", {})
    if yaml_models:
        models = _parse_models(yaml_models)

    # Apply CLI model overrides (e.g. --model summarization=claude-haiku-35-20241022)
    if model_overrides:
        for purpose, model_id in model_overrides.items():
            if purpose in models:
                models[purpose].model = model_id
            else:
                models[purpose] = ModelConfig(model=model_id)

    # Environments
    environments: dict[str, EnvironmentProfile] = {}
    yaml_envs: dict[str, Any] = yaml_data.get("environments", {})
    if yaml_envs:
        environments = _parse_environments(yaml_envs)

    # Apply environment overrides when --env is specified
    active_environment: str | None = None
    if env is not None:
        if env not in environments:
            raise ValueError(
                f"Unknown environment '{env}'. "
                f"Available: {', '.join(sorted(environments)) or '(none)'}"
            )
        active_environment = env
        env_profile = environments[env]

        # Environment base_url (CLI --base-url takes priority)
        if base_url is None and env_profile.base_url:
            resolved_base_url = env_profile.base_url

        # Environment credentials replace global ones (CLI --credential-profile still takes priority)
        if env_profile.credentials:
            credentials = env_profile.credentials

    config = SparkConfig(
        data_dir=resolved_data_dir,
        tasks_dir=resolved_data_dir / "tasks",
        goal_summaries_dir=resolved_data_dir / "goal_summaries",
        runs_dir=resolved_data_dir / "runs",
        base_url=resolved_base_url,
        credentials=credentials if credentials else {"default": CredentialProfile()},
        active_credential_profile=credential_profile or "default",
        models=models if models else {},  # __post_init__ fills defaults
        environments=environments,
        active_environment=active_environment,
        force_unsafe=force_unsafe,
        update_summary=update_summary,
        update_tasks=update_tasks,
        knowledge_reuse=knowledge_reuse,
        auto_close=auto_close,
        headless=headless,
        regenerate_tasks=regenerate_tasks,
    )
    return config


def set_config_file_permissions(config_path: Path) -> None:
    """Set config file permissions to 0600 when credentials are present."""
    try:
        config_path.chmod(0o600)
    except OSError:
        pass
