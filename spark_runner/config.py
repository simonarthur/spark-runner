"""Configuration system: YAML loading, defaults, and environment variable overrides."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import click

from spark_runner.models import CredentialProfile, EnvironmentProfile, ModelConfig, SparkConfig


def _resolve_path(p: str | Path) -> Path:
    """Expand ``~`` and return an absolute Path."""
    return Path(p).expanduser().resolve()


def _resolve_value(value: str) -> str:
    """Resolve environment variable references in a string value.

    Supports ``$VAR`` and ``${VAR}`` syntax. Returns the original string
    if no env var pattern is found or the variable is not set.
    """
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)", value.strip())
    if match:
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, value)
    return value


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
                email=_resolve_value(profile_data.get("email", "")),
                password=_resolve_value(profile_data.get("password", "")),
                extra={
                    k: _resolve_value(v) if isinstance(v, str) else v
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


def resolve_config_path(
    config_path: Path | None = None,
    data_dir: Path | None = None,
) -> Path:
    """Determine the expected config file path without loading it.

    Uses the same resolution logic as ``build_config()`` so that callers can
    check for the file's existence before building the full config.
    """
    if config_path is not None:
        return _resolve_path(config_path)

    env_config = os.environ.get("SPARK_RUNNER_CONFIG")
    if env_config:
        return _resolve_path(env_config)

    default_data_dir = _resolve_path(
        data_dir or os.environ.get("SPARK_RUNNER_DATA_DIR", "~/spark_runner")
    )
    return default_data_dir / "config.yaml"


def _env_var_name(*parts: str) -> str:
    """Build a ``SPARK_RUNNER_…`` environment variable name from parts."""
    suffix = "_".join(p.upper() for p in parts)
    return f"SPARK_RUNNER_{suffix}"


def _build_config_yaml(
    data_dir: str,
    base_url: str,
    email: str,
    password: str,
    extra_credentials: list[dict[str, str]],
    environments: list[dict[str, str]],
    use_env_vars: bool = False,
    anthropic_api_key: str = "",
    use_browseruse_llm: bool = False,
    browseruse_api_key: str = "",
) -> str:
    """Build the config.yaml content string with optional environments."""

    def _secret_value(value: str, env_var: str) -> str:
        """Return an env-var reference or a quoted literal."""
        if use_env_vars and value:
            return f"${env_var}"
        return f'"{value}"'

    lines: list[str] = [
        "general:",
        f"  data_dir: {data_dir}",
        f"  base_url: {base_url}",
        f"  use_browseruse_llm: {str(use_browseruse_llm).lower()}",
        "",
        "api_keys:",
        f'  anthropic: {_secret_value(anthropic_api_key, "ANTHROPIC_API_KEY")}',
        f'  browseruse: {_secret_value(browseruse_api_key, "BROWSER_USE_API_KEY")}',
        "",
        "credentials:",
        "  default:",
        f'    email: {_secret_value(email, _env_var_name("DEFAULT", "EMAIL"))}',
        f'    password: {_secret_value(password, _env_var_name("DEFAULT", "PASSWORD"))}',
    ]

    for cred in extra_credentials:
        cname = cred["name"]
        lines.append(f"  {cname}:")
        lines.append(
            f'    email: {_secret_value(cred.get("email", ""), _env_var_name(cname, "EMAIL"))}'
        )
        lines.append(
            f'    password: {_secret_value(cred.get("password", ""), _env_var_name(cname, "PASSWORD"))}'
        )

    if environments:
        lines.append("")
        lines.append("environments:")
        for env in environments:
            ename = env["name"]
            lines.append(f"  {ename}:")
            lines.append(f"    base_url: {env['base_url']}")
            if env.get("is_production"):
                lines.append("    is_production: true")
            if env.get("email") or env.get("password"):
                lines.append("    credentials:")
                lines.append("      default:")
                lines.append(
                    f'        email: {_secret_value(env.get("email", ""), _env_var_name(ename, "DEFAULT", "EMAIL"))}'
                )
                lines.append(
                    f'        password: {_secret_value(env.get("password", ""), _env_var_name(ename, "DEFAULT", "PASSWORD"))}'
                )
    else:
        lines.extend([
            "",
            "# Uncomment and configure environments for multi-environment support:",
            "# environments:",
            "#   staging:",
            "#     base_url: https://staging.example.com",
            "#     credentials:",
            "#       default:",
            "#         email: staging@example.com",
            "#         password: stagingpass",
            "#   production:",
            "#     base_url: https://app.example.com",
            "#     is_production: true",
            "#     credentials:",
            "#       default:",
            "#         email: prod@example.com",
            "#         password: prodpass",
        ])

    lines.append("")
    return "\n".join(lines)


def _prompt_environment() -> dict[str, str]:
    """Prompt the user for a single environment's settings."""
    name = click.prompt("  Environment name (e.g. staging, production)", type=str)
    env_base_url = click.prompt(f"  Base URL for '{name}'", type=str)
    is_prod_str = "true" if click.confirm(
        f"  Is '{name}' a production environment?", default=False,
    ) else ""
    env_email = click.prompt(f"  Login username for '{name}'", default="", type=str)
    env_password = click.prompt(
        f"  Password for '{name}'", default="", hide_input=True, type=str,
    )
    return {
        "name": name,
        "base_url": env_base_url,
        "is_production": is_prod_str,
        "email": env_email,
        "password": env_password,
    }


def run_setup_wizard(config_path: Path) -> Path:
    """Interactively prompt for essential settings and write ``config.yaml``.

    Creates the data directory (and standard subdirectories) if needed, writes
    the config file, and restricts its permissions when credentials are provided.

    Returns the path to the newly created config file.
    """
    click.echo("Spark Runner – First-time setup\n")

    data_dir = click.prompt(
        "Data directory: this is where your Spark Runner data and configuration will be stored.",
        default="~/spark_runner",
        type=str,
    )
    base_url = click.prompt(
        "Base URL",
        default="https://www.example.com",
        type=str,
    )
    email = click.prompt("Default login username for the test website", default="", type=str)
    password = click.prompt(
        "Default password for the test website", default="", hide_input=True, type=str,
    )

    # Optional additional credential profiles
    extra_credentials: list[dict[str, str]] = []
    if click.confirm("\nSet up additional logins?", default=False):
        while True:
            cred_name = click.prompt("  Profile name (e.g. admin, viewer)", type=str)
            cred_email = click.prompt(f"  Username for '{cred_name}'", default="", type=str)
            cred_password = click.prompt(
                f"  Password for '{cred_name}'", default="", hide_input=True, type=str,
            )
            extra_credentials.append({
                "name": cred_name,
                "email": cred_email,
                "password": cred_password,
            })
            if not click.confirm("\n  Add another login?", default=False):
                break

    # API keys
    click.echo("\n── API Keys ──")
    anthropic_api_key = click.prompt(
        "Anthropic API key", default="", type=str,
    )
    use_browseruse_llm = click.confirm(
        "Use BrowserUse cloud LLM instead of direct Anthropic?", default=False,
    )
    browseruse_api_key = ""
    if use_browseruse_llm:
        browseruse_api_key = click.prompt(
            "BrowserUse API key", default="", type=str,
        )

    # Offer env-var storage when any secrets were provided
    has_any_secret = (
        bool(password)
        or any(c.get("password") for c in extra_credentials)
        or bool(anthropic_api_key)
        or bool(browseruse_api_key)
    )
    use_env_vars = False
    if has_any_secret:
        use_env_vars = click.confirm(
            "\nStore secrets as environment variable references instead of plaintext?",
            default=True,
        )

    # Optional multi-environment setup
    environments: list[dict[str, str]] = []
    if click.confirm(
        "\nDo you want to set up more environments "
        "(development, staging, production...)?",
        default=False,
    ):
        while True:
            click.echo()
            environments.append(_prompt_environment())
            if not click.confirm("\n  Add another environment?", default=False):
                break

    # Create data directory and subdirectories
    resolved_data_dir = _resolve_path(data_dir)
    for subdir in ("tasks", "goal_summaries", "runs"):
        (resolved_data_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Place config.yaml inside the chosen data directory
    config_path = resolved_data_dir / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_content = _build_config_yaml(
        data_dir=data_dir,
        base_url=base_url,
        email=email,
        password=password,
        extra_credentials=extra_credentials,
        environments=environments,
        use_env_vars=use_env_vars,
        anthropic_api_key=anthropic_api_key,
        use_browseruse_llm=use_browseruse_llm,
        browseruse_api_key=browseruse_api_key,
    )
    config_path.write_text(config_content)

    # Restrict permissions when plaintext secrets are present
    has_plaintext_secrets = False
    if not use_env_vars:
        has_plaintext_secrets = bool(
            email or password or anthropic_api_key or browseruse_api_key
        )
        if not has_plaintext_secrets:
            has_plaintext_secrets = any(
                c.get("email") or c.get("password") for c in extra_credentials
            )
        if not has_plaintext_secrets:
            has_plaintext_secrets = any(
                env.get("email") or env.get("password") for env in environments
            )
    if has_plaintext_secrets:
        set_config_file_permissions(config_path)

    click.echo(f"\nConfig written to {config_path}")

    # Print export hints when using env-var mode
    if use_env_vars:
        click.echo(
            "\nAdd these exports to your shell profile "
            "(e.g. ~/.bashrc, ~/.zshrc, or .env):"
        )
        if anthropic_api_key:
            click.echo(f'  export ANTHROPIC_API_KEY="{anthropic_api_key}"')
        if browseruse_api_key:
            click.echo(f'  export BROWSER_USE_API_KEY="{browseruse_api_key}"')
        if email:
            click.echo(f'  export {_env_var_name("DEFAULT", "EMAIL")}="{email}"')
        if password:
            click.echo(f'  export {_env_var_name("DEFAULT", "PASSWORD")}="{password}"')
        for cred in extra_credentials:
            cname = cred["name"]
            if cred.get("email"):
                click.echo(f'  export {_env_var_name(cname, "EMAIL")}="{cred["email"]}"')
            if cred.get("password"):
                click.echo(f'  export {_env_var_name(cname, "PASSWORD")}="{cred["password"]}"')
        for env in environments:
            ename = env["name"]
            if env.get("email"):
                click.echo(f'  export {_env_var_name(ename, "DEFAULT", "EMAIL")}="{env["email"]}"')
            if env.get("password"):
                click.echo(f'  export {_env_var_name(ename, "DEFAULT", "PASSWORD")}="{env["password"]}"')

    # If the user chose a non-default data directory, remind them to set the
    # environment variable so spark-runner can find the config.
    if data_dir != "~/spark_runner":
        click.echo(
            f"\nYou chose a custom data directory. "
            f"Add this to your shell profile so spark-runner can find it:"
        )
        click.echo(f"\n  export SPARK_RUNNER_DATA_DIR=\"{resolved_data_dir}\"")

    click.echo("\nSetup complete! Try running your first task with:")
    click.echo(f"\n  spark-runner run -p \"Log in and verify the dashboard loads\"")
    click.echo()
    return config_path


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
        env_config = os.environ.get("SPARK_RUNNER_CONFIG")
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
    elif os.environ.get("SPARK_RUNNER_BASE_URL"):
        resolved_base_url = os.environ["SPARK_RUNNER_BASE_URL"]
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

    # API keys
    yaml_api_keys: dict[str, Any] = yaml_data.get("api_keys", {})
    anthropic_api_key = _resolve_value(yaml_api_keys.get("anthropic", ""))
    browseruse_api_key = _resolve_value(yaml_api_keys.get("browseruse", ""))
    use_browseruse_llm = bool(general.get("use_browseruse_llm", False))

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
        anthropic_api_key=anthropic_api_key,
        use_browseruse_llm=use_browseruse_llm,
        browseruse_api_key=browseruse_api_key,
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
