"""Multi-profile credential management."""

from __future__ import annotations

from sparky_runner.models import CredentialProfile, SparkyConfig


def get_credentials(config: SparkyConfig) -> CredentialProfile:
    """Return the active credential profile from the config.

    Args:
        config: The current configuration.

    Returns:
        The active ``CredentialProfile``.
    """
    return config.active_credentials


def list_credential_profiles(config: SparkyConfig) -> list[str]:
    """Return the names of all configured credential profiles.

    Args:
        config: The current configuration.

    Returns:
        A sorted list of profile names.
    """
    return sorted(config.credentials.keys())


def switch_profile(config: SparkyConfig, profile_name: str) -> SparkyConfig:
    """Return a new config with the active profile switched.

    Args:
        config: The current configuration.
        profile_name: Name of the profile to switch to.

    Returns:
        A new ``SparkyConfig`` with the updated active profile.

    Raises:
        KeyError: If the profile name doesn't exist.
    """
    if profile_name not in config.credentials:
        raise KeyError(
            f"Profile '{profile_name}' not found. "
            f"Available: {', '.join(sorted(config.credentials.keys()))}"
        )
    config.active_credential_profile = profile_name
    return config
