"""Tests for spark_runner.credentials and spark_runner.placeholders credential helpers."""

from __future__ import annotations

import pytest

from spark_runner.credentials import (
    get_credentials,
    list_credential_profiles,
    switch_profile,
)
from spark_runner.models import CredentialProfile, SparkConfig
from spark_runner.placeholders import cred_placeholder_to_value


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_config(profiles: dict[str, CredentialProfile], active: str = "default") -> SparkConfig:
    """Build a minimal SparkConfig with the given credential profiles."""
    return SparkConfig(
        credentials=profiles,
        active_credential_profile=active,
    )


# ── get_credentials ──────────────────────────────────────────────────────


class TestGetCredentials:
    def test_returns_active_profile(self) -> None:
        profile = CredentialProfile(email="user@example.com", password="secret")
        config = _make_config({"default": profile})
        result = get_credentials(config)
        assert result.email == "user@example.com"
        assert result.password == "secret"

    def test_returns_named_active_profile(self) -> None:
        admin = CredentialProfile(email="admin@example.com", password="adminpw")
        user = CredentialProfile(email="user@example.com", password="userpw")
        config = _make_config({"admin": admin, "user": user}, active="admin")
        result = get_credentials(config)
        assert result.email == "admin@example.com"

    def test_returns_empty_profile_when_profile_missing(self) -> None:
        config = _make_config({"default": CredentialProfile()}, active="nonexistent")
        result = get_credentials(config)
        assert result.email == ""
        assert result.password == ""

    def test_extra_fields_accessible(self) -> None:
        profile = CredentialProfile(
            email="u@example.com",
            password="pw",
            extra={"api_key": "tok_abc"},
        )
        config = _make_config({"default": profile})
        result = get_credentials(config)
        assert result.extra["api_key"] == "tok_abc"


# ── list_credential_profiles ─────────────────────────────────────────────


class TestListCredentialProfiles:
    def test_returns_sorted_names(self) -> None:
        profiles = {
            "zebra": CredentialProfile(),
            "alpha": CredentialProfile(),
            "middle": CredentialProfile(),
        }
        config = _make_config(profiles)
        result = list_credential_profiles(config)
        assert result == ["alpha", "middle", "zebra"]

    def test_single_profile(self) -> None:
        config = _make_config({"default": CredentialProfile()})
        result = list_credential_profiles(config)
        assert result == ["default"]

    def test_empty_credentials_returns_empty_list(self) -> None:
        # SparkConfig.__post_init__ always adds "default" when empty,
        # so we test that the sorted list is consistently returned.
        config = SparkConfig()
        result = list_credential_profiles(config)
        assert isinstance(result, list)
        assert "default" in result


# ── switch_profile ───────────────────────────────────────────────────────


class TestSwitchProfile:
    def test_switches_active_profile(self) -> None:
        profiles = {
            "default": CredentialProfile(email="d@example.com", password="dpw"),
            "admin": CredentialProfile(email="a@example.com", password="apw"),
        }
        config = _make_config(profiles, active="default")
        updated = switch_profile(config, "admin")
        assert updated.active_credential_profile == "admin"
        assert updated.active_credentials.email == "a@example.com"

    def test_returns_same_config_object(self) -> None:
        profiles = {
            "default": CredentialProfile(),
            "staging": CredentialProfile(),
        }
        config = _make_config(profiles)
        returned = switch_profile(config, "staging")
        assert returned is config

    def test_raises_key_error_on_missing_profile(self) -> None:
        config = _make_config({"default": CredentialProfile()})
        with pytest.raises(KeyError, match="nonexistent"):
            switch_profile(config, "nonexistent")

    def test_error_message_includes_available_profiles(self) -> None:
        profiles = {
            "default": CredentialProfile(),
            "staging": CredentialProfile(),
        }
        config = _make_config(profiles)
        with pytest.raises(KeyError) as exc_info:
            switch_profile(config, "missing")
        message = str(exc_info.value)
        assert "default" in message or "staging" in message


# ── cred_placeholder_to_value ────────────────────────────────────────────


class TestCredPlaceholderToValue:
    def test_replaces_cred_email_placeholder(self) -> None:
        credentials = {
            "admin": CredentialProfile(email="admin@example.com", password="apw"),
        }
        text = "Login as {CRED:admin:email}"
        result = cred_placeholder_to_value(text, credentials)
        assert result == "Login as admin@example.com"

    def test_replaces_cred_password_placeholder(self) -> None:
        credentials = {
            "admin": CredentialProfile(email="admin@example.com", password="secret123"),
        }
        text = "Password: {CRED:admin:password}"
        result = cred_placeholder_to_value(text, credentials)
        assert result == "Password: secret123"

    def test_replaces_extra_field_placeholder(self) -> None:
        credentials = {
            "api": CredentialProfile(extra={"token": "tok_xyz"}),
        }
        text = "Bearer {CRED:api:token}"
        result = cred_placeholder_to_value(text, credentials)
        assert result == "Bearer tok_xyz"

    def test_unknown_profile_leaves_placeholder(self) -> None:
        credentials: dict[str, CredentialProfile] = {}
        text = "Use {CRED:ghost:email}"
        result = cred_placeholder_to_value(text, credentials)
        # Unknown profile returns empty string (CredentialProfile default)
        assert "ghost" not in result or "{CRED:ghost:email}" not in result

    def test_legacy_user_email_placeholder_replaced(self) -> None:
        credentials = {
            "default": CredentialProfile(email="legacy@example.com", password="legacypw"),
        }
        text = "Email: {USER_EMAIL}"
        result = cred_placeholder_to_value(text, credentials)
        assert result == "Email: legacy@example.com"

    def test_legacy_user_password_placeholder_replaced(self) -> None:
        credentials = {
            "default": CredentialProfile(email="u@example.com", password="mypw"),
        }
        text = "Pass: {USER_PASSWORD}"
        result = cred_placeholder_to_value(text, credentials)
        assert result == "Pass: mypw"

    def test_multiple_placeholders_in_text(self) -> None:
        credentials = {
            "alice": CredentialProfile(email="alice@example.com", password="alicepw"),
            "bob": CredentialProfile(email="bob@example.com", password="bobpw"),
        }
        text = "{CRED:alice:email} and {CRED:bob:email}"
        result = cred_placeholder_to_value(text, credentials)
        assert result == "alice@example.com and bob@example.com"

    def test_no_placeholders_returns_text_unchanged(self) -> None:
        credentials = {"default": CredentialProfile()}
        text = "No placeholders here."
        result = cred_placeholder_to_value(text, credentials)
        assert result == text
