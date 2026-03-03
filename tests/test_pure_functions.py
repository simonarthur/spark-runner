"""Tests for pure / near-pure helper functions in sparky_runner."""

from __future__ import annotations

import sparky_runner


# ── _host_to_placeholder / _placeholder_to_host ─────────────────────────

class TestHostPlaceholder:
    def test_replaces_host(self) -> None:
        text = "Go to https://test.example.com/dashboard"
        assert sparky_runner._host_to_placeholder(text) == "Go to {BASE_URL}/dashboard"

    def test_strips_trailing_slash(self) -> None:
        text = "Go to https://test.example.com/"
        result = sparky_runner._host_to_placeholder(text)
        assert "{BASE_URL}" in result

    def test_multiple_occurrences(self) -> None:
        text = "https://test.example.com and https://test.example.com/page"
        result = sparky_runner._host_to_placeholder(text)
        assert result.count("{BASE_URL}") == 2

    def test_placeholder_to_host_roundtrip(self) -> None:
        original = "Navigate to https://test.example.com/settings then check https://test.example.com/profile"
        assert sparky_runner._placeholder_to_host(sparky_runner._host_to_placeholder(original)) == original


# ── _credentials_to_placeholders / _placeholders_to_credentials ─────────

class TestCredentialPlaceholders:
    def test_replaces_email(self) -> None:
        text = "Login as test@example.com"
        assert sparky_runner._credentials_to_placeholders(text) == "Login as {USER_EMAIL}"

    def test_replaces_password(self) -> None:
        text = "Password: test-password-123"
        assert sparky_runner._credentials_to_placeholders(text) == "Password: {USER_PASSWORD}"

    def test_replaces_both(self) -> None:
        text = "email=test@example.com&pw=test-password-123"
        result = sparky_runner._credentials_to_placeholders(text)
        assert "{USER_EMAIL}" in result
        assert "{USER_PASSWORD}" in result

    def test_roundtrip(self) -> None:
        original = "Use test@example.com / test-password-123 to log in"
        assert sparky_runner._placeholders_to_credentials(
            sparky_runner._credentials_to_placeholders(original)
        ) == original

    def test_empty_credentials_noop(self, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.setattr(sparky_runner, "USER_EMAIL", "")
        monkeypatch.setattr(sparky_runner, "USER_PASSWORD", "")
        text = "nothing to replace"
        assert sparky_runner._credentials_to_placeholders(text) == text
        assert sparky_runner._placeholders_to_credentials(text) == text


# ── _sanitize_for_storage / _restore_from_storage ────────────────────────

class TestSanitizeRestore:
    def test_roundtrip_identity(self) -> None:
        original = "Visit https://test.example.com and use test@example.com / test-password-123"
        assert sparky_runner._restore_from_storage(sparky_runner._sanitize_for_storage(original)) == original

    def test_sanitize_replaces_all(self) -> None:
        text = "https://test.example.com test@example.com test-password-123"
        result = sparky_runner._sanitize_for_storage(text)
        assert "test.example.com" not in result
        assert "test@example.com" not in result
        assert "test-password-123" not in result


# ── _observation_text ────────────────────────────────────────────────────

class TestObservationText:
    def test_string_passthrough(self) -> None:
        assert sparky_runner._observation_text("some observation") == "some observation"

    def test_dict_extracts_text(self) -> None:
        obs = {"text": "found a bug", "severity": "error"}
        assert sparky_runner._observation_text(obs) == "found a bug"

    def test_dict_missing_text_key(self) -> None:
        assert sparky_runner._observation_text({"severity": "warning"}) == ""
