"""Host and credential placeholder substitution.

All functions take explicit parameters rather than reading module globals,
making them independently testable.
"""

from __future__ import annotations


def host_to_placeholder(text: str, host: str) -> str:
    """Replace the active host URL with ``{BASE_URL}``, normalizing trailing slashes."""
    return text.replace(host.rstrip("/"), "{BASE_URL}")


def placeholder_to_host(text: str, host: str) -> str:
    """Replace ``{BASE_URL}`` with the active host URL, normalizing trailing slashes."""
    return text.replace("{BASE_URL}", host.rstrip("/"))


def credentials_to_placeholders(
    text: str, email: str, password: str
) -> str:
    """Replace literal credentials with ``{USER_EMAIL}``/``{USER_PASSWORD}`` placeholders."""
    if email:
        text = text.replace(email, "{USER_EMAIL}")
    if password:
        text = text.replace(password, "{USER_PASSWORD}")
    return text


def placeholders_to_credentials(
    text: str, email: str, password: str
) -> str:
    """Replace ``{USER_EMAIL}``/``{USER_PASSWORD}`` placeholders with literal credentials."""
    if email:
        text = text.replace("{USER_EMAIL}", email)
    if password:
        text = text.replace("{USER_PASSWORD}", password)
    return text


def cred_placeholder_to_value(
    text: str, credentials: dict[str, "CredentialProfile"]
) -> str:
    """Replace ``{CRED:profile:field}`` placeholders with credential values.

    Also handles the legacy ``{USER_EMAIL}`` and ``{USER_PASSWORD}`` via the
    ``default`` profile.
    """
    from spark_runner.models import CredentialProfile  # avoid circular import

    # Legacy placeholders via default profile
    default = credentials.get("default", CredentialProfile())
    text = placeholders_to_credentials(text, default.email, default.password)

    # New {CRED:profile:field} format
    import re

    def _replace_cred(m: re.Match[str]) -> str:
        profile_name = m.group(1)
        field_name = m.group(2)
        profile = credentials.get(profile_name, CredentialProfile())
        if field_name == "email":
            return profile.email
        elif field_name == "password":
            return profile.password
        else:
            return profile.extra.get(field_name, m.group(0))

    text = re.sub(r"\{CRED:(\w+):(\w+)\}", _replace_cred, text)
    return text


def sanitize_for_storage(
    text: str, host: str, email: str, password: str
) -> str:
    """Replace host URL and credentials with placeholders for safe storage."""
    return credentials_to_placeholders(
        host_to_placeholder(text, host), email, password
    )


def restore_from_storage(
    text: str, host: str, email: str, password: str
) -> str:
    """Restore host URL and credentials from placeholders."""
    return placeholders_to_credentials(
        placeholder_to_host(text, host), email, password
    )
