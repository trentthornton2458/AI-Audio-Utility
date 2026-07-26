"""Storage for the user's Gemini API key, used by app.core.gemini_qa's stem-analysis QA
checkpoint.

The key is kept in the OS credential store via `keyring` (Windows Credential Manager /
macOS Keychain / Secret Service on Linux) rather than a plaintext config file, since it is
the one credential this otherwise-local app handles.
"""

from __future__ import annotations

from typing import Optional

import keyring
import keyring.errors

from app.cache import get_logger

logger = get_logger(__name__)

SERVICE_NAME = "MusicMasteryEnhancer"
USERNAME = "gemini_api_key"


def get_gemini_api_key() -> Optional[str]:
    """Return the stored Gemini API key, or None if none has been saved."""
    try:
        key = keyring.get_password(SERVICE_NAME, USERNAME)
    except keyring.errors.KeyringError as exc:
        logger.warning("Failed to read Gemini API key from OS credential store: %s", exc)
        return None
    return key or None


def set_gemini_api_key(api_key: str) -> None:
    """Save the Gemini API key to the OS credential store, replacing any existing value."""
    keyring.set_password(SERVICE_NAME, USERNAME, api_key)
    logger.info("Saved Gemini API key to OS credential store")


def clear_gemini_api_key() -> None:
    """Remove the stored Gemini API key, if any."""
    try:
        keyring.delete_password(SERVICE_NAME, USERNAME)
        logger.info("Cleared stored Gemini API key")
    except keyring.errors.PasswordDeleteError:
        pass


def has_gemini_api_key() -> bool:
    """Return True if a non-empty Gemini API key is currently stored."""
    return bool(get_gemini_api_key())
