"""Tests for app.models.gemini_settings (OS-credential-store-backed Gemini API key storage)."""

from __future__ import annotations

from unittest.mock import patch

import keyring.errors

from app.models import gemini_settings


def test_get_gemini_api_key_returns_stored_value():
    with patch("app.models.gemini_settings.keyring.get_password", return_value="my-secret-key") as mock_get:
        result = gemini_settings.get_gemini_api_key()

    assert result == "my-secret-key"
    mock_get.assert_called_once_with(gemini_settings.SERVICE_NAME, gemini_settings.USERNAME)


def test_get_gemini_api_key_returns_none_when_unset():
    with patch("app.models.gemini_settings.keyring.get_password", return_value=None):
        assert gemini_settings.get_gemini_api_key() is None


def test_get_gemini_api_key_returns_none_on_keyring_error():
    with patch(
        "app.models.gemini_settings.keyring.get_password",
        side_effect=keyring.errors.KeyringError("no backend available"),
    ):
        assert gemini_settings.get_gemini_api_key() is None


def test_set_gemini_api_key_saves_to_credential_store():
    with patch("app.models.gemini_settings.keyring.set_password") as mock_set:
        gemini_settings.set_gemini_api_key("new-key")

    mock_set.assert_called_once_with(gemini_settings.SERVICE_NAME, gemini_settings.USERNAME, "new-key")


def test_clear_gemini_api_key_deletes_from_credential_store():
    with patch("app.models.gemini_settings.keyring.delete_password") as mock_delete:
        gemini_settings.clear_gemini_api_key()

    mock_delete.assert_called_once_with(gemini_settings.SERVICE_NAME, gemini_settings.USERNAME)


def test_clear_gemini_api_key_ignores_missing_password():
    with patch(
        "app.models.gemini_settings.keyring.delete_password",
        side_effect=keyring.errors.PasswordDeleteError("not found"),
    ):
        gemini_settings.clear_gemini_api_key()  # should not raise


def test_has_gemini_api_key_true_when_present():
    with patch("app.models.gemini_settings.keyring.get_password", return_value="a-key"):
        assert gemini_settings.has_gemini_api_key() is True


def test_has_gemini_api_key_false_when_absent():
    with patch("app.models.gemini_settings.keyring.get_password", return_value=None):
        assert gemini_settings.has_gemini_api_key() is False
