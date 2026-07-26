"""Tests for app.models.settings.Settings, specifically custom_reference_override_path: a
global user preference (not per-track/per-preset) that must stay decoupled from Preset's
schema-validated, persisted fields."""

from __future__ import annotations

from pathlib import Path

from app.models.preset import Preset
from app.models.settings import Settings


def test_custom_reference_override_path_defaults_to_none():
    settings = Settings()

    assert settings.custom_reference_override_path is None


def test_custom_reference_override_path_is_settable():
    override = Path("/some/custom/reference/dir")

    settings = Settings(custom_reference_override_path=override)

    assert settings.custom_reference_override_path == override


def test_from_preset_does_not_populate_override_path():
    """custom_reference_override_path is a global preference, not part of Preset, so building
    Settings from a Preset must not invent a value for it -- it should stay at its default."""
    preset = Preset(vocal_gain_db=3.0)

    settings = Settings.from_preset(preset)

    assert settings.custom_reference_override_path is None
    assert settings.vocal_gain_db == 3.0


def test_to_preset_does_not_leak_override_path():
    """The override path must never end up in the persisted/schema-validated Preset -- round
    tripping through to_preset()/to_dict() should neither fail nor silently include it."""
    settings = Settings(custom_reference_override_path=Path("/some/custom/dir"))

    preset = settings.to_preset()

    assert not hasattr(preset, "custom_reference_override_path")
    preset.to_dict()  # must still validate cleanly against PRESET_SCHEMA
