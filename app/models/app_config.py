"""Application-wide configuration singleton."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR_NAME = "MusicMasteryEnhancer"


@dataclass(frozen=True)
class DSPDefaults:
    """Default DSP parameter values for the vocal/instrumental Pedalboard chains."""

    high_pass_hz: float = 80.0
    low_pass_hz: float = 14500.0
    notch_freq_hz: float = 4000.0
    notch_gain_db: float = -4.5
    de_esser_low_hz: float = 5000.0
    de_esser_high_hz: float = 8000.0


@dataclass(frozen=True)
class AppConfig:
    """Immutable, process-wide application configuration."""

    cache_root: Path = field(default_factory=lambda: _default_cache_root())
    default_lufs_target: float = -14.0
    dsp_defaults: DSPDefaults = field(default_factory=DSPDefaults)


def _default_cache_root() -> Path:
    local_app_data = os.environ.get("LOCALAPPDATA")
    base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
    return base / APP_DIR_NAME


_instance: AppConfig | None = None


def get_app_config() -> AppConfig:
    """Return the process-wide AppConfig singleton, creating it on first use."""
    global _instance
    if _instance is None:
        _instance = AppConfig()
    return _instance
