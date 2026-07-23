"""Preset dataclass covering every adjustable vocal/instrumental/mastering control."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Preset:
    """A named, fully-specified set of pipeline control values.

    Mirrors the granular controls exposed across app.core.vocal_chain, instrumental_chain,
    and remix_master so a preset can fully reproduce a render without extra defaults.
    """

    vocal_denoise_enabled: bool = True
    vocal_denoise_intensity: float = 0.5
    vocal_enhance_enabled: bool = True
    vocal_enhance_intensity: float = 0.5
    vocal_clean_intensity: float = 1.0
    vocal_gain_db: float = 0.0

    instrumental_denoise_enabled: bool = True
    instrumental_denoise_intensity: float = 0.5
    instrumental_enhance_enabled: bool = True
    instrumental_enhance_intensity: float = 0.5
    instrumental_mud_cut_hz: float = 40.0
    instrumental_dehiss_shelf_hz: float = 10000.0
    instrumental_dehiss_gain_db: float = -3.0
    instrumental_gain_db: float = 0.0

    notch_depth_db: float = 4.5
    lufs_target: float = -14.0
