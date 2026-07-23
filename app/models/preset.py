"""Preset dataclass covering every adjustable vocal/instrumental/mastering control."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
import jsonschema

PRESET_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Preset",
    "type": "object",
    "properties": {
        "version": { "type": "string", "default": "1.0" },
        "vocal_denoise_enabled": { "type": "boolean" },
        "vocal_denoise_intensity": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "vocal_enhance_enabled": { "type": "boolean" },
        "vocal_enhance_intensity": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "vocal_clean_intensity": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "vocal_gain_db": { "type": "number", "minimum": -24.0, "maximum": 24.0 },
        "instrumental_denoise_enabled": { "type": "boolean" },
        "instrumental_denoise_intensity": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "instrumental_enhance_enabled": { "type": "boolean" },
        "instrumental_enhance_intensity": { "type": "number", "minimum": 0.0, "maximum": 1.0 },
        "instrumental_mud_cut_hz": { "type": "number", "minimum": 20.0, "maximum": 120.0 },
        "instrumental_dehiss_shelf_hz": { "type": "number", "minimum": 6000.0, "maximum": 16000.0 },
        "instrumental_dehiss_gain_db": { "type": "number", "minimum": -6.0, "maximum": 0.0 },
        "instrumental_gain_db": { "type": "number", "minimum": -24.0, "maximum": 24.0 },
        "notch_depth_db": { "type": "number", "minimum": 3.0, "maximum": 6.0 },
        "lufs_target": { "type": "number", "minimum": -30.0, "maximum": -5.0 }
    },
    "required": [
        "version",
        "vocal_denoise_enabled",
        "vocal_denoise_intensity",
        "vocal_enhance_enabled",
        "vocal_enhance_intensity",
        "vocal_clean_intensity",
        "vocal_gain_db",
        "instrumental_denoise_enabled",
        "instrumental_denoise_intensity",
        "instrumental_enhance_enabled",
        "instrumental_enhance_intensity",
        "instrumental_mud_cut_hz",
        "instrumental_dehiss_shelf_hz",
        "instrumental_dehiss_gain_db",
        "instrumental_gain_db",
        "notch_depth_db",
        "lufs_target"
    ],
    "additionalProperties": False
}


def sanitize_preset_dict(raw_data: dict) -> tuple[dict, list[str]]:
    """Sanitize raw data dictionary to ensure backwards compatibility and strict schema conformance.

    Replaces missing or invalid parameter values with their corresponding safe defaults.
    """
    if not isinstance(raw_data, dict):
        raise ValueError("Preset data must be a dictionary")

    sanitized = {}
    warnings = []

    defaults = {
        "version": "1.0",
        "vocal_denoise_enabled": True,
        "vocal_denoise_intensity": 0.5,
        "vocal_enhance_enabled": True,
        "vocal_enhance_intensity": 0.5,
        "vocal_clean_intensity": 1.0,
        "vocal_gain_db": 0.0,
        "instrumental_denoise_enabled": True,
        "instrumental_denoise_intensity": 0.5,
        "instrumental_enhance_enabled": True,
        "instrumental_enhance_intensity": 0.5,
        "instrumental_mud_cut_hz": 40.0,
        "instrumental_dehiss_shelf_hz": 10000.0,
        "instrumental_dehiss_gain_db": -3.0,
        "instrumental_gain_db": 0.0,
        "notch_depth_db": 4.5,
        "lufs_target": -14.0,
    }

    properties = PRESET_SCHEMA["properties"]

    for key, spec in properties.items():
        if key not in raw_data:
            sanitized[key] = defaults[key]
            warnings.append(f"Missing key {key!r}, filled with default {defaults[key]}")
            continue

        val = raw_data[key]
        expected_type = spec["type"]

        # Type checking
        type_ok = False
        if expected_type == "boolean":
            type_ok = isinstance(val, bool)
        elif expected_type == "number":
            # bool is a subclass of int in Python, so check explicitly
            type_ok = isinstance(val, (int, float)) and not isinstance(val, bool)
        elif expected_type == "string":
            type_ok = isinstance(val, str)

        if not type_ok:
            sanitized[key] = defaults[key]
            warnings.append(
                f"Invalid type for key {key!r} (expected {expected_type}, got {type(val).__name__}), filled with default {defaults[key]}"
            )
            continue

        # Boundary checking for numbers
        if expected_type == "number":
            min_val = spec.get("minimum")
            max_val = spec.get("maximum")
            if min_val is not None and val < min_val:
                sanitized[key] = defaults[key]
                warnings.append(
                    f"Value for {key!r} ({val}) is below minimum ({min_val}), filled with default {defaults[key]}"
                )
                continue
            if max_val is not None and val > max_val:
                sanitized[key] = defaults[key]
                warnings.append(
                    f"Value for {key!r} ({val}) is above maximum ({max_val}), filled with default {defaults[key]}"
                )
                continue

        # If all checks pass, keep the value
        sanitized[key] = val

    return sanitized, warnings


@dataclass
class Preset:
    """A named, fully-specified set of pipeline control values.

    Mirrors the granular controls exposed across app.core.vocal_chain, instrumental_chain,
    and remix_master so a preset can fully reproduce a render without extra defaults.
    """

    version: str = "1.0"
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

    def to_dict(self) -> dict:
        """Convert Preset to a dictionary and validate against JSON Schema."""
        d = asdict(self)
        jsonschema.validate(instance=d, schema=PRESET_SCHEMA)
        return d

    def to_json(self) -> str:
        """Convert Preset to a validated JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> Preset:
        """Create a Preset from a dictionary, with schema validation and sanitization/compatibility fallback."""
        sanitized, _ = sanitize_preset_dict(data)
        # Final safety validation check
        jsonschema.validate(instance=sanitized, schema=PRESET_SCHEMA)
        return cls(**sanitized)

    @classmethod
    def from_json(cls, json_str: str) -> Preset:
        """Create a Preset from a JSON string, with schema validation and sanitization/compatibility fallback."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON syntax: {exc}") from exc
        return cls.from_dict(data)
