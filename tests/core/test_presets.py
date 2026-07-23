"""Tests for Preset persistence, listing, loading, and deletion."""

from pathlib import Path
import pytest
import jsonschema

from app.cache.cache_manager import CacheManager
from app.core.presets import (
    PresetNotFoundError,
    delete_preset,
    list_presets,
    load_preset,
    save_preset,
)
from app.models.app_config import AppConfig
from app.models.preset import Preset, PRESET_SCHEMA


@pytest.fixture
def cache_mgr(tmp_path: Path) -> CacheManager:
    return CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))


def test_preset_lifecycle(cache_mgr: CacheManager):
    p = Preset(
        vocal_denoise_enabled=True,
        vocal_denoise_intensity=0.8,
        notch_depth_db=4.5,
        vocal_gain_db=1.0,
        instrumental_gain_db=-1.5,
    )

    save_preset("my_preset", p, cache_mgr)

    names = list_presets(cache_mgr)
    assert "my_preset" in names

    loaded = load_preset("my_preset", cache_mgr)
    assert loaded.version == "1.0"
    assert loaded.vocal_denoise_enabled is True
    assert loaded.vocal_denoise_intensity == 0.8
    assert loaded.notch_depth_db == 4.5

    delete_preset("my_preset", cache_mgr)
    assert "my_preset" not in list_presets(cache_mgr)


def test_load_nonexistent_preset_raises_error(cache_mgr: CacheManager):
    with pytest.raises(PresetNotFoundError):
        load_preset("non_existent", cache_mgr)


def test_delete_nonexistent_preset_raises_error(cache_mgr: CacheManager):
    with pytest.raises(PresetNotFoundError):
        delete_preset("non_existent", cache_mgr)


def test_preset_serialization_deserialization():
    p = Preset(
        version="1.0",
        vocal_denoise_enabled=False,
        vocal_denoise_intensity=0.2,
        notch_depth_db=5.0,
    )
    d = p.to_dict()
    assert d["version"] == "1.0"
    assert d["vocal_denoise_enabled"] is False
    assert d["vocal_denoise_intensity"] == 0.2
    assert d["notch_depth_db"] == 5.0

    js_str = p.to_json()
    p2 = Preset.from_json(js_str)
    assert p2.version == "1.0"
    assert p2.vocal_denoise_enabled is False
    assert p2.vocal_denoise_intensity == 0.2
    assert p2.notch_depth_db == 5.0


def test_preset_backwards_compatibility_missing_fields():
    # Only provides a couple of fields, missing everything else (including version)
    raw = {
        "vocal_denoise_intensity": 0.7,
        "notch_depth_db": 4.2
    }
    # This should parse successfully, filling in defaults
    p = Preset.from_dict(raw)
    assert p.version == "1.0"  # Filled safe default version
    assert p.vocal_denoise_intensity == 0.7  # Kept
    assert p.notch_depth_db == 4.2  # Kept
    assert p.vocal_denoise_enabled is True  # Default filled
    assert p.lufs_target == -14.0  # Default filled


def test_preset_boundary_and_type_validation_sanitization():
    # Invalid types or out of bounds parameters
    raw = {
        "version": "2.0",
        "vocal_denoise_enabled": "not_a_bool",  # Invalid type, should be defaulted to True
        "vocal_denoise_intensity": 1.5,         # Out of bounds (>1.0), should be defaulted to 0.5
        "vocal_enhance_intensity": -0.1,        # Out of bounds (<0.0), should be defaulted to 0.5
        "vocal_gain_db": 50.0,                  # Out of bounds (>24.0), should be defaulted to 0.0
        "instrumental_mud_cut_hz": "high",      # Invalid type, should be defaulted to 40.0
        "notch_depth_db": 2.0,                  # Out of bounds (<3.0), should be defaulted to 4.5
        "lufs_target": -14.0                    # Valid
    }
    p = Preset.from_dict(raw)
    assert p.version == "2.0"
    assert p.vocal_denoise_enabled is True      # Default
    assert p.vocal_denoise_intensity == 0.5     # Default
    assert p.vocal_enhance_intensity == 0.5     # Default
    assert p.vocal_gain_db == 0.0               # Default
    assert p.instrumental_mud_cut_hz == 40.0    # Default
    assert p.notch_depth_db == 4.5              # Default
    assert p.lufs_target == -14.0               # Kept


def test_preset_malformed_json_raises_value_error():
    with pytest.raises(ValueError, match="Invalid JSON syntax"):
        Preset.from_json("{invalid_json}")


def test_preset_invalid_types_non_dict_raises_value_error():
    with pytest.raises(ValueError, match="Preset data must be a dictionary"):
        Preset.from_dict(["not", "a", "dict"])


def test_schema_is_valid():
    # Verify that the schema is a valid JSON schema dictionary
    assert isinstance(PRESET_SCHEMA, dict)
    assert PRESET_SCHEMA["type"] == "object"
    assert "properties" in PRESET_SCHEMA
