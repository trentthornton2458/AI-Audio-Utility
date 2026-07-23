"""Save/load/list/delete named Presets as JSON files under the cache's presets folder."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.cache import get_logger
from app.cache.cache_manager import CacheManager
from app.models.preset import Preset

logger = get_logger(__name__)

PRESET_FILE_SUFFIX = ".json"


class PresetNotFoundError(Exception):
    """Raised when a named preset does not exist in the cache's presets folder."""


def save_preset(name: str, preset: Preset, cache_manager: CacheManager) -> None:
    """Serialize preset to cache/presets/<name>.json, overwriting any existing file."""
    path = _preset_path(name, cache_manager)
    path.write_text(json.dumps(asdict(preset), indent=2), encoding="utf-8")
    logger.info("Saved preset %r -> %s", name, path)


def load_preset(name: str, cache_manager: CacheManager) -> Preset:
    """Load and deserialize the named preset, raising PresetNotFoundError if it doesn't exist."""
    path = _preset_path(name, cache_manager)
    if not path.is_file():
        raise PresetNotFoundError(f"Preset not found: {name}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return Preset(**data)


def list_presets(cache_manager: CacheManager) -> list[str]:
    """Return the names of all saved presets, sorted alphabetically."""
    return sorted(path.stem for path in cache_manager.presets_dir.glob(f"*{PRESET_FILE_SUFFIX}"))


def delete_preset(name: str, cache_manager: CacheManager) -> None:
    """Delete the named preset, raising PresetNotFoundError if it doesn't exist."""
    path = _preset_path(name, cache_manager)
    if not path.is_file():
        raise PresetNotFoundError(f"Preset not found: {name}")
    path.unlink()
    logger.info("Deleted preset %r", name)


def _preset_path(name: str, cache_manager: CacheManager) -> Path:
    return cache_manager.presets_dir / f"{name}{PRESET_FILE_SUFFIX}"
