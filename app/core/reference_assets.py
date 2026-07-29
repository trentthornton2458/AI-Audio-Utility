"""Factory reference vocal stems for the blind Raw / Humanized / Reference A/B comparison
(app/ui/ab_compare_view.py).

Bundled owned/royalty-free reference stems live under /assets/factory_references/ and are
committed directly to the repo rather than downloaded or user-uploaded per track, per Counsel's
Milestone 4 spec. Four files cover male/female performers in dry (unprocessed) and tuned
(professionally mixed) variants, giving the A/B toggle a fixed "what a human/produced vocal
sounds like" anchor independent of whatever track the user is processing. A single optional
global override directory (app.models.settings.Settings.custom_reference_override_path) may
replace this bundled set wholesale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSettings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FACTORY_REFERENCES_DIR = PROJECT_ROOT / "assets" / "factory_references"

# male/female x dry/tuned: the 4 bundled reference stems Milestone 4 requires.
REFERENCE_STEM_KEYS: tuple[str, ...] = (
    "male_dry",
    "male_tuned",
    "female_dry",
    "female_tuned",
)

REFERENCE_STEM_FILENAME_SUFFIX = ".wav"

ORGANIZATION_NAME = "MusicMasteryEnhancer"
APPLICATION_NAME = "MusicMasteryEnhancer"
SETTINGS_KEY_CUSTOM_OVERRIDE = "custom_reference_override_path"
SETTINGS_KEY_MODAL_SEEN = "missing_reference_modal_seen"


def get_custom_reference_override_path() -> Optional[Path]:
    """Get stored custom reference override path from QSettings if set and valid, else None."""
    settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)
    val = settings.value(SETTINGS_KEY_CUSTOM_OVERRIDE, type=str)
    if val:
        path = Path(val)
        if path.is_dir():
            return path
    return None


def set_custom_reference_override_path(override_dir: Optional[Path]) -> None:
    """Save custom reference override path to QSettings."""
    settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)
    if override_dir is not None:
        settings.setValue(SETTINGS_KEY_CUSTOM_OVERRIDE, str(override_dir))
    else:
        settings.remove(SETTINGS_KEY_CUSTOM_OVERRIDE)


def has_seen_missing_reference_modal() -> bool:
    """Return True if the user has already seen/dismissed the missing reference modal."""
    settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)
    return settings.value(SETTINGS_KEY_MODAL_SEEN, False, type=bool)


def set_seen_missing_reference_modal(seen: bool = True) -> None:
    """Save whether the user has seen/dismissed the missing reference modal."""
    settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)
    settings.setValue(SETTINGS_KEY_MODAL_SEEN, seen)


def reset_missing_reference_modal_seen() -> None:
    """Reset the missing reference modal seen flag in QSettings (useful for testing/resets)."""
    settings = QSettings(ORGANIZATION_NAME, APPLICATION_NAME)
    settings.remove(SETTINGS_KEY_MODAL_SEEN)


def _stem_path(directory: Path, key: str) -> Path:
    return directory / f"{key}{REFERENCE_STEM_FILENAME_SUFFIX}"


def get_reference_stems(
    override_dir: Optional[Path] = None,
) -> dict[str, Optional[Path]]:
    """Return the 4 expected reference stem paths, keyed by REFERENCE_STEM_KEYS.

    Resolves against `override_dir` if given (pass
    Settings.custom_reference_override_path), else checks saved persistent custom override,
    and falls back to the bundled FACTORY_REFERENCES_DIR -- the override takes precedence over
    the bundled factory files, per Counsel's spec. A key's value is its Path if the file exists
    on disk, else None, so callers can skip/disable that slot in the A/B UI instead of failing
    the whole comparison.
    """
    effective_dir = (
        override_dir
        if override_dir is not None
        else get_custom_reference_override_path()
    )
    directory = effective_dir if effective_dir is not None else FACTORY_REFERENCES_DIR
    stems: dict[str, Optional[Path]] = {}
    for key in REFERENCE_STEM_KEYS:
        path = _stem_path(directory, key)
        stems[key] = path if path.is_file() else None
    return stems


def is_reference_assets_missing(directory: Optional[Path] = None) -> bool:
    """True if any of the 4 bundled factory reference stems are absent.

    Drives the one-time first-run fallback modal prompting the user to supply reference files.
    Always evaluates the bundled factory directory (or `directory` when given, e.g. from tests)
    regardless of any custom override -- the override is a separate, optional preference, and
    its presence doesn't change whether the bundled set itself is complete.
    """
    target_dir = directory if directory is not None else FACTORY_REFERENCES_DIR
    return any(not _stem_path(target_dir, key).is_file() for key in REFERENCE_STEM_KEYS)


def select_reference_stem(
    vocal_metadata: Optional[dict] = None,
    override_dir: Optional[Path] = None,
) -> tuple[Optional[str], Optional[Path]]:
    """Select the best matching reference stem key and path based on vocal metadata.

    Returns (key, path) tuple if a valid stem file is found, else (None, None).
    """
    stems = get_reference_stems(override_dir=override_dir)
    if not any(stems.values()):
        return None, None

    if vocal_metadata:
        gender = (
            str(
                vocal_metadata.get("gender") or vocal_metadata.get("vocal_gender") or ""
            )
            .lower()
            .strip()
        )
        v_type = (
            str(vocal_metadata.get("type") or vocal_metadata.get("vocal_type") or "")
            .lower()
            .strip()
        )

        candidates: list[str] = []
        if gender and v_type:
            candidates.append(f"{gender}_{v_type}")
        if gender:
            candidates.extend([f"{gender}_tuned", f"{gender}_dry"])
            for key in stems:
                if key.startswith(gender) and key not in candidates:
                    candidates.append(key)

        for key in candidates:
            if stems.get(key) is not None:
                return key, stems[key]

    # Sensible default selection order
    default_order = ("female_tuned", "male_tuned", "female_dry", "male_dry")
    for key in default_order:
        if stems.get(key) is not None:
            return key, stems[key]

    # Fallback to any available stem
    for key, path in stems.items():
        if path is not None:
            return key, path

    return None, None
