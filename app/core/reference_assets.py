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


def _stem_path(directory: Path, key: str) -> Path:
    return directory / f"{key}{REFERENCE_STEM_FILENAME_SUFFIX}"


def get_reference_stems(override_dir: Optional[Path] = None) -> dict[str, Optional[Path]]:
    """Return the 4 expected reference stem paths, keyed by REFERENCE_STEM_KEYS.

    Resolves against `override_dir` if given (pass
    Settings.custom_reference_override_path), else the bundled FACTORY_REFERENCES_DIR --
    the override takes precedence over the bundled factory files, per Counsel's spec. A key's
    value is its Path if the file exists on disk, else None, so callers can skip/disable that
    slot in the A/B UI instead of failing the whole comparison.
    """
    directory = override_dir if override_dir is not None else FACTORY_REFERENCES_DIR
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
