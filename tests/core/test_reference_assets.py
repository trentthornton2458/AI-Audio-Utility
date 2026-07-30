"""Tests for app.core.reference_assets (bundled factory A/B reference stem lookup + the
custom global override precedence rule from Counsel's Milestone 4 spec)."""

from __future__ import annotations

from pathlib import Path

from app.core import reference_assets


def _touch_stem(directory: Path, key: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.wav"
    path.write_bytes(b"")
    return path


def _touch_all_stems(directory: Path) -> None:
    for key in reference_assets.REFERENCE_STEM_KEYS:
        _touch_stem(directory, key)


def test_is_reference_assets_missing_true_when_directory_empty(tmp_path: Path):
    assert reference_assets.is_reference_assets_missing(tmp_path) is True


def test_is_reference_assets_missing_true_when_partially_populated(tmp_path: Path):
    _touch_stem(tmp_path, reference_assets.REFERENCE_STEM_KEYS[0])

    assert reference_assets.is_reference_assets_missing(tmp_path) is True


def test_is_reference_assets_missing_false_when_all_four_present(tmp_path: Path):
    _touch_all_stems(tmp_path)

    assert reference_assets.is_reference_assets_missing(tmp_path) is False


def test_is_reference_assets_missing_true_when_directory_does_not_exist(tmp_path: Path):
    missing_dir = tmp_path / "does_not_exist"

    assert reference_assets.is_reference_assets_missing(missing_dir) is True


def test_get_reference_stems_returns_none_for_absent_files(tmp_path: Path):
    stems = reference_assets.get_reference_stems(tmp_path)

    assert set(stems.keys()) == set(reference_assets.REFERENCE_STEM_KEYS)
    assert all(path is None for path in stems.values())


def test_get_reference_stems_returns_paths_for_present_files(tmp_path: Path):
    _touch_all_stems(tmp_path)

    stems = reference_assets.get_reference_stems(tmp_path)

    for key in reference_assets.REFERENCE_STEM_KEYS:
        assert stems[key] == tmp_path / f"{key}.wav"


def test_get_reference_stems_mixed_presence(tmp_path: Path):
    present_key = reference_assets.REFERENCE_STEM_KEYS[0]
    _touch_stem(tmp_path, present_key)

    stems = reference_assets.get_reference_stems(tmp_path)

    assert stems[present_key] == tmp_path / f"{present_key}.wav"
    for key in reference_assets.REFERENCE_STEM_KEYS[1:]:
        assert stems[key] is None


def test_override_directory_takes_precedence_over_factory_directory(
    tmp_path: Path, monkeypatch
):
    factory_dir = tmp_path / "factory"
    override_dir = tmp_path / "override"
    _touch_all_stems(factory_dir)
    _touch_all_stems(override_dir)
    monkeypatch.setattr(reference_assets, "FACTORY_REFERENCES_DIR", factory_dir)

    stems = reference_assets.get_reference_stems(override_dir)

    for key in reference_assets.REFERENCE_STEM_KEYS:
        assert stems[key] == override_dir / f"{key}.wav"
        assert stems[key] != factory_dir / f"{key}.wav"


def test_no_override_falls_back_to_factory_directory(tmp_path: Path, monkeypatch):
    factory_dir = tmp_path / "factory"
    _touch_all_stems(factory_dir)
    monkeypatch.setattr(reference_assets, "FACTORY_REFERENCES_DIR", factory_dir)

    stems = reference_assets.get_reference_stems(None)

    for key in reference_assets.REFERENCE_STEM_KEYS:
        assert stems[key] == factory_dir / f"{key}.wav"


def test_override_only_partially_populated_does_not_fall_back_to_factory(
    tmp_path: Path, monkeypatch
):
    """The override, once set, fully replaces the factory directory -- it is not merged with
    it, so a file missing from the override stays missing even if the factory copy exists.
    """
    factory_dir = tmp_path / "factory"
    override_dir = tmp_path / "override"
    _touch_all_stems(factory_dir)
    present_key = reference_assets.REFERENCE_STEM_KEYS[0]
    _touch_stem(override_dir, present_key)
    monkeypatch.setattr(reference_assets, "FACTORY_REFERENCES_DIR", factory_dir)

    stems = reference_assets.get_reference_stems(override_dir)

    assert stems[present_key] == override_dir / f"{present_key}.wav"
    for key in reference_assets.REFERENCE_STEM_KEYS[1:]:
        assert stems[key] is None
