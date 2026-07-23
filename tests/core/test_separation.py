"""Tests for app.core.separation.

Exercises the stem-naming/renaming and caching logic without loading the real BS-RoFormer
model by substituting a fake Separator that writes default-named output files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.core import separation


class _FakeSeparator:
    """Stand-in for audio_separator's Separator that mimics its default output naming.

    On separate(), writes files named like the real library ("<base>_(Vocals)_<model>.wav"
    and "<base>_(Instrumental)_<model>.wav") into output_dir and returns their bare filenames.
    """

    def __init__(self, output_dir: str, model_file_dir: str, *args, **kwargs) -> None:
        self._output_dir = Path(output_dir)

    def load_model(self, model_filename: str) -> None:  # noqa: D401 - test stub
        self._model = model_filename

    def separate(self, audio_file_path: str):
        base = Path(audio_file_path).stem
        names = [
            f"{base}_(Instrumental)_model_bs_roformer.wav",
            f"{base}_(Vocals)_model_bs_roformer.wav",
        ]
        # Generate minimal valid WAV content (0.01 seconds, stereo, 44100Hz)
        data = np.zeros((441, 2))
        for name in names:
            sf.write(str(self._output_dir / name), data, 44100, subtype="PCM_24")
        return names


@pytest.fixture()
def cache_manager(tmp_path, monkeypatch):
    from app.cache.cache_manager import CacheManager
    from app.models.app_config import AppConfig

    config = AppConfig(cache_root=tmp_path / "cache")
    return CacheManager(config=config)


def _make_normalized(cache_manager, track_id: str = "trk") -> Path:
    norm = cache_manager.track_dir(track_id) / "normalized.wav"
    # Generate minimal valid WAV content (0.01 seconds, stereo, 44100Hz)
    data = np.zeros((441, 2))
    sf.write(str(norm), data, 44100, subtype="PCM_24")
    return norm


def test_separate_stems_renames_default_outputs_to_canonical(cache_manager, monkeypatch):
    monkeypatch.setattr(separation, "Separator", _FakeSeparator)
    monkeypatch.setattr(separation, "ensure_ffmpeg_in_path", lambda bin_dir: bin_dir)
    monkeypatch.setattr(separation.torch.cuda, "is_available", lambda: False)

    norm = _make_normalized(cache_manager)
    vocal, instrumental = separation.separate_stems(norm, cache_manager)

    assert vocal.name == "vocal.wav"
    assert instrumental.name == "instrumental.wav"
    assert vocal.is_file() and instrumental.is_file()
    # The default-named intermediates must not linger alongside the canonical files.
    leftovers = [p.name for p in vocal.parent.iterdir() if "roformer" in p.name.lower()]
    assert leftovers == []


def test_separate_stems_uses_cache_when_present(cache_manager, monkeypatch):
    calls = {"n": 0}

    class _CountingSeparator(_FakeSeparator):
        def separate(self, audio_file_path: str):
            calls["n"] += 1
            return super().separate(audio_file_path)

    monkeypatch.setattr(separation, "Separator", _CountingSeparator)
    monkeypatch.setattr(separation, "ensure_ffmpeg_in_path", lambda bin_dir: bin_dir)
    monkeypatch.setattr(separation.torch.cuda, "is_available", lambda: False)

    norm = _make_normalized(cache_manager)
    separation.separate_stems(norm, cache_manager)
    separation.separate_stems(norm, cache_manager)  # second call should hit the cache

    assert calls["n"] == 1


def test_separate_stems_raises_when_no_stems_produced(cache_manager, monkeypatch):
    class _EmptySeparator(_FakeSeparator):
        def separate(self, audio_file_path: str):
            return []  # produced nothing

    monkeypatch.setattr(separation, "Separator", _EmptySeparator)
    monkeypatch.setattr(separation, "ensure_ffmpeg_in_path", lambda bin_dir: bin_dir)
    monkeypatch.setattr(separation.torch.cuda, "is_available", lambda: False)

    norm = _make_normalized(cache_manager)
    with pytest.raises(RuntimeError, match="did not produce expected output"):
        separation.separate_stems(norm, cache_manager)
