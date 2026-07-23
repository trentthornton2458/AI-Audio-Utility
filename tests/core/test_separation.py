"""Tests for app.core.separation.

Exercises the stem-naming/renaming and caching logic without loading the real BS-RoFormer
model by substituting a fake Separator that writes default-named output files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from app.core import separation


class _FakeSeparator:
    """Stand-in for audio_separator's Separator that mimics its default output naming.

    On separate(), writes files named like the real library ("<base>_(Vocals)_<model>.wav"
    and "<base>_(Instrumental)_<model>.wav") into output_dir and returns their bare filenames.
    """

    def __init__(self, output_dir: str, model_file_dir: str) -> None:
        self._output_dir = Path(output_dir)

    def load_model(self, model_filename: str) -> None:  # noqa: D401 - test stub
        self._model = model_filename

    def separate(self, audio_file_path: str):
        base = Path(audio_file_path).stem
        names = [
            f"{base}_(Instrumental)_model_bs_roformer.wav",
            f"{base}_(Vocals)_model_bs_roformer.wav",
        ]
        for name in names:
            (self._output_dir / name).write_bytes(b"RIFFfake")
        return names


@pytest.fixture()
def cache_manager(tmp_path, monkeypatch):
    from app.cache.cache_manager import CacheManager
    from app.models.app_config import AppConfig

    config = AppConfig(cache_root=tmp_path / "cache")
    return CacheManager(config=config)


def _make_normalized(cache_manager, track_id: str = "trk") -> Path:
    norm = cache_manager.track_dir(track_id) / "normalized.wav"
    # Write a small valid mono WAV file to satisfy soundfile validation
    sr = 44100
    duration = 0.05
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t)[:, np.newaxis]
    sf.write(str(norm), audio, sr, subtype="PCM_16")
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


def test_separate_stems_validation_errors(cache_manager, tmp_path):
    # Test 1: Non-existent file
    bogus = tmp_path / "nonexistent.wav"
    with pytest.raises(ValueError, match="does not exist"):
        separation.separate_stems(bogus, cache_manager)

    # Test 2: Corrupt / Non-WAV file content
    corrupt = tmp_path / "corrupt.wav"
    corrupt.write_bytes(b"invalid data")
    with pytest.raises(ValueError, match="corrupt or invalid WAV"):
        separation.separate_stems(corrupt, cache_manager)

    # Test 3: Empty file with 0 frames
    empty = tmp_path / "empty.wav"
    empty.touch()
    with pytest.raises(ValueError, match="corrupt or invalid WAV"):
        separation.separate_stems(empty, cache_manager)


def test_separate_stems_cuda_empty_cache_cleanup(cache_manager, monkeypatch):
    empty_cache_calls = []

    def mock_empty_cache():
        empty_cache_calls.append(True)

    monkeypatch.setattr(torch.cuda, "empty_cache", mock_empty_cache)
    monkeypatch.setattr(separation.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(separation, "ensure_ffmpeg_in_path", lambda bin_dir: bin_dir)
    monkeypatch.setattr(separation, "Separator", _FakeSeparator)

    norm = _make_normalized(cache_manager)

    # Test success flow cleanup
    separation.separate_stems(norm, cache_manager)
    assert len(empty_cache_calls) == 1

    # Test failure flow cleanup
    class _FailingSeparator(_FakeSeparator):
        def separate(self, audio_file_path: str):
            raise RuntimeError("some unexpected model error")

    monkeypatch.setattr(separation, "Separator", _FailingSeparator)

    # We need to recreate the norm (or clear cache stems to make it run)
    import shutil
    shutil.rmtree(cache_manager.stems_dir("trk"), ignore_errors=True)

    with pytest.raises(RuntimeError, match="some unexpected model error"):
        separation.separate_stems(norm, cache_manager)

    assert len(empty_cache_calls) == 2


def test_separate_stems_cuda_oom_fallback_to_cpu(cache_manager, monkeypatch):
    first_run_attempted = {"status": False}
    second_run_attempted = {"status": False}

    class _OOMSeparator:
        def __init__(self, output_dir: str, model_file_dir: str) -> None:
            self._output_dir = Path(output_dir)

        def load_model(self, model_filename: str) -> None:
            pass

        def separate(self, audio_file_path: str):
            if not first_run_attempted["status"]:
                first_run_attempted["status"] = True
                # Mimic Out-Of-Memory error by raising standard PyTorch OOM or RuntimeError
                raise RuntimeError("CUDA out of memory. Tried to allocate 10GB")
            else:
                second_run_attempted["status"] = True
                base = Path(audio_file_path).stem
                names = [
                    f"{base}_(Instrumental)_model_bs_roformer.wav",
                    f"{base}_(Vocals)_model_bs_roformer.wav",
                ]
                for name in names:
                    (self._output_dir / name).write_bytes(b"RIFFfake")
                return names

    monkeypatch.setattr(separation, "Separator", _OOMSeparator)
    monkeypatch.setattr(separation, "ensure_ffmpeg_in_path", lambda bin_dir: bin_dir)
    monkeypatch.setattr(separation.torch.cuda, "is_available", lambda: True)

    norm = _make_normalized(cache_manager)
    vocal, instrumental = separation.separate_stems(norm, cache_manager)

    assert first_run_attempted["status"] is True
    assert second_run_attempted["status"] is True
    assert vocal.name == "vocal.wav"
    assert instrumental.name == "instrumental.wav"
    assert vocal.is_file() and instrumental.is_file()
