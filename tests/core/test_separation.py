"""Tests for app.core.separation.

Exercises the stem-naming/renaming and caching logic without loading the real BS-RoFormer
model by substituting a fake Separator that writes default-named output files.
"""

from __future__ import annotations

from pathlib import Path
import wave

import pytest

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
    norm.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(norm), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"\x00" * 882)  # tiny silence
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


def test_separate_stems_invalid_or_corrupt_wav_file(cache_manager, monkeypatch):
    monkeypatch.setattr(separation, "ensure_ffmpeg_in_path", lambda bin_dir: bin_dir)
    monkeypatch.setattr(separation.torch.cuda, "is_available", lambda: False)

    # 1. Non-existent file
    non_existent = cache_manager.track_dir("nonexistent") / "normalized.wav"
    with pytest.raises(FileNotFoundError, match="Input WAV file does not exist"):
        separation.separate_stems(non_existent, cache_manager)

    # 2. Corrupt WAV file (e.g., zero bytes)
    corrupt = cache_manager.track_dir("corrupt") / "normalized.wav"
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_bytes(b"")  # Empty file
    with pytest.raises(ValueError, match="Invalid or corrupt input WAV file"):
        separation.separate_stems(corrupt, cache_manager)

    # 3. Invalid WAV format (e.g., random bytes)
    invalid = cache_manager.track_dir("invalid") / "normalized.wav"
    invalid.parent.mkdir(parents=True, exist_ok=True)
    invalid.write_bytes(b"invalid header bytes")
    with pytest.raises(ValueError, match="Invalid or corrupt input WAV file"):
        separation.separate_stems(invalid, cache_manager)


def test_separate_stems_gpu_oom_fallback_to_cpu(cache_manager, monkeypatch):
    monkeypatch.setattr(separation, "ensure_ffmpeg_in_path", lambda bin_dir: bin_dir)

    # Spy on torch.cuda.empty_cache
    empty_cache_called = {"count": 0}
    def mock_empty_cache():
        empty_cache_called["count"] += 1
    monkeypatch.setattr(separation.torch.cuda, "empty_cache", mock_empty_cache)

    # Make torch.cuda.is_available() return True initially to trigger GPU path
    is_available_returns = [True]
    def mock_is_available():
        return is_available_returns[0]
    monkeypatch.setattr(separation.torch.cuda, "is_available", mock_is_available)

    # Custom Separator that raises OutOfMemoryError on the first separate call
    calls = []
    class _OOMSeparator(_FakeSeparator):
        def separate(self, audio_file_path: str):
            calls.append("separate")
            if separation.torch.cuda.is_available():
                raise separation.torch.cuda.OutOfMemoryError("CUDA out of memory error mocked!")
            return super().separate(audio_file_path)

    monkeypatch.setattr(separation, "Separator", _OOMSeparator)

    norm = _make_normalized(cache_manager)
    vocal, instrumental = separation.separate_stems(norm, cache_manager)

    assert vocal.name == "vocal.wav"
    assert instrumental.name == "instrumental.wav"
    # Ensure separate was called twice (first time GPU OOM, second time CPU fallback)
    assert len(calls) == 2
    # Ensure empty_cache was called (at least when OOM happened + in the outer finally)
    assert empty_cache_called["count"] >= 2


def test_separate_stems_always_cleans_up_cuda_cache(cache_manager, monkeypatch):
    monkeypatch.setattr(separation, "ensure_ffmpeg_in_path", lambda bin_dir: bin_dir)

    empty_cache_called = {"count": 0}
    def mock_empty_cache():
        empty_cache_called["count"] += 1
    monkeypatch.setattr(separation.torch.cuda, "empty_cache", mock_empty_cache)
    monkeypatch.setattr(separation.torch.cuda, "is_available", lambda: True)

    # 1. Test cleanup on success
    monkeypatch.setattr(separation, "Separator", _FakeSeparator)
    norm = _make_normalized(cache_manager)
    separation.separate_stems(norm, cache_manager)
    assert empty_cache_called["count"] == 1

    # Reset count
    empty_cache_called["count"] = 0

    # 2. Test cleanup on non-OOM exception
    class _FailingSeparator(_FakeSeparator):
        def separate(self, audio_file_path: str):
            raise RuntimeError("Some non-OOM failure")

    monkeypatch.setattr(separation, "Separator", _FailingSeparator)
    norm2 = _make_normalized(cache_manager, track_id="trk2")
    with pytest.raises(RuntimeError, match="Some non-OOM failure"):
        separation.separate_stems(norm2, cache_manager)
    assert empty_cache_called["count"] == 1
