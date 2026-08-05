"""Tests for model downloader checksum validation and reuse logic."""

import hashlib
import io
from pathlib import Path
from unittest.mock import patch

import pytest

from app.cache.cache_manager import CacheManager
from app.models.app_config import AppConfig
from app.setup.model_downloader import (RUBBERBAND_BIN_FILENAME,
                                        ModelDownloader, ModelDownloadError,
                                        ModelSpec,
                                        RubberbandBinaryNotFoundError,
                                        RubberbandDownloadError,
                                        download_rubberband_binary,
                                        get_rubberband_binary_path)


def test_model_downloader_reuses_existing_valid_file(tmp_path: Path):
    cache_mgr = CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))
    downloader = ModelDownloader(cache_mgr)

    dummy_content = b"fake weights content 12345"
    sha256 = hashlib.sha256(dummy_content).hexdigest()

    test_spec = ModelSpec(
        name="TestModel",
        filename="test_model.ckpt",
        url="http://localhost/bogus_url",
        sha256=sha256,
    )

    dest = downloader.models_dir / test_spec.filename
    dest.write_bytes(dummy_content)

    progress_reports = []

    def cb(name, progress):
        progress_reports.append((name, progress))

    # Should pass without making any HTTP request because the file exists and SHA matches
    result_path = downloader._download_one(test_spec, cb)
    assert result_path == dest
    assert progress_reports == [("TestModel", 1.0)]


def test_model_downloader_checksum_mismatch_fails(tmp_path: Path):
    cache_mgr = CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))
    downloader = ModelDownloader(cache_mgr)

    dummy_content = b"fake weights content"
    wrong_sha256 = "0" * 64

    test_spec = ModelSpec(
        name="TestModel",
        filename="test_model.ckpt",
        url="http://localhost/bogus_url",
        sha256=wrong_sha256,
    )

    dest = downloader.models_dir / test_spec.filename
    dest.write_bytes(dummy_content)

    # Since checksum mismatches, it should attempt to download and raise error or checksum failure
    with pytest.raises(ModelDownloadError):
        downloader._download_one(test_spec, lambda n, p: None)


class _FakeHTTPResponse:
    """Minimal stand-in for urllib's response object, usable as a context manager."""

    def __init__(self, content: bytes):
        self._buffer = io.BytesIO(content)
        self.headers = {"Content-Length": str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def read(self, size: int) -> bytes:
        return self._buffer.read(size)


def test_download_rubberband_binary_reuses_existing_valid_file(tmp_path: Path):
    cache_mgr = CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))
    dummy_content = b"fake rubberband exe bytes"
    sha256 = hashlib.sha256(dummy_content).hexdigest()

    dest = cache_mgr.bin_dir / RUBBERBAND_BIN_FILENAME
    dest.write_bytes(dummy_content)

    progress_reports = []

    with (
        patch("app.setup.model_downloader.RUBBERBAND_WINDOWS_SHA256", sha256),
        patch("urllib.request.urlopen") as mock_urlopen,
    ):
        result_path = download_rubberband_binary(
            cache_manager=cache_mgr,
            progress_callback=lambda name, p: progress_reports.append((name, p)),
        )

    mock_urlopen.assert_not_called()
    assert result_path == dest
    assert progress_reports == [("Rubberband CLI", 1.0)]


def test_download_rubberband_binary_downloads_and_verifies(tmp_path: Path):
    cache_mgr = CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))
    dummy_content = b"fake rubberband exe bytes from the network"
    sha256 = hashlib.sha256(dummy_content).hexdigest()

    progress_reports = []

    with (
        patch("app.setup.model_downloader.RUBBERBAND_WINDOWS_SHA256", sha256),
        patch(
            "urllib.request.urlopen", return_value=_FakeHTTPResponse(dummy_content)
        ) as mock_urlopen,
    ):
        result_path = download_rubberband_binary(
            cache_manager=cache_mgr,
            progress_callback=lambda name, p: progress_reports.append((name, p)),
        )

    mock_urlopen.assert_called_once()
    assert result_path == cache_mgr.bin_dir / RUBBERBAND_BIN_FILENAME
    assert result_path.read_bytes() == dummy_content
    assert progress_reports[0] == ("Rubberband CLI", 0.0)
    assert progress_reports[-1] == ("Rubberband CLI", 1.0)
    # No leftover .part file after a successful move.
    assert not result_path.with_name(result_path.name + ".part").exists()


def test_download_rubberband_binary_checksum_mismatch_raises(tmp_path: Path):
    cache_mgr = CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))
    dummy_content = b"fake rubberband exe bytes"
    wrong_sha256 = "0" * 64

    with (
        patch("app.setup.model_downloader.RUBBERBAND_WINDOWS_SHA256", wrong_sha256),
        patch("urllib.request.urlopen", return_value=_FakeHTTPResponse(dummy_content)),
    ):
        with pytest.raises(RubberbandDownloadError):
            download_rubberband_binary(cache_manager=cache_mgr)

    dest = cache_mgr.bin_dir / RUBBERBAND_BIN_FILENAME
    assert not dest.exists()
    assert not dest.with_name(dest.name + ".part").exists()


def test_download_rubberband_binary_respects_cancellation_before_start(tmp_path: Path):
    cache_mgr = CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))

    with patch("urllib.request.urlopen") as mock_urlopen:
        with pytest.raises(RubberbandDownloadError):
            download_rubberband_binary(
                cache_manager=cache_mgr, is_cancelled=lambda: True
            )

    mock_urlopen.assert_not_called()


def test_get_rubberband_binary_path_missing_raises(tmp_path: Path):
    cache_mgr = CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))

    with pytest.raises(RubberbandBinaryNotFoundError):
        get_rubberband_binary_path(cache_mgr)


def test_get_rubberband_binary_path_returns_path_when_present(tmp_path: Path):
    cache_mgr = CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))
    dest = cache_mgr.bin_dir / RUBBERBAND_BIN_FILENAME
    dest.write_bytes(b"fake rubberband exe bytes")

    result_path = get_rubberband_binary_path(cache_mgr)
    assert result_path == dest
