"""Tests for model downloader checksum validation and reuse logic."""

import hashlib
from pathlib import Path
import pytest

from app.cache.cache_manager import CacheManager
from app.models.app_config import AppConfig
from app.setup.model_downloader import (
    ModelDownloadError,
    ModelDownloader,
    ModelSpec,
)


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
