"""Tests for environmental diagnostics and installer helper functions (app.setup.installer)."""

from __future__ import annotations

import sys
import os
import ctypes
from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtCore import QCoreApplication

from app.setup.installer import (
    is_frozen,
    check_pyside6_plugins,
    check_cuda_dll,
    get_system_ram_gb,
    run_diagnostics,
)


def test_is_frozen():
    with patch("sys.frozen", True, create=True):
        assert is_frozen() is True

    # When sys.frozen is missing/False
    if hasattr(sys, "frozen"):
        with patch("sys.frozen", False):
            assert is_frozen() is False
    else:
        assert is_frozen() is False


def test_check_pyside6_plugins_success():
    # Mock library paths returning a valid path containing 'platforms' with files
    mock_paths = ["/mock/qt/plugins"]
    with patch.object(QCoreApplication, "libraryPaths", return_value=mock_paths), \
         patch("os.path.isdir", return_value=True), \
         patch("os.listdir", return_value=["qwindows.dll"]):
        assert check_pyside6_plugins() is True


def test_check_pyside6_plugins_failure_empty_paths():
    with patch.object(QCoreApplication, "libraryPaths", return_value=[]):
        assert check_pyside6_plugins() is False


def test_check_pyside6_plugins_failure_no_dir():
    mock_paths = ["/mock/qt/plugins"]
    with patch.object(QCoreApplication, "libraryPaths", return_value=mock_paths), \
         patch("os.path.isdir", return_value=False):
        assert check_pyside6_plugins() is False


def test_check_cuda_dll_non_windows():
    with patch("sys.platform", "linux"), \
         patch("ctypes.util.find_library", return_value="/usr/lib/libcuda.so"):
        assert check_cuda_dll() is True

    with patch("sys.platform", "linux"), \
         patch("ctypes.util.find_library", return_value=None):
        assert check_cuda_dll() is False


def test_check_cuda_dll_windows_load_library_success():
    # Mock success of loading library on win32
    mock_windll = MagicMock()
    mock_windll.kernel32.LoadLibraryW.return_value = 12345

    with patch("sys.platform", "win32"), \
         patch("ctypes.windll", mock_windll, create=True):
        assert check_cuda_dll() is True
        mock_windll.kernel32.LoadLibraryW.assert_called_with("nvcuda.dll")
        mock_windll.kernel32.FreeLibrary.assert_called_with(12345)


def test_check_cuda_dll_windows_load_library_failure_file_exists():
    # Mock failure of LoadLibraryW but file exists in System32
    mock_windll = MagicMock()
    mock_windll.kernel32.LoadLibraryW.side_effect = Exception("error")

    with patch("sys.platform", "win32"), \
         patch("ctypes.windll", mock_windll, create=True), \
         patch("os.environ", {"SystemRoot": "C:\\Windows"}), \
         patch("os.path.exists", return_value=True) as mock_exists:
        assert check_cuda_dll() is True
        assert mock_exists.called


def test_check_cuda_dll_windows_entire_failure():
    mock_windll = MagicMock()
    mock_windll.kernel32.LoadLibraryW.side_effect = Exception("error")

    with patch("sys.platform", "win32"), \
         patch("ctypes.windll", mock_windll, create=True), \
         patch("os.environ", {"SystemRoot": "C:\\Windows"}), \
         patch("os.path.exists", return_value=False):
        assert check_cuda_dll() is False


def test_get_system_ram_gb_windows_success():
    mock_windll = MagicMock()
    mock_windll.kernel32.GlobalMemoryStatusEx.return_value = True

    class MockStructure:
        def __init__(self, *args, **kwargs):
            self.dwLength = 0
            self.ullTotalPhys = 16 * (1024 ** 3)

    with patch("sys.platform", "win32"), \
         patch("ctypes.windll", mock_windll, create=True), \
         patch("ctypes.Structure", MockStructure), \
         patch("ctypes.sizeof", return_value=64), \
         patch("ctypes.byref", return_value=None):

        assert get_system_ram_gb() == 16.0


def test_get_system_ram_gb_linux_success():
    with patch("sys.platform", "linux"), \
         patch("os.sysconf", side_effect=[4096, 4194304]):  # 4096 * 4194304 = 16 GB
        assert get_system_ram_gb() == 16.0


def test_get_system_ram_gb_proc_meminfo():
    meminfo_content = "MemTotal:       16345672 kB\n"
    with patch("sys.platform", "linux"), \
         patch("os.sysconf", side_effect=ValueError), \
         patch("os.path.exists", return_value=True), \
         patch("builtins.open", patch_open_meminfo(meminfo_content)):
        # 16345672 kB / 1024 / 1024 = 15.58 GB
        assert 15.0 < get_system_ram_gb() < 16.0


def patch_open_meminfo(content):
    mock_file = MagicMock()
    mock_file.__enter__.return_value = content.splitlines()
    return MagicMock(return_value=mock_file)


def test_run_diagnostics():
    with patch("app.setup.installer.check_pyside6_plugins", return_value=True), \
         patch("app.setup.installer.check_cuda_dll", return_value=True), \
         patch("app.setup.installer.get_system_ram_gb", return_value=12.0), \
         patch("app.setup.installer.is_frozen", return_value=True):

        report = run_diagnostics()
        assert report["frozen"] is True
        assert report["pyside6_plugins_ok"] is True
        assert report["cuda_dll_ok"] is True
        assert report["ram_gb"] == 12.0
