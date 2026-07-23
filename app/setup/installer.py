"""Diagnostic and installation checks for frozen environment execution."""

from __future__ import annotations

import os
import sys
import ctypes
import ctypes.util
from typing import Dict, Any

from PySide6.QtCore import QCoreApplication
from app.cache import get_logger

logger = get_logger(__name__)


def is_frozen() -> bool:
    """Check if the application is running in a frozen PyInstaller bundle."""
    return getattr(sys, "frozen", False)


def check_pyside6_plugins() -> bool:
    """Verify that PySide6 Qt plugins (especially platforms) are present and readable."""
    paths = QCoreApplication.libraryPaths()
    if not paths:
        logger.error("No QCoreApplication library paths found.")
        return False

    for p in paths:
        path_str = p if isinstance(p, str) else p.toString()
        platforms_path = os.path.join(path_str, "platforms")
        if os.path.isdir(platforms_path):
            try:
                files = os.listdir(platforms_path)
                if files:
                    logger.info("Found PySide6 platform plugins in %s: %s", platforms_path, files)
                    return True
            except Exception as e:
                logger.error("Failed to list platform files in %s: %s", platforms_path, e)

    logger.error("No valid PySide6 platforms plugins found in library paths: %s", paths)
    return False


def check_cuda_dll() -> bool:
    """Check if the system has CUDA driver DLL (nvcuda.dll on Windows) available."""
    if sys.platform != "win32":
        # On non-Windows platforms, look for libcuda.so
        cuda_lib = ctypes.util.find_library("cuda")
        if cuda_lib:
            logger.info("Found CUDA library on non-Windows: %s", cuda_lib)
            return True
        logger.warning("CUDA library not found on non-Windows platform.")
        return False

    # Windows specific check for nvcuda.dll
    try:
        # LoadLibraryW attempts to load from standard system search path (System32, etc.)
        handle = ctypes.windll.kernel32.LoadLibraryW("nvcuda.dll")
        if handle:
            ctypes.windll.kernel32.FreeLibrary(handle)
            logger.info("Successfully loaded and freed nvcuda.dll via LoadLibraryW.")
            return True
    except Exception as e:
        logger.warning("Failed to load nvcuda.dll via LoadLibraryW: %s", e)

    # Fallback: check in standard System32 directory
    system_root = os.environ.get("SystemRoot", "C:\\Windows")
    system32 = os.path.join(system_root, "System32")
    nvcuda_path = os.path.join(system32, "nvcuda.dll")
    if os.path.exists(nvcuda_path):
        logger.info("Found nvcuda.dll at %s", nvcuda_path)
        return True

    logger.warning("nvcuda.dll was not found in standard system paths.")
    return False


def get_system_ram_gb() -> float:
    """Return total physical RAM in gigabytes (GB)."""
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024 ** 3)
        else:
            # Unix-like fallback
            try:
                pagesize = os.sysconf('SC_PAGE_SIZE')
                pages = os.sysconf('SC_PHYS_PAGES')
                return (pagesize * pages) / (1024 ** 3)
            except (AttributeError, ValueError, OSError):
                pass

            # proc/meminfo fallback
            if os.path.exists("/proc/meminfo"):
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if "MemTotal" in line:
                            parts = line.split()
                            if len(parts) >= 2:
                                return int(parts[1]) / (1024 ** 2)
    except Exception as e:
        logger.warning("Failed to determine system RAM: %s", e)

    return 8.0  # Default fallback


def run_diagnostics() -> Dict[str, Any]:
    """Run all diagnostic checks and return a report dictionary.

    Checks are run regardless of whether frozen or not, but they are crucial for PyInstaller builds.
    """
    pyside_ok = check_pyside6_plugins()
    cuda_ok = check_cuda_dll()
    ram_gb = get_system_ram_gb()

    report = {
        "frozen": is_frozen(),
        "pyside6_plugins_ok": pyside_ok,
        "cuda_dll_ok": cuda_ok,
        "ram_gb": ram_gb,
    }
    logger.info("Diagnostic report generated: %s", report)
    return report
