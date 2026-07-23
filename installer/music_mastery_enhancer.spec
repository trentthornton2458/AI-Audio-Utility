# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Music Mastery Enhancer.

Builds a single MusicMasteryEnhancer.exe from app/main.py.

Build from the repo root with:
    pyinstaller installer/music_mastery_enhancer.spec --noconfirm

See installer/README.md for the full build procedure.
"""

from pathlib import Path

import imageio_ffmpeg
from PyInstaller.utils.hooks import collect_all, copy_metadata

block_cipher = None

# Repo root: this spec lives in installer/, so the project root is one level up.
PROJECT_ROOT = Path(SPECPATH).resolve().parent
ENTRY_POINT = str(PROJECT_ROOT / "app" / "main.py")

# The bundled ffmpeg binary (imageio-ffmpeg) must land at the same relative path
# imageio_ffmpeg.get_ffmpeg_exe() would resolve at runtime: <package_dir>/binaries/<exe>.
# PyInstaller preserves the imageio_ffmpeg package layout inside the bundle, so placing
# the binary under "imageio_ffmpeg/binaries" makes the frozen app find it unchanged.
FFMPEG_EXE_PATH = imageio_ffmpeg.get_ffmpeg_exe()
binaries = [(FFMPEG_EXE_PATH, "imageio_ffmpeg/binaries")]

datas = []
hiddenimports = []

# resemble_enhance's own download() (app/core/neural_common.py, enhancer/download.py) git-clones
# its multi-gigabyte model weights into <site-packages>/resemble_enhance/model_repo the first
# time denoise()/enhance() run -- e.g. while testing locally before a build. collect_all() below
# would otherwise silently bundle that entire multi-GB checkpoint into the installer, which is
# exactly what this app's architecture explicitly avoids: model weights are downloaded by the
# app itself on first run, not shipped in the installer (see ARCHITECTURE.md's
# "download_first_run" decision and installer/README.md). Filter it out defensively so a build
# is safe regardless of whether the build venv has ever run a real denoise/enhance pass.
_RESEMBLE_ENHANCE_MODEL_REPO_MARKER = str(Path("resemble_enhance") / "model_repo")


def _exclude_resemble_enhance_model_repo(entries):
    return [entry for entry in entries if _RESEMBLE_ENHANCE_MODEL_REPO_MARKER not in entry[0]]


# torch, torchaudio, audio-separator, and resemble-enhance all rely on dynamic
# imports / packaged non-Python data that PyInstaller's static analysis misses,
# so pull each in fully rather than hand-listing hidden imports.
for package_name in ("torch", "torchaudio", "audio_separator", "resemble_enhance", "pedalboard"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    if package_name == "resemble_enhance":
        pkg_datas = _exclude_resemble_enhance_model_repo(pkg_datas)
        pkg_binaries = _exclude_resemble_enhance_model_repo(pkg_binaries)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

# collect_all() above does NOT bundle .dist-info/*.dist-info metadata, only code/data/binaries.
# Several packages in this dependency tree (e.g. pandas.compat._optional.import_optional_dependency,
# used by resemble_enhance's Enhancer.summarize() logging via df.to_markdown()) call
# importlib.metadata.version(other_package) at runtime to enforce a minimum-version check. With no
# dist-info bundled, that lookup raises PackageNotFoundError, several packages fall back to the
# literal string "unknown" for their own __version__, and the consuming version-comparison then
# crashes with `InvalidVersion: Invalid version: 'unknown'` (see e.g. onnx/__init__.py's fallback
# and tabulate/__init__.py's). Bundle metadata for every package anywhere in this dependency tree
# that plausibly gets its version queried this way, so those lookups succeed instead of silently
# degrading to "unknown". Wrapped per-package since a couple of names below may not resolve to
# installed distributions in every environment.
_METADATA_PACKAGES = [
    "torch", "torchaudio", "torchvision", "audio-separator", "resemble-enhance", "pedalboard",
    "numpy", "scipy", "numba", "librosa", "matplotlib", "pandas", "tabulate", "rich",
    "onnx", "onnx-weekly", "onnxruntime", "soundfile", "jsonschema", "PySide6",
]
for _metadata_package in _METADATA_PACKAGES:
    try:
        datas += copy_metadata(_metadata_package)
    except Exception:
        pass

a = Analysis(
    [ENTRY_POINT],
    pathex=[str(PROJECT_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="MusicMasteryEnhancer",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
