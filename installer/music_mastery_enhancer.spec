# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Music Mastery Enhancer.

Builds a single MusicMasteryEnhancer.exe from app/main.py.

Build from the repo root with:
    pyinstaller installer/music_mastery_enhancer.spec --noconfirm

See installer/README.md for the full build procedure.
"""

from pathlib import Path

import imageio_ffmpeg
from PyInstaller.utils.hooks import collect_all

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

# torch, torchaudio, audio-separator, and resemble-enhance all rely on dynamic
# imports / packaged non-Python data that PyInstaller's static analysis misses,
# so pull each in fully rather than hand-listing hidden imports.
for package_name in ("torch", "torchaudio", "audio_separator", "resemble_enhance", "pedalboard"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

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
