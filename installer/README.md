# Installer Build

Builds the distributable `MusicMasteryEnhancer.exe` and wraps it in a guided
Windows installer. Model weights are **not** bundled — the app downloads them
itself via the first-run `SetupWizard` (see `app/ui/setup_wizard.py` and the
`download_first_run` decision in `.agents/references/ARCHITECTURE.md`), so the
installer stays small.

## Prerequisites

- Python environment with the project installed (`pip install -e .[dev]` from the repo root)
- `pyinstaller` (`pip install pyinstaller`)
- [Inno Setup](https://jrsoftware.org/isinfo.php) 6.x installed, with `iscc.exe` on your `PATH`

## 1. Build the executable with PyInstaller

From the repo root:

```
pyinstaller installer/music_mastery_enhancer.spec --noconfirm
```

This produces `dist/MusicMasteryEnhancer.exe` — a single-file executable built
from `app/main.py`. The spec:

- Bundles the `imageio-ffmpeg` binary at `imageio_ffmpeg/binaries/` inside the
  frozen app, so `imageio_ffmpeg.get_ffmpeg_exe()` resolves it unchanged at runtime.
- Uses `collect_all` for `torch`, `torchaudio`, `audio_separator`,
  `resemble_enhance`, and `pedalboard` so their dynamic imports and packaged
  non-Python data are included.
- Does **not** include any model weight files.

Delete the `build/` and `dist/` folders (`installer/build`, `installer/dist` if
run from within `installer/`, or the repo-root equivalents if run from the repo
root as shown above) before rebuilding after dependency changes, to avoid stale
PyInstaller caches.

## 2. Build the installer with Inno Setup

From the repo root, after step 1 has produced `dist/MusicMasteryEnhancer.exe`:

```
iscc installer/setup_script.iss
```

This produces `installer/output/MusicMasteryEnhancer-Setup.exe`. It:

- Installs `MusicMasteryEnhancer.exe` to `Program Files\Music Mastery Enhancer`.
- Creates a Start Menu shortcut named **Music Mastery Enhancer** (and an
  optional desktop shortcut via an installer task).
- Does **not** bundle model weights — the app's own `SetupWizard` downloads
  BS-RoFormer and resemble-enhance weights (with progress + checksum
  verification) the first time the installed app is launched.

## Verifying a build

1. Run `dist/MusicMasteryEnhancer.exe` directly (before installing) to confirm
   the frozen app launches and, on a machine/user profile with no
   `%LOCALAPPDATA%\MusicMasteryEnhancer\cache\models` folder, shows the
   first-run `SetupWizard` before `MainWindow`.
2. Run the generated `MusicMasteryEnhancer-Setup.exe`, confirm the Start Menu
   shortcut is named "Music Mastery Enhancer", and confirm no `.ckpt`/`.pth`
   model files were copied under the install directory.
