# Music Mastery Enhancer

A local Windows desktop app (PySide6) for cleaning up AI artifacts, vocal hiss, and metallic
resonance in Suno-generated tracks. Everything runs locally — no subscription, no upload to a
third-party service.

The app is organized into two sections:

## 1. Stem Separation

Upload a Suno track (`.wav` or `.mp3`), which is normalized internally to 44.1kHz/24-bit WAV.
The track is then split into `vocal` and `instrumental` stems using `audio-separator` with a
BS-RoFormer model.

## 2. Artifact Fixing & Mastering

Each stem is run through its own neural cleanup pass (`resemble-enhance` for denoising and
harmonic reconstruction) followed by an adjustable Pedalboard DSP chain (high-pass/low-pass
filtering, a 4kHz notch for metallic resonance, and a de-esser). The cleaned vocal and
instrumental are remixed with user-controlled gain, then LUFS-normalized and limited into a
final mastered 24-bit WAV. Original vs. cleaned renders can be compared side by side before
export.

## Status

This repository currently contains only the project scaffolding (folder layout and
dependencies). No pipeline logic has been implemented yet — see `.agents/references/ROADMAP.md`
for the full milestone plan.

## Project Layout

```
app/
  core/       # Pipeline logic: ingestion, separation, vocal/instrumental chains, remix/master, export
  workers/    # QThread/QRunnable job wrappers around app/core
  ui/         # PySide6 views, widgets, controllers
  models/     # Dataclasses: Settings, Preset, RenderHistoryEntry, TrackSession
  cache/      # Cache manager (%LOCALAPPDATA%\MusicMasteryEnhancer\cache)
  setup/      # First-run CUDA detection + model downloader
tests/        # Mirrors app/ structure
installer/    # PyInstaller spec + Inno Setup script
```

## Setup

```
pip install -e ".[dev]"
```

Requires Python 3.11.
