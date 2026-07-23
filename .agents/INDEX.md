# Sauron AI Directions & Capability Index

> **Source of Truth for Target Project:** AI Audio Utility
> **Last Updated:** 2026-07-23T03:16:34.665Z

---

## 1. Active Architectural Direction
- **Vision:** Act as a senior audio software engineer. I want to build a local Python application that fixes AI artifacts, vocal hiss, and metallic resonances in Suno-generated audio tracks. With an .exe to install and use it with a guided setup.


I want to build an app with 2 separate sections. one part STEM separation, then audio artifact fixing and mastering. The goal is to separate AI generated music, extract the STEMs as cleanly as possible, then analyze them and fix the AI artifacts and master the audio into a perfectly whole song that does not sound like ai. I want to avoid paying a subscription for other services to do this, I want to build my own app that handles this for me. 



1. **File Ingestion:**
   - User uploads a Suno track (`.wav` or `.mp3`).
   - Automatically convert to 44.1kHz 24-bit WAV internally for processing.

2. **Stem Separation:**
   - Use `audio-separator` to split the track into `vocal` and `instrumental` stems using a RoFormer model (e.g., `model_bs_roformer_ep_317_sdr_12.9755.ckpt` or standard BS-RoFormer).

3. **Vocal Processing & Cleaning Pipeline:**
   - **Step A (Neural Denoising & Harmonics):** Process the isolated vocal through `resemble-enhance` to strip top-end diffusion noise and reconstruct missing upper harmonics. Include UI toggles for `Denoise` and `Enhance` intensity.
   - **Step B (DSP Polish via Pedalboard):**
     - High-pass filter at 80 Hz (cuts low-end mud).
     - Steep Low-pass filter at 14.5 kHz (removes remaining digital high-frequency sizzle).
     - Peak EQ notch around 4 kHz (-3dB to -6dB adjustable) to tame metallic/pinched vocal resonances.
     - Dynamic De-Esser targeting 5 kHz - 8 kHz.

4. **Remixing & Export:**
   - Mix the processed vocal stem back over the original instrumental stem.
   - Provide UI controls for:
     - **Vocal Clean Intensity** (Blend original vs. cleaned vocal)
     - **Vocal Gain (dB)**
     - **Instrumental Gain (dB)**
     - **Harshness Cut (4kHz Notch Depth)**
   - Recombine the track and output a downloadable 24-bit WAV file with side-by-side "Original vs. Cleaned" audio players.


- **Selected Direction:** Default Sequential Pipeline
- **Status:** executing

---

## 2. Active User Preferences & Stack
### Tech Stack
- Default Stack

### Code Conventions
- Standard Best Practices

### UI & Aesthetics
- Dark Glassmorphic Theme

### Testing Standard
- **Level:** STANDARD

### Custom Directives
None specified.

---

## 3. Agent Capabilities & Assignments
- ⚙️ **Claude Code**: Backend schema, database models, server API logic
- 👑 **Antigravity CLI**: UI vibe-coding, layouts, styling, components
- 🔍 **Jules**: Async unit testing, bug fixes, & GitHub PR generation

---

## 4. Execution Task Sequence (32 tasks)
- [x] **#1: Set up repo structure and dependency management** [claude] 
- [x] **#2: Build AppConfig, logging, and CacheManager** [claude] 
- [ ] **#3: Write scaffolding smoke tests** [jules] 
- [x] **#4: Implement ingestion module** [claude] 
- [ ] **#5: Write ingestion tests** [jules] 
- [x] **#6: Integrate audio-separator stem separation** [claude] 
- [x] **#7: Build model download manager backend** [claude] 
- [ ] **#8: Test separation and model download logic** [jules] 
- [x] **#9: Integrate resemble-enhance with per-stem caching** [claude] 
- [x] **#10: Build vocal Pedalboard DSP chain** [claude] (Commit: e6322e1)
- [ ] **#11: Test vocal DSP chain correctness** [jules] 
- [x] **#12: Build instrumental neural + DSP chain** [claude] (Commit: c8c299d)
- [ ] **#13: Test instrumental chain** [jules] 
- [x] **#14: Build remix and mastering stage** [claude] (Commit: 0a9dd7c)
- [ ] **#15: Test remix/mastering/export accuracy** [jules] 
- [x] **#16: Implement Preset data model and persistence** [claude] (Commit: 7385825)
- [ ] **#17: Test preset round-tripping** [jules] 
- [x] **#18: Build the render job worker** [claude] (Commit: 00c628b)
- [x] **#19: Build the first-run guided setup wizard UI** [antigravity] (Commit: 706fae8)
- [ ] **#20: Test job cancellation and failure paths** [jules] 
- [x] **#21: Build main window and navigation shell** [antigravity] (Commit: 0fbe18f)
- [ ] **#22: Bug sweep on file load and navigation flow** [jules] 
- [x] **#23: Build vocal control panel** [antigravity] (Commit: c245d6e)
- [x] **#24: Build instrumental control panel** [antigravity] (Commit: 72becc6)
- [ ] **#25: Test control panel state and signal emission** [jules] 
- [/] **#26: Build waveform A/B player widget** [antigravity] 
- [ ] **#27: Build render history panel** [antigravity] 
- [ ] **#28: Test waveform rendering and render history behavior** [jules] 
- [ ] **#29: Configure PyInstaller build and Inno Setup script** [claude] 
- [ ] **#30: Bug sweep on packaged build and first-run flow** [jules] 
- [ ] **#31: Full pipeline integration test** [jules] 
- [ ] **#32: Cross-cutting bug and UX sweep** [jules] 
