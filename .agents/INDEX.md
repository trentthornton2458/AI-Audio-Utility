# Sauron AI Directions & Capability Index

> **Source of Truth for Target Project:** Music Enhancer
> **Last Updated:** 2026-07-26T07:00:37.901Z

---

## 1. Active Architectural Direction
- **Vision:** Original Topic: I am using my music Enhance utility to make suno v5.5 tracks sound more human. It seems to work okay at applying audio tuning to the track to remove the compression and overall ai generated sibilance but I need a better method to make the vocals sound human and authentic while trying my best to keep this from being expensive. I would liek to create my own tool, or get an open source but a paid utility is my last resort.

--- Counsel's Conclusion ---

# Counsel's Conclusion: Technical Synthesis & Architectural Roadmap

## Executive Summary

The evaluation focused on determining an optimal, cost-effective, and fully local processing pipeline to eliminate AI artifacts (sibilance, metallic resonance, compression, micro-timing rigidity) from Suno v5.5 tracks while preserving authentic vocal characteristics (breath dynamics, natural vibrato, and transient detail). 

The existing implementation in `AI-Audio-Utility` (`resemble-enhance` + Pedalboard DSP with BS-RoFormer stem separation) provides a strong foundation. The consensus verdict favors **optimizing and tuning the existing local neural+DSP pipeline**, introducing a controlled **DSP micro-imperfection/jitter layer**, and treating heavy voice conversion models as an optional, secondary wet-blend module.

---

## Technical Evaluation of Proposed Approaches

### 1. Pure Algorithmic DSP & Jitter (PSOLA / Pitch Drift)
* **Strengths**: Lightweight, zero extra neural model overhead, deterministic execution, fits seamlessly into `app/core/vocal_chain.py`.
* **Drawbacks & Risks**: Pitch and timing jitter alone cannot reconstruct missing upper-frequency harmonics or remove metallic formant locking inherent in Suno tracks. Unconditioned random pitch/timing perturbation without real reference data risks introducing phase smearing or a "wobbly/fluttering" artifact rather than natural human micro-variation.

### 2. Full Neural Resynthesis & Voice Conversion (RVC v2 / Seed-VC)
* **Strengths**: Replaces artificial timbre with authentic vocal samples; effectively solves severe formant locking.
* **Drawbacks & Risks**: Highest risk profile across all proposals. Full voice replacement destroys vocal identity, removes subtle sung dynamics/breath, and requires target voice models that match the key, range, and style of the Suno track. Furthermore, aggressive Conditional Flow Matching (CFM) in `resemble-enhance` over-smooths natural vibrato. High GPU VRAM overhead and potential failure surface for a local utility.

### 3. Granular Synthesis & Legacy Stem Separation (Spleeter / Librosa)
* **Verdict**: **Rejected**. Spleeter is technically obsolete compared to the project's active `BS-RoFormer` model (which achieves significantly higher SDR and cleaner isolation). Uncontrolled granular noise injection introduces severe phase issues.

---

## Recommended Architecture & Pipeline Modifications

To achieve human-sounding vocals while remaining 100% local, offline, and zero-cost, the pipeline will expand on the existing `app/core` architecture through a 4-stage processing flow:

```
[Suno Audio] ──► [BS-RoFormer Isolation] ──► [Gated Neural Harmonic Denoise] 
                                                        │
                                                        ▼
[Mastered Output] ◄── [Remix & LUFS Limit] ◄── [DSP Polish & Micro-Jitter]
```

### Stage 1: Optimized Neural Denoising (`app/core/resemble_enhance.py`)
* **CFM Step Gating**: Cap `resemble-enhance` CFM solver steps and restrict `Enhance` intensity to $\le 0.3$ (or Denoise-only mode). This prevents over-smoothing sung vibrato and micro-dynamics while stripping high-frequency diffusion noise.
* **Harmonic Preservation**: Use `resemble-enhance` strictly to clean up top-end phase hiss rather than attempting total timbral re-synthesis.

### Stage 2: Vocal DSP Polish & Micro-Imperfection Layer (`app/core/vocal_chain.py`)
Add a dynamic micro-imperfection stage using `pyrubberband` and `pedalboard` post-denoise:
1. **Micro-Pitch Drift**: Apply subtle, sub-cent random pitch drift ($\pm 3 \text{ to } 5\text{ cents}$) driven by a low-frequency oscillator ($4\text{–}7\text{ Hz}$) to break Suno's robotic micro-pitch locking.
2. **Transient / Micro-Timing Offset**: Introduce mild time-domain jitter ($5\text{–}15\text{ ms}$) on vocal onset transients.
3. **Resonance & Sibilance Sculpting**:
   * High-pass filter at $80\text{ Hz}$.
   * Adjustable 4 kHz peak EQ notch ($-3\text{dB to }-6\text{dB}$) targeting metallic resonance.
   * Dynamic de-esser operating in the $5\text{ kHz}\text{–}8\text{ kHz}$ window.
   * Low-pass filter roll-off around $14.5\text{ kHz}$ to eliminate digital sizzle.

### Stage 3: Optional Timbral Wet Send (RVC / Seed-VC Integration)
* Rather than full timbre substitution, implement pre-trained singing voice conversion (RVC v2 / Seed-VC) as an **optional parallel wet send** (capped at $\le 30\%$ blend).
* Embeddings and inference are cached locally. If enabled by the user, this layers subtle organic vocal formants over the cleaned stem without overriding the original performance.

### Stage 4: Remix, Transient Recovery, & Mastering (`app/core/mastering.py`)
* **Residual Noise/Breath Blend**: Blend back low-level residual noise from the stem separation pass to restore natural breath sounds that neural separation often strips out.
* **Instrumental Stem**: High-pass filter + gentle transient shaping only (no neural pass required for instrumental).
* **Final Limiting**: Sum stems, apply dry/wet vocal blend, LUFS normalization (e.g., $-14\text{ LUFS}$), and true-peak limiting to 24-bit WAV export.

---

## Action Plan & User Controls

To maintain full user agency over varying Suno generation qualities, expose the following controls within the PySide6 UI (`app/ui/vocal_control_panel.py`):

| Control Parameter | Target Stage | Description / Value Range |
| :--- | :--- | :--- |
| **CFM Denoise Intensity** | `resemble-enhance` | $0.0 \text{ to } 0.3$ (Prevents vibrato flattening) |
| **Humanizer Jitter** | `pyrubberband` | $0\% \text{ to } 100\%$ (Maps to $\pm 3\text{--}5\text{ cents}$ drift & micro-timing) |
| **4kHz Metallic Cut** | `pedalboard` Notch | $0\text{dB to }-12\text{dB}$ attenuation |
| **Timbre Resynthesis Send** | RVC / Seed-VC | $0\% \text{ to } 30\%$ parallel wet blend (Optional) |
| **Vocal / Inst Blend** | Remix Stage | Per-stem dB gain + side-by-side A/B preview |

*Active Counsel Quorum: 4/4*
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

## 4. Execution Task Sequence (16 tasks)
- [ ] **#1: Add rubberband binary to model_downloader.py** [claude] (Commit: 99d47ac)
- [ ] **#2: Wire rubberband download into setup_wizard.py UI** [antigravity] (Commit: c8bf985)
- [ ] **#2: Milestone Review: Milestone 1: Rubberband binary provisioning** [jules] 
- [x] **#3: Implement pitch-drift humanizer function** [claude] (Commit: 3cafb1b)
- [x] **#4: Implement automatic breath/noise blend-back** [claude] (Commit: c4a0758)
- [x] **#5: Insert Humanizer stage into vocal_chain.py pipeline** [claude] (Commit: bd3b23c)
- [ ] **#5: Milestone Review: Milestone 2: Humanizer DSP stage (pitch drift + breath blend-back)** [jules] 
- [/] **#6: Add pitch-variance, HF-energy, and crest-factor QA metrics** [claude] 
- [ ] **#7: Surface QA warnings as a caution badge in vocal_panel.py** [antigravity] 
- [ ] **#8: Create factory reference asset loading + fallback logic** [claude] 
- [ ] **#9: First-run fallback modal for missing reference assets** [antigravity] 
- [ ] **#10: Extend A/B compare UI with Raw / Humanized / Reference blind toggle** [antigravity] 
- [ ] **#11: Add stubbed voice-conversion config field** [claude] 
- [ ] **#12: Add disabled 'Coming Soon' RVC slider to vocal_panel.py** [antigravity] 
- [ ] **#13: End-to-end pipeline test for Humanizer stage** [jules] 
- [ ] **#14: Bug sweep across new Humanizer stage and UI** [jules] 
