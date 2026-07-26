"""Pre-DSP stem-analysis QA checkpoint: sends a short, densest-window snippet of an isolated
vocal or instrumental stem to Gemini's generateContent API (native audio understanding, not a
spectrogram) and asks it to suggest baseline values for this app's own neural/DSP parameters.

This runs once per stem right after app.core.separation.separate_stems and before the
resemble-enhance neural pass, so its output can seed app.ui.vocal_panel/instrumental_panel's
sliders with track-appropriate defaults instead of a static preset. See app.workers.
stem_analysis_job for the QThread that calls this off the UI thread, and app.models.
gemini_settings for where the API key comes from.

Uses the stable `generateContent` API (google-genai SDK), not the Beta "Interactions" API --
Google's own docs recommend generateContent for production use as of this writing.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.cache import get_logger
from app.models.preset import PRESET_SCHEMA

logger = get_logger(__name__)

MODEL_NAME = "gemini-2.5-flash"
SNIPPET_SECONDS = 20.0
RMS_WINDOW_SECONDS = 0.5

_BOUNDS = PRESET_SCHEMA["properties"]


class GeminiAnalysisError(Exception):
    """Raised when a stem could not be analyzed (network failure, bad API key, malformed
    response, etc). Callers should treat this as non-fatal -- the render can proceed with
    whatever preset values are already on the sliders."""


def _extract_loudest_window(stem_path: Path, window_seconds: float = SNIPPET_SECONDS) -> tuple[np.ndarray, int]:
    """Return (audio, samplerate) for the loudest `window_seconds` window of stem_path.

    Falls back to the whole file if it is shorter than window_seconds. "Loudest" is measured
    by RMS energy over RMS_WINDOW_SECONDS hops, since the densest/most active part of a track
    (e.g. the chorus) is the most informative snippet for a fixed-size analysis request.
    """
    audio, samplerate = sf.read(str(stem_path), always_2d=True, dtype="float32")
    total_frames = audio.shape[0]
    window_frames = int(window_seconds * samplerate)

    if total_frames <= window_frames:
        return audio, samplerate

    hop_frames = max(1, int(RMS_WINDOW_SECONDS * samplerate))
    mono = audio.mean(axis=1)
    energy = mono.astype(np.float64) ** 2

    cumsum = np.concatenate(([0.0], np.cumsum(energy)))
    hop_starts = np.arange(0, total_frames - window_frames + 1, hop_frames)
    window_sums = cumsum[hop_starts + window_frames] - cumsum[hop_starts]
    best_start = int(hop_starts[int(np.argmax(window_sums))])

    return audio[best_start : best_start + window_frames], samplerate


def _wav_bytes(audio: np.ndarray, samplerate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, samplerate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _clamp(key: str, value: float) -> float:
    bounds: dict = _BOUNDS[key] # type: ignore
    return max(bounds["minimum"], min(bounds["maximum"], float(value)))


def _call_gemini(
    api_key: str,
    snippet_bytes: bytes,
    prompt: str,
    schema: type[BaseModel],
) -> dict:
    """Call generateContent with an inline audio snippet and a Pydantic response_schema.

    response.parsed only reliably deserializes into a model instance when response_schema is
    a Pydantic class (rather than a raw types.Schema/dict) -- see google-genai's structured
    output docs -- so every caller here defines its schema as a BaseModel subclass.
    """
    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=snippet_bytes, mime_type="audio/wav"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - any SDK/network failure becomes GeminiAnalysisError
        raise GeminiAnalysisError(f"Gemini request failed: {exc}") from exc

    if response.parsed is None:
        raise GeminiAnalysisError(f"Gemini returned no parsable JSON payload (raw text: {response.text!r})")

    return response.parsed.model_dump()


class _VocalAnalysis(BaseModel):
    vocal_denoise_intensity: float = Field(
        description="0.0-1.0 dry/wet strength for resemble-enhance's denoise stage. Lower for "
        "a stem that is already clean (avoids watery/gated artifacts); higher for audible "
        "hiss or noise floor."
    )
    vocal_enhance_intensity: float = Field(
        description="0.0-1.0 dry/wet strength for resemble-enhance's harmonic-reconstruction "
        "stage. Lower if the stem shows signs of hi-hat bleed or sibilance that could be "
        "misread as vocal consonants (risk of metallic ringing); higher for a thin/dull vocal "
        "that needs harmonic fill-in."
    )
    notch_depth_db: float = Field(
        description="3.0-6.0 dB depth for a 4kHz peak notch cutting harsh/pinched resonance "
        "typical of AI-generated vocals. Higher for a harsher/more nasal stem."
    )


class _InstrumentalAnalysis(BaseModel):
    instrumental_denoise_intensity: float = Field(
        description="0.0-1.0 dry/wet strength for resemble-enhance's denoise stage on the "
        "instrumental stem. Lower if the stem is already clean; higher for audible "
        "separation-artifact noise or hiss."
    )
    instrumental_enhance_intensity: float = Field(
        description="0.0-1.0 dry/wet strength for resemble-enhance's harmonic-reconstruction "
        "stage on the instrumental stem."
    )
    instrumental_mud_cut_hz: float = Field(
        description="20.0-120.0 Hz highpass cutoff trimming low-end mud/rumble, including ghost "
        "bass bleed left over from stem separation in the 80-150Hz range. Higher for a muddier "
        "low end."
    )
    instrumental_dehiss_gain_db: float = Field(
        description="-6.0 to 0.0 dB high-shelf gain cutting hiss/noise in the top end. More "
        "negative for audible hiss."
    )


_VOCAL_PROMPT = (
    "You are an expert mastering engineer reviewing an isolated vocal stem. It was extracted "
    "from an AI-generated (Suno-style) song via BS-RoFormer stem separation and is about to go "
    "through a resemble-enhance neural denoise/enhance pass followed by a fixed DSP chain "
    "(highpass, lowpass, 4kHz notch, de-esser). Listen for: background noise/hiss, sibilance "
    "or harshness, thin/metallic/ringing timbre, and low-end bleed from the instrumental. "
    "Return conservative baseline parameter values that avoid over-processing -- err toward "
    "gentler settings when the stem already sounds clean, since heavy denoise/enhance settings "
    "on a clean stem cause watery or metallic artifacts."
)

_INSTRUMENTAL_PROMPT = (
    "You are an expert mastering engineer reviewing an isolated instrumental stem. It was "
    "extracted from an AI-generated (Suno-style) song via BS-RoFormer stem separation and is "
    "about to go through a resemble-enhance neural denoise/enhance pass followed by a gentle "
    "DSP EQ chain (low-end highpass, de-hiss high-shelf). Listen for: background noise/hiss, "
    "low-end mud or rumble (including ghost bass/vocal bleed left over from stem separation), "
    "and harsh top-end. Return conservative baseline parameter values that avoid "
    "over-processing -- err toward gentler settings when the stem already sounds clean."
)


class _WindowDiagnosis(BaseModel):
    verdict: str = Field(
        description="One of: 'clean', 'mild_artifact', 'hallucinated_tone', 'excessive_sibilance', "
        "or another short label describing what (if anything) is audibly wrong with the enhanced "
        "window compared to the DSP-only window."
    )
    recommended_gain_multiplier: float = Field(
        description="0.0-1.0 multiplier further attenuating the deterministic QA-gate blend gain "
        "already computed for this window. 1.0 = no extra attenuation needed (false alarm), "
        "0.0 = fully mute the AI-enhance contribution for this window."
    )


_WINDOW_DIAGNOSTIC_PROMPT = (
    "You are reviewing a short window flagged by an automated spectral QA gate as a candidate "
    "artifact introduced by a neural harmonic-enhancement pass on a {stem_label} stem. You are "
    "given two clips of the same window: the first is the DSP-only (pre-enhancement) signal, "
    "the second is the fully-enhanced (post-neural-pass) signal. The automated gate flagged this "
    "window for: {reason}. Listen to both and judge whether the enhanced clip actually contains "
    "an audible artifact (metallic ringing, a hallucinated/invented tone, excessive sibilance, "
    "watery/gated texture) that was NOT present in the DSP-only clip, versus the flag being a "
    "false alarm (e.g. legitimate musical brightness or a dense/loud passage). Recommend a gain "
    "multiplier to further attenuate the enhancement blend only if a real artifact is present."
)


def diagnose_qa_window(
    dsp_window_audio,
    enhanced_window_audio,
    samplerate: int,
    api_key: str,
    stem_label: str,
    reason: str,
) -> tuple[float, str]:
    """Diagnostic 2nd-pass QA check for a single window flagged by app.core.qa_gate's
    deterministic spectral gate. Sends both the DSP-only and enhanced clips for that window to
    Gemini and asks for a verdict plus a recommended additional gain multiplier.

    This is strictly diagnostic and never triggers re-enhancement -- it only ever recommends
    further attenuating (never boosting beyond 1.0x) the deterministic gain already computed by
    the QA gate. Raises GeminiAnalysisError on any failure (network, auth, malformed response);
    callers (app.core.qa_gate) must treat this as non-fatal and fall back to the deterministic
    gain alone.

    Returns (recommended_gain_multiplier clamped to [0.0, 1.0], verdict).
    """
    dsp_bytes = _wav_bytes(np.asarray(dsp_window_audio, dtype=np.float32), samplerate)
    enhanced_bytes = _wav_bytes(np.asarray(enhanced_window_audio, dtype=np.float32), samplerate)
    prompt = _WINDOW_DIAGNOSTIC_PROMPT.format(stem_label=stem_label, reason=reason)

    client = genai.Client(api_key=api_key)
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=dsp_bytes, mime_type="audio/wav"),
                types.Part.from_bytes(data=enhanced_bytes, mime_type="audio/wav"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_WindowDiagnosis,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - any SDK/network failure becomes GeminiAnalysisError
        raise GeminiAnalysisError(f"Gemini QA-window diagnostic failed: {exc}") from exc

    if response.parsed is None:
        raise GeminiAnalysisError(f"Gemini returned no parsable JSON payload (raw text: {response.text!r})")

    multiplier = max(0.0, min(1.0, float(response.parsed.recommended_gain_multiplier)))
    return multiplier, response.parsed.verdict


def analyze_vocal_stem(stem_path: Path, api_key: str) -> dict:
    """Analyze the isolated vocal stem and return clamped suggested parameter values for
    vocal_denoise_intensity, vocal_enhance_intensity, and notch_depth_db.

    Raises GeminiAnalysisError on any failure (network, auth, malformed response).
    """
    audio, samplerate = _extract_loudest_window(stem_path)
    snippet_bytes = _wav_bytes(audio, samplerate)
    result = _call_gemini(api_key, snippet_bytes, _VOCAL_PROMPT, _VocalAnalysis)

    return {
        "vocal_denoise_intensity": _clamp("vocal_denoise_intensity", result["vocal_denoise_intensity"]),
        "vocal_enhance_intensity": _clamp("vocal_enhance_intensity", result["vocal_enhance_intensity"]),
        "notch_depth_db": _clamp("notch_depth_db", result["notch_depth_db"]),
    }


def analyze_instrumental_stem(stem_path: Path, api_key: str) -> dict:
    """Analyze the isolated instrumental stem and return clamped suggested parameter values
    for instrumental_denoise_intensity, instrumental_enhance_intensity,
    instrumental_mud_cut_hz, and instrumental_dehiss_gain_db.

    Raises GeminiAnalysisError on any failure (network, auth, malformed response).
    """
    audio, samplerate = _extract_loudest_window(stem_path)
    snippet_bytes = _wav_bytes(audio, samplerate)
    result = _call_gemini(api_key, snippet_bytes, _INSTRUMENTAL_PROMPT, _InstrumentalAnalysis)

    return {
        "instrumental_denoise_intensity": _clamp(
            "instrumental_denoise_intensity", result["instrumental_denoise_intensity"]
        ),
        "instrumental_enhance_intensity": _clamp(
            "instrumental_enhance_intensity", result["instrumental_enhance_intensity"]
        ),
        "instrumental_mud_cut_hz": _clamp("instrumental_mud_cut_hz", result["instrumental_mud_cut_hz"]),
        "instrumental_dehiss_gain_db": _clamp(
            "instrumental_dehiss_gain_db", result["instrumental_dehiss_gain_db"]
        ),
    }
