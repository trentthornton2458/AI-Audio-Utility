"""Tests for app.core.gemini_qa: densest-window snippet slicing and Gemini structured-output
stem analysis (Gemini's SDK client itself is mocked; no real network calls)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from app.core import gemini_qa


def _write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    sf.write(str(path), samples, sr, subtype="PCM_16")


def test_extract_loudest_window_returns_whole_file_when_shorter_than_window(tmp_path):
    sr = 8000
    audio = (
        np.random.default_rng(0).uniform(-0.1, 0.1, size=(sr * 2, 1)).astype(np.float32)
    )  # 2s
    path = tmp_path / "short.wav"
    _write_wav(path, audio, sr)

    result_audio, result_sr = gemini_qa._extract_loudest_window(
        path, window_seconds=20.0
    )

    assert result_sr == sr
    assert result_audio.shape[0] == audio.shape[0]


def test_extract_loudest_window_picks_the_loud_segment(tmp_path):
    sr = 8000
    total_seconds = 10.0
    window_seconds = 2.0
    n = int(sr * total_seconds)

    audio = np.full((n, 1), 0.01, dtype=np.float32)  # quiet everywhere
    loud_start = int(6.0 * sr)
    loud_end = loud_start + int(window_seconds * sr)
    audio[loud_start:loud_end, 0] = 0.9  # loud region between 6s-8s

    path = tmp_path / "loud.wav"
    _write_wav(path, audio, sr)

    result_audio, result_sr = gemini_qa._extract_loudest_window(
        path, window_seconds=window_seconds
    )

    assert result_sr == sr
    assert result_audio.shape[0] == int(window_seconds * sr)
    # The extracted window should be dominated by the loud region, not the quiet background.
    assert np.abs(result_audio).mean() > 0.5


def test_wav_bytes_round_trips_through_soundfile():
    sr = 8000
    audio = np.zeros((100, 1), dtype=np.float32)
    raw = gemini_qa._wav_bytes(audio, sr)

    assert raw[:4] == b"RIFF"
    assert len(raw) > 44  # more than just a WAV header


def test_clamp_respects_preset_bounds():
    assert gemini_qa._clamp("vocal_denoise_intensity", 1.5) == 1.0
    assert gemini_qa._clamp("vocal_denoise_intensity", -0.5) == 0.0
    assert gemini_qa._clamp("notch_depth_db", 100.0) == 6.0
    assert gemini_qa._clamp("notch_depth_db", -100.0) == 3.0
    assert gemini_qa._clamp("instrumental_mud_cut_hz", 40.0) == 40.0


def _mock_client(parsed_model):
    mock_response = MagicMock()
    mock_response.parsed = parsed_model
    mock_response.text = "irrelevant raw text"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


def test_analyze_vocal_stem_clamps_out_of_range_suggestions(tmp_path):
    sr = 8000
    audio = np.zeros((sr, 1), dtype=np.float32)
    stem_path = tmp_path / "vocal.wav"
    _write_wav(stem_path, audio, sr)

    parsed = gemini_qa._VocalAnalysis(
        vocal_denoise_intensity=1.8,
        vocal_enhance_intensity=-0.3,
        notch_depth_db=99.0,
    )

    with patch("app.core.gemini_qa.genai.Client", return_value=_mock_client(parsed)):
        result = gemini_qa.analyze_vocal_stem(stem_path, api_key="fake-key")

    assert result == {
        "vocal_denoise_intensity": 1.0,
        "vocal_enhance_intensity": 0.0,
        "notch_depth_db": 6.0,
    }


def test_analyze_instrumental_stem_returns_clamped_values(tmp_path):
    sr = 8000
    audio = np.zeros((sr, 1), dtype=np.float32)
    stem_path = tmp_path / "instrumental.wav"
    _write_wav(stem_path, audio, sr)

    parsed = gemini_qa._InstrumentalAnalysis(
        instrumental_denoise_intensity=0.4,
        instrumental_enhance_intensity=0.3,
        instrumental_mud_cut_hz=500.0,
        instrumental_dehiss_gain_db=-20.0,
    )

    with patch("app.core.gemini_qa.genai.Client", return_value=_mock_client(parsed)):
        result = gemini_qa.analyze_instrumental_stem(stem_path, api_key="fake-key")

    assert result == {
        "instrumental_denoise_intensity": 0.4,
        "instrumental_enhance_intensity": 0.3,
        "instrumental_mud_cut_hz": 120.0,
        "instrumental_dehiss_gain_db": -6.0,
    }


def test_analyze_vocal_stem_raises_on_network_failure(tmp_path):
    sr = 8000
    audio = np.zeros((sr, 1), dtype=np.float32)
    stem_path = tmp_path / "vocal.wav"
    _write_wav(stem_path, audio, sr)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = ConnectionError("no network")

    with patch("app.core.gemini_qa.genai.Client", return_value=mock_client):
        with pytest.raises(gemini_qa.GeminiAnalysisError):
            gemini_qa.analyze_vocal_stem(stem_path, api_key="fake-key")


def test_analyze_vocal_stem_raises_on_unparsable_response(tmp_path):
    sr = 8000
    audio = np.zeros((sr, 1), dtype=np.float32)
    stem_path = tmp_path / "vocal.wav"
    _write_wav(stem_path, audio, sr)

    with patch("app.core.gemini_qa.genai.Client", return_value=_mock_client(None)):
        with pytest.raises(gemini_qa.GeminiAnalysisError):
            gemini_qa.analyze_vocal_stem(stem_path, api_key="fake-key")


def test_diagnose_qa_window_returns_clamped_multiplier():
    sr = 8000
    dsp_window = np.zeros(sr, dtype=np.float32)
    enhanced_window = np.zeros(sr, dtype=np.float32)

    parsed = gemini_qa._WindowDiagnosis(
        verdict="hallucinated_tone", recommended_gain_multiplier=1.5
    )

    with patch("app.core.gemini_qa.genai.Client", return_value=_mock_client(parsed)):
        multiplier, verdict = gemini_qa.diagnose_qa_window(
            dsp_window,
            enhanced_window,
            sr,
            api_key="fake-key",
            stem_label="vocal",
            reason="hallucination_proxy",
        )

    assert multiplier == 1.0  # clamped down from 1.5
    assert verdict == "hallucinated_tone"


def test_diagnose_qa_window_raises_gemini_analysis_error_on_sdk_failure():
    sr = 8000
    dsp_window = np.zeros(sr, dtype=np.float32)
    enhanced_window = np.zeros(sr, dtype=np.float32)

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = ConnectionError("no network")

    with patch("app.core.gemini_qa.genai.Client", return_value=mock_client):
        with pytest.raises(gemini_qa.GeminiAnalysisError):
            gemini_qa.diagnose_qa_window(
                dsp_window,
                enhanced_window,
                sr,
                api_key="fake-key",
                stem_label="vocal",
                reason="silence",
            )
