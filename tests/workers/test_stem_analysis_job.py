"""Tests for app.workers.stem_analysis_job.StemAnalysisJob.

Verifies both stems are analyzed and that a failure on one stem does not block the other --
analysisFinished always fires with whatever succeeded plus per-stem error messages.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.core.gemini_qa import GeminiAnalysisError
from app.workers.stem_analysis_job import StemAnalysisJob


def _run_directly(job: StemAnalysisJob) -> None:
    """Invoke the job body synchronously (QThread.run) without spawning a Qt thread."""
    job.run()


def test_stem_analysis_job_emits_both_results_on_success(tmp_path):
    vocal_path = tmp_path / "vocal.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    job = StemAnalysisJob(vocal_path=vocal_path, instrumental_path=instrumental_path, api_key="fake-key")

    results: list[tuple[dict, dict, list]] = []
    job.analysisFinished.connect(lambda v, i, e: results.append((v, i, e)))

    vocal_result = {"vocal_denoise_intensity": 0.3, "vocal_enhance_intensity": 0.2, "notch_depth_db": 4.0}
    instrumental_result = {
        "instrumental_denoise_intensity": 0.1,
        "instrumental_enhance_intensity": 0.1,
        "instrumental_mud_cut_hz": 45.0,
        "instrumental_dehiss_gain_db": -2.0,
    }

    with patch(
        "app.workers.stem_analysis_job.gemini_qa.analyze_vocal_stem", return_value=vocal_result
    ) as mock_vocal, patch(
        "app.workers.stem_analysis_job.gemini_qa.analyze_instrumental_stem", return_value=instrumental_result
    ) as mock_instrumental:
        _run_directly(job)

        mock_vocal.assert_called_once_with(vocal_path, "fake-key")
        mock_instrumental.assert_called_once_with(instrumental_path, "fake-key")

    assert results == [(vocal_result, instrumental_result, [])]


def test_stem_analysis_job_vocal_failure_does_not_block_instrumental(tmp_path):
    vocal_path = tmp_path / "vocal.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    job = StemAnalysisJob(vocal_path=vocal_path, instrumental_path=instrumental_path, api_key="fake-key")

    results: list[tuple[dict, dict, list]] = []
    job.analysisFinished.connect(lambda v, i, e: results.append((v, i, e)))

    instrumental_result = {
        "instrumental_denoise_intensity": 0.1,
        "instrumental_enhance_intensity": 0.1,
        "instrumental_mud_cut_hz": 45.0,
        "instrumental_dehiss_gain_db": -2.0,
    }

    with patch(
        "app.workers.stem_analysis_job.gemini_qa.analyze_vocal_stem",
        side_effect=GeminiAnalysisError("network error"),
    ), patch(
        "app.workers.stem_analysis_job.gemini_qa.analyze_instrumental_stem",
        return_value=instrumental_result,
    ):
        _run_directly(job)

    assert len(results) == 1
    vocal_updates, instrumental_updates, errors = results[0]
    assert vocal_updates == {}
    assert instrumental_updates == instrumental_result
    assert len(errors) == 1
    assert "Vocal stem analysis failed" in errors[0]


def test_stem_analysis_job_both_fail_emits_empty_dicts_and_two_errors(tmp_path):
    vocal_path = tmp_path / "vocal.wav"
    instrumental_path = tmp_path / "instrumental.wav"
    job = StemAnalysisJob(vocal_path=vocal_path, instrumental_path=instrumental_path, api_key="fake-key")

    results: list[tuple[dict, dict, list]] = []
    job.analysisFinished.connect(lambda v, i, e: results.append((v, i, e)))

    with patch(
        "app.workers.stem_analysis_job.gemini_qa.analyze_vocal_stem",
        side_effect=GeminiAnalysisError("bad key"),
    ), patch(
        "app.workers.stem_analysis_job.gemini_qa.analyze_instrumental_stem",
        side_effect=GeminiAnalysisError("bad key"),
    ):
        _run_directly(job)

    assert len(results) == 1
    vocal_updates, instrumental_updates, errors = results[0]
    assert vocal_updates == {}
    assert instrumental_updates == {}
    assert len(errors) == 2
