"""Tests for app.workers.separation_job.SeparationJob.

Verifies the 'Extract Stems' background job runs ONLY ingestion + separation (no neural
denoise/enhance/master stages) and reports its results via separationFinished/failed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from app.workers.separation_job import SeparationJob


def _run_directly(job: SeparationJob) -> None:
    """Invoke the job body synchronously (QThread.run) without spawning a thread."""
    job.run()


def test_separation_job_runs_ingestion_then_separation(tmp_path):
    input_path = tmp_path / "input.wav"
    normalized = tmp_path / "normalized.wav"
    vocal = tmp_path / "vocal.wav"
    instrumental = tmp_path / "instrumental.wav"

    cache = MagicMock()
    job = SeparationJob(input_path=input_path, cache_manager=cache)

    finished: list[tuple[Path, Path]] = []
    stages: list[str] = []
    job.separationFinished.connect(lambda v, i: finished.append((v, i)))
    job.stageChanged.connect(stages.append)

    with patch("app.workers.separation_job.ingestion") as mock_ing, patch(
        "app.workers.separation_job.separation"
    ) as mock_sep:
        mock_ing.load_and_normalize_track.return_value = normalized
        mock_sep.separate_stems.return_value = (vocal, instrumental)

        _run_directly(job)

        mock_ing.load_and_normalize_track.assert_called_once_with(input_path, cache)
        mock_sep.separate_stems.assert_called_once_with(normalized, cache)

    assert finished == [(vocal, instrumental)]
    assert stages == ["Normalizing", "Separating"]


def test_separation_job_emits_failed_on_error(tmp_path):
    cache = MagicMock()
    job = SeparationJob(input_path=tmp_path / "input.wav", cache_manager=cache)

    errors: list[str] = []
    job.failed.connect(errors.append)

    with patch("app.workers.separation_job.ingestion") as mock_ing, patch(
        "app.workers.separation_job.separation"
    ) as mock_sep:
        mock_ing.load_and_normalize_track.return_value = tmp_path / "normalized.wav"
        mock_sep.separate_stems.side_effect = RuntimeError("no stems produced")

        _run_directly(job)

    assert errors == ["no stems produced"]


def test_separation_job_cancels_before_separation(tmp_path):
    cache = MagicMock()
    job = SeparationJob(input_path=tmp_path / "input.wav", cache_manager=cache)

    cancelled: list[bool] = []
    job.cancelled.connect(lambda: cancelled.append(True))

    with patch("app.workers.separation_job.ingestion") as mock_ing, patch(
        "app.workers.separation_job.separation"
    ) as mock_sep, patch.object(SeparationJob, "isInterruptionRequested", return_value=True):
        mock_ing.load_and_normalize_track.return_value = tmp_path / "normalized.wav"

        _run_directly(job)

        mock_sep.separate_stems.assert_not_called()

    assert cancelled == [True]
