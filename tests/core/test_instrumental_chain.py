"""Tests for instrumental DSP chain."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from app.core.instrumental_chain import InstrumentalEqParams, apply_dsp_chain


def test_apply_instrumental_dsp_chain(tmp_path: Path):
    sr = 44100
    duration = 0.2
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 100 * t)[:, np.newaxis]
    audio = np.repeat(audio, 2, axis=1)  # Stereo

    in_file = tmp_path / "inst_in.wav"
    sf.write(str(in_file), audio, sr, subtype="PCM_24")

    out_file = tmp_path / "inst_dsp.wav"
    params = InstrumentalEqParams(mud_cut_hz=50.0, dehiss_shelf_hz=8000.0, dehiss_gain_db=-3.0)

    res_path = apply_dsp_chain(in_file, params, out_file)
    assert res_path.is_file()

    data, out_sr = sf.read(str(res_path))
    assert out_sr == sr
    assert data.shape == audio.shape


def test_apply_instrumental_dsp_chain_stability_edge_cases(tmp_path: Path):
    sr = 44100
    num_samples = int(sr * 0.1)

    # 1. Pure silence (all zeros)
    silence = np.zeros((num_samples, 2), dtype=np.float32)
    in_file = tmp_path / "silence.wav"
    sf.write(str(in_file), silence, sr, subtype="PCM_24")
    out_file = tmp_path / "silence_out.wav"
    apply_dsp_chain(in_file, InstrumentalEqParams(), out_file)
    out_data, out_sr = sf.read(str(out_file))
    assert np.allclose(out_data, 0.0, atol=1e-5)

    # 2. 0dB clipped square wave (exactly 1.0 and -1.0)
    square = np.sign(np.sin(2 * np.pi * 100 * np.linspace(0, 0.1, num_samples, endpoint=False)))[:, np.newaxis]
    square = np.repeat(square, 2, axis=1).astype(np.float32)
    in_file = tmp_path / "square.wav"
    sf.write(str(in_file), square, sr, subtype="PCM_24")
    out_file = tmp_path / "square_out.wav"
    apply_dsp_chain(in_file, InstrumentalEqParams(), out_file)
    out_data, out_sr = sf.read(str(out_file))
    assert not np.any(np.isnan(out_data))
    assert not np.any(np.isinf(out_data))
    # Filter overshoot may occur, but output must remain bounded
    assert np.max(np.abs(out_data)) < 3.0

    # 3. Audio containing NaN and Inf values
    corrupt_audio = np.random.uniform(-1.0, 1.0, (num_samples, 2)).astype(np.float32)
    corrupt_audio[100, 0] = np.nan
    corrupt_audio[200, 1] = np.inf
    corrupt_audio[300, :] = -np.inf
    in_file = tmp_path / "corrupt.wav"
    sf.write(str(in_file), corrupt_audio, sr, subtype="PCM_24")
    out_file = tmp_path / "corrupt_out.wav"
    apply_dsp_chain(in_file, InstrumentalEqParams(), out_file)
    out_data, out_sr = sf.read(str(out_file))
    # The output must be fully sanitized and contain no NaNs/Infs
    assert not np.any(np.isnan(out_data))
    assert not np.any(np.isinf(out_data))

    # 4. Extreme/low sampling rate (e.g., 8000 Hz, where Nyquist is 4000 Hz, below DEHISS shelf range)
    low_sr = 8000
    low_sr_samples = int(low_sr * 0.1)
    low_sr_audio = np.random.uniform(-1.0, 1.0, (low_sr_samples, 2)).astype(np.float32)
    in_file = tmp_path / "low_sr.wav"
    sf.write(str(in_file), low_sr_audio, low_sr, subtype="PCM_24")
    out_file = tmp_path / "low_sr_out.wav"
    # Even with high DEHISS_SHELF (e.g. default 10000 Hz), it should safely clamp below Nyquist and run
    apply_dsp_chain(in_file, InstrumentalEqParams(), out_file)
    out_data, out_sr = sf.read(str(out_file))
    assert out_sr == low_sr
    assert not np.any(np.isnan(out_data))

    # 5. Extreme parameter settings (which will be clamped internally)
    in_file = tmp_path / "extreme.wav"
    noise = np.random.uniform(-0.5, 0.5, (num_samples, 2)).astype(np.float32)
    sf.write(str(in_file), noise, sr, subtype="PCM_24")
    out_file = tmp_path / "extreme_out.wav"
    extreme_params = InstrumentalEqParams(
        mud_cut_hz=-5000.0,       # well below minimum
        dehiss_shelf_hz=100000.0,  # well above Nyquist / max
        dehiss_gain_db=-100.0     # well below minimum
    )
    apply_dsp_chain(in_file, extreme_params, out_file)
    out_data, out_sr = sf.read(str(out_file))
    assert not np.any(np.isnan(out_data))


def test_apply_instrumental_dsp_chain_memory_leak(tmp_path: Path):
    import gc
    import psutil

    sr = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 100 * t)[:, np.newaxis]
    audio = np.repeat(audio, 2, axis=1)  # Stereo

    in_file = tmp_path / "leak_in.wav"
    sf.write(str(in_file), audio, sr, subtype="PCM_24")

    # Warm up run to initialize buffers and dll loading
    out_file = tmp_path / "leak_out_warmup.wav"
    params = InstrumentalEqParams()
    apply_dsp_chain(in_file, params, out_file)

    gc.collect()
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    # Repeatedly process audio
    for i in range(50):
        out_file_i = tmp_path / f"leak_out_{i}.wav"
        apply_dsp_chain(in_file, params, out_file_i)
        # clean up files to not consume disk or leak file handles
        out_file_i.unlink()

    gc.collect()
    final_memory = process.memory_info().rss

    # Memory growth should be negligible (e.g., less than 5 MB)
    memory_diff_mb = (final_memory - initial_memory) / (1024 * 1024)
    assert memory_diff_mb < 5.0, f"Memory leak detected: RSS grew by {memory_diff_mb:.2f} MB"


def test_apply_instrumental_dsp_chain_frequency_response_smoothness(tmp_path: Path):
    sr = 44100
    num_samples = int(sr * 0.5)  # 0.5s for better FFT frequency resolution

    # Generate stable white noise with a fixed seed
    np.random.seed(12345)
    noise = np.random.uniform(-0.5, 0.5, (num_samples, 2)).astype(np.float32)
    in_file = tmp_path / "noise.wav"
    sf.write(str(in_file), noise, sr, subtype="PCM_24")

    # Helper to calculate band energy using FFT
    def get_band_energy(audio_data, freq_min, freq_max):
        mono = np.mean(audio_data, axis=1)
        fft_vals = np.abs(np.fft.rfft(mono))
        freqs = np.fft.rfftfreq(len(mono), d=1.0/sr)
        indices = np.where((freqs >= freq_min) & (freqs <= freq_max))[0]
        return float(np.sum(fft_vals[indices] ** 2))

    # 1. Sweep mud_cut_hz [20.0, 50.0, 80.0, 120.0]
    # More mud_cut cutoff should monotonically DECREASE the energy in low-frequency band [0, 40] Hz.
    low_band_energies = []
    mud_cut_steps = [20.0, 50.0, 80.0, 120.0]
    for cut in mud_cut_steps:
        out_file = tmp_path / f"mud_{cut}.wav"
        params = InstrumentalEqParams(mud_cut_hz=cut, dehiss_shelf_hz=12000.0, dehiss_gain_db=0.0)
        apply_dsp_chain(in_file, params, out_file)
        out_data, _ = sf.read(str(out_file))
        energy = get_band_energy(out_data, 0.0, 40.0)
        low_band_energies.append(energy)

    # Verify strict monotonic decrease
    for idx in range(len(low_band_energies) - 1):
        assert low_band_energies[idx + 1] <= low_band_energies[idx], \
            f"Low-band energy did not decrease monotonically during mud-cut sweep: {low_band_energies}"

    # 2. Sweep dehiss_gain_db [0.0, -1.5, -3.0, -4.5, -6.0]
    # More negative gain should monotonically DECREASE high-frequency band [12000, 20000] Hz.
    high_band_energies_gain = []
    gain_steps = [0.0, -1.5, -3.0, -4.5, -6.0]
    for gain in gain_steps:
        out_file = tmp_path / f"gain_{gain}.wav"
        params = InstrumentalEqParams(mud_cut_hz=20.0, dehiss_shelf_hz=10000.0, dehiss_gain_db=gain)
        apply_dsp_chain(in_file, params, out_file)
        out_data, _ = sf.read(str(out_file))
        energy = get_band_energy(out_data, 12000.0, 20000.0)
        high_band_energies_gain.append(energy)

    # Verify strict monotonic decrease
    for idx in range(len(high_band_energies_gain) - 1):
        assert high_band_energies_gain[idx + 1] <= high_band_energies_gain[idx], \
            f"High-band energy did not decrease monotonically during dehiss gain sweep: {high_band_energies_gain}"

    # 3. Sweep dehiss_shelf_hz [6000.0, 9000.0, 12000.0, 15000.0] with gain -6.0 dB
    # Higher shelf cutoff frequency shifts the filter high-cut band further up,
    # meaning the mid-high band [6000, 11000] Hz should be LESS attenuated (energy INCREASES monotonically).
    mid_high_band_energies = []
    shelf_steps = [6000.0, 9000.0, 12000.0, 15000.0]
    for shelf in shelf_steps:
        out_file = tmp_path / f"shelf_{shelf}.wav"
        params = InstrumentalEqParams(mud_cut_hz=20.0, dehiss_shelf_hz=shelf, dehiss_gain_db=-6.0)
        apply_dsp_chain(in_file, params, out_file)
        out_data, _ = sf.read(str(out_file))
        energy = get_band_energy(out_data, 6000.0, 11000.0)
        mid_high_band_energies.append(energy)

    # Verify strict monotonic increase
    for idx in range(len(mid_high_band_energies) - 1):
        assert mid_high_band_energies[idx + 1] >= mid_high_band_energies[idx], \
            f"Mid-high-band energy did not increase monotonically during shelf sweep: {mid_high_band_energies}"
