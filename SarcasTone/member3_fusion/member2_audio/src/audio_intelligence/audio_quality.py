"""Audio quality indicators and 6-dimensional quality feature extraction."""

from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    from member2_audio.src.audio_processing.audio_loader import load_audio
except ImportError:
    try:
        from ..audio_processing.audio_loader import load_audio
    except (ImportError, ValueError):
        import sys
        _root = Path(__file__).resolve().parents[2]
        if str(_root) not in sys.path:
            sys.path.insert(0, str(_root))
        from src.audio_processing.audio_loader import load_audio

from shared.constants import (
    MIN_AUDIO_SECONDS,
    MAX_AUDIO_SECONDS,
    QUALITY_DIM,
    QUALITY_FEATURES,
)


def _compute_spectral_flatness(waveform: np.ndarray, n_fft: int = 2048, hop_length: int = 512) -> float:
    """Compute overall spectral flatness (geometric mean / arithmetic mean of power spectrum)."""
    if len(waveform) < n_fft:
        return 0.0
    num_frames = max(1, (len(waveform) - n_fft) // hop_length + 1)
    flatness_values = []
    window = np.hanning(n_fft)
    for i in range(num_frames):
        start = i * hop_length
        frame = waveform[start : start + n_fft] * window
        fft_vals = np.abs(np.fft.rfft(frame)) ** 2 + 1e-12
        geom_mean = np.exp(np.mean(np.log(fft_vals)))
        arith_mean = np.mean(fft_vals)
        if arith_mean > 0:
            flatness_values.append(geom_mean / arith_mean)
    return float(np.mean(flatness_values)) if flatness_values else 0.0


def get_audio_quality(audio_path):
    """
    Assess basic audio quality indicators.
    """
    waveform, sr, metadata = load_audio(
        audio_path,
        mono=True,
        normalize=False,
    )

    waveform = np.asarray(waveform, dtype=np.float32)

    peak = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(waveform ** 2)))

    clipping_ratio = float(
        np.mean(np.abs(waveform) >= 0.999)
    )

    silence_ratio = float(
        np.mean(np.abs(waveform) < 0.005)
    )

    # Approximate signal-to-noise indicator.
    snr_db = 0.0
    if rms > 0:
        snr_db = float(
            20 * np.log10(rms / max(peak * 0.01, 1e-10))
        )

    duration = float(metadata["duration_seconds"])
    spectral_flatness = _compute_spectral_flatness(waveform)

    warnings = []
    if duration < 0.5:
        warnings.append("Audio is very short.")

    if clipping_ratio > 0.01:
        warnings.append("Possible audio clipping detected.")

    if silence_ratio > 0.5:
        warnings.append("High proportion of near-silent samples.")

    if rms < 0.005:
        warnings.append("Very low audio energy.")

    duration_in_range = 1.0 if (MIN_AUDIO_SECONDS <= duration <= MAX_AUDIO_SECONDS) else 0.0

    quality_features = {
        "snr_db_estimate": float(snr_db),
        "clipping_ratio": float(clipping_ratio),
        "silence_ratio": float(silence_ratio),
        "duration_in_range": float(duration_in_range),
        "rms": float(rms),
        "spectral_flatness": float(spectral_flatness),
    }

    quality_vector = np.array(
        [quality_features[k] for k in QUALITY_FEATURES],
        dtype=np.float32,
    )

    return {
        "duration_seconds": duration,
        "sample_rate": int(sr),
        "channels": int(metadata.get("original_channels", 1)),
        "peak_amplitude": peak,
        "rms_energy": rms,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "approximate_level_indicator_db": snr_db,
        "warnings": warnings,
        "is_valid": len(warnings) == 0,
        "spectral_flatness": spectral_flatness,
        "duration_in_range": duration_in_range,
        "quality_features": quality_features,
        "quality_vector": quality_vector,
    }


def extract_quality_features(audio_path) -> tuple[np.ndarray, dict[str, float]]:
    """Return the 6-d float32 quality vector and feature dict matching shared.constants.QUALITY_FEATURES."""
    report = get_audio_quality(audio_path)
    return report["quality_vector"], report["quality_features"]
