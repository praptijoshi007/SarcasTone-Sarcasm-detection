"""Audio loading and optional preprocessing."""

from __future__ import annotations

import os
from pathlib import Path
from math import gcd
import numpy as np

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from scipy.signal import resample_poly
except ImportError:
    resample_poly = None


def _load_with_wave(audio_path: Path) -> tuple[np.ndarray, int]:
    """Fallback WAV loader using Python's standard library wave module."""
    import wave

    with wave.open(str(audio_path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        raw_bytes = wf.readframes(num_frames)

    if sample_width == 1:
        # 8-bit unsigned PCM
        data = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        # 16-bit signed PCM
        data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 3:
        # 24-bit PCM
        raw_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        reshaped = raw_array.reshape(-1, 3)
        padded = np.pad(reshaped, ((0, 0), (1, 0)), mode="constant", constant_values=0)
        int32_arr = padded.view("<i4").reshape(-1) >> 8
        data = int32_arr.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        # 32-bit signed PCM or float
        try:
            data = np.frombuffer(raw_bytes, dtype=np.float32)
        except Exception:
            data = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if num_channels > 1:
        data = data.reshape(-1, num_channels)
    else:
        data = data.reshape(-1, 1)

    return data, sample_rate


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample 1D waveform from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return waveform

    if resample_poly is not None:
        common = gcd(orig_sr, target_sr)
        return resample_poly(waveform, target_sr // common, orig_sr // common, axis=0).astype(np.float32)

    # Clean linear interpolation fallback
    duration = len(waveform) / orig_sr
    target_length = max(1, int(round(duration * target_sr)))
    x_orig = np.linspace(0.0, duration, len(waveform), endpoint=False)
    x_target = np.linspace(0.0, duration, target_length, endpoint=False)
    return np.interp(x_target, x_orig, waveform).astype(np.float32)


def load_audio(
    audio_path,
    mono=True,
    normalize=False,
    target_sr=None,
):
    """
    Load an audio file and optionally process it.

    Parameters
    ----------
    audio_path : str or Path
        Path to the audio file.
    mono : bool
        Convert multi-channel audio to mono.
    normalize : bool
        Normalize peak amplitude to 1.0.
    target_sr : int or None
        Optional target sample rate. Original rate is preserved
        when None.

    Returns
    -------
    waveform : numpy.ndarray
        Audio samples as float32.
    sample_rate : int
        Final sample rate.
    metadata : dict
        Audio properties and processing information.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if audio_path.suffix.lower() != ".wav":
        raise ValueError("Currently, only WAV files are supported.")

    waveform = None
    original_sr = None

    if sf is not None:
        try:
            waveform, original_sr = sf.read(
                str(audio_path),
                dtype="float32",
                always_2d=True,
            )
        except Exception:
            waveform = None

    if waveform is None:
        try:
            waveform, original_sr = _load_with_wave(audio_path)
        except Exception as error:
            raise ValueError(f"Unable to read audio {audio_path}: {error}") from error

    if waveform.size == 0:
        raise ValueError("Audio file is empty.")

    if not np.isfinite(waveform).all():
        raise ValueError("Audio contains invalid numerical values.")

    original_channels = waveform.shape[1] if waveform.ndim > 1 else 1
    original_duration = waveform.shape[0] / original_sr

    # Convert stereo or multi-channel audio to mono
    if mono and original_channels > 1:
        waveform = np.mean(waveform, axis=1, keepdims=True)

    if mono and waveform.ndim > 1:
        waveform = waveform[:, 0]

    # Optional amplitude normalization
    peak_before = float(np.max(np.abs(waveform)))

    if normalize and peak_before > 0:
        waveform = waveform / peak_before

    # Optional resampling
    final_sr = original_sr

    if target_sr is not None:
        if not isinstance(target_sr, int) or target_sr <= 0:
            raise ValueError("target_sr must be a positive integer.")

        if target_sr != original_sr:
            waveform = _resample(waveform, original_sr, target_sr)
            final_sr = target_sr

    waveform = np.asarray(waveform, dtype=np.float32)
    final_duration = len(waveform) / final_sr

    metadata = {
        "file_name": audio_path.name,
        "original_sample_rate": int(original_sr),
        "final_sample_rate": int(final_sr),
        "original_channels": int(original_channels),
        "duration_seconds": float(final_duration),
        "original_duration_seconds": float(original_duration),
        "samples": int(len(waveform)),
        "peak_amplitude": float(np.max(np.abs(waveform))),
        "normalized": bool(normalize),
        "mono": bool(mono),
    }

    return waveform, final_sr, metadata
