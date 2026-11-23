"""Acoustic prosody and spectral feature extraction.

Implements the 32-dimensional PROSODY_FEATURES contract defined in shared.constants,
as well as the detailed 116-dimensional acoustic feature matrix for emotion models.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any
import numpy as np

try:
    import librosa
except ImportError:
    librosa = None

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
    PROSODY_DIM,
    PROSODY_FEATURES,
    TARGET_SAMPLE_RATE,
)

ROOT = Path(__file__).resolve().parents[2]
RAVDESS_DIR = ROOT / "data" / "raw" / "RAVDESS" / "Audio_Speech_Actors_01-24"
OUTPUT_FILE = ROOT / "data" / "processed" / "ravdess_acoustic_features.csv"

N_MFCC = 13
N_MELS = 40
N_FFT = 2048
HOP_LENGTH = 512


def mean_std(values: Any, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
    }


def _estimate_f0_autocorr(waveform: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Fallback pitch estimator using normalized autocorrelation when librosa is absent."""
    min_lag = int(sr / 500)  # max pitch 500 Hz
    max_lag = int(sr / 65)   # min pitch 65 Hz
    num_frames = max(1, (len(waveform) - frame_length) // hop_length + 1)
    f0_values = []
    window = np.hanning(frame_length)

    for i in range(num_frames):
        start = i * hop_length
        frame = waveform[start : start + frame_length] * window
        energy = np.sum(frame ** 2)
        if energy < 1e-4:
            f0_values.append(0.0)
            continue
        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(frame) - 1 :]
        search_region = corr[min_lag : min(max_lag, len(corr))]
        if len(search_region) == 0:
            f0_values.append(0.0)
            continue
        peak_idx = int(np.argmax(search_region)) + min_lag
        peak_val = corr[peak_idx]
        if corr[0] > 0 and (peak_val / corr[0]) > 0.3:
            pitch = float(sr / peak_idx)
            f0_values.append(pitch)
        else:
            f0_values.append(0.0)

    return np.array(f0_values, dtype=np.float32)


def _compute_mfcc_numpy(waveform: np.ndarray, sr: int, n_mfcc: int = 13, n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Lightweight MFCC estimation with pure NumPy."""
    num_frames = max(1, (len(waveform) - n_fft) // hop_length + 1)
    window = np.hanning(n_fft)
    coeffs = []
    for i in range(num_frames):
        start = i * hop_length
        frame = waveform[start : start + n_fft] * window
        mag = np.abs(np.fft.rfft(frame))
        # Log energy spectrum pooled into 20 bands
        bands = np.array_split(mag, 20)
        band_energies = np.array([np.log(np.mean(b ** 2) + 1e-10) for b in bands])
        # Simple DCT-II
        k = np.arange(n_mfcc)[:, None]
        n = np.arange(len(band_energies))[None, :]
        dct_vals = np.sum(band_energies * np.cos(np.pi * k * (n + 0.5) / len(band_energies)), axis=1)
        coeffs.append(dct_vals)
    if not coeffs:
        return np.zeros((n_mfcc, 1), dtype=np.float32)
    return np.array(coeffs, dtype=np.float32).T


def extract_prosody_features(audio_path: str | Path) -> tuple[np.ndarray, dict[str, float]]:
    """
    Extract the exact 32-dimensional PROSODY_FEATURES vector defined in shared.constants.
    Returns:
        prosody_vector: np.ndarray of shape (32,) dtype float32
        prosody_dict: dictionary of {feature_name: value}
    """
    waveform, sr, metadata = load_audio(
        audio_path,
        mono=True,
        normalize=False,
        target_sr=TARGET_SAMPLE_RATE,
    )
    waveform = np.asarray(waveform, dtype=np.float32)
    duration_sec = float(metadata["duration_seconds"])

    # 1. Energy / RMS
    if librosa is not None:
        rms = librosa.feature.rms(y=waveform, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    else:
        num_frames = max(1, (len(waveform) - N_FFT) // HOP_LENGTH + 1)
        rms = np.array([
            np.sqrt(np.mean(waveform[i * HOP_LENGTH : i * HOP_LENGTH + N_FFT] ** 2))
            for i in range(num_frames)
        ], dtype=np.float32)

    rms_mean = float(np.mean(rms)) if len(rms) > 0 else 0.0
    rms_std = float(np.std(rms)) if len(rms) > 0 else 0.0
    rms_max = float(np.max(rms)) if len(rms) > 0 else 0.0
    rms_min = float(np.min(rms)) if len(rms) > 0 else 0.0

    # Active speech dynamic range (P90 vs P10 among active frames to prevent silence floor bias)
    active_rms = rms[rms > max(0.15 * rms_mean, 0.005)]
    if len(active_rms) >= 5:
        p90 = float(np.percentile(active_rms, 90))
        p10 = float(np.percentile(active_rms, 10))
        energy_dynamic_range_db = float(20.0 * np.log10(max(p90, 1e-6) / max(p10, 1e-6)))
    else:
        energy_dynamic_range_db = float(20.0 * np.log10(max(rms_max, 1e-7) / max(rms_min, 1e-7)))
    shimmer_proxy = float(np.mean(np.abs(np.diff(rms))) / (rms_mean + 1e-6)) if len(rms) >= 2 else 0.0
    emphasis_proxy = float(np.mean(rms > (rms_mean + rms_std))) if len(rms) > 0 else 0.0

    # 2. Fundamental Frequency (F0 / Pitch)
    if librosa is not None:
        try:
            f0_arr = librosa.yin(waveform, fmin=65, fmax=500, sr=sr, frame_length=N_FFT, hop_length=HOP_LENGTH)
        except Exception:
            f0_arr = _estimate_f0_autocorr(waveform, sr, N_FFT, HOP_LENGTH)
    else:
        f0_arr = _estimate_f0_autocorr(waveform, sr, N_FFT, HOP_LENGTH)

    valid_mask = np.isfinite(f0_arr) & (f0_arr > 65) & (f0_arr < 500)
    voiced = f0_arr[valid_mask]

    if len(voiced) > 0:
        f0_mean = float(np.mean(voiced))
        f0_std = float(np.std(voiced))
        f0_min = float(np.min(voiced))
        f0_max = float(np.max(voiced))
        f0_range = float(f0_max - f0_min)
        if len(voiced) >= 2:
            x = np.arange(len(voiced))
            slope = float(np.polyfit(x, voiced, 1)[0])
            jitter_local = float(np.mean(np.abs(np.diff(voiced))) / (f0_mean + 1e-6))
        else:
            slope = 0.0
            jitter_local = 0.0
        intonation_range_semitones = float(12.0 * np.log2(max(f0_max, 1e-6) / max(f0_min, 1e-6))) if f0_min > 0 else 0.0
        pitch_variation_coeff = float(f0_std / (f0_mean + 1e-6))
        voiced_ratio = float(len(voiced) / max(len(f0_arr), 1))
    else:
        f0_mean = 0.0
        f0_std = 0.0
        f0_min = 0.0
        f0_max = 0.0
        f0_range = 0.0
        slope = 0.0
        jitter_local = 0.0
        intonation_range_semitones = 0.0
        pitch_variation_coeff = 0.0
        voiced_ratio = 0.0

    # 3. Spectral Descriptors
    if librosa is not None:
        zcr = librosa.feature.zero_crossing_rate(y=waveform, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        flatness = librosa.feature.spectral_flatness(y=waveform, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
    else:
        # Pure numpy approximations
        num_frames = max(1, (len(waveform) - N_FFT) // HOP_LENGTH + 1)
        zcr_list = []
        cent_list = []
        roll_list = []
        bw_list = []
        flat_list = []
        freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
        window = np.hanning(N_FFT)

        for i in range(num_frames):
            frame = waveform[i * HOP_LENGTH : i * HOP_LENGTH + N_FFT] * window
            # ZCR
            zc = np.mean(np.abs(np.diff(np.sign(frame) > 0)))
            zcr_list.append(zc)
            # Spectrum
            mag = np.abs(np.fft.rfft(frame))
            total_mag = np.sum(mag) + 1e-12
            c = np.sum(freqs * mag) / total_mag
            cent_list.append(c)
            bw = np.sqrt(np.sum(((freqs - c) ** 2) * mag) / total_mag)
            bw_list.append(bw)
            cumsum = np.cumsum(mag)
            roll_idx = np.searchsorted(cumsum, 0.85 * total_mag)
            roll_list.append(freqs[min(roll_idx, len(freqs) - 1)])
            geom = np.exp(np.mean(np.log(mag ** 2 + 1e-12)))
            arith = np.mean(mag ** 2) + 1e-12
            flat_list.append(geom / arith)

        zcr = np.array(zcr_list)
        centroid = np.array(cent_list)
        rolloff = np.array(roll_list)
        bandwidth = np.array(bw_list)
        flatness = np.array(flat_list)
        mfcc = _compute_mfcc_numpy(waveform, sr, N_MFCC, N_FFT, HOP_LENGTH)

    zcr_mean = float(np.mean(zcr)) if len(zcr) > 0 else 0.0
    zcr_std = float(np.std(zcr)) if len(zcr) > 0 else 0.0
    spectral_centroid_mean = float(np.mean(centroid)) if len(centroid) > 0 else 0.0
    spectral_centroid_std = float(np.std(centroid)) if len(centroid) > 0 else 0.0
    spectral_rolloff_mean = float(np.mean(rolloff)) if len(rolloff) > 0 else 0.0
    spectral_rolloff_std = float(np.std(rolloff)) if len(rolloff) > 0 else 0.0
    spectral_bandwidth_mean = float(np.mean(bandwidth)) if len(bandwidth) > 0 else 0.0
    spectral_flatness_mean = float(np.mean(flatness)) if len(flatness) > 0 else 0.0

    mfcc1_mean = float(np.mean(mfcc[0])) if mfcc.shape[0] > 0 else 0.0
    mfcc2_mean = float(np.mean(mfcc[1])) if mfcc.shape[0] > 1 else 0.0
    mfcc1_std = float(np.std(mfcc[0])) if mfcc.shape[0] > 0 else 0.0

    # 4. Rhythm / Pause / Speaking Rate
    frame_dur = HOP_LENGTH / sr
    pause_threshold = max(0.1 * rms_mean, 0.005)
    pause_mask = rms < pause_threshold
    pause_ratio = float(np.mean(pause_mask)) if len(rms) > 0 else 0.0

    # Contiguous pause segments
    pause_segments = []
    current_run = 0
    for is_pause in pause_mask:
        if is_pause:
            current_run += 1
        else:
            if current_run >= 2:
                pause_segments.append(current_run)
            current_run = 0
    if current_run >= 2:
        pause_segments.append(current_run)

    pause_count_per_sec = float(len(pause_segments) / max(duration_sec, 0.1))
    mean_pause_sec = float(np.mean([p * frame_dur for p in pause_segments])) if pause_segments else 0.0

    # Speaking rate proxy: local RMS peaks
    if len(rms) >= 3:
        peaks = [i for i in range(1, len(rms) - 1) if rms[i] > rms[i - 1] and rms[i] > rms[i + 1] and rms[i] > rms_mean]
        speaking_rate_proxy = float(len(peaks) / max(duration_sec, 0.1))
    else:
        speaking_rate_proxy = 0.0

    prosody_dict: dict[str, float] = {
        "duration_sec": duration_sec,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_min": f0_min,
        "f0_max": f0_max,
        "f0_range": f0_range,
        "f0_slope": slope,
        "jitter_local": jitter_local,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "rms_max": rms_max,
        "energy_dynamic_range_db": energy_dynamic_range_db,
        "zcr_mean": zcr_mean,
        "zcr_std": zcr_std,
        "spectral_centroid_mean": spectral_centroid_mean,
        "spectral_centroid_std": spectral_centroid_std,
        "spectral_rolloff_mean": spectral_rolloff_mean,
        "spectral_rolloff_std": spectral_rolloff_std,
        "spectral_bandwidth_mean": spectral_bandwidth_mean,
        "spectral_flatness_mean": spectral_flatness_mean,
        "mfcc1_mean": mfcc1_mean,
        "mfcc2_mean": mfcc2_mean,
        "mfcc1_std": mfcc1_std,
        "speaking_rate_proxy": speaking_rate_proxy,
        "pause_ratio": pause_ratio,
        "pause_count_per_sec": pause_count_per_sec,
        "mean_pause_sec": mean_pause_sec,
        "voiced_ratio": voiced_ratio,
        "intonation_range_semitones": intonation_range_semitones,
        "shimmer_proxy": shimmer_proxy,
        "pitch_variation_coeff": pitch_variation_coeff,
        "emphasis_proxy": emphasis_proxy,
    }

    prosody_vector = np.array([prosody_dict[name] for name in PROSODY_FEATURES], dtype=np.float32)
    return prosody_vector, prosody_dict


def extract_features(audio_path: str | Path) -> dict[str, Any]:
    """Extract raw feature dictionary for legacy emotion model compatibility."""
    audio_path = Path(audio_path)
    waveform, sr, metadata = load_audio(audio_path, mono=True, normalize=False)
    if len(waveform) < N_FFT:
        raise ValueError("Audio is too short for feature extraction.")

    features = {
        "sample_id": audio_path.stem,
        "sample_rate": sr,
        "duration_seconds": metadata["duration_seconds"],
    }

    if librosa is not None:
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        for i in range(N_MFCC):
            features.update(mean_std(mfcc[i], f"mfcc_{i+1}"))
        mel = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        for i in range(N_MELS):
            features.update(mean_std(mel_db[i], f"mel_{i+1}"))
        centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
        zcr = librosa.feature.zero_crossing_rate(waveform, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        rms = librosa.feature.rms(y=waveform, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        pitch = librosa.yin(waveform, fmin=65, fmax=500, sr=sr, frame_length=N_FFT, hop_length=HOP_LENGTH)
        features.update(mean_std(centroid, "spectral_centroid"))
        features.update(mean_std(rolloff, "spectral_rolloff"))
        features.update(mean_std(zcr, "zero_crossing_rate"))
        features.update(mean_std(rms, "rms_energy"))
        features.update(mean_std(pitch, "pitch_f0"))
    else:
        # Fallback populate prosody
        _, pdict = extract_prosody_features(audio_path)
        features.update(pdict)

    return features
