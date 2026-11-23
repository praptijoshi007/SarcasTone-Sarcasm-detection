"""Audio preprocessing for Wav2Vec2 inference."""

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


def prepare_audio_for_wav2vec2(audio_path):
    """
    Load WAV audio and resample to 16 kHz mono
    for Wav2Vec2 inference.
    """
    waveform, sr, metadata = load_audio(
        audio_path,
        mono=True,
        normalize=False,
        target_sr=16000
    )

    waveform = np.asarray(waveform, dtype=np.float32)

    if len(waveform) < 400:
        raise ValueError("Audio is too short for Wav2Vec2.")

    if not np.isfinite(waveform).all():
        raise ValueError("Audio contains invalid values.")

    if np.max(np.abs(waveform)) == 0:
        raise ValueError("Audio contains only silence.")

    return waveform, sr, metadata
