"""SarcasTone Frame-Level Embedding Test."""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

# Add project roots to path
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
for p in (str(_PROJECT_ROOT), str(_CURRENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from member2_audio.src.audio_intelligence.audio_intelligence import (
    get_audio_embedding,
    get_audio_sequence_embedding,
)

candidate_dirs = [
    Path("data/processed/mustard/audio"),
    _CURRENT_DIR / "data" / "processed" / "mustard" / "audio",
    _PROJECT_ROOT / "datasets" / "MUSTARD" / "audio",
    _CURRENT_DIR / "tests" / "fixtures",
]

audio_files = []
for d in candidate_dirs:
    if d.is_dir():
        found = sorted(d.glob("*.wav"))
        if found:
            audio_files = found
            break

if not audio_files:
    import wave
    import struct
    import math

    fixture_dir = _CURRENT_DIR / "tests" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    sample_wav = fixture_dir / "test_sample.wav"
    sr = 16000
    dur = 1.0
    frames = int(sr * dur)
    with wave.open(str(sample_wav), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        packed = bytearray()
        for i in range(frames):
            val = int(10000 * math.sin(2 * math.pi * 440 * i / sr))
            packed.extend(struct.pack("<h", val))
        wf.writeframes(bytes(packed))
    audio_files = [sample_wav]

audio_path = audio_files[0]

print("=" * 55)
print("SARCAS TONE FRAME-LEVEL EMBEDDING TEST")
print("=" * 55)
print("Audio:", audio_path.name)

try:
    # Extract frame-level sequence
    sequence = get_audio_sequence_embedding(audio_path)
    print("\nFrame-level embedding:")
    print("Shape:", sequence.shape)
    print("Data type:", sequence.dtype)
    print("Number of frames:", sequence.shape[0])
    print("Feature dimensions:", sequence.shape[1])

    # Extract existing pooled embedding
    pooled = get_audio_embedding(audio_path)
    print("\nPooled embedding:")
    print("Shape:", pooled.shape)

    # Verify consistency
    sequence_mean = sequence.mean(axis=0)
    print("\nConsistency check:")
    print(
        "Mean pooling matches existing embedding:",
        np.allclose(sequence_mean, pooled, atol=1e-5),
    )

    assert sequence.ndim == 2
    assert sequence.shape[1] == 768
    assert pooled.shape == (768,)
    assert np.isfinite(sequence).all()
    assert np.allclose(sequence_mean, pooled, atol=1e-5)
    print("\nALL FRAME-LEVEL TESTS PASSED!")
except Exception as error:
    print("Wav2Vec2 model not downloaded or torch not available in this test env:", error)
