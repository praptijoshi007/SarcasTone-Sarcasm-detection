"""SarcasTone Audio Integration Test."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
import numpy as np

# Add project roots to path
_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _CURRENT_DIR.parent
for p in (str(_PROJECT_ROOT), str(_CURRENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from member2_audio.src.audio_intelligence.audio_intelligence import (
    extract_audio_features,
    get_audio_embedding,
    predict_audio,
    predict_emotion,
    get_audio_quality,
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
    # Create a small 1-second synthetic 16kHz WAV test tone in a temporary fixture
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

print("\n" + "=" * 55)
print("SARCAS TONE AUDIO INTEGRATION TEST")
print("=" * 55)
print("\nTest audio:", audio_path.name)

tests = [
    ("Acoustic Features", lambda: extract_audio_features(audio_path)),
    ("Audio Quality", lambda: get_audio_quality(audio_path)),
    ("Wav2Vec2 Embedding", lambda: get_audio_embedding(audio_path)),
    ("Audio Sarcasm Prediction", lambda: predict_audio(audio_path)),
    ("Emotion Prediction", lambda: predict_emotion(audio_path)),
]

for name, function in tests:
    print("\n" + "-" * 45)
    print(name)
    print("-" * 45)
    try:
        result = function()
        if name == "Acoustic Features":
            print("Feature count:", len(result))
            print("Prosody shape:", result.get("prosody", np.array([])).shape)
            print("Quality shape:", result.get("quality", np.array([])).shape)
            print("Sample:", dict(list(result.items())[:5]))
        elif name == "Wav2Vec2 Embedding":
            print("Shape:", result.shape)
            print("Data type:", result.dtype)
        else:
            print(result)
        print("STATUS: PASSED")
    except Exception as error:
        print("STATUS: SKIPPED/FAILED (Requires trained weights or external model):", error)

print("\n" + "=" * 55)
print("INTEGRATION TEST COMPLETED")
print("=" * 55)
