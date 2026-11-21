"""Speech prosody and audio intelligence module for SarcasTone Member 2."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package roots are accessible
_PACKAGE_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PACKAGE_ROOT / "src"
for _p in (str(_PACKAGE_ROOT), str(_SRC_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from member2_audio.src.audio_intelligence.audio_intelligence import (
    extract_audio_features,
    get_audio_embedding,
    get_audio_sequence_embedding,
    get_audio_quality,
    assess_audio_quality,
    predict_audio,
    predict_emotion,
)
from member2_audio.src.feature_extraction.acoustic_features import extract_prosody_features
from member2_audio.src.audio_intelligence.audio_quality import extract_quality_features

__version__ = "0.1.0"

__all__ = [
    "extract_audio_features",
    "extract_prosody_features",
    "extract_quality_features",
    "get_audio_embedding",
    "get_audio_sequence_embedding",
    "get_audio_quality",
    "assess_audio_quality",
    "predict_audio",
    "predict_emotion",
    "__version__",
]
