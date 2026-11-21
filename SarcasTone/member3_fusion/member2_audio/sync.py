"""Multimodal synchronization bridge for Member 2."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shared.sync import MultimodalFeatureBundle, sync_utterance, verify_corpus_sync

__all__ = ["MultimodalFeatureBundle", "sync_utterance", "verify_corpus_sync"]
