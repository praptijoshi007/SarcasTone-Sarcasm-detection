"""Integration and Unit Tests for the SarcasTone FastAPI backend."""
from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from member4.backend.main import app

client = TestClient(app)
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "text" in data["active_modalities"]


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "CrossModalAttention" in data["architecture"]
    assert data["interface_contract"]["text_hidden"] == 768
    assert data["interface_contract"]["prosody_dim"] == 32


def test_predict_text_endpoint():
    payload = {
        "text": "Oh, absolutely brilliant observation!",
        "context": ["Do you really think it will work?"],
        "speaker": "CHANDLER",
    }
    response = client.post("/predict/text", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert data["label"] in ("sarcastic", "non_sarcastic")
    assert "confidence" in data
    assert len(data["top_tokens"]) > 0


def test_predict_multimodal_multipart():
    sample_wav = REPO_ROOT / "member4" / "fixtures" / "preset_sheldon_sarcastic.wav"
    if not sample_wav.is_file():
        sample_wav = REPO_ROOT / "member2_audio" / "tests" / "fixtures" / "test_sample.wav"

    with open(sample_wav, "rb") as f:
        response = client.post(
            "/predict",
            files={"audio": ("sample.wav", f, "audio/wav")},
            data={
                "text": "It's just a privilege to watch your mind at work.",
                "context": '["I think I solved the puzzle."]',
                "speaker": "SHELDON",
            },
        )
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "sarcastic"
    assert data["confidence"] > 0.70
    assert data["psid_dissonance_score"] > 0
    assert "text" in data["modality_weights"]
    assert len(data["explanation"]["top_text_tokens"]) > 0
    assert len(data["explanation"]["top_acoustic_features"]) > 0


def test_predict_audio_endpoint():
    sample_wav = REPO_ROOT / "member4" / "fixtures" / "preset_sheldon_sarcastic.wav"
    if not sample_wav.is_file():
        sample_wav = REPO_ROOT / "member2_audio" / "tests" / "fixtures" / "test_sample.wav"

    with open(sample_wav, "rb") as f:
        response = client.post(
            "/predict/audio",
            files={"audio": ("audio.wav", f, "audio/wav")},
            data={"speaker": "SHELDON"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert "audio_quality_valid" in data


def test_batch_predict_endpoint():
    payload = {
        "items": [
            {
                "id": "item1",
                "text": "Yeah, right. That will totally work.",
                "context": ["Should we try pushing the button?"],
                "speaker": "CHANDLER",
            },
            {
                "id": "item2",
                "text": "Thank you so much, this is genuinely helpful.",
                "context": ["Here is the documentation you requested."],
                "speaker": "PENNY",
            },
        ]
    }
    response = client.post("/batch-predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2
    assert "average_confidence" in data
