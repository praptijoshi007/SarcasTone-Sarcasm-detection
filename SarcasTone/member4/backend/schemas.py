"""Pydantic schemas for the SarcasTone FastAPI backend."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    model_loaded: bool = True
    device: str = "cpu"
    active_modalities: list[str] = Field(default_factory=lambda: ["text", "audio", "prosody", "fusion"])


class ModelInfoResponse(BaseModel):
    model_name: str = "SarcasTone Multimodal Sarcasm Classifier"
    architecture: str = "CrossModalAttention + PSID + ConfidenceGate"
    upstream_modules: dict[str, str] = Field(default_factory=lambda: {
        "member1_text": "RoBERTa-base Span Pooling [768-d]",
        "member2_audio": "Wav2Vec2.0 + 32-d Prosody + 6-d Quality [768-d]",
        "member3_fusion": "Stacked Bidirectional Cross-Attention + PSID [512-d]",
        "member4_xai": "Captum Integrated Gradients + Uncertainty Calibration",
    })
    interface_contract: dict[str, Any] = Field(default_factory=lambda: {
        "text_hidden": 768,
        "audio_hidden": 768,
        "prosody_dim": 32,
        "quality_dim": 6,
        "sample_rate_hz": 16000,
        "classes": ["non_sarcastic", "sarcastic"],
    })
    checkpoint_status: str = "production_ready"


class TokenAttributionSchema(BaseModel):
    token: str
    score: float
    position: int


class AcousticAttributionSchema(BaseModel):
    feature_name: str
    importance: float
    value: float
    direction: str


class ExplanationSchema(BaseModel):
    label: str
    confidence: float
    entropy: float
    uncertain: bool
    modality_weights: dict[str, float]
    psid_dissonance_score: float
    cosine_agreement: Optional[float] = None
    top_text_tokens: list[TokenAttributionSchema] = Field(default_factory=list)
    top_acoustic_features: list[AcousticAttributionSchema] = Field(default_factory=list)
    methodology_note: str
    narrative: str


class BaselineResultSchema(BaseModel):
    model_name: str
    model_type: str
    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]
    description: str
    transcript: Optional[str] = None
    extra_metrics: dict[str, Any] = Field(default_factory=dict)


class MultimodalPredictionResponse(BaseModel):
    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]
    entropy: float
    uncertain: bool
    text_weight: float
    audio_weight: float
    modality_weights: dict[str, float]
    psid_dissonance_score: float
    cosine_agreement: Optional[float] = None
    transcript: str
    transcript_is_ground_truth: bool
    audio_available: bool
    text_available: bool
    speaker: Optional[str] = None
    explanation: ExplanationSchema
    processing_time_ms: float = 0.0
    text_only: Optional[BaselineResultSchema] = None
    audio_only: Optional[BaselineResultSchema] = None
    direct_concat: Optional[BaselineResultSchema] = None
    cross_modal_fusion: Optional[BaselineResultSchema] = None
    comparison_summary: list[BaselineResultSchema] = Field(default_factory=list)


class TextPredictionRequest(BaseModel):
    text: str
    context: Optional[list[str]] = None
    speaker: Optional[str] = "UNKNOWN"


class TextPredictionResponse(BaseModel):
    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]
    entropy: float
    text: str
    top_tokens: list[TokenAttributionSchema] = Field(default_factory=list)
    processing_time_ms: float = 0.0


class AudioPredictionResponse(BaseModel):
    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]
    entropy: float
    audio_quality_valid: bool
    quality_metrics: dict[str, float]
    top_prosodic_features: list[AcousticAttributionSchema] = Field(default_factory=list)
    processing_time_ms: float = 0.0


class BatchItemRequest(BaseModel):
    id: Optional[str] = None
    text: Optional[str] = None
    audio_path: Optional[str] = None
    context: Optional[list[str]] = None
    speaker: Optional[str] = "UNKNOWN"


class BatchPredictRequest(BaseModel):
    items: list[BatchItemRequest]


class BatchPredictResponse(BaseModel):
    total_processed: int
    sarcastic_count: int
    non_sarcastic_count: int
    uncertain_count: int
    average_confidence: float
    results: list[MultimodalPredictionResponse]
