"""FastAPI Application for SarcasTone Multimodal Sarcasm Detection.

Exposes RESTful endpoints:
- POST /predict: Full multimodal audio + text prediction
- POST /predict/text: Text-only sarcasm classification
- POST /predict/audio: Audio-only sarcasm analysis
- POST /batch-predict: Batch prediction over multiple samples
- GET /health: System health and component status
- GET /model-info: Model architecture & contract metadata
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from member4.backend.pipeline import MultimodalPipeline
from member4.backend.schemas import (
    AudioPredictionResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    MultimodalPredictionResponse,
    TextPredictionRequest,
    TextPredictionResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sarcastone.api")

# Configuration constants
MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}

# Initialize FastAPI App
app = FastAPI(
    title="SarcasTone Multimodal Sarcasm Detection API",
    description="Production REST API serving multimodal sarcasm detection using textual semantics and speech prosody.",
    version="1.0.0",
)

# CORS middleware for seamless Streamlit/React frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[MultimodalPipeline] = None


def get_pipeline() -> MultimodalPipeline:
    """Retrieve or lazily initialize the multimodal pipeline."""
    global pipeline
    if pipeline is None:
        try:
            pipeline = MultimodalPipeline()
        except Exception as e:
            log.error("Lazy pipeline init failed: %s", e)
            pipeline = MultimodalPipeline()
    return pipeline


@app.on_event("startup")
def startup_event():
    """Load model checkpoints and initialize pipeline on application startup."""
    get_pipeline()


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health_check() -> HealthResponse:
    """Return health status of the API, runtime device, and active modalities."""
    pipe = get_pipeline()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        model_loaded=pipe is not None,
        device=pipe.device if pipe else "cpu",
        active_modalities=["text", "audio", "prosody", "fusion", "xai"],
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Diagnostics"])
def model_info() -> ModelInfoResponse:
    """Return architecture specifications, layer dimensions, and interface contract."""
    return ModelInfoResponse()


@app.post("/predict", response_model=MultimodalPredictionResponse, tags=["Inference"])
async def predict_multimodal_endpoint(
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    context: Optional[str] = Form(None),
    speaker: Optional[str] = Form("UNKNOWN"),
) -> MultimodalPredictionResponse:
    """Multimodal prediction supporting both file upload and transcript."""
    pipe = get_pipeline()

    if audio is None and (text is None or not text.strip()):
        raise HTTPException(
            status_code=400,
            detail="At least one modality (audio file or text transcript) must be provided.",
        )

    temp_audio_path: Optional[Path] = None
    try:
        if audio is not None:
            # Validate extension
            filename = audio.filename or "uploaded_audio.wav"
            ext = Path(filename).suffix.lower()
            if ext not in ALLOWED_AUDIO_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported audio format '{ext}'. Allowed: {list(ALLOWED_AUDIO_EXTENSIONS)}",
                )

            # Write file and check size limit
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                shutil.copyfileobj(audio.file, tmp)
                temp_audio_path = Path(tmp.name)

            if temp_audio_path.stat().st_size > MAX_UPLOAD_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"File exceeds maximum allowable size of {MAX_UPLOAD_SIZE_BYTES // (1024*1024)} MB.",
                )

        # Parse context list if passed as JSON string
        parsed_context: Optional[list[str]] = None
        if context:
            try:
                loaded = json.loads(context)
                if isinstance(loaded, list):
                    parsed_context = [str(x) for x in loaded]
                else:
                    parsed_context = [str(context)]
            except Exception:
                parsed_context = [str(context)]

        result = pipeline.predict(
            audio_path=temp_audio_path,
            text=text,
            context=parsed_context,
            speaker=speaker,
        )

        log.info(
            "Predict [Multimodal] -> %s (conf: %.2f%%, text_w: %.2f, psid: %.2f)",
            result.label,
            result.confidence * 100.0,
            result.text_weight,
            result.psid_dissonance_score,
        )
        return MultimodalPredictionResponse(**result.to_dict())

    finally:
        # Clean up temporary audio file
        if temp_audio_path and temp_audio_path.is_file():
            try:
                temp_audio_path.unlink()
            except Exception:
                pass


@app.post("/predict/text", response_model=TextPredictionResponse, tags=["Inference"])
def predict_text_endpoint(req: TextPredictionRequest) -> TextPredictionResponse:
    """Dedicated endpoint for text-only sarcasm detection."""
    pipe = get_pipeline()
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text field cannot be empty.")

    t0 = time.perf_counter()
    res = pipe.predict(audio_path=None, text=req.text, context=req.context, speaker=req.speaker)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    return TextPredictionResponse(
        label=res.label,
        label_index=res.label_index,
        confidence=res.confidence,
        probabilities=res.probabilities,
        entropy=res.entropy,
        text=req.text,
        top_tokens=[
            {"token": t.token, "score": t.score, "position": t.position}
            for t in res.explanation.top_text_tokens
        ],
        processing_time_ms=round(dt_ms, 2),
    )


@app.post("/predict/audio", response_model=AudioPredictionResponse, tags=["Inference"])
async def predict_audio_endpoint(
    audio: UploadFile = File(...),
    speaker: Optional[str] = Form("UNKNOWN"),
) -> AudioPredictionResponse:
    """Dedicated endpoint for audio-only sarcasm and prosody analysis."""
    pipe = get_pipeline()
    ext = Path(audio.filename or "sample.wav").suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {list(ALLOWED_AUDIO_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(audio.file, tmp)
        temp_audio_path = Path(tmp.name)

    try:
        t0 = time.perf_counter()
        res = pipe.predict(audio_path=temp_audio_path, text=None, speaker=speaker)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        return AudioPredictionResponse(
            label=res.label,
            label_index=res.label_index,
            confidence=res.confidence,
            probabilities=res.probabilities,
            entropy=res.entropy,
            audio_quality_valid=res.audio_available,
            quality_metrics={"is_valid": 1.0 if res.audio_available else 0.0, "clipping_ratio": 0.0, "silence_ratio": 0.0},
            top_prosodic_features=[
                {"feature_name": a.feature_name, "importance": a.importance, "value": a.value, "direction": a.direction}
                for a in res.explanation.top_acoustic_features
            ],
            processing_time_ms=round(dt_ms, 2),
        )
    finally:
        if temp_audio_path.is_file():
            try:
                temp_audio_path.unlink()
            except Exception:
                pass


@app.post("/batch-predict", response_model=BatchPredictResponse, tags=["Inference"])
def batch_predict_endpoint(req: BatchPredictRequest) -> BatchPredictResponse:
    """Process a batch of text/audio samples and produce aggregated analytics."""
    pipe = get_pipeline()
    if not req.items:
        raise HTTPException(status_code=400, detail="Batch request contains no items.")

    results: list[MultimodalPredictionResponse] = []
    sarcastic_cnt = 0
    non_sarcastic_cnt = 0
    uncertain_cnt = 0
    conf_sum = 0.0

    for item in req.items:
        res = pipe.predict(
            audio_path=item.audio_path,
            text=item.text,
            context=item.context,
            speaker=item.speaker or "UNKNOWN",
        )
        if res.label == "sarcastic":
            sarcastic_cnt += 1
        else:
            non_sarcastic_cnt += 1
        if res.uncertain:
            uncertain_cnt += 1
        conf_sum += res.confidence
        results.append(MultimodalPredictionResponse(**res.to_dict()))

    avg_conf = conf_sum / len(req.items) if req.items else 0.0

    return BatchPredictResponse(
        total_processed=len(results),
        sarcastic_count=sarcastic_cnt,
        non_sarcastic_count=non_sarcastic_cnt,
        uncertain_count=uncertain_cnt,
        average_confidence=round(avg_conf, 4),
        results=results,
    )
