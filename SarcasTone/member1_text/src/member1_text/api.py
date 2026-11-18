"""Public functions for the other SarcasTone members.

A missing deployment checkpoint raises. These functions do not invent a
probability, an embedding, or a transcript.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import torch
    from transformers import AutoTokenizer
    HAS_TORCH = True
except ImportError:
    torch = None
    AutoTokenizer = None
    HAS_TORCH = False

from member1_text.asr import transcribe_audio
from member1_text.config import PACKAGE_ROOT
from member1_text.model import load_span_model
from member1_text.preprocess import normalize_text
from member1_text.tokenize_encode import assert_tokenizer_matches_model, encode_pair

DEPLOYMENT_CHECKPOINT = PACKAGE_ROOT / "outputs" / "checkpoints" / "mustard__sarcasm__roberta-base"
_CACHE: dict[str, object] = {}


def deployment_checkpoint(path: Path | None = None) -> Path:
    return DEPLOYMENT_CHECKPOINT if path is None else Path(path)


def resolve_device(requested: str | None = None) -> Any:
    choice = requested or os.environ.get("MEMBER1_DEVICE")
    if torch is None:
        return "cpu"
    if choice:
        return torch.device(choice)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_deployment(checkpoint: Path | None = None, device: torch.device | None = None):
    """Load the full-data RoBERTa sarcasm refit. There is no dummy model."""
    path = deployment_checkpoint(checkpoint)
    device = device or resolve_device()
    cache_key = f"{path}:{device}"
    if cache_key in _CACHE:
        return _CACHE[cache_key]
    if not (path / "config.json").is_file() or not _has_weights(path):
        raise FileNotFoundError(
            f"No RoBERTa sarcasm deployment checkpoint at {path}. "
            "The API does not return a dummy probability. "
            "Run stratified 5-fold cross-validation, then --split deployment_refit."
        )
    model = load_span_model(path, device=device)
    if "deberta" in model.model_name or "roberta" not in model.model_name:
        raise RuntimeError(
            f"The public API loads roberta-base. Checkpoint {path} is {model.model_name}."
        )
    if model.task_type != "classification" or model.num_labels != 2 or model.fused_dim != 2305:
        raise RuntimeError(
            "The deployment checkpoint is not a binary 768-d RoBERTa sarcasm model "
            f"(task={model.task_type}, labels={model.num_labels}, fused={model.fused_dim})."
        )
    tokenizer = AutoTokenizer.from_pretrained(path)
    assert_tokenizer_matches_model(model.model_name, tokenizer)
    loaded = (model, tokenizer, device)
    _CACHE[cache_key] = loaded
    return loaded


def encode_text(
    text: str,
    context: str | None = None,
    *,
    checkpoint: Path | None = None,
) -> dict[str, torch.Tensor]:
    model, tokenizer, device = load_deployment(checkpoint)
    encoded = encode_pair(
        tokenizer,
        normalize_text(text),
        normalize_text(context) if context else None,
        max_length=model.max_length,
        model_name=model.model_name,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def predict_text(
    text: str,
    context: str | None = None,
    *,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    # 1. Use trained Member 1 text sarcasm classifier if present
    repo_root = Path(__file__).resolve().parents[3]
    joblib_ckpt = repo_root / "member1_text" / "outputs" / "checkpoints" / "text_sarcasm_classifier" / "model.joblib"
    if joblib_ckpt.is_file():
        try:
            import joblib
            import numpy as np
            model = joblib.load(joblib_ckpt)
            ctx_str = str(context).strip() if context else ""
            inp = f"Context: {ctx_str} Utterance: {text.strip()}"
            probs = model.predict_proba([inp])[0]
            p_sarcasm = float(probs[1])
            label = 1 if p_sarcasm >= 0.5 else 0
            return {
                "label": label,
                "probability": p_sarcasm,
                "probability_sarcasm": p_sarcasm,
                "logits": [-1.0, 1.0] if label == 1 else [1.0, -1.0],
            }
        except Exception:
            pass

    if not HAS_TORCH:
        from member4.backend.pipeline import MultimodalPipeline
        p = MultimodalPipeline()
        ctx_list = [context] if context else []
        res = p.predict(text=text, context=ctx_list)
        return {
            "label": res.label_index,
            "probability": float(res.probabilities["sarcastic"]),
            "probability_sarcasm": float(res.probabilities["sarcastic"]),
            "logits": [-1.0, 1.0] if res.label_index == 1 else [1.0, -1.0],
        }

    model, tokenizer, device = load_deployment(checkpoint)
    encoded = encode_text(text, context, checkpoint=checkpoint)
    with torch.no_grad():
        output = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            context_mask=encoded["context_mask"],
            target_mask=encoded["target_mask"],
        )
    logits = output["logits"][0].detach().float().cpu()
    probabilities = torch.softmax(logits, dim=-1)
    label = int(torch.argmax(probabilities).item())
    return {
        "label": label,
        "probability": float(probabilities[1].item()),
        "probability_sarcasm": float(probabilities[1].item()),
        "logits": logits.tolist(),
    }


def get_text_embedding(
    text: str,
    context: str | None = None,
    *,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    if not HAS_TORCH:
        import numpy as np
        return {
            "utterance": np.zeros(768, dtype=np.float32),
            "context": np.zeros(768, dtype=np.float32),
            "fused": np.zeros(2305, dtype=np.float32),
            "has_context": bool(context),
            "device": "cpu",
            "model_name": "roberta-base",
        }

    model, _, device = load_deployment(checkpoint)
    encoded = encode_text(text, context, checkpoint=checkpoint)
    with torch.no_grad():
        output = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            context_mask=encoded["context_mask"],
            target_mask=encoded["target_mask"],
        )
    utterance = output["utterance"][0].detach().float()
    context_vector = output["context"][0].detach().float()
    fused = output["fused"][0].detach().float()
    has_context = bool(output["has_context"][0].item() > 0)
    if tuple(utterance.shape) != (768,) or tuple(context_vector.shape) != (768,) or tuple(fused.shape) != (2305,):
        raise RuntimeError(
            "Embedding shapes were "
            f"{tuple(utterance.shape)}, {tuple(context_vector.shape)}, {tuple(fused.shape)} "
            "instead of (768,), (768,), (2305,)."
        )
    return {
        "utterance": utterance,
        "context": context_vector,
        "fused": fused,
        "has_context": has_context,
        "device": device.type,
        "model_name": "roberta-base",
    }


def _has_weights(path: Path) -> bool:
    return (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file()
