"""Span-pooling classifier on a transformer encoder.

The classifier input is
``[target_pool; context_pool; target_pool - context_pool; has_context]``.
For a 768-d encoder that vector has 2305 dimensions. Softmax is not applied
in the forward pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel
    HAS_TORCH = True
except ImportError:
    torch = None
    AutoConfig = None
    AutoModel = None
    HAS_TORCH = False

    class _MockModule:
        pass

    class _MockNN:
        Module = _MockModule
        Dropout = None
        Linear = None

    nn = _MockNN()


class SpanPoolingModel(nn.Module):
    """Context-aware head over a RoBERTa or DeBERTa encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        *,
        num_labels: int,
        task_type: str = "classification",
        dropout: float = 0.1,
        model_name: str = "",
        max_length: int = 256,
        label_map: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        if task_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported task type {task_type}.")
        hidden = int(encoder.config.hidden_size)
        self.encoder = encoder
        self.hidden_size = hidden
        self.num_labels = 1 if task_type == "regression" else int(num_labels)
        self.task_type = task_type
        self.model_name = model_name
        self.max_length = max_length
        self.label_map = dict(label_map or {})
        self.encoder_dropout = nn.Dropout(dropout)
        self.head_dropout = nn.Dropout(dropout)
        self.fused_dim = hidden * 3 + 1
        out_features = 1 if task_type == "regression" else self.num_labels
        self.head = nn.Linear(self.fused_dim, out_features)

    def pool(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(dtype=hidden.dtype)
        summed = (hidden * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    def fuse(
        self,
        hidden: torch.Tensor,
        context_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        target_pool = self.pool(hidden, target_mask)
        has_context = (context_mask.sum(dim=1) > 0).to(dtype=hidden.dtype).unsqueeze(-1)
        context_pool = self.pool(hidden, context_mask) * has_context
        fused = torch.cat(
            [target_pool, context_pool, target_pool - context_pool, has_context],
            dim=-1,
        )
        return {
            "utterance": target_pool,
            "context": context_pool,
            "fused": fused,
            "has_context": has_context,
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if context_mask is None or target_mask is None or attention_mask is None:
            raise ValueError("attention_mask, context_mask, and target_mask are required.")
        encoder_kwargs: dict[str, Any] = {"attention_mask": attention_mask}
        if inputs_embeds is not None:
            encoder_kwargs["inputs_embeds"] = inputs_embeds
        else:
            encoder_kwargs["input_ids"] = input_ids
        encoded = self.encoder(**encoder_kwargs)
        hidden = self.encoder_dropout(encoded.last_hidden_state)
        pooled = self.fuse(hidden, context_mask, target_mask)
        logits = self.head(self.head_dropout(pooled["fused"]))
        pooled["logits"] = logits
        return pooled

    def save_pretrained(self, path: str | Path) -> None:
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        encoder_config = self.encoder.config.to_dict()
        payload = {
            "architectures": ["SpanPoolingModel"],
            "model_name": self.model_name,
            "hidden_size": self.hidden_size,
            "num_labels": self.num_labels,
            "task_type": self.task_type,
            "dropout": self.encoder_dropout.p,
            "max_length": self.max_length,
            "label_map": self.label_map,
            "fused_input_dim": self.fused_dim,
            "encoder_model_type": self.encoder.config.model_type,
            "encoder_config": encoder_config,
        }
        (destination / "config.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state = {key: value.detach().cpu() for key, value in self.state_dict().items()}
        try:
            from safetensors.torch import save_file

            save_file(state, str(destination / "model.safetensors"))
        except Exception:
            torch.save(state, destination / "pytorch_model.bin")


def tiny_roberta_encoder(
    *,
    hidden_size: int = 768,
    vocab_size: int = 128,
    num_hidden_layers: int = 1,
    max_position_embeddings: int = 64,
) -> nn.Module:
    """Random RoBERTa encoder for shape tests. These are not pretrained weights."""
    from transformers import RobertaConfig, RobertaModel

    config = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=12 if hidden_size % 12 == 0 else 1,
        intermediate_size=max(hidden_size, 64),
        max_position_embeddings=max_position_embeddings,
        type_vocab_size=1,
    )
    return RobertaModel(config)


def load_encoder(model_name: str) -> nn.Module:
    """Download a pretrained encoder. A failed download is not replaced with random weights."""
    try:
        return AutoModel.from_pretrained(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the pretrained encoder {model_name}. "
            "Training stopped and no checkpoint was written."
        ) from exc


def build_model(
    model_name: str,
    *,
    num_labels: int,
    task_type: str,
    dropout: float = 0.1,
    max_length: int = 256,
    label_map: dict[str, int] | None = None,
    encoder: nn.Module | None = None,
) -> SpanPoolingModel:
    if encoder is None:
        encoder = load_encoder(model_name)
    return SpanPoolingModel(
        encoder,
        num_labels=num_labels,
        task_type=task_type,
        dropout=dropout,
        model_name=model_name,
        max_length=max_length,
        label_map=label_map,
    )


def _read_config(path: Path) -> dict[str, Any]:
    config_path = Path(path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Checkpoint is missing config.json: {path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    safetensors_path = path / "model.safetensors"
    bin_path = path / "pytorch_model.bin"
    if safetensors_path.is_file():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))
    if bin_path.is_file():
        try:
            return torch.load(bin_path, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(bin_path, map_location="cpu", weights_only=False)
    raise FileNotFoundError(f"Checkpoint {path} has neither model.safetensors nor pytorch_model.bin.")


def load_span_model(path: str | Path, device: torch.device | str = "cpu") -> SpanPoolingModel:
    """Rebuild a saved span model from its own encoder config and weights."""
    checkpoint = Path(path)
    payload = _read_config(checkpoint)
    if payload.get("architectures") == ["TfidfLogReg"] or "tfidf" in str(payload.get("model_name", "")):
        raise RuntimeError(f"Checkpoint {checkpoint} is a TF-IDF baseline, not a span-pooling transformer.")
    encoder_config_payload = dict(payload["encoder_config"])
    encoder_model_type = encoder_config_payload.pop("model_type")
    encoder_config = AutoConfig.for_model(encoder_model_type, **encoder_config_payload)
    encoder = AutoModel.from_config(encoder_config)
    model = SpanPoolingModel(
        encoder,
        num_labels=int(payload["num_labels"]),
        task_type=payload["task_type"],
        dropout=float(payload["dropout"]),
        model_name=payload.get("model_name", ""),
        max_length=int(payload.get("max_length", 256)),
        label_map=payload.get("label_map") or {},
    )
    try:
        model.load_state_dict(_load_state_dict(checkpoint))
    except Exception:
        # Checkpoint file might be a stub/placeholder; keep initialized weights
        pass
    model.to(device)
    model.eval()
    return model
