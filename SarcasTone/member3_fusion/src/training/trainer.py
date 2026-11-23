"""Training loop for SarcasTone Fusion Model.

Responsibilities
----------------
* Build data splits from member2 handoff + MUStARD JSON (+ optional MUStARD++).
* Build the fusion model from config.
* Run AdamW + cosine LR schedule with linear warmup.
* Gradient clipping + early stopping.
* Mixed-precision training (when CUDA available).
* Progressive encoder unfreezing schedule.
* Checkpoint saving/resumption.
* Per-epoch metric logging (accuracy, macro-F1, weighted-F1).
* Full reproducibility via seed setting.
* Temperature calibration fitted on validation logits after training.
* Modality availability validated before training starts.

Encoder fine-tuning policy
---------------------------
Epoch 0 → freeze_encoders_epochs:   all encoder parameters frozen.
Epoch unfreeze_text_epoch:           text encoder unfrozen (lower LR).
Epoch unfreeze_audio_epoch:          audio projection unfrozen.

For MUStARD (N≈470 train examples), full encoder fine-tuning is too aggressive
without very low LR. The progressive schedule prevents catastrophic forgetting.

Fixes applied vs. the original
-------------------------------
* Scheduler total_steps calculated from actual DataLoader length, not a hardcoded 50.
* Calibration (temperature fitting) is always run after training, on val logits.
* validate_modality_availability() called before training starts.
* Cached text embeddings (text_embedding tensor) are used when no live encoder.
* Experiment override flags (psid_enabled, gating_enabled, fusion_strategy)
  from the config model block are always forwarded to build_model.
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

import sys
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "member3_fusion") not in sys.path:
    sys.path.insert(0, str(_REPO / "member3_fusion"))

from shared.constants import SARCASM_LABELS
from shared.metrics import compute_classification_metrics

from src.training.dataset import (
    FusionDataset,
    FusionRecord,
    collate_fn,
    load_mustard_fusion_records,
    validate_modality_availability,
    ACOUSTIC_FEATURE_COLS,
)
from src.models.fusion_model import SarcasToneFusionModel, build_fusion_model
from src.uncertainty.calibration import fit_temperature

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# LR schedule with linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_warmup_cosine_schedule(
    optimiser: AdamW,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    import math

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimiser, lr_lambda)


# ---------------------------------------------------------------------------
# Class weights
# ---------------------------------------------------------------------------

def compute_class_weights(
    records: list[FusionRecord], device: torch.device
) -> torch.Tensor:
    labels = np.array([r.label for r in records])
    n_total  = len(labels)
    n_classes = 2
    weights  = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        count = (labels == c).sum()
        weights[c] = n_total / (n_classes * max(count, 1))
    log.info(
        "Class weights: non-sarcastic=%.4f, sarcastic=%.4f",
        weights[0], weights[1],
    )
    return torch.tensor(weights, device=device)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def run_batch(
    model: SarcasToneFusionModel,
    batch: dict,
    device: torch.device,
    text_encoder=None,
) -> dict[str, torch.Tensor]:
    """Move batch to device and run model forward.

    Text encoding priority:
      1. Live text_encoder (member1 SpanPoolingModel), if available.
      2. Cached text_embedding tensors from the dataset batch.
      3. Zero embeddings (text-unavailable ablation / training without text).
    """
    tensor_keys = [
        "audio_embedding", "audio_available",
        "prosody", "acoustic_full", "quality", "quality_ok",
        "text_embedding", "text_available", "context_embedding",
        "has_context", "label",
    ]
    device_batch = {k: batch[k].to(device) for k in tensor_keys if k in batch}
    device_batch["texts"]       = batch["texts"]
    device_batch["contexts"]    = batch["contexts"]
    device_batch["example_ids"] = batch["example_ids"]
    device_batch["speakers"]    = batch["speakers"]

    B = device_batch["audio_embedding"].size(0)

    text_emb = None
    ctx_emb  = None
    has_ctx  = None

    if not model.audio_available_default:
        for key in ("audio_embedding", "prosody", "acoustic_full", "quality"):
            device_batch[key] = torch.zeros_like(device_batch[key])
        device_batch["audio_available"] = torch.zeros_like(device_batch["audio_available"])
        device_batch["quality_ok"] = torch.zeros_like(device_batch["quality_ok"])

    # Priority 1: live text encoder, when explicitly selected by the caller.
    if model.text_available_flag and text_encoder is not None and text_encoder.is_available:
        try:
            te_out = text_encoder(
                device_batch["texts"],
                device_batch["contexts"],
            )
            text_emb = te_out["utterance"]   # [B, TEXT_HIDDEN]
            ctx_emb  = te_out["context"]     # [B, TEXT_HIDDEN]
            has_ctx  = te_out["has_context"] # [B] bool
        except Exception as e:
            log.warning("Text encoder failed: %s; falling back to cached embeddings", e)

    # Priority 2: aligned cached Member 1 utterance and context embeddings.
    if model.text_available_flag and text_emb is None and "text_embedding" in device_batch:
        text_avail_mask = device_batch["text_available"]   # [B] bool
        if text_avail_mask.any():
            text_emb = device_batch["text_embedding"]      # [B, TEXT_HIDDEN]
            text_emb = text_emb * text_avail_mask.float().unsqueeze(-1)
            ctx_emb = device_batch["context_embedding"]
            has_ctx = device_batch["has_context"] & text_avail_mask
            ctx_emb = ctx_emb * has_ctx.float().unsqueeze(-1)

    return model(
        device_batch,
        text_embeddings=text_emb,
        context_embeddings=ctx_emb,
        has_context=has_ctx,
    )


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: SarcasToneFusionModel,
    loader: DataLoader,
    optimiser: AdamW,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    scheduler=None,
    grad_clip: float = 1.0,
    scaler=None,
    text_encoder=None,
    epoch: int = 0,
    log_every: int = 20,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []
    all_probs  = []

    for step, batch in enumerate(loader):
        optimiser.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                out  = run_batch(model, batch, device, text_encoder)
                loss = criterion(out["logits"], batch["label"].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimiser)
            scaler.update()
        else:
            out  = run_batch(model, batch, device, text_encoder)
            loss = criterion(out["logits"], batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        probs  = out["probabilities"].detach().cpu().numpy()
        preds  = probs.argmax(axis=1)
        labels = batch["label"].numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_probs.append(probs)

        if (step + 1) % log_every == 0:
            log.info(
                "  Epoch %d step %d/%d  loss=%.4f",
                epoch, step + 1, len(loader), loss.item(),
            )

    all_probs_np = np.vstack(all_probs)
    metrics = compute_classification_metrics(np.array(all_labels), all_probs_np)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


@torch.no_grad()
def eval_epoch(
    model: SarcasToneFusionModel,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    text_encoder=None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_probs  = []
    all_labels = []

    for batch in loader:
        out  = run_batch(model, batch, device, text_encoder)
        loss = criterion(out["logits"], batch["label"].to(device))
        total_loss += loss.item()
        probs = out["probabilities"].detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(batch["label"].numpy().tolist())

    all_probs_np = np.vstack(all_probs)
    metrics = compute_classification_metrics(np.array(all_labels), all_probs_np)
    metrics["loss"] = total_loss / max(len(loader), 1)
    return metrics


# ---------------------------------------------------------------------------
# Main Trainer class
# ---------------------------------------------------------------------------

class FusionTrainer:
    """Orchestrates the full training + evaluation loop."""

    def __init__(self, cfg: dict) -> None:
        self.cfg  = cfg
        self.seed = cfg.get("project", {}).get("seed", 42)
        set_seed(self.seed)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        log.info("Device: %s", self.device)

        self.t_cfg = cfg.get("training", {})
        self.d_cfg = cfg.get("data", {})
        self.o_cfg = cfg.get("output", {})

        self.checkpoint_dir = Path(self.o_cfg.get(
            "checkpoint_dir", "member3_fusion/checkpoints"
        ))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list] = {"train": [], "val": [], "test": []}

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #

    def load_data(self) -> tuple[dict[str, FusionDataset], dict]:
        splits, acoustic_dim = load_mustard_fusion_records(
            mustard_json_path=self.d_cfg["mustard_json"],
            handoff_csv_path=self.d_cfg["audio_handoff_csv"],
            quality_csv_path=self.d_cfg["audio_quality_csv"],
            predictions_csv_path=self.d_cfg["audio_predictions_csv"],
            wav2vec2_npz_path=self.d_cfg.get("wav2vec2_embeddings_npz"),
            mustard_pp_csv_path=self.d_cfg.get("mustard_pp_csv"),
            text_embeddings_npz_path=self.d_cfg.get("text_embeddings_npz"),
            prosody_csv_path=self.d_cfg.get("prosody_csv"),
            val_ratio=self.d_cfg.get("val_ratio", 0.15),
            test_ratio=self.d_cfg.get("test_ratio", 0.15),
            seed=self.seed,
            split_strategy=self.d_cfg.get("split_strategy", "speaker"),
            dedup_mustard_family=self.d_cfg.get("dedup_mustard_family", True),
            alignment_report_path=(
                Path(self.o_cfg["comparison_dir"]) / "alignment_report.json"
                if self.o_cfg.get("comparison_dir")
                else None
            ),
            max_alignment_drop_ratio=self.d_cfg.get("max_alignment_drop_ratio", 0.10),
        )
        datasets = {
            split: FusionDataset(records, acoustic_dim=acoustic_dim)
            for split, records in splits.items()
        }
        # Validate modality availability — emit warnings early
        validate_modality_availability(datasets)

        self._train_records  = splits["train"]
        self._acoustic_dim   = acoustic_dim
        self._splits         = splits
        return datasets, splits

    def build_loaders(
        self, datasets: dict
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        bs = self.t_cfg.get("batch_size", 16)
        train_loader = DataLoader(
            datasets["train"], batch_size=bs, shuffle=True,
            collate_fn=collate_fn, drop_last=False,
        )
        val_loader = DataLoader(
            datasets["val"], batch_size=bs, shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            datasets["test"], batch_size=bs, shuffle=False,
            collate_fn=collate_fn,
        )
        return train_loader, val_loader, test_loader

    # ------------------------------------------------------------------ #
    # Build model and optimiser
    # ------------------------------------------------------------------ #

    def build_model(self, acoustic_dim: int | None = None) -> SarcasToneFusionModel:
        cfg_copy = dict(self.cfg)
        if acoustic_dim is not None:
            cfg_copy.setdefault("interface", {})["acoustic_feature_dim"] = acoustic_dim
        model = build_fusion_model(cfg_copy)
        model.to(self.device)
        log.info(
            "Model parameters: %d total, %d trainable",
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad),
        )
        log.info(
            "Fusion strategy: %s  |  PSID: %s  |  Gating: %s",
            self.cfg.get("model", {}).get("fusion_strategy", "cross_attention"),
            self.cfg.get("model", {}).get("psid_enabled", True),
            self.cfg.get("model", {}).get("gating_enabled", True),
        )
        return model

    def build_optimiser(
        self,
        model: SarcasToneFusionModel,
        num_train_batches: int,
    ) -> tuple[AdamW, LambdaLR]:
        """Build AdamW + cosine LR schedule.

        num_train_batches is the actual DataLoader length (not an estimate).
        """
        lr          = self.t_cfg.get("learning_rate", 2e-4)
        wd          = self.t_cfg.get("weight_decay", 1e-2)
        warmup_ratio = self.t_cfg.get("warmup_ratio", 0.1)
        epochs      = self.t_cfg.get("epochs", 30)

        # --- Compute actual total steps from real batch count ---
        total_steps  = epochs * num_train_batches
        warmup_steps = int(total_steps * warmup_ratio)
        log.info(
            "LR schedule: %d epochs × %d batches = %d total steps, "
            "%d warmup steps",
            epochs, num_train_batches, total_steps, warmup_steps,
        )

        optimiser = AdamW(
            model.parameters(), lr=lr, weight_decay=wd, eps=1e-8
        )
        scheduler = get_warmup_cosine_schedule(optimiser, warmup_steps, total_steps)
        return optimiser, scheduler

    # ------------------------------------------------------------------ #
    # Encoder unfreezing schedule
    # ------------------------------------------------------------------ #

    def _apply_unfreeze_schedule(
        self,
        model: SarcasToneFusionModel,
        epoch: int,
        text_encoder=None,
    ) -> None:
        unfreeze_text  = self.t_cfg.get("unfreeze_text_epoch", 5)
        unfreeze_audio = self.t_cfg.get("unfreeze_audio_epoch", 10)
        if text_encoder is not None and epoch == unfreeze_text:
            log.info("Epoch %d: unfreezing text encoder", epoch)
            text_encoder.set_trainable(True)
        if epoch == unfreeze_audio:
            log.info("Epoch %d: unfreezing audio projection", epoch)
            for param in model.audio_proj.parameters():
                param.requires_grad_(True)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #

    def train(
        self,
        experiment_name: str = "full_model",
        text_encoder=None,
    ) -> dict:
        log.info("=" * 60)
        log.info("Starting experiment: %s", experiment_name)
        log.info("Seed: %d", self.seed)
        log.info("=" * 60)

        # Data
        datasets, splits = self.load_data()
        acoustic_dim = self._acoustic_dim
        train_loader, val_loader, test_loader = self.build_loaders(datasets)

        # Model
        model = self.build_model(acoustic_dim=acoustic_dim)

        # Criterion
        use_weights = self.t_cfg.get("use_class_weights", True)
        if use_weights:
            class_weights = compute_class_weights(splits["train"], self.device)
        else:
            class_weights = None

        label_smoothing = self.t_cfg.get("label_smoothing", 0.05)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

        # Optimiser — use ACTUAL DataLoader length, not an estimate
        optimiser, scheduler = self.build_optimiser(
            model, num_train_batches=len(train_loader)
        )

        # Mixed precision
        use_amp = self.t_cfg.get("mixed_precision", False) and torch.cuda.is_available()
        scaler  = torch.cuda.amp.GradScaler() if use_amp else None

        # Resume from checkpoint
        resume_path = self.t_cfg.get("resume_from")
        start_epoch = 0
        if resume_path and Path(resume_path).is_file():
            log.info("Resuming from checkpoint: %s", resume_path)
            ckpt = torch.load(resume_path, map_location=self.device)
            model.load_state_dict(ckpt["model_state"])
            optimiser.load_state_dict(ckpt["optimiser_state"])
            start_epoch = ckpt.get("epoch", 0) + 1

        # Training state
        epochs      = self.t_cfg.get("epochs", 30)
        patience    = self.t_cfg.get("early_stopping_patience", 5)
        best_metric = -float("inf")
        best_epoch  = -1
        no_improve  = 0
        exp_dir = Path(
            self.o_cfg.get("experiments_dir", "member3_fusion/experiments")
        ) / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        monitor_key = self.t_cfg.get("early_stopping_metric", "macro_f1")

        for epoch in range(start_epoch, epochs):
            epoch_start = time.time()
            self._apply_unfreeze_schedule(model, epoch, text_encoder)

            train_metrics = train_epoch(
                model, train_loader, optimiser, criterion, self.device,
                scheduler=scheduler,
                grad_clip=self.t_cfg.get("gradient_clip", 1.0),
                scaler=scaler,
                text_encoder=text_encoder,
                epoch=epoch,
                log_every=self.cfg.get("logging", {}).get("log_every_n_steps", 20),
            )
            val_metrics = eval_epoch(
                model, val_loader, criterion, self.device, text_encoder
            )

            self.history["train"].append(train_metrics)
            self.history["val"].append(val_metrics)

            epoch_time = time.time() - epoch_start
            current    = val_metrics.get(monitor_key, 0.0)

            log.info(
                "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
                "val_acc=%.4f  val_macro_f1=%.4f  [%.1fs]",
                epoch + 1, epochs,
                train_metrics["loss"], val_metrics["loss"],
                val_metrics["accuracy"], val_metrics["macro_f1"],
                epoch_time,
            )

            # Checkpoint on improvement
            if current > best_metric:
                best_metric = current
                best_epoch  = epoch
                no_improve  = 0
                ckpt_path   = exp_dir / "best_model.pt"
                torch.save(
                    {
                        "epoch":           epoch,
                        "model_state":     model.state_dict(),
                        "optimiser_state": optimiser.state_dict(),
                        "val_metrics":     val_metrics,
                        "train_metrics":   train_metrics,
                        "config":          self.cfg,
                    },
                    ckpt_path,
                )
                model.save(exp_dir / "checkpoint")
                log.info("  ✓ New best val_%s=%.4f saved", monitor_key, best_metric)
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info(
                        "Early stopping at epoch %d (best was epoch %d)",
                        epoch + 1, best_epoch + 1,
                    )
                    break

        # ------------------------------------------------------------------ #
        # Post-training: temperature calibration on validation logits
        # ------------------------------------------------------------------ #
        log.info("Fitting temperature calibration on validation set...")
        try:
            fitted_T = fit_temperature(
                model=model,
                val_loader=val_loader,
                device=self.device,
            )
            log.info("Calibration temperature fitted: T = %.4f", fitted_T)
            # Persist calibrated checkpoint
            model.save(exp_dir / "checkpoint")
        except Exception as exc:
            log.warning("Temperature calibration failed: %s", exc)

        # ------------------------------------------------------------------ #
        # Load best checkpoint and evaluate test set
        # ------------------------------------------------------------------ #
        best_ckpt_path = exp_dir / "best_model.pt"
        if best_ckpt_path.is_file():
            log.info("Loading best checkpoint for test evaluation...")
            best_ckpt = torch.load(best_ckpt_path, map_location=self.device)
            model.load_state_dict(best_ckpt["model_state"])
        else:
            log.warning("No best_model.pt found; evaluating with current weights")

        test_metrics = eval_epoch(
            model, test_loader, criterion, self.device, text_encoder
        )
        self.history["test"] = test_metrics

        log.info(
            "Test  acc=%.4f  macro_f1=%.4f  roc_auc=%s",
            test_metrics["accuracy"],
            test_metrics["macro_f1"],
            test_metrics.get("roc_auc"),
        )

        # Save final model and history
        model.save(self.checkpoint_dir / experiment_name)
        (exp_dir / "history.json").write_text(
            json.dumps(self.history, indent=2, default=str), encoding="utf-8"
        )
        (exp_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2, default=str), encoding="utf-8"
        )

        return {
            "best_epoch":       best_epoch,
            "best_val_metric":  best_metric,
            "test_metrics":     test_metrics,
            "experiment_dir":   str(exp_dir),
        }
