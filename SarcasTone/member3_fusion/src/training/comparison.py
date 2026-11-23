"""Reproducible, speaker-aware text/audio comparison experiments."""
from __future__ import annotations

import copy
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.metrics import compute_classification_metrics
from shared.splits import speaker_group
from src.training.dataset import FusionRecord


EXPERIMENTS = ("text_only", "audio_only", "direct_concat", "psid_concat")


class ComparisonClassifier(nn.Module):
    """One classifier architecture with optional learned normalized PSID."""

    def __init__(
        self,
        mode: str,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int = 128,
        psid_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if mode not in EXPERIMENTS:
            raise ValueError(f"Unknown comparison mode: {mode}")
        self.mode = mode
        self.text_projection = (
            nn.Linear(text_dim, psid_dim, bias=False) if mode == "psid_concat" else None
        )
        self.audio_projection = (
            nn.Linear(audio_dim, psid_dim, bias=False) if mode == "psid_concat" else None
        )
        classifier_dim = {
            "text_only": text_dim,
            "audio_only": audio_dim,
            "direct_concat": text_dim + audio_dim,
            "psid_concat": text_dim + audio_dim + psid_dim,
        }[mode]
        self.classifier = nn.Sequential(
            nn.Linear(classifier_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, text: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        if self.mode == "text_only":
            features = text
        elif self.mode == "audio_only":
            features = audio
        elif self.mode == "direct_concat":
            features = torch.cat((text, audio), dim=-1)
        else:
            assert self.text_projection is not None
            assert self.audio_projection is not None
            z_text = F.normalize(self.text_projection(text), p=2, dim=-1)
            z_audio = F.normalize(self.audio_projection(audio), p=2, dim=-1)
            mismatch = torch.abs(z_text - z_audio)
            features = torch.cat((text, audio, mismatch), dim=-1)
        return self.classifier(features)


def build_representations(
    records: list[FusionRecord],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create explicit context-aware text and quality-aware audio vectors."""
    text_rows: list[np.ndarray] = []
    audio_rows: list[np.ndarray] = []
    labels: list[int] = []
    reliable: list[bool] = []
    for record in records:
        if record.text_embedding is None or record.context_embedding is None:
            raise ValueError(f"Missing cached text/context embedding for {record.example_id}")
        if record.audio_embedding is None:
            raise ValueError(f"Missing Wav2Vec2 embedding for {record.example_id}")
        text = np.concatenate((
            record.text_embedding,
            record.context_embedding * float(record.has_context),
            np.asarray([float(record.has_context)], dtype=np.float32),
        ))
        audio_is_reliable = bool(record.audio_available and record.quality_ok)
        audio_core = np.concatenate((
            record.audio_embedding,
            record.acoustic_features_full,
            record.prosody_vector,
        )).astype(np.float32)
        audio_core *= float(audio_is_reliable)
        audio = np.concatenate((
            audio_core,
            record.quality_vector,
            np.asarray([float(audio_is_reliable)], dtype=np.float32),
        ))
        text_rows.append(text.astype(np.float32))
        audio_rows.append(audio.astype(np.float32))
        labels.append(record.label)
        reliable.append(audio_is_reliable)
    return (
        np.stack(text_rows),
        np.stack(audio_rows),
        np.asarray(labels, dtype=np.int64),
        np.asarray(reliable, dtype=bool),
    )


def _standardize(
    train: np.ndarray,
    arrays: list[np.ndarray],
) -> tuple[list[np.ndarray], dict[str, list[float]]]:
    mean = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = train.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    transformed = [(array - mean) / scale for array in arrays]
    return transformed, {"mean": mean.tolist(), "scale": scale.tolist()}


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(
    model: ComparisonClassifier,
    text: torch.Tensor,
    audio: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(text, audio), dim=-1).cpu().numpy()
    metrics = compute_classification_metrics(labels.cpu().numpy(), probabilities)
    return metrics, probabilities


def _bootstrap_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    scores = {"accuracy": [], "macro_f1": []}
    for _ in range(n_bootstrap):
        indexes = rng.integers(0, len(labels), size=len(labels))
        metrics = compute_classification_metrics(labels[indexes], probabilities[indexes])
        for key in scores:
            scores[key].append(metrics[key])
    return {
        key: {
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "bootstrap_mean": float(np.mean(values)),
            "n_bootstrap": n_bootstrap,
            "seed": seed,
        }
        for key, values in scores.items()
    }


def _paired_bootstrap(
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    deltas: dict[str, list[float]] = {"accuracy": [], "macro_f1": []}
    for _ in range(n_bootstrap):
        indexes = rng.integers(0, len(labels), size=len(labels))
        first_metrics = compute_classification_metrics(labels[indexes], first[indexes])
        second_metrics = compute_classification_metrics(labels[indexes], second[indexes])
        for key in deltas:
            deltas[key].append(second_metrics[key] - first_metrics[key])
    output: dict[str, Any] = {"n_bootstrap": n_bootstrap, "seed": seed}
    for key, values in deltas.items():
        values_array = np.asarray(values)
        output[key] = {
            "delta_mean": float(values_array.mean()),
            "delta_ci_95": [
                float(np.quantile(values_array, 0.025)),
                float(np.quantile(values_array, 0.975)),
            ],
            "two_sided_bootstrap_p": float(
                min(1.0, 2 * min(np.mean(values_array <= 0), np.mean(values_array >= 0)))
            ),
        }
    return output


def run_comparison(
    splits: dict[str, list[FusionRecord]],
    output_dir: Path | str,
    *,
    evaluation_status: dict[str, Any] | None = None,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-2,
    patience: int = 6,
    hidden_dim: int = 128,
    psid_dim: int = 128,
    dropout: float = 0.2,
    n_bootstrap: int = 500,
) -> dict[str, Any]:
    """Train and evaluate the four required models on identical split IDs."""
    if not splits["train"] or not splits["val"] or not splits["test"]:
        raise ValueError("Train, validation, and test splits must all be non-empty")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    split_data = {name: build_representations(rows) for name, rows in splits.items()}
    text_scaler_input = split_data["train"][0]
    audio_scaler_input = split_data["train"][1]
    text_arrays, text_scaler = _standardize(
        text_scaler_input, [split_data[name][0] for name in ("train", "val", "test")]
    )
    audio_arrays, audio_scaler = _standardize(
        audio_scaler_input, [split_data[name][1] for name in ("train", "val", "test")]
    )

    reliable_by_split = [split_data[name][3] for name in ("train", "val", "test")]
    # Keep unavailable/unreliable waveform and feature vectors zero after scaling.
    audio_base_width = 768 + 116 + 32
    for array, masks in zip(audio_arrays, reliable_by_split):
        array[~masks, :audio_base_width] = 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for index, name in enumerate(("train", "val", "test")):
        tensors[name] = (
            torch.as_tensor(text_arrays[index], dtype=torch.float32, device=device),
            torch.as_tensor(audio_arrays[index], dtype=torch.float32, device=device),
            torch.as_tensor(split_data[name][2], dtype=torch.long, device=device),
        )

    config = {
        "seed": seed,
        "split_strategy": "speaker-aware; IDs assigned by shared.splits.assign_speaker_splits",
        "experiments": list(EXPERIMENTS),
        "text_representation": (
            "[cached utterance 768; cached context 768 masked by has_context; has_context 1]"
        ),
        "audio_representation": (
            "[Wav2Vec2 768; acoustic 116; prosody 32; quality 6; audio_reliable 1]"
        ),
        "audio_reliability_rule": (
            "Member 2 duration_in_range >= 0.5, silence_ratio < 0.95, rms > 1e-8"
        ),
        "psid": {
            "z_t": "normalize(W_t h_t, p=2)",
            "z_a": "normalize(W_a h_a, p=2)",
            "mismatch": "|z_t - z_a|",
            "classifier_input": "[h_t; h_a; mismatch]",
            "projection_status": (
                "W_t and W_a are trainable linear projections learned only in this "
                "sarcasm objective; they are not pretrained sentiment/emotion features."
            ),
            "projection_dim": psid_dim,
        },
        "dimensions": {
            "text": int(text_arrays[0].shape[1]),
            "audio": int(audio_arrays[0].shape[1]),
            "acoustic": 116,
            "prosody": 32,
            "quality": 6,
        },
        "normalization": {
            "method": "train-split mean/std; standard deviations below 1e-6 set to 1",
            "text": text_scaler,
            "audio": audio_scaler,
            "fit_split": "train",
        },
        "training": {
            "epochs_max": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "early_stopping": "validation macro-F1",
            "patience": patience,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "loss": "class-weighted cross entropy",
            "device": str(device),
        },
        "evaluation_status": evaluation_status or {
            "status": "unverified_upstream_text_provenance",
            "warning": (
                "The cached Member 1 embedding archive does not identify its source "
                "checkpoint or training IDs. Do not interpret these scores as "
                "leakage-safe generalization results."
            ),
        },
    }
    (output / "comparison_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    (output / "split_ids.json").write_text(json.dumps({
        name: [
            {
                "utterance_id": row.example_id,
                "dataset": row.dataset,
                "speaker": row.speaker,
                "show": row.show,
                "speaker_group": speaker_group(row),
                "label": row.label,
            }
            for row in rows
        ]
        for name, rows in splits.items()
    }, indent=2), encoding="utf-8")

    y_train = split_data["train"][2]
    class_counts = np.bincount(y_train, minlength=2)
    class_weights = len(y_train) / (2 * np.maximum(class_counts, 1))
    weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    train_text, train_audio, train_labels = tensors["train"]
    val_text, val_audio, val_labels = tensors["val"]
    test_text, test_audio, test_labels = tensors["test"]

    results: dict[str, Any] = {}
    test_probabilities: dict[str, np.ndarray] = {}
    for experiment in EXPERIMENTS:
        _set_seed(seed)
        model = ComparisonClassifier(
            experiment,
            text_dim=int(train_text.shape[1]),
            audio_dim=int(train_audio.shape[1]),
            hidden_dim=hidden_dim,
            psid_dim=psid_dim,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
        best_state = copy.deepcopy(model.state_dict())
        best_val_f1 = -1.0
        best_epoch = -1
        no_improvement = 0
        history: list[dict[str, float]] = []

        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(train_labels), device=device)
            total_loss = 0.0
            for start in range(0, len(order), batch_size):
                indexes = order[start:start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                logits = model(train_text[indexes], train_audio[indexes])
                loss = loss_fn(logits, train_labels[indexes])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += float(loss.item()) * len(indexes)

            val_metrics, _ = _evaluate(model, val_text, val_audio, val_labels)
            train_loss = total_loss / len(train_labels)
            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_macro_f1": float(val_metrics["macro_f1"]),
                "val_accuracy": float(val_metrics["accuracy"]),
            })
            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = float(val_metrics["macro_f1"])
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement >= patience:
                    break

        model.load_state_dict(best_state)
        test_metrics, probabilities = _evaluate(model, test_text, test_audio, test_labels)
        test_metrics["bootstrap_ci_95"] = _bootstrap_metrics(
            split_data["test"][2],
            probabilities,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        test_metrics["best_epoch"] = best_epoch + 1
        test_metrics["test_ids"] = len(splits["test"])
        test_metrics["evaluation_status"] = config["evaluation_status"]["status"]
        test_probabilities[experiment] = probabilities
        results[experiment] = test_metrics

        experiment_dir = output / experiment
        experiment_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": best_state,
                "mode": experiment,
                "config": config,
                "text_scaler": text_scaler,
                "audio_scaler": audio_scaler,
            },
            experiment_dir / "best_model.pt",
        )
        (experiment_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        (experiment_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )

    paired = {}
    baseline = test_probabilities["text_only"]
    for experiment in EXPERIMENTS[1:]:
        paired[experiment] = _paired_bootstrap(
            split_data["test"][2],
            baseline,
            test_probabilities[experiment],
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    (output / "paired_comparison_vs_text.json").write_text(
        json.dumps(paired, indent=2), encoding="utf-8"
    )
    paired_psid_vs_direct = _paired_bootstrap(
        split_data["test"][2],
        test_probabilities["direct_concat"],
        test_probabilities["psid_concat"],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    (output / "paired_psid_vs_direct.json").write_text(
        json.dumps(paired_psid_vs_direct, indent=2), encoding="utf-8"
    )
    (output / "comparison_metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    prediction_path = output / "test_predictions.csv"
    with prediction_path.open("w", newline="", encoding="utf-8") as file:
        columns = ["utterance_id", "dataset", "speaker", "true_label"]
        for experiment in EXPERIMENTS:
            columns.extend([f"{experiment}_prediction", f"{experiment}_prob_sarcastic"])
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for index, record in enumerate(splits["test"]):
            row: dict[str, Any] = {
                "utterance_id": record.example_id,
                "dataset": record.dataset,
                "speaker": record.speaker,
                "true_label": record.label,
            }
            for experiment in EXPERIMENTS:
                probs = test_probabilities[experiment][index]
                row[f"{experiment}_prediction"] = int(np.argmax(probs))
                row[f"{experiment}_prob_sarcastic"] = float(probs[1])
            writer.writerow(row)

    with (output / "comparison_table.csv").open("w", newline="", encoding="utf-8") as file:
        fields = [
            "experiment", "accuracy", "macro_f1", "weighted_f1",
            "precision_macro", "recall_macro", "roc_auc", "pr_auc",
            "evaluation_status",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for experiment, metrics in results.items():
            writer.writerow({"experiment": experiment, **{
                field: metrics.get(field) for field in fields[1:]
            }})
    return results
