"""evaluate_fusion.py — Evaluate a trained fusion model checkpoint.

Usage
-----
# Evaluate best model from an experiment
python member3_fusion/evaluate_fusion.py --experiment full_model

# Evaluate on specific split
python member3_fusion/evaluate_fusion.py --experiment full_model --split test

# Run all ablation experiments and produce comparison table
python member3_fusion/evaluate_fusion.py --ablation
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "member3_fusion"))
sys.path.insert(0, str(_REPO / "member3_fusion" / "src"))

import torch
import numpy as np
from torch.utils.data import DataLoader

from member3_fusion.src.fusion.config import load_config
from member3_fusion.src.models.fusion_model import SarcasToneFusionModel
from member3_fusion.src.training.dataset import (
    FusionDataset,
    collate_fn,
    load_mustard_fusion_records,
)
from member3_fusion.src.evaluation.evaluator import FusionEvaluator
from member3_fusion.src.training.trainer import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("evaluate_fusion")

ABLATION_EXPERIMENTS = [
    "text_only",
    "audio_only",
    "early_fusion",
    "late_fusion",
    "concat_fusion",
    "cross_attn",
    "cross_attn_psid",
    "full_model",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SarcasTone Fusion Model")
    p.add_argument("--experiment", default="full_model")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--config", default=None)
    p.add_argument("--ablation", action="store_true",
                   help="Evaluate all ablation experiments and produce table")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--bootstrap", type=int, default=500)
    return p.parse_args()


def load_data(cfg: dict, seed: int) -> dict:
    splits, acoustic_dim = load_mustard_fusion_records(
        mustard_json_path=cfg["data"]["mustard_json"],
        handoff_csv_path=cfg["data"]["audio_handoff_csv"],
        quality_csv_path=cfg["data"]["audio_quality_csv"],
        predictions_csv_path=cfg["data"]["audio_predictions_csv"],
        wav2vec2_npz_path=cfg["data"].get("wav2vec2_embeddings_npz"),
        val_ratio=cfg["data"].get("val_ratio", 0.15),
        test_ratio=cfg["data"].get("test_ratio", 0.15),
        seed=seed,
        split_strategy=cfg["data"].get("split_strategy", "speaker"),
        text_embeddings_npz_path=cfg["data"].get("text_embeddings_npz"),
        prosody_csv_path=cfg["data"].get("prosody_csv"),
        alignment_report_path=Path(cfg["output"]["comparison_dir"]) / "alignment_report.json",
        max_alignment_drop_ratio=cfg["data"].get("max_alignment_drop_ratio", 0.10),
    )
    return {
        split: FusionDataset(records, acoustic_dim=acoustic_dim)
        for split, records in splits.items()
    }, acoustic_dim


def evaluate_experiment(
    experiment: str,
    cfg: dict,
    split: str,
    device: torch.device,
    n_bootstrap: int,
    save_preds: bool,
) -> dict | None:
    exp_dir = Path(cfg["output"]["experiments_dir"]) / experiment
    ckpt_dir = exp_dir / "checkpoint"

    if not ckpt_dir.is_dir():
        log.warning("No checkpoint found for experiment '%s' at %s", experiment, ckpt_dir)
        return None

    try:
        model = SarcasToneFusionModel.load(ckpt_dir, device=device)
    except Exception as e:
        log.error("Failed to load model for '%s': %s", experiment, e)
        return None

    seed = cfg.get("project", {}).get("seed", 42)
    datasets, _ = load_data(cfg, seed)
    loader = DataLoader(
        datasets[split], batch_size=16, shuffle=False, collate_fn=collate_fn
    )

    evaluator = FusionEvaluator(model, device)
    metrics = evaluator.evaluate(loader, split_name=split, n_bootstrap=n_bootstrap)
    metrics["experiment"] = experiment

    out_dir = Path(cfg["output"]["metrics_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{experiment}_{split}_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    if save_preds:
        evaluator.save_predictions(
            loader,
            output_path=out_dir / f"{experiment}_{split}_predictions.json",
        )

    return metrics


def print_ablation_table(all_metrics: dict[str, dict]) -> None:
    """Print a formatted comparison table to stdout."""
    cols = ["accuracy", "macro_f1", "weighted_f1", "roc_auc", "pr_auc"]
    header = f"{'Experiment':<30}" + "".join(f"{c:>14}" for c in cols)
    print("\n" + "=" * len(header))
    print("ABLATION RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for exp in ABLATION_EXPERIMENTS:
        m = all_metrics.get(exp)
        if m is None:
            row = f"{exp:<30}" + "".join(f"{'N/A':>14}" for _ in cols)
        else:
            row = f"{exp:<30}"
            for c in cols:
                val = m.get(c)
                row += f"{val:>14.4f}" if isinstance(val, float) else f"{'N/A':>14}"
        print(row)
    print("=" * len(header))


def main() -> None:
    args = parse_args()
    cfg = load_config(config_path=args.config)
    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ablation:
        all_results: dict[str, dict] = {}
        for exp in ABLATION_EXPERIMENTS:
            log.info("Evaluating experiment: %s", exp)
            m = evaluate_experiment(
                exp, cfg, args.split, device, args.bootstrap, args.save_predictions
            )
            if m:
                all_results[exp] = m

        print_ablation_table(all_results)

        # Save ablation table
        abl_dir = Path(cfg["output"]["ablation_dir"])
        abl_dir.mkdir(parents=True, exist_ok=True)
        (abl_dir / f"ablation_{args.split}.json").write_text(
            json.dumps(all_results, indent=2, default=str), encoding="utf-8"
        )
        log.info("Ablation results saved to %s", abl_dir)
    else:
        m = evaluate_experiment(
            args.experiment, cfg, args.split,
            device, args.bootstrap, args.save_predictions
        )
        if m:
            log.info(
                "%s | acc=%.4f  macro_f1=%.4f  roc_auc=%s",
                args.experiment,
                m.get("accuracy", 0),
                m.get("macro_f1", 0),
                m.get("roc_auc"),
            )


if __name__ == "__main__":
    main()
