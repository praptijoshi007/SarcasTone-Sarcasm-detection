"""Train the four reproducible Member 3 text/audio comparison models."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "member3_fusion"))

from member3_fusion.src.fusion.config import load_config
from member3_fusion.src.training.comparison import run_comparison
from member3_fusion.src.training.dataset import load_mustard_fusion_records


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compare_fusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--allow-unverified-text-provenance",
        action="store_true",
        help=(
            "Run an exploratory comparison when the text handoff lacks evidence "
            "that its encoder excluded validation/test labels."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(config_path=args.config)
    data = cfg["data"]
    output_dir = Path(args.output_dir or cfg["output"]["comparison_dir"])
    seed = args.seed if args.seed is not None else cfg.get("project", {}).get("seed", 42)

    splits, acoustic_dim = load_mustard_fusion_records(
        mustard_json_path=data["mustard_json"],
        handoff_csv_path=data["audio_handoff_csv"],
        quality_csv_path=data["audio_quality_csv"],
        predictions_csv_path=data.get("audio_predictions_csv"),
        wav2vec2_npz_path=data["wav2vec2_embeddings_npz"],
        text_embeddings_npz_path=data["text_embeddings_npz"],
        prosody_csv_path=data["prosody_csv"],
        val_ratio=data.get("val_ratio", 0.15),
        test_ratio=data.get("test_ratio", 0.15),
        seed=seed,
        split_strategy=data.get("split_strategy", "speaker"),
        alignment_report_path=output_dir / "alignment_report.json",
        max_alignment_drop_ratio=data.get("max_alignment_drop_ratio", 0.10),
    )
    log.info("Loaded split counts %s; acoustic dimension=%d", {
        key: len(value) for key, value in splits.items()
    }, acoustic_dim)
    text_npz = Path(data["text_embeddings_npz"])
    with np.load(text_npz, allow_pickle=False) as archive:
        npz_keys = list(archive.keys())
        has_oof_provenance = "is_out_of_fold" in archive and bool(archive["is_out_of_fold"].item())

    checkpoint_state_path = (
        _REPO / "member1_text" / "checkpoints" / "mustard__sarcasm__roberta-base"
        / "trainer_state.pt"
    )
    checkpoint_training_ids: set[str] = set()
    checkpoint_metadata: dict = {"path": str(checkpoint_state_path), "available": False}
    if checkpoint_state_path.is_file():
        state = torch.load(checkpoint_state_path, map_location="cpu", weights_only=True)
        checkpoint_config = state.get("config", {})
        checkpoint_training_ids = {
            str(uid).strip().casefold() for uid in checkpoint_config.get("train_ids", [])
        }
        test_ids = {record.example_id for record in splits["test"]}
        checkpoint_metadata = {
            "path": str(checkpoint_state_path),
            "available": True,
            "fold_index": checkpoint_config.get("fold_index"),
            "training_id_count": len(checkpoint_training_ids),
            "test_id_overlap_count": len(checkpoint_training_ids & test_ids),
        }

    provenance_status = {
        "status": "unverified_upstream_text_provenance",
        "text_embedding_npz": str(text_npz),
        "npz_keys": npz_keys,
        "npz_contains_oof_provenance": has_oof_provenance,
        "nearby_member1_checkpoint": checkpoint_metadata,
        "warning": (
            "The text handoff has no source-checkpoint or training-ID manifest. "
            "A nearby Member 1 checkpoint records fold_index=null and "
            f"{checkpoint_metadata.get('training_id_count', 0)} training IDs, with "
            f"{checkpoint_metadata.get('test_id_overlap_count', 0)} test-ID overlap. "
            "Treat all results as exploratory unless the embedding source is proven "
            "out-of-fold."
        ),
        "required_for_leakage_safe_evaluation": (
            "Provide per-utterance out-of-fold Member 1 embeddings or fold-specific "
            "Member 1 checkpoints plus their held-out ID mapping."
        ),
    }
    if args.allow_unverified_text_provenance:
        provenance_status["status"] = "exploratory_upstream_provenance_unverified"
    (output_dir / "upstream_text_provenance.json").write_text(
        json.dumps(provenance_status, indent=2), encoding="utf-8"
    )
    if not has_oof_provenance and not args.allow_unverified_text_provenance:
        raise RuntimeError(
            f"Cannot certify a leakage-safe comparison: {text_npz} has no out-of-fold "
            "or training-ID metadata. Supply OOF embeddings or rerun only for "
            "exploratory analysis with --allow-unverified-text-provenance."
        )
    results = run_comparison(
        splits,
        output_dir,
        evaluation_status=provenance_status,
        seed=seed,
        epochs=args.epochs or cfg["training"]["epochs"],
        batch_size=args.batch_size or cfg["training"]["batch_size"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        patience=cfg["training"]["early_stopping_patience"],
        hidden_dim=cfg["model"]["classifier_hidden"],
        psid_dim=cfg["model"]["psid_proj_dim"],
        dropout=cfg["model"]["dropout"],
        n_bootstrap=args.bootstrap,
    )
    print("\nExperiment       Accuracy  Macro-F1  Weighted-F1  ROC-AUC  PR-AUC")
    for experiment, metrics in results.items():
        print(
            f"{experiment:<16} {metrics['accuracy']:.4f}    "
            f"{metrics['macro_f1']:.4f}    {metrics['weighted_f1']:.4f}       "
            f"{metrics['roc_auc']}  {metrics['pr_auc']}"
        )
    print(f"\nArtifacts: {output_dir}")
    print(json.dumps({key: value["confusion_matrix"] for key, value in results.items()}, indent=2))


if __name__ == "__main__":
    main()
