"""train_fusion.py — entry point for training SarcasTone Fusion Model.

Usage
-----
# Full model (cross-attention + PSID + adaptive gating)
python member3_fusion/train_fusion.py --experiment full_model

# Text-only ablation
python member3_fusion/train_fusion.py --experiment text_only

# Custom config
python member3_fusion/train_fusion.py --config my_config.yaml --experiment cross_attn_psid

# Resume from checkpoint
python member3_fusion/train_fusion.py --experiment full_model --resume member3_fusion/experiments/full_model/best_model.pt

# No text encoder (use cached embeddings or zero text)
python member3_fusion/train_fusion.py --experiment full_model --no-text-encoder

All experiments are logged to member3_fusion/experiments/<experiment_name>/
Calibration temperature is fitted automatically on the validation set after training.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Insert repo root
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "member3_fusion"))
sys.path.insert(0, str(_REPO / "member3_fusion" / "src"))

from member3_fusion.src.fusion.config import load_config
from member3_fusion.src.training.trainer import FusionTrainer, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_fusion")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SarcasTone Fusion Model")
    p.add_argument(
        "--config", default=None,
        help="Path to YAML config (default: member3_fusion/configs/fusion_config.yaml)",
    )
    p.add_argument(
        "--experiment", default="full_model",
        help="Experiment name matching a key in configs.experiments (default: full_model)",
    )
    p.add_argument("--epochs",      type=int, default=None, help="Override epochs in config")
    p.add_argument("--batch-size",  type=int, default=None, dest="batch_size",
                   help="Override batch size")
    p.add_argument("--seed",        type=int, default=None, help="Override random seed")
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint (best_model.pt) to resume from")
    p.add_argument(
        "--no-text-encoder", action="store_true",
        help="Disable live text encoder; use cached text embeddings",
    )
    p.add_argument(
        "--live-text-encoder", action="store_true",
        help="Explicitly prefer the Member 1 checkpoint over configured cached embeddings",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Load config with experiment overlay
    cfg = load_config(
        config_path=args.config,
        experiment=args.experiment,
    )

    # CLI overrides
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["project"]["seed"] = args.seed
    if args.resume is not None:
        cfg["training"]["resume_from"] = args.resume

    seed = cfg.get("project", {}).get("seed", 42)
    set_seed(seed)

    log.info("Experiment : %s", args.experiment)
    log.info("Seed       : %d", seed)
    log.info("Config     : %s", args.config or "default")

    # Prefer the exact cached Member 1 handoff unless live encoding was requested.
    text_encoder = None
    cached_embeddings = cfg.get("data", {}).get("text_embeddings_npz")
    cache_available = bool(cached_embeddings and Path(cached_embeddings).is_file())
    if args.live_text_encoder and args.no_text_encoder:
        raise ValueError("--live-text-encoder and --no-text-encoder cannot be used together")
    use_live_encoder = args.live_text_encoder or (not args.no_text_encoder and not cache_available)
    if cache_available and not args.live_text_encoder:
        log.info("Using cached Member 1 embeddings from %s", cached_embeddings)
    elif not use_live_encoder:
        log.info("Live text encoder disabled; text cache is unavailable")
    if use_live_encoder:
        m1_checkpoint = Path(
            cfg.get("interface", {}).get(
                "member1_checkpoint",
                "member1_text/outputs/checkpoints/mustard__sarcasm__roberta-base",
            )
        )
        # Resolve relative path against repo root
        if not m1_checkpoint.is_absolute():
            m1_checkpoint = _REPO / m1_checkpoint

        if m1_checkpoint.is_dir():
            try:
                import torch
                from member3_fusion.src.fusion.encoders import TextEncoderInterface
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                text_encoder = TextEncoderInterface(
                    checkpoint_path=m1_checkpoint,
                    device=device,
                    freeze=True,
                )
                if text_encoder.is_available:
                    log.info("Text encoder loaded from %s", m1_checkpoint)
                else:
                    log.warning(
                        "TextEncoderInterface loaded but reports not available; "
                        "will fall back to cached embeddings or zeros."
                    )
                    text_encoder = None
            except Exception as e:
                log.warning("Could not load text encoder: %s; will use cached/zeros", e)
        else:
            log.info(
                "Member1 checkpoint not found at %s; text encoder disabled. "
                "Provide text_embeddings_npz in config for text signal.",
                m1_checkpoint,
            )

    # Train
    trainer = FusionTrainer(cfg)
    result  = trainer.train(
        experiment_name=args.experiment,
        text_encoder=text_encoder,
    )

    log.info("=" * 60)
    log.info("Training complete.")
    log.info("Best epoch      : %d", result["best_epoch"] + 1)
    log.info("Best val metric : %.4f", result["best_val_metric"])
    test_m = result["test_metrics"]
    log.info("Test accuracy   : %.4f", test_m.get("accuracy", 0))
    log.info("Test macro-F1   : %.4f", test_m.get("macro_f1", 0))
    log.info("Test ROC-AUC    : %s",   test_m.get("roc_auc"))
    log.info("Results in      : %s",   result["experiment_dir"])
    log.info("=" * 60)


if __name__ == "__main__":
    main()
