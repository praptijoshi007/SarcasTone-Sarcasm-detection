"""Train Member 2 Acoustic Prosody Sarcasm Classifier on MUStARD.

Trains a Gradient Boosting + Scaler pipeline on the 12 key acoustic prosodic features:
  - F0 pitch mean & std
  - RMS energy mean & std
  - Spectral centroid mean & std
  - Spectral rolloff mean & std
  - Zero-crossing rate mean & std
  - Speaking rate proxy & pause ratio

Saves model to:
  member2_audio/checkpoints/sarcasm/acoustic_sarcasm_model.joblib
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_audio_model")

REPO_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLS = [
    "pitch_f0_mean", "pitch_f0_std", "rms_energy_mean", "rms_energy_std",
    "spectral_centroid_mean", "spectral_centroid_std", "spectral_rolloff_mean",
    "spectral_rolloff_std", "zero_crossing_rate_mean", "zero_crossing_rate_std",
    "speaking_rate_proxy", "pause_ratio"
]


def train_and_save():
    csv_path = REPO_ROOT / "member2_audio" / "outputs" / "handoff" / "mustard_audio_handoff.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Handoff CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].values
    y = df["sarcasm"].values

    log.info("Running 5-fold cross-validation on %d acoustic records...", len(df))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    for tr, te in skf.split(X, y):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42))
        ])
        pipe.fit(X[tr], y[tr])
        preds = pipe.predict(X[te])
        accs.append(accuracy_score(y[te], preds))
        f1s.append(f1_score(y[te], preds, average="macro"))

    log.info("5-Fold CV Accuracy: %.2f%% (+/- %.2f%%)", np.mean(accs) * 100.0, np.std(accs) * 100.0)
    log.info("5-Fold CV Macro-F1: %.2f%%", np.mean(f1s) * 100.0)

    # Train deployment pipeline on full dataset
    final_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42))
    ])
    final_pipe.fit(X, y)

    out_dir = REPO_ROOT / "member2_audio" / "checkpoints" / "sarcasm"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "acoustic_sarcasm_model.joblib"
    meta_path = out_dir / "acoustic_sarcasm_meta.json"

    joblib.dump(final_pipe, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": "GradientBoosting_StandardScaler",
            "feature_cols": FEATURE_COLS,
            "cv_accuracy": float(np.mean(accs)),
            "cv_macro_f1": float(np.mean(f1s)),
            "n_samples": len(df),
        }, f, indent=2)

    log.info("Saved trained acoustic model to %s", model_path)


if __name__ == "__main__":
    train_and_save()
