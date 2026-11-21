
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "emotion_dataset"

TRAIN_FILE = DATA_DIR / "train.csv"
VAL_FILE = DATA_DIR / "validation.csv"

CHECKPOINT_DIR = ROOT / "checkpoints"
METRICS_DIR = ROOT / "outputs" / "metrics"

MODEL_FILE = CHECKPOINT_DIR / "ravdess_emotion_random_forest.joblib"

SEED = 42

EXCLUDED_COLUMNS = [
    "sample_id",
    "file_name",
    "file_path",
    "actor_id",
    "emotion_code",
    "emotion",
    "intensity",
    "statement",
    "repetition",
    "sample_rate",
    "duration_seconds"
]


def load_data():

    train = pd.read_csv(TRAIN_FILE)
    validation = pd.read_csv(VAL_FILE)

    # Verify speaker independence
    train_actors = set(train["actor_id"])
    val_actors = set(validation["actor_id"])

    if not train_actors.isdisjoint(val_actors):
        raise ValueError("Speaker overlap detected!")

    feature_columns = [
        col for col in train.select_dtypes(
            include=[np.number]
        ).columns
        if col not in EXCLUDED_COLUMNS
    ]

    if len(feature_columns) != 116:
        raise ValueError(
            f"Expected 116 features, found {len(feature_columns)}"
        )

    X_train = train[feature_columns]
    X_val = validation[feature_columns]

    y_train = train["emotion"].astype(str)
    y_val = validation["emotion"].astype(str)

    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("Invalid values in training features.")

    if not np.isfinite(X_val.to_numpy()).all():
        raise ValueError("Invalid values in validation features.")

    if set(y_train.unique()) != set(y_val.unique()):
        raise ValueError("Training and validation classes differ.")

    return (
        X_train, X_val,
        y_train, y_val,
        feature_columns
    )


def train_model():

    print("Loading emotion datasets...")

    X_train, X_val, y_train, y_val, feature_columns = load_data()

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Feature dimensions:", len(feature_columns))

    print("\nTraining Random Forest...")

    model = RandomForestClassifier(
        n_estimators=400,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("\nEvaluating on validation set...")

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_val,
        predictions,
        average="weighted"
    )

    report = classification_report(
        y_val,
        predictions,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_val,
        predictions,
        labels=model.classes_
    )

    CHECKPOINT_DIR.mkdir(
        parents=True, exist_ok=True
    )

    METRICS_DIR.mkdir(
        parents=True, exist_ok=True
    )

    # Save model and feature configuration
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "classes": model.classes_.tolist(),
            "feature_count": len(feature_columns),
            "random_state": SEED
        },
        MODEL_FILE
    )

    # Save validation metrics
    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
        "feature_count": len(feature_columns),
        "model": "RandomForestClassifier"
    }

    with open(
        METRICS_DIR / "ravdess_emotion_validation.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    pd.DataFrame(report).transpose().to_csv(
        METRICS_DIR / "ravdess_emotion_classification_report.csv"
    )

    pd.DataFrame(
        matrix,
        index=model.classes_,
        columns=model.classes_
    ).to_csv(
        METRICS_DIR / "ravdess_emotion_confusion_matrix.csv"
    )

    print("\nEMOTION MODEL TRAINING COMPLETE")
    print("=" * 45)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_val,
        predictions,
        zero_division=0
    ))

    print("\nModel saved:")
    print(MODEL_FILE)

    print("\nValidation metrics saved:")
    print(METRICS_DIR)


if __name__ == "__main__":
    train_model()
