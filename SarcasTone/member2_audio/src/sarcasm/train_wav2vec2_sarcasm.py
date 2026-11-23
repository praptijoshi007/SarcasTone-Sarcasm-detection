
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_DIR = Path(os.environ.get("SARCASTONE_EMBEDDINGS", ROOT / "outputs" / "embeddings"))
EMBEDDING_FILE = EMBEDDING_DIR / "wav2vec2_embeddings.npz"

MANIFEST_FILE = (
    ROOT / "data" / "processed" / "mustard"
    / "mustard_audio_manifest.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "metrics" / "wav2vec2"
CHECKPOINT_DIR = ROOT / "checkpoints" / "sarcasm"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD EMBEDDINGS
# ==================================================

print("=" * 65)
print("SARCAS TONE: WAV2VEC2 SARCASM CLASSIFICATION")
print("=" * 65)

data = np.load(EMBEDDING_FILE, allow_pickle=False)

embeddings = data["embeddings"]
utterance_ids = data["utterance_ids"].astype(str)

print("Embedding shape:", embeddings.shape)
print("Number of utterances:", len(utterance_ids))

if embeddings.shape != (690, 768):
    print("Warning: Unexpected embedding dimensions:", embeddings.shape)

if not np.isfinite(embeddings).all():
    raise ValueError("Non-finite embedding values detected.")

if len(set(utterance_ids)) != len(utterance_ids):
    raise ValueError("Duplicate utterance IDs detected.")


# ==================================================
# LOAD LABELS AND SPEAKERS
# ==================================================

manifest = pd.read_csv(MANIFEST_FILE)

required_columns = [
    "utterance_id",
    "speaker",
    "sarcasm"
]

missing = [
    column for column in required_columns
    if column not in manifest.columns
]

if missing:
    raise ValueError(
        f"Missing manifest columns: {missing}. "
        f"Available columns: {manifest.columns.tolist()}"
    )

manifest["utterance_id"] = (
    manifest["utterance_id"].astype(str)
)

manifest["sarcasm"] = manifest["sarcasm"].astype(int)

if manifest["utterance_id"].duplicated().any():
    raise ValueError("Duplicate IDs in manifest.")

labels = pd.DataFrame({
    "utterance_id": utterance_ids
})

labels = labels.merge(
    manifest[required_columns],
    on="utterance_id",
    how="left",
    validate="one_to_one"
)

if labels[required_columns].isnull().any().any():
    raise ValueError("Missing labels or speakers after matching.")

if len(labels) != len(embeddings):
    raise ValueError("Embedding and label counts do not match.")

if labels["sarcasm"].nunique() != 2:
    raise ValueError("Expected two sarcasm classes.")

y = labels["sarcasm"].to_numpy()
groups = labels["speaker"].to_numpy()

X = embeddings

print("\nDataset verified.")
print("Samples:", len(X))
print("Speakers:", len(set(groups)))
print("Sarcastic:", int(np.sum(y == 1)))
print("Non-sarcastic:", int(np.sum(y == 0)))


# ==================================================
# MODEL
# ==================================================

def create_model():

    return Pipeline([
        ("scaler", StandardScaler()),

        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced"
        ))
    ])


# ==================================================
# SPEAKER-INDEPENDENT CROSS-VALIDATION
# ==================================================

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_predictions = np.full(len(y), -1, dtype=int)

fold_results = []

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    print("\n" + "=" * 65)
    print(f"FOLD {fold}/5")
    print("=" * 65)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_speakers = set(groups[train_idx])
    test_speakers = set(groups[test_idx])

    if train_speakers.intersection(test_speakers):
        raise RuntimeError("Speaker leakage detected!")

    model = create_model()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    oof_predictions[test_idx] = predictions

    metrics = {
        "fold": fold,
        "train_samples": len(train_idx),
        "test_samples": len(test_idx),
        "test_speakers": len(test_speakers),

        "accuracy": accuracy_score(
            y_test, predictions
        ),

        "precision_macro": precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_test,
            predictions
        )
    }

    fold_results.append(metrics)

    print("Training samples:", len(train_idx))
    print("Testing samples:", len(test_idx))
    print("Unseen speakers:", len(test_speakers))
    print("Accuracy:", round(metrics["accuracy"], 4))
    print("Macro-F1:", round(metrics["macro_f1"], 4))


# ==================================================
# VERIFY OOF PREDICTIONS
# ==================================================

if np.any(oof_predictions == -1):
    raise RuntimeError("Some samples lack OOF predictions.")


# ==================================================
# FOLD RESULTS
# ==================================================

fold_df = pd.DataFrame(fold_results)

fold_df.to_csv(
    OUTPUT_DIR / "wav2vec2_fold_results.csv",
    index=False
)

print("\n" + "=" * 65)
print("FOLD-AVERAGED RESULTS")
print("=" * 65)

print(
    fold_df[
        [
            "fold",
            "accuracy",
            "macro_f1",
            "balanced_accuracy"
        ]
    ].to_string(index=False)
)

print("\nMean Macro-F1:", round(
    fold_df["macro_f1"].mean(), 4
))

print("Std Macro-F1:", round(
    fold_df["macro_f1"].std(), 4
))


# ==================================================
# POOLED OOF RESULTS
# ==================================================

pooled_metrics = {
    "accuracy": accuracy_score(y, oof_predictions),

    "precision_macro": precision_score(
        y,
        oof_predictions,
        average="macro",
        zero_division=0
    ),

    "recall_macro": recall_score(
        y,
        oof_predictions,
        average="macro",
        zero_division=0
    ),

    "macro_f1": f1_score(
        y,
        oof_predictions,
        average="macro",
        zero_division=0
    ),

    "balanced_accuracy": balanced_accuracy_score(
        y,
        oof_predictions
    )
}

print("\n" + "=" * 65)
print("POOLED OUT-OF-FOLD RESULTS")
print("=" * 65)

for metric, value in pooled_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\nClassification report:")

print(
    classification_report(
        y,
        oof_predictions,
        target_names=["Non-sarcastic", "Sarcastic"],
        zero_division=0
    )
)

print("Confusion matrix:")
print(confusion_matrix(y, oof_predictions))


# ==================================================
# SAVE PREDICTIONS
# ==================================================

prediction_df = labels.copy()

prediction_df["predicted"] = oof_predictions

prediction_df.to_csv(
    OUTPUT_DIR / "wav2vec2_oof_predictions.csv",
    index=False
)

pd.DataFrame([pooled_metrics]).to_csv(
    OUTPUT_DIR / "wav2vec2_pooled_metrics.csv",
    index=False
)


# ==================================================
# TRAIN FINAL DEPLOYMENT MODEL
# ==================================================

print("\nTraining final model on all 690 samples...")

final_model = create_model()

final_model.fit(X, y)

joblib.dump(
    final_model,
    CHECKPOINT_DIR / "wav2vec2_sarcasm_svm.joblib"
)

print("\n" + "=" * 65)
print("WAV2VEC2 CLASSIFIER TRAINING COMPLETED")
print("=" * 65)

print("Model saved:")
print(CHECKPOINT_DIR / "wav2vec2_sarcasm_svm.joblib")

print("Metrics saved:")
print(OUTPUT_DIR)
