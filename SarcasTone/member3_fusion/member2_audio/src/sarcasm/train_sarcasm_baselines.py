
import json
import joblib
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
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


ROOT = Path(__file__).resolve().parents[2]

DATA_FILE = (
    ROOT / "data/processed/mustard/"
    "mustard_acoustic_features.csv"
)

CHECKPOINT_DIR = ROOT / "checkpoints/sarcasm"
METRIC_DIR = ROOT / "outputs/metrics/sarcasm"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def evaluate_predictions(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        )
    }


def main():

    print("=" * 65)
    print("SARCAS TONE: MUStARD AUDIO SARCASM BASELINES")
    print("=" * 65)

    df = pd.read_csv(DATA_FILE)

    metadata_columns = [
        "utterance_id",
        "speaker",
        "sarcasm",
        "duration_seconds"
    ]

    feature_columns = [
        col for col in df.columns
        if col not in metadata_columns
    ]

    X = df[feature_columns].astype(float)
    y = df["sarcasm"].astype(int)
    groups = df["speaker"]

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Feature matrix contains non-finite values.")

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced"
            ))
        ])
    }

    comparison = []
    prediction_records = []

    for model_name, model in models.items():

        print(f"\nTRAINING: {model_name}")
        print("-" * 55)

        oof_predictions = np.full(len(df), -1, dtype=int)

        for fold, (train_idx, test_idx) in enumerate(
            splitter.split(X, y, groups),
            start=1
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            train_speakers = set(groups.iloc[train_idx])
            test_speakers = set(groups.iloc[test_idx])

            if train_speakers & test_speakers:
                raise RuntimeError(
                    f"Speaker leakage in fold {fold}"
                )

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            oof_predictions[test_idx] = predictions

            fold_f1 = f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0
            )

            print(
                f"Fold {fold}: "
                f"Test samples={len(test_idx)}, "
                f"Macro-F1={fold_f1:.4f}"
            )

        if np.any(oof_predictions == -1):
            raise RuntimeError(
                "Some samples did not receive an OOF prediction."
            )

        metrics = evaluate_predictions(
            y.to_numpy(),
            oof_predictions
        )

        metrics["model"] = model_name
        comparison.append(metrics)

        print("\nOVERALL OUT-OF-FOLD RESULTS")
        for key, value in metrics.items():
            if key != "model":
                print(f"{key}: {value:.4f}")

        print("\nClassification report:")
        print(
            classification_report(
                y,
                oof_predictions,
                target_names=["Non-sarcastic", "Sarcastic"],
                zero_division=0
            )
        )

        cm = confusion_matrix(y, oof_predictions)

        pd.DataFrame(
            cm,
            index=["Actual_NonSarcastic", "Actual_Sarcastic"],
            columns=["Predicted_NonSarcastic", "Predicted_Sarcastic"]
        ).to_csv(
            METRIC_DIR / f"{model_name}_confusion_matrix.csv"
        )

        report = classification_report(
            y,
            oof_predictions,
            target_names=["Non-sarcastic", "Sarcastic"],
            output_dict=True,
            zero_division=0
        )

        pd.DataFrame(report).transpose().to_csv(
            METRIC_DIR / f"{model_name}_classification_report.csv"
        )

        for i, prediction in enumerate(oof_predictions):
            prediction_records.append({
                "model": model_name,
                "utterance_id": df.iloc[i]["utterance_id"],
                "speaker": groups.iloc[i],
                "actual": int(y.iloc[i]),
                "predicted": int(prediction)
            })

        # Fit final model on the complete dataset for later inference.
        # This model is NOT used to calculate the OOF evaluation metrics.
        model.fit(X, y)

        joblib.dump(
            {
                "model": model,
                "feature_columns": feature_columns,
                "model_name": model_name,
                "random_state": RANDOM_STATE
            },
            CHECKPOINT_DIR / f"{model_name}.joblib"
        )

    comparison_df = pd.DataFrame(comparison)

    comparison_df = comparison_df[
        [
            "model",
            "accuracy",
            "precision_macro",
            "recall_macro",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy"
        ]
    ].sort_values("macro_f1", ascending=False)

    comparison_df.to_csv(
        METRIC_DIR / "baseline_comparison.csv",
        index=False
    )

    pd.DataFrame(prediction_records).to_csv(
        METRIC_DIR / "out_of_fold_predictions.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("FINAL MODEL COMPARISON")
    print("=" * 65)
    print(comparison_df.to_string(index=False))

    print("\nSaved checkpoints:", CHECKPOINT_DIR)
    print("Saved evaluation files:", METRIC_DIR)


if __name__ == "__main__":
    main()
