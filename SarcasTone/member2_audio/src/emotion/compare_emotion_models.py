
from pathlib import Path

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report
)

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "emotion_dataset"
OUTPUT_DIR = ROOT / "outputs" / "metrics"

EXCLUDED = [
    "sample_id", "file_name", "file_path",
    "actor_id", "emotion_code", "emotion",
    "intensity", "statement", "repetition",
    "sample_rate", "duration_seconds"
]


def main():

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")

    # Confirm speaker independence
    assert set(train.actor_id).isdisjoint(set(val.actor_id))

    features = [
        c for c in train.select_dtypes(include=np.number).columns
        if c not in EXCLUDED
    ]

    X_train = train[features]
    y_train = train["emotion"]

    X_val = val[features]
    y_val = val["emotion"]

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "SVM RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                C=1.0,
                class_weight="balanced"
            ))
        ]),

        "SVM Linear": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="linear",
                C=1.0,
                class_weight="balanced"
            ))
        ])
    }

    results = []

    print("COMPARING EMOTION CLASSIFIERS")
    print("=" * 50)

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val, predictions, average="macro"
        )
        weighted_f1 = f1_score(
            y_val, predictions, average="weighted"
        )
        balanced_acc = balanced_accuracy_score(
            y_val, predictions
        )

        results.append({
            "model": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "balanced_accuracy": balanced_acc
        })

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro F1: {macro_f1:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")

        report = classification_report(
            y_val,
            predictions,
            output_dict=True,
            zero_division=0
        )

        pd.DataFrame(report).transpose().to_csv(
            OUTPUT_DIR / (
                name.lower().replace(" ", "_")
                + "_comparison_report.csv"
            )
        )

    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values(
        "macro_f1", ascending=False
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    comparison.to_csv(
        OUTPUT_DIR / "emotion_model_comparison.csv",
        index=False
    )

    print("\nFINAL MODEL COMPARISON")
    print("=" * 70)
    print(comparison.to_string(index=False))

    print("\nBest validation model:")
    print(comparison.iloc[0]["model"])

    print("\nComparison saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
