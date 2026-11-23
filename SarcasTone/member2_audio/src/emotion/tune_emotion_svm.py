
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "emotion_dataset"
CHECKPOINT_DIR = ROOT / "checkpoints"
METRICS_DIR = ROOT / "outputs" / "metrics"

EXCLUDED = {
    "sample_id", "file_name", "file_path",
    "actor_id", "emotion_code", "emotion",
    "intensity", "statement", "repetition",
    "sample_rate", "duration_seconds"
}


def get_features(df):
    return [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in EXCLUDED
    ]


def main():

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "validation.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")

    # Verify speaker independence
    assert set(train.actor_id).isdisjoint(set(val.actor_id))
    assert set(train.actor_id).isdisjoint(set(test.actor_id))
    assert set(val.actor_id).isdisjoint(set(test.actor_id))

    features = get_features(train)

    if get_features(val) != features or get_features(test) != features:
        raise ValueError("Feature columns do not match.")

    X_train = train[features]
    y_train = train["emotion"]

    X_val = val[features]
    y_val = val["emotion"]

    X_test = test[features]
    y_test = test["emotion"]

    # Hyperparameter search using validation only
    candidates = []

    for c in [0.1, 1, 10, 100]:
        for gamma in ["scale", 0.001, 0.01, 0.1]:

            model = Pipeline([
                ("scaler", StandardScaler()),
                ("classifier", SVC(
                    kernel="rbf",
                    C=c,
                    gamma=gamma,
                    class_weight="balanced"
                ))
            ])

            model.fit(X_train, y_train)
            predictions = model.predict(X_val)

            candidates.append({
                "C": c,
                "gamma": gamma,
                "accuracy": accuracy_score(y_val, predictions),
                "macro_f1": f1_score(
                    y_val, predictions, average="macro"
                ),
                "weighted_f1": f1_score(
                    y_val, predictions, average="weighted"
                )
            })

    results = pd.DataFrame(candidates)
    results = results.sort_values(
        ["macro_f1", "accuracy"],
        ascending=False
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        METRICS_DIR / "svm_hyperparameter_search.csv",
        index=False
    )

    best = results.iloc[0]

    best_c = float(best["C"])
    best_gamma = best["gamma"]

    print("\nHYPERPARAMETER SEARCH")
    print("=" * 45)
    print(results.to_string(index=False))
    print("\nSelected C:", best_c)
    print("Selected gamma:", best_gamma)
    print("Validation macro F1:", round(best["macro_f1"], 4))

    # Retrain selected configuration on train + validation
    development = pd.concat(
        [train, val],
        ignore_index=True
    )

    X_development = development[features]
    y_development = development["emotion"]

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="rbf",
            C=best_c,
            gamma=best_gamma,
            class_weight="balanced"
        ))
    ])

    final_model.fit(X_development, y_development)

    # Evaluate held-out test actors ONCE
    test_predictions = final_model.predict(X_test)

    metrics = {
        "model": "SVM RBF",
        "C": best_c,
        "gamma": best_gamma,
        "accuracy": float(
            accuracy_score(y_test, test_predictions)
        ),
        "macro_f1": float(
            f1_score(y_test, test_predictions, average="macro")
        ),
        "weighted_f1": float(
            f1_score(y_test, test_predictions, average="weighted")
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, test_predictions)
        ),
        "training_plus_validation_samples": len(development),
        "test_samples": len(test),
        "test_actors": sorted(test["actor_id"].unique().tolist()),
        "feature_count": len(features)
    }

    # Save model and configuration
    joblib.dump({
        "model": final_model,
        "feature_columns": features,
        "classes": final_model.classes_.tolist(),
        "model_name": "SVM RBF",
        "hyperparameters": {
            "C": best_c,
            "gamma": best_gamma,
            "class_weight": "balanced"
        }
    }, CHECKPOINT_DIR / "ravdess_emotion_svm_rbf.joblib")

    with open(
        METRICS_DIR / "ravdess_emotion_test_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metrics, f, indent=4)

    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        METRICS_DIR / "ravdess_emotion_test_report.csv"
    )

    cm = confusion_matrix(
        y_test,
        test_predictions,
        labels=final_model.classes_
    )

    pd.DataFrame(
        cm,
        index=final_model.classes_,
        columns=final_model.classes_
    ).to_csv(
        METRICS_DIR / "ravdess_emotion_test_confusion_matrix.csv"
    )

    print("\nFINAL HELD-OUT TEST RESULTS")
    print("=" * 45)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        test_predictions,
        zero_division=0
    ))

    print("\nFinal model saved.")
    print("Test metrics saved.")


if __name__ == "__main__":
    main()
