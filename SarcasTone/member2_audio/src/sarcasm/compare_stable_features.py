
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "mustard"
    / "mustard_acoustic_features.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "analysis"
    / "stable_feature_comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
OUTER_SPLITS = 5
INNER_SPLITS = 4
TOP_K = 40
MIN_SELECTION_FREQUENCY = 2
PERMUTATION_REPEATS = 3

LABELS = [0, 1]


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


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true, y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        )
    }


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

metadata = [
    "utterance_id",
    "speaker",
    "sarcasm",
    "duration_seconds"
]

feature_columns = [
    c for c in df.columns if c not in metadata
]

X = df[feature_columns]
y = df["sarcasm"].astype(int)
groups = df["speaker"]

print("=" * 70)
print("SARCAS TONE: STABLE FEATURE COMPARISON")
print("=" * 70)

print("Samples:", len(df))
print("Speakers:", groups.nunique())
print("Features:", len(feature_columns))


# --------------------------------------------------
# OUTER SPEAKER-INDEPENDENT CV
# --------------------------------------------------

outer_cv = StratifiedGroupKFold(
    n_splits=OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

results = []
predictions_output = []
stable_feature_output = []

experiments = ["All_116", "Top_40", "Stable_Inner"]

oof_predictions = {
    name: np.full(len(df), -1, dtype=int)
    for name in experiments
}


for outer_fold, (train_idx, test_idx) in enumerate(
    outer_cv.split(X, y, groups),
    start=1
):

    print("\n" + "=" * 70)
    print(f"OUTER FOLD {outer_fold}/{OUTER_SPLITS}")
    print("=" * 70)

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]
    groups_test = groups.iloc[test_idx]

    if set(groups_train).intersection(set(groups_test)):
        raise RuntimeError("Outer speaker leakage detected!")

    # --------------------------------------------------
    # INNER FEATURE RANKING
    # --------------------------------------------------

    inner_cv = StratifiedGroupKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE + outer_fold
    )

    inner_rankings = []

    for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(
        inner_cv.split(X_train, y_train, groups_train),
        start=1
    ):

        X_inner_train = X_train.iloc[inner_train_idx]
        X_inner_valid = X_train.iloc[inner_valid_idx]

        y_inner_train = y_train.iloc[inner_train_idx]
        y_inner_valid = y_train.iloc[inner_valid_idx]

        train_groups = set(
            groups_train.iloc[inner_train_idx]
        )
        valid_groups = set(
            groups_train.iloc[inner_valid_idx]
        )

        if train_groups.intersection(valid_groups):
            raise RuntimeError("Inner speaker leakage detected!")

        model = create_model()
        model.fit(X_inner_train, y_inner_train)

        importance = permutation_importance(
            model,
            X_inner_valid,
            y_inner_valid,
            scoring="f1_macro",
            n_repeats=PERMUTATION_REPEATS,
            random_state=(
                RANDOM_STATE + outer_fold * 10 + inner_fold
            ),
            n_jobs=-1
        )

        ranking = pd.Series(
            importance.importances_mean,
            index=feature_columns
        ).sort_values(ascending=False)

        inner_rankings.append(ranking)

        print(
            f"Inner fold {inner_fold}: "
            f"feature ranking completed"
        )

    # --------------------------------------------------
    # BUILD STABLE FEATURE SET
    # --------------------------------------------------

    selection_counts = {}

    for ranking in inner_rankings:

        selected = ranking.head(TOP_K).index.tolist()

        for feature in selected:
            selection_counts[feature] = (
                selection_counts.get(feature, 0) + 1
            )

    stable_features = [
        feature
        for feature, count in selection_counts.items()
        if count >= MIN_SELECTION_FREQUENCY
    ]

    # If fewer than 10 features survive, use the
    # most frequently selected features as a fallback.
    if len(stable_features) < 10:

        stable_features = [
            feature
            for feature, count in sorted(
                selection_counts.items(),
                key=lambda item: item[1],
                reverse=True
            )[:10]
        ]

    # Order stable features by selection frequency,
    # then by their average inner-fold importance.
    average_importance = pd.concat(
        inner_rankings,
        axis=1
    ).mean(axis=1)

    stable_features = sorted(
        stable_features,
        key=lambda f: (
            selection_counts[f],
            average_importance[f]
        ),
        reverse=True
    )

    top40_features = (
        average_importance
        .sort_values(ascending=False)
        .head(TOP_K)
        .index.tolist()
    )

    print("Stable feature count:", len(stable_features))

    for feature in stable_features:
        stable_feature_output.append({
            "outer_fold": outer_fold,
            "feature": feature,
            "selection_count": selection_counts[feature]
        })

    # --------------------------------------------------
    # EVALUATE THREE CONFIGURATIONS
    # --------------------------------------------------

    feature_sets = {
        "All_116": feature_columns,
        "Top_40": top40_features,
        "Stable_Inner": stable_features
    }

    for experiment, selected_features in feature_sets.items():

        model = create_model()

        model.fit(
            X_train[selected_features],
            y_train
        )

        y_pred = model.predict(
            X_test[selected_features]
        )

        metrics = calculate_metrics(y_test, y_pred)

        results.append({
            "outer_fold": outer_fold,
            "experiment": experiment,
            "features_used": len(selected_features),
            **metrics
        })

        oof_predictions[experiment][test_idx] = y_pred

        for position, prediction in zip(test_idx, y_pred):
            predictions_output.append({
                "experiment": experiment,
                "utterance_id": df.iloc[position]["utterance_id"],
                "speaker": df.iloc[position]["speaker"],
                "actual": int(y.iloc[position]),
                "predicted": int(prediction),
                "outer_fold": outer_fold
            })

        print(
            f"{experiment}: "
            f"{len(selected_features)} features | "
            f"Macro-F1={metrics['macro_f1']:.4f}"
        )


# --------------------------------------------------
# CHECK PREDICTIONS
# --------------------------------------------------

for name, prediction in oof_predictions.items():
    if np.any(prediction == -1):
        raise RuntimeError(f"Incomplete OOF predictions: {name}")


# --------------------------------------------------
# SAVE FOLD RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_DIR / "stable_feature_fold_results.csv",
    index=False
)


# --------------------------------------------------
# FOLD-AVERAGED SUMMARY
# --------------------------------------------------

summary = (
    results_df.groupby("experiment")
    .agg(
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        mean_accuracy=("accuracy", "mean"),
        mean_precision_macro=("precision_macro", "mean"),
        mean_recall_macro=("recall_macro", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        mean_features=("features_used", "mean")
    )
    .reset_index()
    .sort_values("mean_macro_f1", ascending=False)
)

summary.to_csv(
    OUTPUT_DIR / "stable_feature_summary.csv",
    index=False
)


# --------------------------------------------------
# POOLED OOF METRICS
# --------------------------------------------------

pooled_results = []

for experiment in experiments:

    metrics = calculate_metrics(
        y,
        oof_predictions[experiment]
    )

    pooled_results.append({
        "experiment": experiment,
        **metrics
    })

pooled_df = pd.DataFrame(pooled_results)

pooled_df.to_csv(
    OUTPUT_DIR / "stable_feature_pooled_results.csv",
    index=False
)


# --------------------------------------------------
# SAVE FEATURE SELECTION AND PREDICTIONS
# --------------------------------------------------

pd.DataFrame(stable_feature_output).to_csv(
    OUTPUT_DIR / "stable_features_by_outer_fold.csv",
    index=False
)

pd.DataFrame(predictions_output).to_csv(
    OUTPUT_DIR / "stable_feature_oof_predictions.csv",
    index=False
)


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("FOLD-AVERAGED RESULTS")
print("=" * 70)

print(summary.to_string(index=False))

print("\n" + "=" * 70)
print("POOLED OOF RESULTS")
print("=" * 70)

print(pooled_df.to_string(index=False))

print("\nResults saved to:")
print(OUTPUT_DIR)
