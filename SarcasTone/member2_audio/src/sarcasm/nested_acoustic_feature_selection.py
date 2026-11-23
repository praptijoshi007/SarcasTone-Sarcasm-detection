
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


# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "mustard"
    / "mustard_acoustic_features.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "analysis"
    / "nested_feature_selection"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

OUTER_SPLITS = 5
INNER_SPLITS = 4

PERMUTATION_REPEATS = 3

FEATURE_COUNTS = [20, 40, 60]

LABELS = [0, 1]

EXPERIMENTS = [
    "All_116",
    "Top_20",
    "Top_40",
    "Top_60"
]


# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(INPUT_FILE)

metadata_columns = [
    "utterance_id",
    "speaker",
    "sarcasm",
    "duration_seconds"
]

feature_columns = [
    column for column in df.columns
    if column not in metadata_columns
]

X = df[feature_columns].copy()

y = df["sarcasm"].astype(int)

groups = df["speaker"].copy()

if X.isnull().any().any():
    raise ValueError("Missing values found in acoustic features.")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Non-finite values found in acoustic features.")

if y.nunique() != 2:
    raise ValueError("Expected two sarcasm classes.")

if groups.nunique() < OUTER_SPLITS:
    raise ValueError("Not enough speakers for outer cross-validation.")

print("=" * 70)
print("SARCAS TONE: NESTED ACOUSTIC FEATURE SELECTION")
print("=" * 70)

print(f"Samples: {len(df)}")
print(f"Speakers: {groups.nunique()}")
print(f"Acoustic features: {len(feature_columns)}")


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
# METRIC CALCULATION
# ==================================================

def calculate_metrics(y_true, y_pred):

    return {
        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        )
    }


# ==================================================
# OUTER CROSS-VALIDATION
# ==================================================

outer_cv = StratifiedGroupKFold(
    n_splits=OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

fold_results = []

prediction_records = []

selected_feature_records = []

# Store one prediction per utterance for each experiment.
oof_predictions = {
    name: np.full(len(df), -1, dtype=int)
    for name in EXPERIMENTS
}


for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(
    outer_cv.split(X, y, groups),
    start=1
):

    print("\n" + "=" * 70)
    print(f"OUTER FOLD {outer_fold}/{OUTER_SPLITS}")
    print("=" * 70)

    X_outer_train = X.iloc[outer_train_idx]
    X_outer_test = X.iloc[outer_test_idx]

    y_outer_train = y.iloc[outer_train_idx]
    y_outer_test = y.iloc[outer_test_idx]

    groups_outer_train = groups.iloc[outer_train_idx]
    groups_outer_test = groups.iloc[outer_test_idx]

    train_speakers = set(groups_outer_train)
    test_speakers = set(groups_outer_test)

    if train_speakers.intersection(test_speakers):
        raise RuntimeError("Outer speaker leakage detected!")

    print(f"Training samples: {len(outer_train_idx)}")
    print(f"Testing samples: {len(outer_test_idx)}")
    print(f"Training speakers: {len(train_speakers)}")
    print(f"Testing speakers: {len(test_speakers)}")

    # --------------------------------------------------
    # INNER CV: RANK FEATURES USING TRAINING SPEAKERS
    # --------------------------------------------------

    inner_cv = StratifiedGroupKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE + outer_fold
    )

    inner_importances = []

    for inner_fold, (inner_train_idx, inner_valid_idx) in enumerate(
        inner_cv.split(
            X_outer_train,
            y_outer_train,
            groups_outer_train
        ),
        start=1
    ):

        X_inner_train = X_outer_train.iloc[inner_train_idx]
        X_inner_valid = X_outer_train.iloc[inner_valid_idx]

        y_inner_train = y_outer_train.iloc[inner_train_idx]
        y_inner_valid = y_outer_train.iloc[inner_valid_idx]

        inner_train_speakers = set(
            groups_outer_train.iloc[inner_train_idx]
        )

        inner_valid_speakers = set(
            groups_outer_train.iloc[inner_valid_idx]
        )

        if inner_train_speakers.intersection(inner_valid_speakers):
            raise RuntimeError("Inner speaker leakage detected!")

        print(
            f"  Inner fold {inner_fold}/{INNER_SPLITS}: "
            f"training model and ranking features"
        )

        ranking_model = create_model()

        ranking_model.fit(
            X_inner_train,
            y_inner_train
        )

        importance_result = permutation_importance(
            ranking_model,
            X_inner_valid,
            y_inner_valid,
            scoring="f1_macro",
            n_repeats=PERMUTATION_REPEATS,
            random_state=RANDOM_STATE + outer_fold + inner_fold,
            n_jobs=-1
        )

        inner_importances.append(
            importance_result.importances_mean
        )

    # Average importance across inner validation folds.
    mean_importance = np.mean(
        np.vstack(inner_importances),
        axis=0
    )

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "mean_importance": mean_importance
    }).sort_values(
        "mean_importance",
        ascending=False
    )

    ranked_features = importance_df["feature"].tolist()

    # Save ranking for audit and reproducibility.
    importance_df.to_csv(
        OUTPUT_DIR / f"outer_fold_{outer_fold}_feature_ranking.csv",
        index=False
    )

    # --------------------------------------------------
    # SELECT FEATURES AND EVALUATE OUTER TEST FOLD
    # --------------------------------------------------

    selected_feature_sets = {
        "All_116": feature_columns
    }

    for count in FEATURE_COUNTS:

        selected_feature_sets[f"Top_{count}"] = (
            ranked_features[:count]
        )

    for experiment_name, selected_features in (
        selected_feature_sets.items()
    ):

        final_model = create_model()

        final_model.fit(
            X_outer_train[selected_features],
            y_outer_train
        )

        predictions = final_model.predict(
            X_outer_test[selected_features]
        )

        metrics = calculate_metrics(
            y_outer_test,
            predictions
        )

        fold_result = {
            "fold": outer_fold,
            "experiment": experiment_name,
            "features_used": len(selected_features),
            **metrics
        }

        fold_results.append(fold_result)

        # Store predictions at their original dataset positions.
        oof_predictions[experiment_name][outer_test_idx] = predictions

        for position, prediction in zip(
            outer_test_idx,
            predictions
        ):

            prediction_records.append({
                "experiment": experiment_name,
                "utterance_id": df.iloc[position]["utterance_id"],
                "speaker": df.iloc[position]["speaker"],
                "actual": int(y.iloc[position]),
                "predicted": int(prediction),
                "outer_fold": outer_fold
            })

        for feature in selected_features:

            selected_feature_records.append({
                "outer_fold": outer_fold,
                "experiment": experiment_name,
                "feature": feature
            })

        print(
            f"  {experiment_name}: "
            f"{len(selected_features)} features, "
            f"Macro-F1={metrics['macro_f1']:.4f}"
        )


# ==================================================
# VALIDATE OUT-OF-FOLD PREDICTIONS
# ==================================================

for experiment_name, predictions in oof_predictions.items():

    if np.any(predictions == -1):
        raise RuntimeError(
            f"Incomplete predictions for {experiment_name}"
        )


# ==================================================
# SAVE FOLD RESULTS
# ==================================================

fold_df = pd.DataFrame(fold_results)

fold_df.to_csv(
    OUTPUT_DIR / "nested_fold_results.csv",
    index=False
)


# ==================================================
# FOLD-AVERAGED SUMMARY
# ==================================================

summary = (
    fold_df
    .groupby("experiment")
    .agg(
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        mean_accuracy=("accuracy", "mean"),
        mean_precision_macro=("precision_macro", "mean"),
        mean_recall_macro=("recall_macro", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        features_used=("features_used", "mean")
    )
    .reset_index()
    .sort_values(
        "mean_macro_f1",
        ascending=False
    )
)

summary.to_csv(
    OUTPUT_DIR / "nested_feature_selection_summary.csv",
    index=False
)


# ==================================================
# POOLED OUT-OF-FOLD METRICS
# ==================================================

pooled_results = []

for experiment_name in EXPERIMENTS:

    predictions = oof_predictions[experiment_name]

    metrics = calculate_metrics(
        y,
        predictions
    )

    pooled_results.append({
        "experiment": experiment_name,
        "features_used": (
            len(feature_columns)
            if experiment_name == "All_116"
            else int(experiment_name.split("_")[1])
        ),
        **metrics
    })

pooled_df = pd.DataFrame(pooled_results)

pooled_df.to_csv(
    OUTPUT_DIR / "nested_pooled_oof_results.csv",
    index=False
)


# ==================================================
# SAVE ALL PREDICTIONS AND FEATURE SELECTIONS
# ==================================================

predictions_df = pd.DataFrame(prediction_records)

predictions_df.to_csv(
    OUTPUT_DIR / "nested_oof_predictions.csv",
    index=False
)

selected_df = pd.DataFrame(selected_feature_records)

selected_df.to_csv(
    OUTPUT_DIR / "selected_features_by_fold.csv",
    index=False
)


# ==================================================
# FEATURE SELECTION FREQUENCY
# ==================================================

selected_only = selected_df[
    selected_df["experiment"] != "All_116"
]

frequency = (
    selected_only
    .groupby(["experiment", "feature"])
    .size()
    .reset_index(name="selection_count")
)

frequency.to_csv(
    OUTPUT_DIR / "feature_selection_frequency.csv",
    index=False
)


# ==================================================
# DISPLAY RESULTS
# ==================================================

print("\n" + "=" * 70)
print("NESTED CROSS-VALIDATION: FOLD-AVERAGED RESULTS")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        formatters={
            "mean_macro_f1": "{:.4f}".format,
            "std_macro_f1": "{:.4f}".format,
            "mean_accuracy": "{:.4f}".format,
            "mean_precision_macro": "{:.4f}".format,
            "mean_recall_macro": "{:.4f}".format,
            "mean_balanced_accuracy": "{:.4f}".format,
            "features_used": "{:.1f}".format
        }
    )
)

print("\n" + "=" * 70)
print("POOLED OUT-OF-FOLD RESULTS")
print("=" * 70)

print(
    pooled_df.to_string(
        index=False,
        formatters={
            "accuracy": "{:.4f}".format,
            "precision_macro": "{:.4f}".format,
            "recall_macro": "{:.4f}".format,
            "macro_f1": "{:.4f}".format,
            "balanced_accuracy": "{:.4f}".format
        }
    )
)

print("\n" + "=" * 70)
print("NESTED FEATURE SELECTION COMPLETED")
print("=" * 70)

print(f"Results saved in:\n{OUTPUT_DIR}")
