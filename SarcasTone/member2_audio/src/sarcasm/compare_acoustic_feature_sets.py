
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    balanced_accuracy_score
)
from sklearn.inspection import permutation_importance


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "mustard"
    / "mustard_acoustic_features.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "analysis" / "feature_selection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5

LABELS = [0, 1]

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

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

X = df[feature_columns].copy()
y = df["sarcasm"].astype(int)
groups = df["speaker"]

if X.isnull().any().any():
    raise ValueError("Missing values found in features.")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Non-finite feature values found.")

# --------------------------------------------------
# FEATURE GROUPS
# --------------------------------------------------

def get_feature_family(feature):

    if feature.startswith("mfcc_"):
        return "MFCC"

    if feature.startswith("mel_"):
        return "Mel"

    if feature.startswith("rms_energy"):
        return "RMS"

    if feature.startswith("spectral_centroid"):
        return "Spectral_Centroid"

    if feature.startswith("spectral_rolloff"):
        return "Spectral_Rolloff"

    if feature.startswith("zero_crossing_rate"):
        return "Zero_Crossing_Rate"

    if feature.startswith("pitch_f0"):
        return "Pitch"

    return "Other"


# --------------------------------------------------
# MODEL
# --------------------------------------------------

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


# --------------------------------------------------
# EXPERIMENT DEFINITIONS
# --------------------------------------------------

experiment_names = [
    "All_116",
    "Top_20",
    "Top_40",
    "Selected_Families"
]

fold_results = []

# --------------------------------------------------
# SPEAKER-INDEPENDENT CV
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

for fold, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    print("\n" + "=" * 65)
    print(f"PROCESSING FOLD {fold}/{N_SPLITS}")
    print("=" * 65)

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_groups = groups.iloc[train_idx]
    test_groups = groups.iloc[test_idx]

    if set(train_groups).intersection(set(test_groups)):
        raise RuntimeError("Speaker leakage detected!")

    # --------------------------------------------------
    # SELECT FEATURES USING TRAINING DATA ONLY
    # --------------------------------------------------

    importance_model = create_model()
    importance_model.fit(X_train, y_train)

    importance = permutation_importance(
        importance_model,
        X_train,
        y_train,
        scoring="f1_macro",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": importance.importances_mean
    }).sort_values(
        "importance",
        ascending=False
    )

    top_20 = importance_df.head(20)["feature"].tolist()
    top_40 = importance_df.head(40)["feature"].tolist()

    # Family selection is also based on training data only.
    importance_df["family"] = (
        importance_df["feature"].apply(get_feature_family)
    )

    selected_families = [
        "MFCC",
        "Mel",
        "RMS"
    ]

    family_features = importance_df[
        importance_df["family"].isin(selected_families)
    ]["feature"].tolist()

    experiment_features = {
        "All_116": feature_columns,
        "Top_20": top_20,
        "Top_40": top_40,
        "Selected_Families": family_features
    }

    # --------------------------------------------------
    # TRAIN AND EVALUATE EACH FEATURE SET
    # --------------------------------------------------

    for experiment_name, selected_features in (
        experiment_features.items()
    ):

        model = create_model()

        model.fit(
            X_train[selected_features],
            y_train
        )

        predictions = model.predict(
            X_test[selected_features]
        )

        result = {
            "fold": fold,
            "experiment": experiment_name,
            "features_used": len(selected_features),
            "accuracy": accuracy_score(
                y_test, predictions
            ),
            "macro_f1": f1_score(
                y_test,
                predictions,
                labels=LABELS,
                average="macro",
                zero_division=0
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            )
        }

        fold_results.append(result)

        print(
            f"{experiment_name}: "
            f"Features={len(selected_features)}, "
            f"Macro-F1={result['macro_f1']:.4f}"
        )


# --------------------------------------------------
# SAVE FOLD RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(fold_results)

results_df.to_csv(
    OUTPUT_DIR / "feature_selection_fold_results.csv",
    index=False
)

# --------------------------------------------------
# AGGREGATE RESULTS
# --------------------------------------------------

summary = (
    results_df
    .groupby("experiment")
    .agg(
        mean_macro_f1=("macro_f1", "mean"),
        std_macro_f1=("macro_f1", "std"),
        mean_accuracy=("accuracy", "mean"),
        mean_balanced_accuracy=("balanced_accuracy", "mean"),
        features_used=("features_used", "mean")
    )
    .reset_index()
    .sort_values("mean_macro_f1", ascending=False)
)

summary.to_csv(
    OUTPUT_DIR / "feature_selection_comparison.csv",
    index=False
)

# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

print("\n" + "=" * 70)
print("FINAL FEATURE SELECTION COMPARISON")
print("=" * 70)

print(
    summary.to_string(
        index=False,
        formatters={
            "mean_macro_f1": "{:.4f}".format,
            "std_macro_f1": "{:.4f}".format,
            "mean_accuracy": "{:.4f}".format,
            "mean_balanced_accuracy": "{:.4f}".format,
            "features_used": "{:.1f}".format
        }
    )
)

print("\nResults saved in:")
print(OUTPUT_DIR)
