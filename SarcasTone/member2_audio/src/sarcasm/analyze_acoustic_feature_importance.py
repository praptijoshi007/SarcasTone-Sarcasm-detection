
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score


# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "data" / "processed" / "mustard"
    / "mustard_acoustic_features.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "analysis" / "feature_importance"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10

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
    raise ValueError("Missing values found in acoustic features.")

if not np.isfinite(X.to_numpy()).all():
    raise ValueError("Non-finite values found in acoustic features.")

print("=" * 70)
print("SARCAS TONE: ACOUSTIC FEATURE IMPORTANCE ANALYSIS")
print("=" * 70)

print(f"Total samples: {len(df)}")
print(f"Total speakers: {groups.nunique()}")
print(f"Total acoustic features: {len(feature_columns)}")


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
# SPEAKER-INDEPENDENT CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedGroupKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)

importance_records = []
fold_scores = []

for fold_number, (train_idx, test_idx) in enumerate(
    cv.split(X, y, groups),
    start=1
):

    print(f"\nProcessing Fold {fold_number}/{N_SPLITS}")

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    train_speakers = set(groups.iloc[train_idx])
    test_speakers = set(groups.iloc[test_idx])

    if train_speakers.intersection(test_speakers):
        raise RuntimeError("Speaker leakage detected!")

    model = create_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    fold_f1 = f1_score(
        y_test,
        predictions,
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    fold_scores.append({
        "fold": fold_number,
        "test_samples": len(test_idx),
        "macro_f1": fold_f1
    })

    print(f"Test samples: {len(test_idx)}")
    print(f"Macro-F1: {fold_f1:.4f}")

    # ----------------------------------------------
    # PERMUTATION IMPORTANCE
    # ----------------------------------------------

    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="f1_macro",
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    for index, feature_name in enumerate(feature_columns):

        importance_records.append({
            "fold": fold_number,
            "feature": feature_name,
            "importance_mean": result.importances_mean[index],
            "importance_std": result.importances_std[index]
        })


# --------------------------------------------------
# SAVE FOLD RESULTS
# --------------------------------------------------

fold_df = pd.DataFrame(fold_scores)

fold_df.to_csv(
    OUTPUT_DIR / "fold_performance.csv",
    index=False
)

print("\nFold performance:")
print(fold_df.to_string(index=False))

print(
    f"\nMean Macro-F1: {fold_df['macro_f1'].mean():.4f}"
)

print(
    f"Standard Deviation: {fold_df['macro_f1'].std():.4f}"
)


# --------------------------------------------------
# AGGREGATE FEATURE IMPORTANCE
# --------------------------------------------------

importance_df = pd.DataFrame(importance_records)

summary = (
    importance_df
    .groupby("feature")
    .agg(
        mean_importance=("importance_mean", "mean"),
        std_importance=("importance_mean", "std"),
        mean_repeat_std=("importance_std", "mean")
    )
    .reset_index()
)

summary = summary.sort_values(
    "mean_importance",
    ascending=False
)

summary.to_csv(
    OUTPUT_DIR / "acoustic_feature_importance.csv",
    index=False
)

print("\nTop 20 acoustic features:")

print(
    summary.head(20).to_string(
        index=False,
        formatters={
            "mean_importance": "{:.6f}".format,
            "std_importance": "{:.6f}".format,
            "mean_repeat_std": "{:.6f}".format
        }
    )
)


# --------------------------------------------------
# TOP 20 FEATURE VISUALIZATION
# --------------------------------------------------

top_features = summary.head(20).sort_values(
    "mean_importance",
    ascending=True
)

plt.figure(figsize=(11, 8))

plt.barh(
    top_features["feature"],
    top_features["mean_importance"],
    xerr=top_features["std_importance"]
)

plt.axvline(0, linestyle="--", linewidth=1)

plt.title("Top 20 Acoustic Features - SVM Permutation Importance")
plt.xlabel("Mean Decrease in Macro-F1 After Permutation")
plt.ylabel("Acoustic Feature")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "top_20_acoustic_features.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# FEATURE FAMILY ANALYSIS
# --------------------------------------------------

def get_feature_family(feature):

    if feature.startswith("mfcc_"):
        return "MFCC"

    if feature.startswith("mel_"):
        return "Mel Spectrogram"

    if feature.startswith("spectral_centroid"):
        return "Spectral Centroid"

    if feature.startswith("spectral_rolloff"):
        return "Spectral Rolloff"

    if feature.startswith("zero_crossing_rate"):
        return "Zero Crossing Rate"

    if feature.startswith("rms_energy"):
        return "RMS Energy"

    if feature.startswith("pitch_f0"):
        return "Pitch F0"

    return "Other"


summary["feature_family"] = summary["feature"].apply(
    get_feature_family
)

family_summary = (
    summary
    .groupby("feature_family")
    .agg(
        mean_importance=("mean_importance", "mean"),
        total_features=("feature", "count")
    )
    .reset_index()
    .sort_values("mean_importance", ascending=False)
)

family_summary.to_csv(
    OUTPUT_DIR / "feature_family_importance.csv",
    index=False
)

print("\nFeature family importance:")
print(family_summary.to_string(index=False))


# --------------------------------------------------
# FEATURE FAMILY VISUALIZATION
# --------------------------------------------------

family_plot = family_summary.sort_values(
    "mean_importance",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    family_plot["feature_family"],
    family_plot["mean_importance"]
)

plt.axvline(0, linestyle="--", linewidth=1)

plt.title("Acoustic Feature Family Importance")
plt.xlabel("Average Permutation Importance")
plt.ylabel("Feature Family")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "feature_family_importance.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE ANALYSIS COMPLETED")
print("=" * 70)

print(f"Features analyzed: {len(feature_columns)}")
print(f"Cross-validation folds: {N_SPLITS}")
print(f"Permutation repeats per feature per fold: {N_REPEATS}")

print(f"\nResults saved in:\n{OUTPUT_DIR}")
