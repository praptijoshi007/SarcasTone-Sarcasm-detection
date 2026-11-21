
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "outputs" / "analysis"
    / "nested_feature_selection"
    / "selected_features_by_fold.csv"
)

OUTPUT_DIR = (
    ROOT / "outputs" / "analysis"
    / "nested_feature_selection"
)

df = pd.read_csv(INPUT_FILE)

# Analyze only the Top 40 features from each outer fold.
top40 = df[df["experiment"] == "Top_40"].copy()

frequency = (
    top40.groupby("feature")
    .agg(
        selection_count=("outer_fold", "nunique"),
        selected_folds=(
            "outer_fold",
            lambda x: ", ".join(map(str, sorted(x.unique())))
        )
    )
    .reset_index()
)

frequency = frequency.sort_values(
    ["selection_count", "feature"],
    ascending=[False, True]
)

frequency["stability_percentage"] = (
    frequency["selection_count"] / 5 * 100
)

# Features selected in at least 3 of the 5 outer folds.
stable_features = frequency[
    frequency["selection_count"] >= 3
].copy()

frequency.to_csv(
    OUTPUT_DIR / "top40_feature_stability.csv",
    index=False
)

stable_features.to_csv(
    OUTPUT_DIR / "stable_acoustic_features.csv",
    index=False
)

print("=" * 65)
print("TOP 40 FEATURE STABILITY")
print("=" * 65)

print(f"Unique features selected: {len(frequency)}")
print(f"Stable features (3 or more folds): {len(stable_features)}")

print("\nMost consistently selected features:")
print(frequency.head(20).to_string(index=False))

print("\nStable feature list:")
print(stable_features["feature"].tolist())

print("\nSaved:")
print(OUTPUT_DIR / "top40_feature_stability.csv")
print(OUTPUT_DIR / "stable_acoustic_features.csv")
