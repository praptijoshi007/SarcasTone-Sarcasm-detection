
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DATA_FILE = (
    "data/processed/mustard/"
    "mustard_acoustic_features.csv"
)

df = pd.read_csv(DATA_FILE)

X = df.drop(
    columns=["utterance_id", "speaker", "sarcasm", "duration_seconds"]
)

y = df["sarcasm"].astype(int)
groups = df["speaker"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("=" * 65)
print("STRATIFIED GROUP 5-FOLD SPLIT ANALYSIS")
print("=" * 65)

all_test_indices = []

for fold, (train_idx, test_idx) in enumerate(
    splitter.split(X, y, groups),
    start=1
):

    train = df.iloc[train_idx]
    test = df.iloc[test_idx]

    train_speakers = set(train["speaker"])
    test_speakers = set(test["speaker"])

    overlap = train_speakers & test_speakers

    all_test_indices.extend(test_idx.tolist())

    print(f"\nFOLD {fold}")
    print("-" * 45)

    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")

    print(f"Train speakers: {len(train_speakers)}")
    print(f"Test speakers: {len(test_speakers)}")

    print(
        "Train labels:",
        train["sarcasm"].value_counts().sort_index().to_dict()
    )

    print(
        "Test labels:",
        test["sarcasm"].value_counts().sort_index().to_dict()
    )

    print("Held-out speakers:", sorted(test_speakers))
    print("Speaker overlap:", overlap)

    if test["sarcasm"].nunique() < 2:
        print("WARNING: Test fold contains only one class!")

print("\n" + "=" * 65)
print("OVERALL SPLIT CHECK")
print("=" * 65)

print("Total records:", len(df))
print("Unique test assignments:", len(set(all_test_indices)))
print("All samples tested exactly once:", 
      len(set(all_test_indices)) == len(df))

if len(set(all_test_indices)) != len(df):
    raise RuntimeError("Cross-validation coverage check failed.")
