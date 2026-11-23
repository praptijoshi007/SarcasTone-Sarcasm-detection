
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[2]

METADATA_FILE = ROOT / "data" / "metadata" / "ravdess_metadata.csv"

FEATURE_FILE = ROOT / "data" / "processed" / "ravdess_acoustic_features.csv"

OUTPUT_DIR = ROOT / "data" / "processed" / "emotion_dataset"

SEED = 42


def prepare_dataset():

    print("Loading RAVDESS metadata and acoustic features...")

    metadata = pd.read_csv(METADATA_FILE)
    features = pd.read_csv(FEATURE_FILE)

    # Check unique sample IDs
    if metadata["sample_id"].duplicated().any():
        raise ValueError("Duplicate IDs in metadata.")

    if features["sample_id"].duplicated().any():
        raise ValueError("Duplicate IDs in features.")

    # Merge using exact sample IDs
    dataset = metadata.merge(
        features,
        on="sample_id",
        how="inner",
        validate="one_to_one"
    )

    if len(dataset) != len(metadata) or len(dataset) != len(features):
        raise ValueError("Some samples were not matched.")

    # Define actual acoustic model features
    excluded = {
        "sample_id",
        "file_name",
        "file_path",
        "actor_id",
        "emotion_code",
        "emotion",
        "intensity",
        "statement",
        "repetition",
        "sample_rate",
        "duration_seconds"
    }

    feature_columns = [
        column for column in dataset.columns
        if column not in excluded
    ]

    X = dataset[feature_columns]

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values found in features.")

    print("\nDATASET SUMMARY")
    print("=" * 45)
    print("Total samples:", len(dataset))
    print("Unique actors:", dataset["actor_id"].nunique())
    print("Emotion classes:", dataset["emotion"].nunique())
    print("Acoustic feature dimensions:", len(feature_columns))

    print("\nEmotion distribution:")
    print(dataset["emotion"].value_counts().sort_index())

    # First split: 4 actors for testing, 20 for development
    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=4 / 24,
        random_state=SEED
    )

    development_idx, test_idx = next(
        first_split.split(
            dataset,
            groups=dataset["actor_id"]
        )
    )

    development = dataset.iloc[development_idx].copy()
    test = dataset.iloc[test_idx].copy()

    # Second split: 4 actors for validation, 16 for training
    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=4 / 20,
        random_state=SEED
    )

    train_idx, val_idx = next(
        second_split.split(
            development,
            groups=development["actor_id"]
        )
    )

    train = development.iloc[train_idx].copy()
    validation = development.iloc[val_idx].copy()

    # Verify speaker independence
    train_actors = set(train["actor_id"])
    val_actors = set(validation["actor_id"])
    test_actors = set(test["actor_id"])

    assert train_actors.isdisjoint(val_actors)
    assert train_actors.isdisjoint(test_actors)
    assert val_actors.isdisjoint(test_actors)

    # Save datasets
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    dataset.to_csv(
        OUTPUT_DIR / "ravdess_emotion_dataset.csv",
        index=False
    )

    # Save split information
    print("\nSPEAKER-INDEPENDENT SPLIT")
    print("=" * 45)

    for name, frame in [
        ("Training", train),
        ("Validation", validation),
        ("Testing", test)
    ]:
        print(
            f"{name}: {len(frame)} samples | "
            f"{frame['actor_id'].nunique()} actors"
        )

        print("Emotion counts:")
        print(frame["emotion"].value_counts().sort_index())

    print("\nSpeaker overlap:", 
          train_actors & val_actors |
          train_actors & test_actors |
          val_actors & test_actors)

    print("\nSaved files in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    prepare_dataset()
