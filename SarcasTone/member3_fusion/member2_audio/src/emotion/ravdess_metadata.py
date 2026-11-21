
from pathlib import Path
import csv
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

RAVDESS_DIR = (
    ROOT / "data" / "raw" / "RAVDESS"
    / "Audio_Speech_Actors_01-24"
)

OUTPUT_FILE = (
    ROOT / "data" / "metadata"
    / "ravdess_metadata.csv"
)

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

INTENSITY_MAP = {
    "01": "normal",
    "02": "strong"
}


def parse_filename(file_path):

    parts = file_path.stem.split("-")

    if len(parts) != 7:
        raise ValueError(
            f"Invalid filename: {file_path.name}"
        )

    modality, channel, emotion, intensity, statement, repetition, actor = parts

    if modality != "03" or channel != "01":
        raise ValueError(
            f"Unexpected audio format: {file_path.name}"
        )

    if emotion not in EMOTION_MAP:
        raise ValueError(
            f"Unknown emotion: {file_path.name}"
        )

    if intensity not in INTENSITY_MAP:
        raise ValueError(
            f"Unknown intensity: {file_path.name}"
        )

    return {
        "sample_id": file_path.stem,
        "file_name": file_path.name,
        "file_path": str(file_path.relative_to(ROOT)),
        "actor_id": int(actor),
        "emotion_code": emotion,
        "emotion": EMOTION_MAP[emotion],
        "intensity": INTENSITY_MAP[intensity],
        "statement": statement,
        "repetition": repetition
    }


def build_metadata():

    if not RAVDESS_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAVDESS_DIR}"
        )

    audio_files = sorted(RAVDESS_DIR.rglob("*.wav"))

    records = [
        parse_filename(path)
        for path in audio_files
    ]

    if not records:
        raise ValueError("No WAV files found.")

    OUTPUT_FILE.parent.mkdir(
        parents=True, exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=records[0].keys()
        )

        writer.writeheader()
        writer.writerows(records)

    emotion_counts = Counter(
        row["emotion"] for row in records
    )

    actor_counts = Counter(
        row["actor_id"] for row in records
    )

    print("\nRAVDESS METADATA SUCCESS")
    print("=" * 40)
    print("Total audio files:", len(records))
    print("Unique actors:", len(actor_counts))

    print("\nEmotion distribution:")

    for emotion, count in sorted(emotion_counts.items()):
        print(f"{emotion}: {count}")

    print("\nActor distribution:")
    print("Minimum files per actor:", min(actor_counts.values()))
    print("Maximum files per actor:", max(actor_counts.values()))

    print("\nSaved metadata:", OUTPUT_FILE)


if __name__ == "__main__":
    build_metadata()
