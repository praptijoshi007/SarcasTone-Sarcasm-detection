
import json
from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANNOTATION_FILE = (
    PROJECT_ROOT
    / "data/raw/MUSTARD/MUStARD-master/data/sarcasm_data.json"
)

VIDEO_DIR = (
    PROJECT_ROOT
    / "data/raw/MUSTARD/mmsd_raw_data/utterances_final"
)

OUTPUT_DIR = PROJECT_ROOT / "data/processed/mustard"
OUTPUT_FILE = OUTPUT_DIR / "mustard_audio_manifest.csv"
REPORT_FILE = OUTPUT_DIR / "mustard_matching_report.txt"


def main():

    print("=" * 55)
    print("MUStARD ANNOTATION-VIDEO MATCHING")
    print("=" * 55)

    # Check input paths
    if not ANNOTATION_FILE.exists():
        raise FileNotFoundError(
            f"Annotation file not found: {ANNOTATION_FILE}"
        )

    if not VIDEO_DIR.exists():
        raise FileNotFoundError(
            f"Video directory not found: {VIDEO_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load annotations
    with open(ANNOTATION_FILE, "r", encoding="utf-8") as file:
        annotations = json.load(file)

    print(f"\nAnnotation records: {len(annotations)}")

    # Build video ID lookup
    video_files = list(VIDEO_DIR.glob("*.mp4"))

    video_lookup = {}

    for video_path in video_files:
        video_id = video_path.stem

        if video_id in video_lookup:
            raise ValueError(
                f"Duplicate video ID found: {video_id}"
            )

        video_lookup[video_id] = video_path

    print(f"Extracted video files: {len(video_lookup)}")

    # Compare annotation IDs and video IDs
    annotation_ids = set(annotations.keys())
    video_ids = set(video_lookup.keys())

    matched_ids = annotation_ids & video_ids
    missing_video_ids = annotation_ids - video_ids
    extra_video_ids = video_ids - annotation_ids

    # Build manifest
    rows = []

    for utterance_id, record in annotations.items():

        video_path = video_lookup.get(utterance_id)

        context = record.get("context", [])
        context_speakers = record.get("context_speakers", [])

        rows.append({
            "utterance_id": utterance_id,
            "utterance": record.get("utterance", ""),
            "speaker": record.get("speaker", ""),
            "show": record.get("show", ""),
            "sarcasm": int(bool(record.get("sarcasm", False))),
            "context": " || ".join(context),
            "context_speakers": " || ".join(context_speakers),
            "video_path": str(video_path.resolve()) if video_path else "",
            "video_exists": video_path is not None
        })

    manifest = pd.DataFrame(rows)

    # Save manifest
    manifest.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    # Matching statistics
    sarcasm_counts = manifest["sarcasm"].value_counts().to_dict()

    report = [
        "MUStARD MATCHING REPORT",
        "=" * 40,
        f"Total annotations: {len(annotation_ids)}",
        f"Total videos: {len(video_ids)}",
        f"Matched IDs: {len(matched_ids)}",
        f"Missing videos: {len(missing_video_ids)}",
        f"Extra videos: {len(extra_video_ids)}",
        f"Sarcasm samples: {sarcasm_counts.get(1, 0)}",
        f"Non-sarcasm samples: {sarcasm_counts.get(0, 0)}",
        "",
        f"Manifest: {OUTPUT_FILE}",
        ""
    ]

    if missing_video_ids:
        report.append("ANNOTATIONS WITHOUT VIDEOS:")
        report.extend(sorted(missing_video_ids))

    if extra_video_ids:
        report.append("VIDEOS WITHOUT ANNOTATIONS:")
        report.extend(sorted(extra_video_ids))

    report_text = "\n".join(report)

    REPORT_FILE.write_text(report_text, encoding="utf-8")

    print("\nMATCHING RESULTS")
    print("=" * 40)
    print(f"Matched IDs: {len(matched_ids)}")
    print(f"Missing videos: {len(missing_video_ids)}")
    print(f"Extra videos: {len(extra_video_ids)}")
    print(f"Sarcasm samples: {sarcasm_counts.get(1, 0)}")
    print(f"Non-sarcasm samples: {sarcasm_counts.get(0, 0)}")

    print("\nFiles saved:")
    print(OUTPUT_FILE)
    print(REPORT_FILE)

    print("\nMatching completed successfully.")


if __name__ == "__main__":
    main()
