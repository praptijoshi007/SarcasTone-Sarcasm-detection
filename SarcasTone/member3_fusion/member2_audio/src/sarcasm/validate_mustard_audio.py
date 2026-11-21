
import csv
import soundfile as sf
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

AUDIO_DIR = PROJECT_ROOT / "data/processed/mustard/audio"

OUTPUT_DIR = PROJECT_ROOT / "data/processed/mustard"

REPORT_FILE = OUTPUT_DIR / "mustard_audio_quality_report.csv"


def main():

    files = sorted(AUDIO_DIR.glob("*.wav"))

    print("=" * 55)
    print("MUStARD AUDIO VALIDATION")
    print("=" * 55)

    print(f"Total WAV files: {len(files)}")

    results = []
    valid_count = 0
    invalid_count = 0

    for index, path in enumerate(files, start=1):

        try:
            info = sf.info(str(path))

            audio, sample_rate = sf.read(
                str(path),
                dtype="float32",
                always_2d=True
            )

            finite = bool(np.isfinite(audio).all())
            non_empty = audio.size > 0 and info.frames > 0
            valid_rate = sample_rate > 0
            mono = audio.shape[1] == 1

            valid = (
                finite and
                non_empty and
                valid_rate and
                mono
            )

            if valid:
                valid_count += 1
                status = "valid"
                error = ""
            else:
                invalid_count += 1
                status = "invalid"
                error = (
                    f"finite={finite}, "
                    f"non_empty={non_empty}, "
                    f"valid_rate={valid_rate}, "
                    f"mono={mono}"
                )

            results.append({
                "utterance_id": path.stem,
                "file_path": str(path.resolve()),
                "sample_rate": sample_rate,
                "channels": audio.shape[1],
                "duration_seconds": round(
                    len(audio) / sample_rate, 4
                ) if sample_rate > 0 else 0,
                "peak_amplitude": round(
                    float(np.max(np.abs(audio))), 6
                ) if audio.size else 0,
                "status": status,
                "error": error
            })

        except Exception as exc:

            invalid_count += 1

            results.append({
                "utterance_id": path.stem,
                "file_path": str(path.resolve()),
                "sample_rate": "",
                "channels": "",
                "duration_seconds": "",
                "peak_amplitude": "",
                "status": "invalid",
                "error": str(exc)
            })

        if index % 50 == 0 or index == len(files):
            print(
                f"Processed {index}/{len(files)}"
            )

    fieldnames = [
        "utterance_id",
        "file_path",
        "sample_rate",
        "channels",
        "duration_seconds",
        "peak_amplitude",
        "status",
        "error"
    ]

    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nVALIDATION SUMMARY")
    print("=" * 40)
    print(f"Total files: {len(files)}")
    print(f"Valid files: {valid_count}")
    print(f"Invalid files: {invalid_count}")
    print(f"\nReport saved: {REPORT_FILE}")


if __name__ == "__main__":
    main()
