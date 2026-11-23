
from pathlib import Path
import csv
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]

RAVDESS_DIR = (
    ROOT / "data" / "raw" / "RAVDESS"
    / "Audio_Speech_Actors_01-24"
)

OUTPUT_FILE = (
    ROOT / "data" / "metadata"
    / "ravdess_audio_quality_report.csv"
)


def validate_audio(audio_path):

    result = {
        "file_name": audio_path.name,
        "status": "valid",
        "sample_rate": None,
        "channels": None,
        "duration_seconds": None,
        "samples": None,
        "issue": ""
    }

    try:
        info = sf.info(str(audio_path))

        result["sample_rate"] = info.samplerate
        result["channels"] = info.channels
        result["duration_seconds"] = info.duration
        result["samples"] = info.frames

        if info.frames <= 0:
            raise ValueError("Empty audio file")

        if info.samplerate <= 0:
            raise ValueError("Invalid sample rate")

        if info.channels <= 0:
            raise ValueError("Invalid channel count")

        waveform, _ = sf.read(
            str(audio_path),
            dtype="float32"
        )

        if waveform.size == 0:
            raise ValueError("No audio samples")

        if not np.isfinite(waveform).all():
            raise ValueError("NaN or Inf values found")

    except Exception as error:

        result["status"] = "invalid"
        result["issue"] = str(error)

    return result


def run_validation():

    audio_files = sorted(
        RAVDESS_DIR.rglob("*.wav")
    )

    if not audio_files:
        raise FileNotFoundError(
            f"No WAV files found in {RAVDESS_DIR}"
        )

    results = [
        validate_audio(path)
        for path in audio_files
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    valid = sum(
        row["status"] == "valid"
        for row in results
    )

    invalid = len(results) - valid

    print("\nRAVDESS AUDIO VALIDATION")
    print("=" * 40)
    print("Total files:", len(results))
    print("Valid files:", valid)
    print("Invalid files:", invalid)

    print("\nSample rates:")

    rates = sorted({
        row["sample_rate"]
        for row in results
        if row["sample_rate"] is not None
    })

    for rate in rates:
        count = sum(
            row["sample_rate"] == rate
            for row in results
        )
        print(f"{rate} Hz: {count} files")

    print("\nQuality report saved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    run_validation()
