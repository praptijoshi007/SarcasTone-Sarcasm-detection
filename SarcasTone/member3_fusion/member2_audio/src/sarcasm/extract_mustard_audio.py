
import csv
import subprocess
import wave
from pathlib import Path

from imageio_ffmpeg import get_ffmpeg_exe


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data/processed/mustard/mustard_audio_manifest.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/mustard/audio"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "data/processed/mustard/audio_extraction_report.csv"
)


def extract_audio(ffmpeg, video_path, output_path):
    """Extract mono PCM WAV while preserving the source sample rate."""

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
        "-y",
        "-i", str(video_path),
        "-map", "0:a:0",
        "-vn",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        str(output_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        if output_path.exists():
            output_path.unlink()

        error = result.stderr.strip().splitlines()
        return False, error[-1] if error else "FFmpeg extraction failed"

    # Verify the resulting WAV file
    try:
        with wave.open(str(output_path), "rb") as audio:
            sample_rate = audio.getframerate()
            channels = audio.getnchannels()
            frames = audio.getnframes()
            duration = frames / sample_rate if sample_rate else 0

        if frames <= 0 or sample_rate <= 0:
            raise ValueError("Empty or invalid WAV file")

        return True, {
            "sample_rate": sample_rate,
            "channels": channels,
            "duration": round(duration, 4),
            "frames": frames
        }

    except Exception as error:
        if output_path.exists():
            output_path.unlink()

        return False, str(error)


def main():

    print("=" * 60)
    print("MUStARD VIDEO TO AUDIO EXTRACTION")
    print("=" * 60)

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ffmpeg = get_ffmpeg_exe()
    print(f"\nFFmpeg executable: {ffmpeg}")

    with open(
        MANIFEST_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        records = list(csv.DictReader(file))

    print(f"Manifest records: {len(records)}")

    results = []

    successful = 0
    failed = 0

    for index, record in enumerate(records, start=1):

        utterance_id = record["utterance_id"]
        video_path = Path(record["video_path"])

        output_path = OUTPUT_DIR / f"{utterance_id}.wav"

        print(
            f"[{index}/{len(records)}] "
            f"Processing {utterance_id}...",
            end=" "
        )

        if not video_path.exists():
            success = False
            details = "Source video not found"

        else:
            success, details = extract_audio(
                ffmpeg,
                video_path,
                output_path
            )

        if success:
            successful += 1

            print(
                f"OK | {details['sample_rate']} Hz | "
                f"{details['duration']} sec"
            )

            results.append({
                "utterance_id": utterance_id,
                "video_path": str(video_path),
                "audio_path": str(output_path.resolve()),
                "status": "success",
                "sample_rate": details["sample_rate"],
                "channels": details["channels"],
                "duration_seconds": details["duration"],
                "error": ""
            })

        else:
            failed += 1
            print(f"FAILED | {details}")

            results.append({
                "utterance_id": utterance_id,
                "video_path": str(video_path),
                "audio_path": "",
                "status": "failed",
                "sample_rate": "",
                "channels": "",
                "duration_seconds": "",
                "error": str(details)
            })

    # Save extraction report
    fieldnames = [
        "utterance_id",
        "video_path",
        "audio_path",
        "status",
        "sample_rate",
        "channels",
        "duration_seconds",
        "error"
    ]

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 60)
    print("AUDIO EXTRACTION SUMMARY")
    print("=" * 60)

    print(f"Total videos: {len(records)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    print(f"\nAudio folder: {OUTPUT_DIR}")
    print(f"Extraction report: {REPORT_FILE}")

    print("\nExtraction process completed.")


if __name__ == "__main__":
    main()
