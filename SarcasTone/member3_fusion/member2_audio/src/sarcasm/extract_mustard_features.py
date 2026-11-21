
import numpy as np
import pandas as pd
import librosa

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data/processed/mustard/mustard_audio_manifest.csv"
)

AUDIO_DIR = (
    PROJECT_ROOT
    / "data/processed/mustard/audio"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/mustard"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "mustard_acoustic_features.csv"
)

TARGET_SR = 16000
N_MFCC = 13
N_MELS = 40
N_FFT = 1024
HOP_LENGTH = 256


def calculate_statistics(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0
        }

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values))
    }


def extract_features(audio_path):

    # Load and resample to a common sample rate.
    audio, sr = librosa.load(
        str(audio_path),
        sr=TARGET_SR,
        mono=True
    )

    if audio.size == 0:
        raise ValueError("Empty audio recording")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains non-finite values")

    features = {}

    # MFCC features
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    for i in range(N_MFCC):
        features.update(
            calculate_statistics(
                mfcc[i],
                f"mfcc_{i + 1}"
            )
        )

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    for i in range(N_MELS):
        features.update(
            calculate_statistics(
                mel_db[i],
                f"mel_{i + 1}"
            )
        )

    # Spectral centroid
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.update(
        calculate_statistics(
            centroid[0],
            "spectral_centroid"
        )
    )

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.update(
        calculate_statistics(
            rolloff[0],
            "spectral_rolloff"
        )
    )

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.update(
        calculate_statistics(
            zcr[0],
            "zero_crossing_rate"
        )
    )

    # RMS energy
    rms = librosa.feature.rms(
        y=audio,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.update(
        calculate_statistics(
            rms[0],
            "rms_energy"
        )
    )

    # Fundamental frequency (F0)
    f0 = librosa.yin(
        audio,
        fmin=65,
        fmax=500,
        sr=sr,
        frame_length=N_FFT,
        hop_length=HOP_LENGTH
    )

    features.update(
        calculate_statistics(
            f0,
            "pitch_f0"
        )
    )

    return features, len(audio) / sr


def main():

    print("=" * 60)
    print("MUStARD ACOUSTIC FEATURE EXTRACTION")
    print("=" * 60)

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(MANIFEST_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(
        MANIFEST_FILE,
        encoding="utf-8-sig"
    )

    results = []
    failures = []

    total = len(manifest)

    print(f"\nTotal recordings: {total}")
    print(f"Target sample rate: {TARGET_SR} Hz")

    for index, record in manifest.iterrows():

        utterance_id = str(record["utterance_id"])
        audio_path = AUDIO_DIR / f"{utterance_id}.wav"

        try:

            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Audio not found: {audio_path}"
                )

            features, duration = extract_features(audio_path)

            row = {
                "utterance_id": utterance_id,
                "speaker": record["speaker"],
                "sarcasm": int(record["sarcasm"]),
                "duration_seconds": round(duration, 4)
            }

            row.update(features)
            results.append(row)

        except Exception as error:

            failures.append({
                "utterance_id": utterance_id,
                "error": str(error)
            })

        processed = index + 1

        if processed % 50 == 0 or processed == total:
            print(
                f"Processed {processed}/{total} | "
                f"Successful: {len(results)} | "
                f"Failed: {len(failures)}"
            )

    # Save successful feature records
    feature_df = pd.DataFrame(results)

    if not feature_df.empty:
        feature_df.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig"
        )

    # Save failures separately
    failure_file = OUTPUT_DIR / "mustard_feature_failures.csv"

    pd.DataFrame(
        failures,
        columns=["utterance_id", "error"]
    ).to_csv(
        failure_file,
        index=False,
        encoding="utf-8-sig"
    )

    # Final summary
    feature_columns = [
        col for col in feature_df.columns
        if col not in [
            "utterance_id",
            "speaker",
            "sarcasm",
            "duration_seconds"
        ]
    ]

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION SUMMARY")
    print("=" * 60)

    print(f"Total recordings: {total}")
    print(f"Successful: {len(results)}")
    print(f"Failed: {len(failures)}")
    print(f"Acoustic feature dimensions: {len(feature_columns)}")

    if not feature_df.empty:
        print(f"Output shape: {feature_df.shape}")
        print(
            "Missing feature values:",
            int(feature_df[feature_columns].isna().sum().sum())
        )
        print(
            "Duplicate utterance IDs:",
            int(feature_df["utterance_id"].duplicated().sum())
        )

    print(f"\nFeature file: {OUTPUT_FILE}")
    print(f"Failure report: {failure_file}")


if __name__ == "__main__":
    main()