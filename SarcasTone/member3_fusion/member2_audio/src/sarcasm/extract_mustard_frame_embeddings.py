
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import numpy as np
import pandas as pd
import traceback
import gc
import time

from member2_audio.src.audio_intelligence.audio_intelligence import (
    get_audio_sequence_embedding
)

# ==================================================
# PATHS
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

AUDIO_DIR = (
    ROOT / "data" / "processed"
    / "mustard" / "audio"
)

OUTPUT_DIR = Path(os.environ.get("SARCASTONE_EMBEDDINGS", ROOT / "outputs" / "embeddings"))

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_NPZ = (
    OUTPUT_DIR / "mustard_frame_embeddings.npz"
)

OUTPUT_MANIFEST = (
    OUTPUT_DIR / "mustard_frame_embedding_manifest.csv"
)

# ==================================================
# SETTINGS
# ==================================================

SAVE_EVERY = 25

# ==================================================
# AUDIO FILES
# ==================================================

audio_files = sorted(
    AUDIO_DIR.glob("*.wav")
)

if not audio_files:
    raise FileNotFoundError(
        f"No WAV files found in {AUDIO_DIR}"
    )

print("=" * 65)
print("SARCAS TONE: MUSTARD FRAME EMBEDDING EXTRACTION")
print("=" * 65)

print("Total audio files:", len(audio_files))
print("Output directory:", OUTPUT_DIR)

# ==================================================
# EXTRACTION STORAGE
# ==================================================

embeddings = {}
manifest = []

success_count = 0
failure_count = 0

start_time = time.time()

# ==================================================
# EXTRACTION LOOP
# ==================================================

for index, audio_path in enumerate(audio_files, start=1):

    utterance_id = audio_path.stem

    try:

        sequence = get_audio_sequence_embedding(
            audio_path
        )

        pooled = sequence.mean(axis=0)

        if sequence.shape[1] != 768:
            raise ValueError(
                "Unexpected embedding dimension."
            )

        if not np.isfinite(sequence).all():
            raise ValueError(
                "Invalid embedding values."
            )

        # Store each utterance's variable-length sequence
        embeddings[utterance_id] = sequence

        manifest.append({
            "utterance_id": utterance_id,
            "audio_file": audio_path.name,
            "num_frames": sequence.shape[0],
            "embedding_dim": sequence.shape[1],
            "pooled_dim": pooled.shape[0],
            "duration_seconds": len(sequence) / 50.0,
            "status": "success"
        })

        success_count += 1

    except Exception as error:

        failure_count += 1

        manifest.append({
            "utterance_id": utterance_id,
            "audio_file": audio_path.name,
            "num_frames": 0,
            "embedding_dim": 768,
            "pooled_dim": 768,
            "duration_seconds": None,
            "status": f"failed: {error}"
        })

        print(
            f"\nFAILED: {utterance_id}: {error}"
        )

    # Progress
    if index % 10 == 0 or index == len(audio_files):

        elapsed = time.time() - start_time

        print(
            f"\rProcessed {index}/{len(audio_files)} | "
            f"Success: {success_count} | "
            f"Failed: {failure_count} | "
            f"Time: {elapsed/60:.1f} min",
            end="",
            flush=True
        )

    # Save checkpoint
    if index % SAVE_EVERY == 0 or index == len(audio_files):

        if embeddings:

            np.savez_compressed(
                OUTPUT_NPZ,
                **embeddings
            )

        pd.DataFrame(manifest).to_csv(
            OUTPUT_MANIFEST,
            index=False
        )

        gc.collect()

# ==================================================
# FINAL SUMMARY
# ==================================================

elapsed = time.time() - start_time

print("\n\n" + "=" * 65)
print("EXTRACTION COMPLETED")
print("=" * 65)

print("Total files:", len(audio_files))
print("Successful:", success_count)
print("Failed:", failure_count)
print(f"Elapsed time: {elapsed/60:.2f} minutes")

if embeddings:

    frame_counts = [
        array.shape[0]
        for array in embeddings.values()
    ]

    print(
        "Total embedding frames:",
        sum(frame_counts)
    )

    print(
        "Average frames per utterance:",
        round(np.mean(frame_counts), 2)
    )

    print(
        "Embedding dimension:",
        768
    )

print("\nSaved files:")
print(OUTPUT_NPZ)
print(OUTPUT_MANIFEST)

print("\n" + "=" * 65)
