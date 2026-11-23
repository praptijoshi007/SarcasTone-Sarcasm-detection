
from pathlib import Path
import numpy as np
import pandas as pd

import os

ROOT = Path(__file__).resolve().parents[2]

AUDIO_DIR = ROOT / "data" / "processed" / "mustard" / "audio"

OUTPUT_DIR = Path(os.environ.get("SARCASTONE_EMBEDDINGS", ROOT / "outputs" / "embeddings"))

NPZ_PATH = OUTPUT_DIR / "mustard_frame_embeddings.npz"

MANIFEST_PATH = (
    OUTPUT_DIR / "mustard_frame_embedding_manifest.csv"
)

print("=" * 60)
print("MUSTARD FRAME EMBEDDING VALIDATION")
print("=" * 60)

assert NPZ_PATH.exists(), "NPZ archive not found."
assert MANIFEST_PATH.exists(), "Manifest not found."

manifest = pd.read_csv(MANIFEST_PATH)

audio_ids = {
    p.stem for p in AUDIO_DIR.glob("*.wav")
}

archive_ids = set()
total_frames = 0
errors = []

with np.load(NPZ_PATH, allow_pickle=False) as archive:

    archive_ids = set(archive.files)

    print("Archive entries:", len(archive_ids))

    for index, utterance_id in enumerate(
        sorted(archive_ids), start=1
    ):

        try:
            sequence = archive[utterance_id]

            assert sequence.ndim == 2
            assert sequence.shape[1] == 768
            assert sequence.shape[0] > 0
            assert sequence.dtype == np.float32
            assert np.isfinite(sequence).all()

            total_frames += sequence.shape[0]

        except Exception as error:
            errors.append((utterance_id, str(error)))

        if index % 100 == 0:
            print(f"Validated {index}/{len(archive_ids)}")

manifest_ids = set(
    manifest.loc[
        manifest["status"] == "success",
        "utterance_id"
    ].astype(str)
)

print("\n" + "-" * 45)
print("VALIDATION SUMMARY")
print("-" * 45)

print("Audio files:", len(audio_ids))
print("Archive entries:", len(archive_ids))
print("Successful manifest entries:", len(manifest_ids))
print("Total frames:", total_frames)
print("Validation errors:", len(errors))

print("Missing from archive:", len(audio_ids - archive_ids))
print("Extra archive IDs:", len(archive_ids - audio_ids))
print("Missing from manifest:", len(archive_ids - manifest_ids))

if errors:
    print("\nFirst errors:", errors[:5])

assert len(audio_ids) == 690
assert len(archive_ids) == 690
assert audio_ids == archive_ids
assert archive_ids == manifest_ids
assert len(errors) == 0
assert total_frames == 179165

print("\nALL VALIDATION CHECKS PASSED!")
