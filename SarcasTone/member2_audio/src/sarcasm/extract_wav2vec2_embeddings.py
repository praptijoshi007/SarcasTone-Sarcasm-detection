
import os

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = str(ROOT / ".cache")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import librosa
import torch

from transformers import (
    Wav2Vec2Model,
    Wav2Vec2FeatureExtractor
)

AUDIO_DIR = (
    ROOT / "data" / "processed" / "mustard" / "audio"
)

MANIFEST_FILE = (
    ROOT / "data" / "processed" / "mustard"
    / "mustard_audio_manifest.csv"
)

OUTPUT_DIR = Path(os.environ.get("SARCASTONE_EMBEDDINGS", ROOT / "outputs" / "embeddings"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "wav2vec2_embeddings.npz"
REPORT_FILE = OUTPUT_DIR / "wav2vec2_extraction_report.csv"

MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SR = 16000

torch.set_num_threads(2)


def main():

    print("=" * 65)
    print("SARCAS TONE: WAV2VEC2 EMBEDDING EXTRACTION")
    print("=" * 65)

    manifest = pd.read_csv(MANIFEST_FILE)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    model = Wav2Vec2Model.from_pretrained(
        MODEL_NAME,
        low_cpu_mem_usage=True
    )

    model.eval()
    model.to("cpu")

    embeddings = []
    utterance_ids = []
    reports = []

    total = len(manifest)

    for index, row in manifest.iterrows():

        utterance_id = str(row["utterance_id"])

        audio_path = AUDIO_DIR / f"{utterance_id}.wav"

        try:

            if not audio_path.exists():
                raise FileNotFoundError(str(audio_path))

            audio, sr = librosa.load(
                audio_path,
                sr=TARGET_SR,
                mono=True
            )

            if len(audio) == 0:
                raise ValueError("Empty audio")

            inputs = processor(
                audio,
                sampling_rate=TARGET_SR,
                return_tensors="pt",
                padding=True
            )

            with torch.inference_mode():

                outputs = model(
                    inputs.input_values
                )

                # Mean-pool the contextual frame embeddings.
                embedding = outputs.last_hidden_state.mean(
                    dim=1
                ).squeeze(0)

            embedding = embedding.cpu().numpy().astype(
                np.float32
            )

            if not np.isfinite(embedding).all():
                raise ValueError("Non-finite embedding")

            embeddings.append(embedding)
            utterance_ids.append(utterance_id)

            reports.append({
                "utterance_id": utterance_id,
                "status": "success",
                "error": ""
            })

        except Exception as error:

            reports.append({
                "utterance_id": utterance_id,
                "status": "failed",
                "error": str(error)
            })

        finally:

            if (index + 1) % 25 == 0 or index + 1 == total:

                print(
                    f"Processed {index + 1}/{total} "
                    f"| Successful: {len(embeddings)}"
                )

    if not embeddings:
        raise RuntimeError("No embeddings were extracted.")

    embedding_matrix = np.vstack(embeddings)

    np.savez_compressed(
        OUTPUT_FILE,
        embeddings=embedding_matrix,
        utterance_ids=np.array(utterance_ids)
    )

    pd.DataFrame(reports).to_csv(
        REPORT_FILE,
        index=False
    )

    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETED")
    print("=" * 65)

    print("Successful:", len(embeddings))
    print("Failed:", total - len(embeddings))
    print("Embedding shape:", embedding_matrix.shape)
    print("Output:", OUTPUT_FILE)
    print("Report:", REPORT_FILE)


if __name__ == "__main__":
    main()
