
import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import json
import time
import gc

import joblib
import numpy as np
import pandas as pd

from member2_audio.src.audio_intelligence.audio_intelligence import (
    get_audio_quality
)

# ==================================================
# PATH CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

AUDIO_DIR = (
    ROOT / "data" / "processed"
    / "mustard" / "audio"
)

FEATURE_PATH = (
    ROOT / "data" / "processed"
    / "mustard" / "mustard_acoustic_features.csv"
)

EMBEDDING_DIR = Path(os.environ.get("SARCASTONE_EMBEDDINGS", ROOT / "outputs" / "embeddings"))

POOLED_PATH = (
    EMBEDDING_DIR / "wav2vec2_embeddings.npz"
)

FRAME_PATH = (
    EMBEDDING_DIR / "mustard_frame_embeddings.npz"
)

FRAME_MANIFEST_PATH = (
    EMBEDDING_DIR
    / "mustard_frame_embedding_manifest.csv"
)

EMOTION_MODEL_PATH = (
    ROOT / "checkpoints"
    / "ravdess_emotion_svm_rbf.joblib"
)

SARCASM_MODEL_PATH = (
    ROOT / "checkpoints" / "sarcasm"
    / "wav2vec2_sarcasm_svm.joblib"
)

OUTPUT_DIR = EMBEDDING_DIR / "member2_handoff"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==================================================
# LOAD DATA
# ==================================================

print("=" * 65)
print("SARCAS TONE: MEMBER 2 AUDIO HANDOFF EXPORT")
print("=" * 65)

print("\nLoading acoustic features...")

features_df = pd.read_csv(FEATURE_PATH)

features_df["utterance_id"] = (
    features_df["utterance_id"].astype(str)
)

assert len(features_df) == 690
assert features_df["utterance_id"].is_unique

print("Acoustic feature shape:", features_df.shape)

print("\nLoading pooled Wav2Vec2 embeddings...")

pooled_archive = np.load(
    POOLED_PATH,
    allow_pickle=False
)

pooled_embeddings = pooled_archive["embeddings"]
pooled_ids = pooled_archive["utterance_ids"].astype(str)

assert pooled_embeddings.shape == (690, 768)
assert len(pooled_ids) == 690
assert len(set(pooled_ids)) == 690

pooled_map = {
    uid: embedding
    for uid, embedding in zip(
        pooled_ids,
        pooled_embeddings
    )
}

print("Pooled embedding shape:", pooled_embeddings.shape)

print("\nLoading frame-level manifest...")

frame_manifest = pd.read_csv(
    FRAME_MANIFEST_PATH
)

frame_manifest["utterance_id"] = (
    frame_manifest["utterance_id"].astype(str)
)

assert frame_manifest["utterance_id"].is_unique

frame_map = frame_manifest.set_index(
    "utterance_id"
)["num_frames"].to_dict()

print("Frame manifest entries:", len(frame_map))

print("\nLoading trained models...")

emotion_bundle = joblib.load(
    EMOTION_MODEL_PATH
)

emotion_model = emotion_bundle["model"]

emotion_columns = emotion_bundle["feature_columns"]

sarcasm_model = joblib.load(
    SARCASM_MODEL_PATH
)

print("Emotion model loaded.")
print("Sarcasm model loaded.")

# ==================================================
# VALIDATE UTTERANCE ALIGNMENT
# ==================================================

feature_ids = set(
    features_df["utterance_id"]
)

pooled_id_set = set(pooled_ids)

frame_id_set = set(frame_map)

assert feature_ids == pooled_id_set, (
    "Mismatch between acoustic features and pooled embeddings."
)

assert feature_ids == frame_id_set, (
    "Mismatch between acoustic features and frame manifest."
)

print("\nAll 690 utterance IDs are aligned.")

# ==================================================
# PREPARE EMOTION FEATURES
# ==================================================

missing_columns = [
    column for column in emotion_columns
    if column not in features_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing emotion feature columns: {missing_columns}"
    )

# Preserve exact feature order expected by the trained model
emotion_X = features_df[
    emotion_columns
].copy()

if not np.isfinite(
    emotion_X.to_numpy(dtype=np.float64)
).all():
    raise ValueError(
        "Acoustic feature matrix contains invalid values."
    )

# ==================================================
# BATCH SARCASM PREDICTION
# ==================================================

print("\nGenerating sarcasm predictions...")

X_pooled = np.stack([
    pooled_map[uid]
    for uid in features_df["utterance_id"]
])

assert X_pooled.shape == (690, 768)

sarcasm_predictions = sarcasm_model.predict(
    X_pooled
)

sarcasm_scores = sarcasm_model.decision_function(
    X_pooled
)

# ==================================================
# BATCH EMOTION PREDICTION
# ==================================================

print("Generating emotion predictions...")

emotion_predictions = emotion_model.predict(
    emotion_X
)

emotion_scores = emotion_model.decision_function(
    emotion_X
)

# Store multiclass decision scores as JSON strings.
# They are not calibrated probabilities.

# ==================================================
# AUDIO QUALITY ASSESSMENT
# ==================================================

print("\nAssessing audio quality...")

quality_records = []

audio_files = sorted(
    AUDIO_DIR.glob("*.wav")
)

audio_map = {
    path.stem: path
    for path in audio_files
}

assert set(audio_map) == feature_ids, (
    "Audio files and feature IDs do not match."
)

start_time = time.time()

for index, uid in enumerate(
    features_df["utterance_id"],
    start=1
):

    try:

        quality = get_audio_quality(
            audio_map[uid]
        )

        quality_records.append({
            "utterance_id": uid,
            "audio_duration_seconds":
                quality.get("duration_seconds"),
            "sample_rate":
                quality.get("sample_rate"),
            "channels":
                quality.get("channels"),
            "peak_amplitude":
                quality.get("peak_amplitude"),
            "rms_energy":
                quality.get("rms_energy"),
            "clipping_ratio":
                quality.get("clipping_ratio"),
            "silence_ratio":
                quality.get("silence_ratio"),
            "audio_valid":
                quality.get("is_valid"),
            "audio_warnings":
                json.dumps(quality.get("warnings", [])),
            "quality_status": "success"
        })

    except Exception as error:

        quality_records.append({
            "utterance_id": uid,
            "audio_valid": False,
            "audio_warnings": str(error),
            "quality_status": "failed"
        })

    if index % 50 == 0 or index == len(features_df):

        print(
            f"\rQuality checked: {index}/690",
            end="",
            flush=True
        )

print()

quality_df = pd.DataFrame(
    quality_records
)

# ==================================================
# BUILD PREDICTION TABLE
# ==================================================

prediction_records = []

for index, uid in enumerate(
    features_df["utterance_id"]
):

    score = np.asarray(
        sarcasm_scores[index]
    ).reshape(-1)

    emotion_score = np.asarray(
        emotion_scores[index]
    ).reshape(-1)

    prediction_records.append({

        "utterance_id": uid,

        "audio_sarcasm_prediction":
            int(sarcasm_predictions[index]),

        "audio_sarcasm_label": (
            "sarcastic"
            if int(sarcasm_predictions[index]) == 1
            else "non-sarcastic"
        ),

        "audio_sarcasm_decision_score":
            float(score[0]),

        "sarcasm_score_type":
            "raw_svm_decision_score_not_probability",

        "predicted_emotion":
            str(emotion_predictions[index]),

        "emotion_decision_scores":
            json.dumps(
                emotion_score.tolist()
            ),

        "emotion_score_type":
            "raw_svm_decision_scores_not_probabilities",

        "pooled_embedding_dim": 768,

        "frame_count":
            int(frame_map[uid]),

        "frame_embedding_dim": 768
    })

predictions_df = pd.DataFrame(
    prediction_records
)

# ==================================================
# CREATE CONSOLIDATED HANDOFF TABLE
# ==================================================

print("\nCombining audio outputs...")

handoff_df = features_df.merge(
    predictions_df,
    on="utterance_id",
    how="inner",
    validate="one_to_one"
)

handoff_df = handoff_df.merge(
    quality_df,
    on="utterance_id",
    how="left",
    validate="one_to_one"
)

assert len(handoff_df) == 690
assert handoff_df["utterance_id"].is_unique

# ==================================================
# SAVE OUTPUTS
# ==================================================

print("\nSaving handoff files...")

HANDOFF_PATH = (
    OUTPUT_DIR / "mustard_audio_handoff.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR / "mustard_audio_predictions.csv"
)

QUALITY_PATH = (
    OUTPUT_DIR / "mustard_audio_quality.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "mustard_audio_handoff_summary.txt"
)

handoff_df.to_csv(
    HANDOFF_PATH,
    index=False
)

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)

quality_df.to_csv(
    QUALITY_PATH,
    index=False
)

# ==================================================
# FINAL SUMMARY
# ==================================================

sarcasm_counts = (
    handoff_df["audio_sarcasm_label"]
    .value_counts()
    .to_dict()
)

emotion_counts = (
    handoff_df["predicted_emotion"]
    .value_counts()
    .to_dict()
)

quality_success = int(
    (handoff_df["quality_status"] == "success").sum()
)

summary = f"""
SARCAS TONE - MEMBER 2 AUDIO HANDOFF

Total utterances: {len(handoff_df)}

Pooled embedding dimensions: 768
Frame embedding dimensions: 768
Acoustic feature dimensions: 116

Total frame embeddings:
{int(handoff_df['frame_count'].sum())}

Sarcasm prediction distribution:
{sarcasm_counts}

Emotion prediction distribution:
{emotion_counts}

Successful quality assessments:
{quality_success}

Failed quality assessments:
{len(handoff_df) - quality_success}

All utterance IDs aligned: True

Handoff directory:
{OUTPUT_DIR}
"""

SUMMARY_PATH.write_text(
    summary,
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("MEMBER 2 HANDOFF EXPORT COMPLETED")
print("=" * 65)

print("Total utterances:", len(handoff_df))
print("Handoff table shape:", handoff_df.shape)
print("Quality checks successful:", quality_success)

print("\nOutput files:")

for path in [
    HANDOFF_PATH,
    PREDICTIONS_PATH,
    QUALITY_PATH,
    SUMMARY_PATH
]:
    print(path)

print("\n" + summary)

gc.collect()
