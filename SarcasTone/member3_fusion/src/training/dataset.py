"""Multimodal dataset for member3_fusion.

Loads MUStARD (and optionally MUStARD++) from:
  - datasets/MUSTARD/sarcasm_data.json          (labels, text, context, speaker)
  - datasets/MUSTARD_PP/mustard++_text.csv      (MUStARD++ labelled CSV)
  - member2_audio/outputs/handoff/mustard_audio_handoff.csv    (acoustic features)
  - member2_audio/outputs/handoff/mustard_audio_quality.csv    (audio quality)
  - member2_audio/outputs/handoff/mustard_audio_predictions.csv (audio preds)
  - (optional) wav2vec2_embeddings.npz            (member2 pooled audio embeddings)
  - (optional) text_embeddings.npz                (member1 cached text embeddings)

Design principles
-----------------
* Every row has a sarcasm label that came from the actual dataset file.
* Missing modalities are represented by zeros and a boolean mask.
* Acoustic feature columns are aligned positionally to shared.constants.PROSODY_FEATURES
  where available; full 116-column vectors are also stored for the classifier.
* Speaker-aware splitting uses shared.splits utilities.
* The class never fabricates a probability or an embedding.
* Training cannot silently proceed with zero text — a warning is emitted
  and the batch carries a ``text_available`` bool mask.
* MUStARD++ rows are loaded via a separate CSV loader and merged before splitting.
"""
from __future__ import annotations

import json
import logging
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    class Dataset:  # type: ignore
        pass

# shared utilities (repo-level)
import sys
_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from shared.constants import (
    PROSODY_FEATURES,
    PROSODY_DIM,
    QUALITY_FEATURES,
    QUALITY_DIM,
    TEXT_HIDDEN,
    AUDIO_HIDDEN,
)
from shared.splits import assign_speaker_splits
from shared.examples import UtteranceExample

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mapping from MUSTARD++ / handoff CSV columns to shared quality/prosody names
# ---------------------------------------------------------------------------
_HANDOFF_PROSODY_MAP: dict[str, str] = {
    "pitch_f0_mean":          "f0_mean",
    "pitch_f0_std":           "f0_std",
    "rms_energy_mean":        "rms_mean",
    "rms_energy_std":         "rms_std",
    "spectral_centroid_mean": "spectral_centroid_mean",
    "spectral_centroid_std":  "spectral_centroid_std",
    "spectral_rolloff_mean":  "spectral_rolloff_mean",
    "spectral_rolloff_std":   "spectral_rolloff_std",
    "zero_crossing_rate_mean":"zcr_mean",
    "zero_crossing_rate_std": "zcr_std",
    "duration_seconds":       "duration_sec",
}

# MUStARD++ CSV required columns
_MUSTARD_PP_LABEL_COLS = {"sarcasm", "label", "is_sarcastic"}  # any one accepted
_MUSTARD_PP_TEXT_COLS  = {"utterance", "text", "sentence"}


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

class FusionRecord:
    """One training/evaluation sample with all modalities."""

    __slots__ = (
        "example_id",
        "dataset",
        "text",
        "context",
        "context_embedding",
        "has_context",
        "speaker",
        "show",
        "label",
        "split",
        # Acoustic
        "audio_embedding",        # np.ndarray [768] or None
        "audio_available",        # bool
        "prosody_vector",         # np.ndarray [PROSODY_DIM]
        "acoustic_features_full", # np.ndarray [116] (all handoff cols)
        "quality_vector",         # np.ndarray [QUALITY_DIM]
        "quality_ok",             # bool
        "audio_duration_sec",     # float | None
        # Text (member1 cached embeddings)
        "text_embedding",         # np.ndarray [TEXT_HIDDEN] or None
        "text_available",         # bool
    )

    def __init__(
        self,
        example_id: str,
        dataset: str,
        text: str,
        context: list[str],
        speaker: str,
        label: int,
        show: str | None = None,
        split: str = "train",
        audio_embedding: np.ndarray | None = None,
        audio_available: bool = False,
        prosody_vector: np.ndarray | None = None,
        acoustic_features_full: np.ndarray | None = None,
        quality_vector: np.ndarray | None = None,
        quality_ok: bool = False,
        audio_duration_sec: float | None = None,
        text_embedding: np.ndarray | None = None,
        context_embedding: np.ndarray | None = None,
        has_context: bool = False,
        text_available: bool = False,
    ) -> None:
        self.example_id = example_id
        self.dataset = dataset
        self.text = text
        self.context = context
        self.context_embedding = context_embedding
        self.has_context = bool(has_context)
        self.speaker = speaker
        self.show = show
        self.label = int(label)
        self.split = split
        self.audio_embedding = audio_embedding
        self.audio_available = audio_available
        self.prosody_vector = (
            prosody_vector if prosody_vector is not None
            else np.zeros(PROSODY_DIM, dtype=np.float32)
        )
        self.acoustic_features_full = (
            acoustic_features_full if acoustic_features_full is not None
            else np.zeros(116, dtype=np.float32)
        )
        self.quality_vector = (
            quality_vector if quality_vector is not None
            else np.zeros(QUALITY_DIM, dtype=np.float32)
        )
        self.quality_ok = quality_ok
        self.audio_duration_sec = audio_duration_sec
        self.text_embedding = text_embedding
        self.text_available = text_available


# ---------------------------------------------------------------------------
# Loader helpers — MUStARD JSON
# ---------------------------------------------------------------------------

def _load_mustard_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Loader helpers — MUStARD++ labelled CSV
# ---------------------------------------------------------------------------

def _load_mustard_pp_csv(csv_path: Path) -> list[UtteranceExample]:
    """Load MUStARD++ from a labelled CSV file.

    The CSV must have at minimum:
      - A unique ID column (first column, or named 'clip_id' / 'ID' / 'id')
      - A label column (one of: 'sarcasm', 'label', 'is_sarcastic')
      - A text column  (one of: 'utterance', 'text', 'sentence')

    Optional columns: 'speaker', 'context', 'show'
    """
    if not csv_path.is_file():
        log.warning("MUStARD++ CSV not found at %s; skipping", csv_path)
        return []

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # --- Identify label column ---
    label_col = None
    for c in _MUSTARD_PP_LABEL_COLS:
        if c in df.columns:
            label_col = c
            break
    if label_col is None:
        log.warning("MUStARD++ CSV has no recognised label column; skipping")
        return []

    # --- Identify text column ---
    text_col = None
    for c in _MUSTARD_PP_TEXT_COLS:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        log.warning("MUStARD++ CSV has no recognised text column; skipping")
        return []

    # --- Identify ID column ---
    id_col = None
    for cand in ("clip_id", "ID", "id", "example_id"):
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        # Use first column as ID
        id_col = df.columns[0]

    records: list[UtteranceExample] = []
    for _, row in df.iterrows():
        uid = str(row[id_col]).strip()
        # Prefix to avoid collision with MUStARD ids
        uid = f"pp_{uid}" if not uid.startswith("pp_") else uid

        raw_label = str(row[label_col]).strip().lower()
        from shared.constants import POSITIVE_SARCASM
        label = 1 if raw_label in POSITIVE_SARCASM else 0

        text = str(row.get(text_col, "")).strip()
        speaker = str(row.get("speaker", "UNKNOWN")).strip()
        show = str(row.get("show", "")) if "show" in df.columns else None

        # Context: may be a stringified list or a plain string
        raw_ctx = row.get("context", "")
        if isinstance(raw_ctx, str) and raw_ctx.startswith("["):
            try:
                import ast
                context = [str(c) for c in ast.literal_eval(raw_ctx)]
            except Exception:
                context = [raw_ctx] if raw_ctx else []
        elif isinstance(raw_ctx, str) and raw_ctx:
            context = [raw_ctx]
        else:
            context = []

        ue = UtteranceExample(
            example_id=uid,
            dataset="mustard_pp",
            text=text,
            speaker=speaker,
            task="sarcasm",
            context=context,
            context_speakers=[],
            label=label,
            show=show,
        )
        records.append(ue)

    log.info("Loaded %d examples from MUStARD++ CSV", len(records))
    return records


# ---------------------------------------------------------------------------
# Loader helpers — member2 handoff CSVs
# ---------------------------------------------------------------------------

def _load_handoff_csvs(
    handoff_csv: Path,
    quality_csv: Path,
    predictions_csv: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three member2 handoff CSVs. Index by utterance_id."""
    df_h = pd.read_csv(handoff_csv, dtype={"utterance_id": str})
    df_q = pd.read_csv(quality_csv, dtype={"utterance_id": str})
    df_p = pd.read_csv(predictions_csv, dtype={"utterance_id": str})
    df_h = df_h.set_index("utterance_id")
    df_q = df_q.set_index("utterance_id")
    df_p = df_p.set_index("utterance_id")
    return df_h, df_q, df_p


# ---------------------------------------------------------------------------
# Loader helpers — wav2vec2 / text npz embeddings
# ---------------------------------------------------------------------------

def _load_wav2vec2_npz(npz_path: Path | None) -> dict[str, np.ndarray] | None:
    """Load pooled Wav2Vec2 embeddings from the Member 2 handoff."""
    return _load_embedding_npz(
        npz_path,
        modality="Wav2Vec2",
        id_key_options=("utterance_ids", "ids"),
        embedding_key_options=("embeddings",),
        width=AUDIO_HIDDEN,
    )


def _load_text_npz(npz_path: Path | None) -> dict[str, np.ndarray] | None:
    """Load cached Member 1 utterance/context vectors, keyed by normalized ID."""
    if npz_path is None or not Path(npz_path).is_file():
        return None
    with np.load(npz_path, allow_pickle=False) as archive:
        id_key = "utterance_ids" if "utterance_ids" in archive else "ids"
        utterance_key = "embeddings" if "embeddings" in archive else "utterance"
        if id_key not in archive or utterance_key not in archive or "context" not in archive and "context_embeddings" not in archive:
            raise ValueError(
                f"Text NPZ {npz_path} must contain ids, utterance embeddings, and context embeddings"
            )
        ids = [_normalize_utterance_id(value) for value in archive[id_key]]
        utterance = np.asarray(archive[utterance_key], dtype=np.float32)
        context_key = "context_embeddings" if "context_embeddings" in archive else "context"
        context = np.asarray(archive[context_key], dtype=np.float32)
        has_context = (
            np.asarray(archive["has_context"], dtype=bool)
            if "has_context" in archive
            else np.ones(len(ids), dtype=bool)
        )
    if len(set(ids)) != len(ids) or any(uid is None for uid in ids):
        raise ValueError(f"Text NPZ {npz_path} has null or duplicate normalized IDs")
    expected_shape = (len(ids), TEXT_HIDDEN)
    if utterance.shape != expected_shape or context.shape != expected_shape:
        raise ValueError(
            f"Text NPZ {npz_path} must have utterance/context shape {expected_shape}; "
            f"got {utterance.shape} and {context.shape}"
        )
    if has_context.shape != (len(ids),):
        raise ValueError(f"Text NPZ {npz_path} has invalid has_context shape {has_context.shape}")
    if not np.isfinite(utterance).all() or not np.isfinite(context).all():
        raise ValueError(f"Text NPZ {npz_path} contains non-finite embeddings")
    return {
        uid: np.concatenate((utterance[index], context[index], [float(has_context[index])])).astype(np.float32)
        for index, uid in enumerate(ids)
    }


def _normalize_utterance_id(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    normalized = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return normalized or None


def _read_feature_csv(
    path: Path | str,
    *,
    modality: str,
    required_features: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, list[str], set[str], int]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required {modality} handoff not found: {path}")
    frame = pd.read_csv(path, dtype={"utterance_id": str})
    if "utterance_id" not in frame.columns:
        raise ValueError(f"{modality} handoff {path} is missing 'utterance_id'")
    if frame.columns.duplicated().any():
        raise ValueError(f"{modality} handoff {path} has duplicate column names")
    ids = [_normalize_utterance_id(value) for value in frame["utterance_id"]]
    if any(uid is None for uid in ids):
        raise ValueError(f"{modality} handoff {path} contains null/blank utterance_id values")
    duplicates = sorted(uid for uid, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"{modality} handoff {path} has duplicate normalized IDs: {duplicates[:10]}")

    if required_features is not None:
        missing_columns = [column for column in required_features if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"{modality} handoff {path} is missing columns: {missing_columns}")
        feature_cols = list(required_features)
    else:
        metadata = {
            "utterance_id", "speaker", "sarcasm", "audio_sarcasm_prediction",
            "audio_sarcasm_label", "audio_sarcasm_decision_score", "sarcasm_score_type",
            "predicted_emotion", "emotion_decision_scores", "emotion_score_type",
            "pooled_embedding_dim", "frame_count", "frame_embedding_dim",
            "audio_duration_seconds", "duration_seconds", "sample_rate", "channels",
            "peak_amplitude", "rms_energy", "clipping_ratio", "silence_ratio",
            "audio_valid", "audio_warnings", "quality_status",
        }
        feature_cols = [column for column in frame.columns if column not in metadata]
    if not feature_cols:
        raise ValueError(f"{modality} handoff {path} has no feature columns")

    numeric = frame[feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    valid_rows = np.isfinite(numeric).all(axis=1)
    invalid_ids = {ids[index] for index, valid in enumerate(valid_rows) if not valid}
    frame = frame.copy()
    frame["utterance_id"] = ids
    frame.loc[:, feature_cols] = numeric.astype(np.float32)
    frame.index = ids
    return frame, feature_cols, invalid_ids, len(ids)


def _load_embedding_npz(
    npz_path: Path | str | None,
    *,
    modality: str,
    id_key_options: tuple[str, ...],
    embedding_key_options: tuple[str, ...],
    width: int,
) -> dict[str, np.ndarray] | None:
    if npz_path is None:
        return None
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Required {modality} embeddings not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        id_key = next((key for key in id_key_options if key in archive), None)
        embedding_key = next((key for key in embedding_key_options if key in archive), None)
        if id_key is None or embedding_key is None:
            raise ValueError(
                f"{modality} NPZ {path} requires one of {id_key_options} and {embedding_key_options}; "
                f"found {list(archive.keys())}"
            )
        ids = [_normalize_utterance_id(value) for value in archive[id_key]]
        embeddings = np.asarray(archive[embedding_key], dtype=np.float32)
    if any(uid is None for uid in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{modality} NPZ {path} has null or duplicate normalized IDs")
    if embeddings.shape != (len(ids), width):
        raise ValueError(
            f"{modality} NPZ {path} must have shape ({len(ids)}, {width}); got {embeddings.shape}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError(f"{modality} NPZ {path} contains non-finite embeddings")
    return {uid: embeddings[index] for index, uid in enumerate(ids)}


# ---------------------------------------------------------------------------
# Feature-vector builders
# ---------------------------------------------------------------------------

def _build_prosody_vector(row: pd.Series) -> np.ndarray:
    """Build a PROSODY_DIM vector aligned to shared.constants.PROSODY_FEATURES."""
    rev = {v: k for k, v in _HANDOFF_PROSODY_MAP.items()}
    vec = np.zeros(PROSODY_DIM, dtype=np.float32)
    for i, feat_name in enumerate(PROSODY_FEATURES):
        handoff_col = rev.get(feat_name)
        if handoff_col and handoff_col in row.index:
            val = row[handoff_col]
            try:
                fval = float(val)
                if np.isfinite(fval):
                    vec[i] = fval
            except (TypeError, ValueError):
                pass
    return vec


def _build_quality_vector(qrow: pd.Series) -> np.ndarray:
    """Build a QUALITY_DIM vector aligned to shared.constants.QUALITY_FEATURES."""
    vec = np.zeros(QUALITY_DIM, dtype=np.float32)
    mapping = {
        "clipping_ratio": 1,
        "silence_ratio": 2,
        "rms_energy": 4,
    }
    for col, idx in mapping.items():
        if col in qrow.index:
            val = qrow[col]
            if pd.notna(val):
                try:
                    fval = float(val)
                    if np.isfinite(fval):
                        vec[idx] = fval
                except (TypeError, ValueError):
                    pass
    if "audio_valid" in qrow.index:
        vec[3] = 1.0 if bool(qrow["audio_valid"]) else 0.0
    return vec


def _build_acoustic_full(row: pd.Series, feature_cols: list[str]) -> np.ndarray:
    """Build acoustic feature vector from handoff CSV row."""
    vec = np.zeros(len(feature_cols), dtype=np.float32)
    for i, col in enumerate(feature_cols):
        val = row.get(col, 0.0)
        try:
            fval = float(val)
            vec[i] = fval if np.isfinite(fval) else 0.0
        except (TypeError, ValueError):
            vec[i] = 0.0
    return vec


# ---------------------------------------------------------------------------
# Main loader function
# ---------------------------------------------------------------------------

ACOUSTIC_FEATURE_COLS: list[str] = []  # populated on first load


def _load_legacy_mustard_fusion_records(
    mustard_json_path: Path | str,
    handoff_csv_path: Path | str,
    quality_csv_path: Path | str,
    predictions_csv_path: Path | str,
    wav2vec2_npz_path: Path | str | None = None,
    mustard_pp_csv_path: Path | str | None = None,
    text_embeddings_npz_path: Path | str | None = None,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    split_strategy: str = "speaker",
    dedup_mustard_family: bool = True,
) -> tuple[dict[str, list[FusionRecord]], int]:
    """Build train/val/test FusionRecord dicts from MUStARD + member2 handoff.

    Args:
        mustard_json_path:       Path to datasets/MUSTARD/sarcasm_data.json
        handoff_csv_path:        member2 audio handoff CSV
        quality_csv_path:        member2 audio quality CSV
        predictions_csv_path:    member2 audio predictions CSV
        wav2vec2_npz_path:       optional member2 pooled wav2vec2 embeddings (.npz)
        mustard_pp_csv_path:     optional MUStARD++ labelled text CSV
        text_embeddings_npz_path: optional member1 cached text embeddings (.npz)
        val_ratio / test_ratio:  split proportions
        seed:                    random seed for splits
        split_strategy:          "speaker" | "stratified"
        dedup_mustard_family:    if True, remove MUStARD rows that appear in MUStARD++

    Returns:
        (splits_dict, acoustic_dim)
        splits_dict has keys "train", "val", "test"
        acoustic_dim is the number of acoustic feature columns detected
    """
    global ACOUSTIC_FEATURE_COLS

    mustard_json_path    = Path(mustard_json_path)
    handoff_csv_path     = Path(handoff_csv_path)
    quality_csv_path     = Path(quality_csv_path)
    predictions_csv_path = Path(predictions_csv_path)

    # ------------------------------------------------------------------ #
    # 1. Load MUStARD JSON
    # ------------------------------------------------------------------ #
    log.info("Loading MUStARD JSON from %s", mustard_json_path)
    mustard_data = _load_mustard_json(mustard_json_path)
    log.info("  -> %d utterances in MUStARD", len(mustard_data))

    ue_list: list[UtteranceExample] = []
    for uid, record in mustard_data.items():
        ue = UtteranceExample(
            example_id=str(uid),
            dataset="mustard",
            text=record["utterance"],
            speaker=record.get("speaker", ""),
            task="sarcasm",
            context=record.get("context", []),
            context_speakers=record.get("context_speakers", []),
            label=1 if record["sarcasm"] else 0,
            show=record.get("show"),
        )
        ue_list.append(ue)

    # ------------------------------------------------------------------ #
    # 2. Load MUStARD++ (if path provided)
    # ------------------------------------------------------------------ #
    pp_ue_list: list[UtteranceExample] = []
    if mustard_pp_csv_path is not None:
        pp_ue_list = _load_mustard_pp_csv(Path(mustard_pp_csv_path))

    # De-duplicate: remove MUStARD rows whose IDs appear in MUStARD++
    if dedup_mustard_family and pp_ue_list:
        try:
            pp_ids = {ue.example_id for ue in pp_ue_list}
            before = len(ue_list)
            ue_list = [ue for ue in ue_list if ue.example_id not in pp_ids]
            removed = before - len(ue_list)
            if removed:
                log.info(
                    "De-duplicated %d MUStARD rows that appear in MUStARD++",
                    removed,
                )
        except Exception as exc:
            log.warning("deduplicate_mustard_family failed: %s; skipping dedup", exc)

    all_ue = ue_list + pp_ue_list
    log.info(
        "Total examples after merge: %d (MUStARD=%d, MUStARD++=%d)",
        len(all_ue), len(ue_list), len(pp_ue_list),
    )

    # ------------------------------------------------------------------ #
    # 3. Load member2 handoff CSVs
    # ------------------------------------------------------------------ #
    log.info("Loading member2 handoff CSVs")
    _handoff_ok = (
        handoff_csv_path.is_file()
        and quality_csv_path.is_file()
        and predictions_csv_path.is_file()
    )
    if _handoff_ok:
        df_h, df_q, df_p = _load_handoff_csvs(
            handoff_csv_path, quality_csv_path, predictions_csv_path
        )
        # Identify acoustic feature columns (all numeric columns except meta)
        _meta_cols = {
            "utterance_id", "speaker", "sarcasm",
            "audio_sarcasm_prediction", "audio_sarcasm_label",
            "audio_sarcasm_decision_score", "sarcasm_score_type",
            "predicted_emotion", "emotion_decision_scores", "emotion_score_type",
            "pooled_embedding_dim", "frame_count", "frame_embedding_dim",
            "audio_duration_seconds", "sample_rate", "channels",
            "peak_amplitude", "rms_energy", "clipping_ratio", "silence_ratio",
            "audio_valid", "audio_warnings", "quality_status",
        }
        ACOUSTIC_FEATURE_COLS = [
            c for c in df_h.columns
            if c not in _meta_cols
            and df_h[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        ]
        log.info("  -> %d acoustic feature columns detected", len(ACOUSTIC_FEATURE_COLS))
    else:
        log.warning(
            "One or more member2 handoff CSVs not found; running in audio-features-unavailable mode"
        )
        df_h = df_q = df_p = pd.DataFrame()
        ACOUSTIC_FEATURE_COLS = []

    # ------------------------------------------------------------------ #
    # 4. Optional embeddings
    # ------------------------------------------------------------------ #
    wav2vec2_map = _load_wav2vec2_npz(wav2vec2_npz_path)
    if not wav2vec2_map:
        log.info("No wav2vec2 npz — using zero audio embeddings (feature-only mode)")

    text_emb_map = _load_text_npz(text_embeddings_npz_path)
    if not text_emb_map:
        log.info(
            "No cached text embeddings npz — text_available=False in all records. "
            "Provide member1 checkpoint or pre-extract embeddings for text signal."
        )

    # ------------------------------------------------------------------ #
    # 5. Split
    # ------------------------------------------------------------------ #
    if split_strategy == "speaker":
        buckets = assign_speaker_splits(
            all_ue, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
    else:
        from sklearn.model_selection import train_test_split
        labels = [ue.label for ue in all_ue]
        train_val, test_ue = train_test_split(
            all_ue, test_size=test_ratio, stratify=labels, random_state=seed
        )
        tv_labels = [ue.label for ue in train_val]
        relative_val = val_ratio / (1.0 - test_ratio)
        train_ue, val_ue = train_test_split(
            train_val, test_size=relative_val, stratify=tv_labels, random_state=seed
        )
        buckets = {"train": train_ue, "val": val_ue, "test": test_ue}

    # ------------------------------------------------------------------ #
    # 6. Build FusionRecords
    # ------------------------------------------------------------------ #
    result: dict[str, list[FusionRecord]] = {"train": [], "val": [], "test": []}
    missing_in_handoff = 0
    missing_text = 0

    for split_name, examples in buckets.items():
        for ue in examples:
            uid = ue.example_id

            # --- Audio embedding (member2 wav2vec2) ---
            if wav2vec2_map and uid in wav2vec2_map:
                audio_emb       = wav2vec2_map[uid]
                audio_available = True
            else:
                audio_emb       = None
                audio_available = False

            # --- Acoustic features from handoff CSV ---
            if not df_h.empty and uid in df_h.index:
                hrow       = df_h.loc[uid]
                prosody_vec  = _build_prosody_vector(hrow)
                acoustic_full = _build_acoustic_full(hrow, ACOUSTIC_FEATURE_COLS)
            else:
                prosody_vec   = None
                acoustic_full = None
                if not df_h.empty:
                    missing_in_handoff += 1

            # --- Audio quality ---
            if not df_q.empty and uid in df_q.index:
                qrow      = df_q.loc[uid]
                quality_vec = _build_quality_vector(qrow)
                quality_ok  = bool(qrow.get("audio_valid", False))
                audio_dur   = float(qrow.get("audio_duration_seconds", 0.0) or 0.0)
            else:
                quality_vec = None
                quality_ok  = False
                audio_dur   = None

            # --- Cached text embeddings (member1) ---
            if text_emb_map and uid in text_emb_map:
                text_emb      = text_emb_map[uid]
                text_available = True
            else:
                text_emb      = None
                text_available = False
                missing_text += 1

            rec = FusionRecord(
                example_id=uid,
                dataset=ue.dataset,
                text=ue.text,
                context=ue.context,
                speaker=ue.speaker,
                label=ue.label,
                split=split_name,
                audio_embedding=audio_emb,
                audio_available=audio_available,
                prosody_vector=prosody_vec,
                acoustic_features_full=acoustic_full,
                quality_vector=quality_vec,
                quality_ok=quality_ok,
                audio_duration_sec=audio_dur,
                text_embedding=text_emb,
                text_available=text_available,
            )
            result[split_name].append(rec)

    if missing_in_handoff > 0:
        log.warning(
            "%d utterances had no entry in member2 handoff CSV", missing_in_handoff
        )

    # Warn if text embeddings are absent for the whole training set
    if missing_text == sum(len(v) for v in result.values()):
        log.warning(
            "ALL %d examples have text_available=False. "
            "Provide text_embeddings_npz_path or a live text encoder to use text signal.",
            missing_text,
        )
    elif missing_text > 0:
        log.info(
            "%d / %d examples lack cached text embeddings (text_available=False for those).",
            missing_text, sum(len(v) for v in result.values()),
        )

    for split_name, recs in result.items():
        n_sarc = sum(r.label for r in recs)
        log.info(
            "Split '%s': %d examples (%d sarcastic, %d non-sarcastic)",
            split_name, len(recs), n_sarc, len(recs) - n_sarc,
        )

    return result, len(ACOUSTIC_FEATURE_COLS) if ACOUSTIC_FEATURE_COLS else 116


# ---------------------------------------------------------------------------
# Current Member 1/2 handoff loader
# ---------------------------------------------------------------------------

def load_mustard_fusion_records(
    mustard_json_path: Path | str,
    handoff_csv_path: Path | str,
    quality_csv_path: Path | str,
    predictions_csv_path: Path | str | None = None,
    wav2vec2_npz_path: Path | str | None = None,
    mustard_pp_csv_path: Path | str | None = None,
    text_embeddings_npz_path: Path | str | None = None,
    *,
    prosody_csv_path: Path | str | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    split_strategy: str = "speaker",
    dedup_mustard_family: bool = True,
    alignment_report_path: Path | str | None = None,
    max_alignment_drop_ratio: float = 0.10,
) -> tuple[dict[str, list[FusionRecord]], int]:
    """Synchronize current text/audio handoffs by normalized utterance ID."""
    del predictions_csv_path, dedup_mustard_family
    global ACOUSTIC_FEATURE_COLS
    if not 0 <= max_alignment_drop_ratio < 1:
        raise ValueError("max_alignment_drop_ratio must be in [0, 1)")
    if mustard_pp_csv_path is not None:
        raise ValueError(
            "MUStARD++ requires matching Member 1/2 exports; this loader only "
            "synchronizes the supplied MUStARD handoffs."
        )

    mustard_path = Path(mustard_json_path)
    mustard_data = _load_mustard_json(mustard_path)
    source_records: dict[str, dict[str, Any]] = {}
    labels: dict[str, int] = {}
    invalid_labels: set[str] = set()
    for raw_id, record in mustard_data.items():
        uid = _normalize_utterance_id(raw_id)
        if uid is None or uid in source_records:
            raise ValueError(f"MUStARD JSON has null/duplicate normalized ID: {raw_id!r}")
        source_records[uid] = record
        raw_label = record.get("sarcasm")
        if isinstance(raw_label, (bool, np.bool_)):
            labels[uid] = int(raw_label)
        elif isinstance(raw_label, (int, np.integer)) and raw_label in (0, 1):
            labels[uid] = int(raw_label)
        else:
            invalid_labels.add(uid)
        if not isinstance(record.get("utterance"), str) or not record["utterance"].strip():
            invalid_labels.add(uid)

    acoustic, acoustic_cols, bad_acoustic, acoustic_count = _read_feature_csv(
        handoff_csv_path, modality="acoustic"
    )
    prosody_path = prosody_csv_path or handoff_csv_path
    prosody, prosody_cols, bad_prosody, prosody_count = _read_feature_csv(
        prosody_path, modality="prosody", required_features=PROSODY_FEATURES
    )
    quality, quality_cols, bad_quality, quality_count = _read_feature_csv(
        quality_csv_path, modality="quality", required_features=QUALITY_FEATURES
    )
    if len(acoustic_cols) != 116:
        raise ValueError(
            f"Acoustic handoff must provide 116 named features; found {len(acoustic_cols)}"
        )
    if len(prosody_cols) != PROSODY_DIM or len(quality_cols) != QUALITY_DIM:
        raise ValueError(
            f"Expected prosody/quality dimensions {PROSODY_DIM}/{QUALITY_DIM}; "
            f"got {len(prosody_cols)}/{len(quality_cols)}"
        )
    ACOUSTIC_FEATURE_COLS = acoustic_cols

    wav2vec2_map = _load_wav2vec2_npz(
        Path(wav2vec2_npz_path) if wav2vec2_npz_path is not None else None
    )
    text_emb_map = _load_text_npz(
        Path(text_embeddings_npz_path)
        if text_embeddings_npz_path is not None
        else None
    )
    if wav2vec2_map is None or text_emb_map is None:
        raise ValueError(
            "Synchronization requires cached Member 1 text and Member 2 Wav2Vec2 "
            "embeddings; configure both NPZ paths."
        )
    with np.load(text_embeddings_npz_path, allow_pickle=False) as text_archive:
        text_has_context_key = "has_context" in text_archive

    label_ids = set(labels) - invalid_labels
    modality_ids = {
        "text": set(text_emb_map),
        "wav2vec2": set(wav2vec2_map),
        "acoustic": set(acoustic.index) - bad_acoustic,
        "prosody": set(prosody.index) - bad_prosody,
        "quality": set(quality.index) - bad_quality,
    }
    common_ids = set(label_ids)
    for ids in modality_ids.values():
        common_ids.intersection_update(ids)

    report: dict[str, Any] = {
        "label_source": str(mustard_path),
        "speaker_source": "MUStARD JSON",
        "label_mapping": {"False": 0, "True": 1},
        "sources": {
            "labels": {"count": len(source_records), "valid_count": len(label_ids)},
            "text_embeddings": {
                "path": str(text_embeddings_npz_path),
                "count": len(text_emb_map),
                "utterance_dim": TEXT_HIDDEN,
                "context_dim": TEXT_HIDDEN,
                "has_context_source": (
                    "Member 1 has_context array"
                    if text_has_context_key
                    else "non-empty context text in MUStARD JSON"
                ),
            },
            "wav2vec2_embeddings": {
                "path": str(wav2vec2_npz_path),
                "count": len(wav2vec2_map),
                "embedding_dim": AUDIO_HIDDEN,
            },
            "acoustic_csv": {
                "path": str(handoff_csv_path),
                "count": acoustic_count,
                "feature_count": len(acoustic_cols),
                "feature_columns": acoustic_cols,
            },
            "prosody_csv": {
                "path": str(prosody_path),
                "count": prosody_count,
                "feature_count": len(prosody_cols),
                "feature_columns": prosody_cols,
            },
            "quality_csv": {
                "path": str(quality_csv_path),
                "count": quality_count,
                "feature_count": len(quality_cols),
                "feature_columns": quality_cols,
            },
        },
        "id_validation": {
            "normalization": "Unicode NFKC, trim whitespace, casefold",
            "null_ids": 0,
            "duplicate_normalized_ids": 0,
        },
        "invalid_label_or_text": {
            "count": len(invalid_labels),
            "examples": sorted(invalid_labels)[:20],
        },
        "invalid_feature_ids": {
            "acoustic": {"count": len(bad_acoustic), "examples": sorted(bad_acoustic)[:20]},
            "prosody": {"count": len(bad_prosody), "examples": sorted(bad_prosody)[:20]},
            "quality": {"count": len(bad_quality), "examples": sorted(bad_quality)[:20]},
        },
        "unmatched_ids": {},
        "common_valid_labelled_count": len(common_ids),
        "dropped_labelled_count": len(label_ids - common_ids),
        "max_alignment_drop_ratio": max_alignment_drop_ratio,
        "audio_quality_rule": (
            "unreliable when duration_in_range < 0.5, silence_ratio >= 0.95, "
            "or rms <= 1e-8"
        ),
    }
    for name, ids in modality_ids.items():
        missing_ids = label_ids - ids
        extra_ids = ids - label_ids
        report["unmatched_ids"][name] = {
            "missing_from_modality_count": len(missing_ids),
            "missing_from_modality_examples": sorted(missing_ids)[:10],
            "extra_not_in_labels_count": len(extra_ids),
            "extra_not_in_labels_examples": sorted(extra_ids)[:10],
        }

    dropped_ratio = (len(label_ids) - len(common_ids)) / max(1, len(label_ids))
    report_path = Path(alignment_report_path) if alignment_report_path is not None else None
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not common_ids:
        raise ValueError("No common valid labelled IDs across Member 1/2 handoffs")
    if dropped_ratio > max_alignment_drop_ratio:
        raise ValueError(
            f"Only {len(common_ids)}/{len(label_ids)} labelled utterances aligned "
            f"({dropped_ratio:.1%} dropped; limit {max_alignment_drop_ratio:.1%}). "
            f"See synchronization report: {report_path}"
        )

    records_by_id: dict[str, FusionRecord] = {}
    examples: list[UtteranceExample] = []
    for uid, source in source_records.items():
        if uid not in common_ids:
            continue
        context = source.get("context") or []
        if isinstance(context, str):
            context = [context] if context.strip() else []
        has_context = bool(context)
        cached_text = text_emb_map[uid]
        text_embedding = cached_text[:TEXT_HIDDEN]
        context_embedding = cached_text[TEXT_HIDDEN:2 * TEXT_HIDDEN]
        has_context = has_context and bool(cached_text[-1])

        quality_row = quality.loc[uid]
        quality_vector = quality_row[quality_cols].to_numpy(dtype=np.float32)
        audio_quality_ok = bool(
            quality_row["duration_in_range"] >= 0.5
            and quality_row["silence_ratio"] < 0.95
            and quality_row["rms"] > 1e-8
        )
        record = FusionRecord(
            example_id=uid,
            dataset="mustard",
            text=source["utterance"],
            context=context,
            speaker=str(source.get("speaker") or "UNKNOWN"),
            show=source.get("show"),
            label=labels[uid],
            audio_embedding=wav2vec2_map[uid],
            audio_available=audio_quality_ok,
            prosody_vector=prosody.loc[uid, prosody_cols].to_numpy(dtype=np.float32),
            acoustic_features_full=acoustic.loc[uid, acoustic_cols].to_numpy(dtype=np.float32),
            quality_vector=quality_vector,
            quality_ok=audio_quality_ok,
            audio_duration_sec=float(prosody.loc[uid, "duration_sec"]),
            text_embedding=text_embedding,
            context_embedding=context_embedding,
            has_context=has_context,
            text_available=True,
        )
        records_by_id[uid] = record
        examples.append(UtteranceExample(
            example_id=uid,
            dataset="mustard",
            text=record.text,
            speaker=record.speaker,
            task="sarcasm",
            context=context,
            context_speakers=source.get("context_speakers", []),
            label=record.label,
            show=source.get("show"),
        ))

    if split_strategy == "speaker":
        buckets = assign_speaker_splits(
            examples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
    elif split_strategy == "stratified":
        from sklearn.model_selection import train_test_split
        labels_array = [example.label for example in examples]
        train_val, test_examples = train_test_split(
            examples, test_size=test_ratio, stratify=labels_array, random_state=seed
        )
        relative_val = val_ratio / (1.0 - test_ratio)
        train_labels = [example.label for example in train_val]
        train_examples, val_examples = train_test_split(
            train_val, test_size=relative_val, stratify=train_labels, random_state=seed
        )
        buckets = {"train": train_examples, "val": val_examples, "test": test_examples}
    else:
        raise ValueError(f"Unknown split_strategy {split_strategy!r}")

    result: dict[str, list[FusionRecord]] = {}
    for split_name, split_examples in buckets.items():
        result[split_name] = []
        for example in split_examples:
            record = records_by_id[example.example_id]
            record.split = split_name
            result[split_name].append(record)
        log.info(
            "Split '%s': %d examples (%d sarcastic, %d non-sarcastic)",
            split_name,
            len(result[split_name]),
            sum(record.label for record in result[split_name]),
            sum(1 - record.label for record in result[split_name]),
        )
    if report_path is not None:
        report["speaker_split_counts"] = {
            split: len(split_records) for split, split_records in result.items()
        }
        report["audio_unreliable_count"] = sum(
            not record.quality_ok for record in records_by_id.values()
        )
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(
        "Synchronized %d/%d labelled utterances by normalized ID",
        len(common_ids), len(label_ids),
    )
    return result, len(acoustic_cols)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class FusionDataset(Dataset):
    """PyTorch dataset wrapping FusionRecord lists.

    Text is returned as raw string; the text encoder interface handles
    tokenisation. Cached text embeddings (from member1 npz) are also
    returned as tensors when available.

    Tensors returned
    ----------------
    audio_embedding:   float32 [AUDIO_HIDDEN]   (zeros if unavailable)
    audio_available:   bool scalar
    prosody:           float32 [PROSODY_DIM]
    acoustic_full:     float32 [acoustic_dim]
    quality:           float32 [QUALITY_DIM]
    quality_ok:        bool scalar
    text_embedding:    float32 [TEXT_HIDDEN]     (zeros if unavailable)
    text_available:    bool scalar
    label:             int64 scalar
    """

    def __init__(
        self,
        records: list[FusionRecord],
        acoustic_dim: int | None = None,
    ) -> None:
        self.records = records
        self.acoustic_dim = acoustic_dim or (
            len(ACOUSTIC_FEATURE_COLS) if ACOUSTIC_FEATURE_COLS else 116
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # Audio embedding
        if rec.audio_embedding is not None:
            audio_emb = torch.tensor(rec.audio_embedding, dtype=torch.float32)
        else:
            audio_emb = torch.zeros(AUDIO_HIDDEN, dtype=torch.float32)

        # Cached text embedding (member1)
        if rec.text_embedding is not None:
            text_emb = torch.tensor(rec.text_embedding, dtype=torch.float32)
        else:
            text_emb = torch.zeros(TEXT_HIDDEN, dtype=torch.float32)
        if rec.context_embedding is not None:
            context_emb = torch.tensor(rec.context_embedding, dtype=torch.float32)
        else:
            context_emb = torch.zeros(TEXT_HIDDEN, dtype=torch.float32)

        prosody = torch.tensor(rec.prosody_vector, dtype=torch.float32)
        quality = torch.tensor(rec.quality_vector, dtype=torch.float32)

        # Acoustic full (pad/truncate to acoustic_dim)
        af = rec.acoustic_features_full
        if af is None:
            af = np.zeros(self.acoustic_dim, dtype=np.float32)
        if len(af) < self.acoustic_dim:
            af = np.pad(af, (0, self.acoustic_dim - len(af)))
        elif len(af) > self.acoustic_dim:
            af = af[: self.acoustic_dim]
        acoustic_full = torch.tensor(af, dtype=torch.float32)

        return {
            "example_id":      rec.example_id,
            "text":            rec.text,
            "context":         rec.context,              # list[str]
            "speaker":         rec.speaker,
            "audio_embedding": audio_emb,                # [AUDIO_HIDDEN]
            "audio_available": torch.tensor(rec.audio_available, dtype=torch.bool),
            "prosody":         prosody,                  # [PROSODY_DIM]
            "acoustic_full":   acoustic_full,            # [acoustic_dim]
            "quality":         quality,                  # [QUALITY_DIM]
            "quality_ok":      torch.tensor(rec.quality_ok, dtype=torch.bool),
            "text_embedding":  text_emb,                 # [TEXT_HIDDEN]
            "text_available":  torch.tensor(rec.text_available, dtype=torch.bool),
            "context_embedding": context_emb,            # [CONTEXT_HIDDEN]
            "has_context":     torch.tensor(rec.has_context, dtype=torch.bool),
            "label":           torch.tensor(rec.label, dtype=torch.long),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that handles variable-length context lists."""
    keys_tensor = [
        "audio_embedding", "audio_available",
        "prosody", "acoustic_full",
        "quality", "quality_ok",
        "text_embedding", "text_available",
        "context_embedding", "has_context",
        "label",
    ]
    out: dict = {}
    for key in keys_tensor:
        out[key] = torch.stack([b[key] for b in batch])

    out["example_ids"] = [b["example_id"] for b in batch]
    out["texts"]       = [b["text"]       for b in batch]
    out["contexts"]    = [b["context"]    for b in batch]   # list[list[str]]
    out["speakers"]    = [b["speaker"]    for b in batch]
    return out


def validate_modality_availability(
    datasets: dict[str, "FusionDataset"],
) -> None:
    """Log warnings if entire splits have no text or audio signal."""
    for split_name, ds in datasets.items():
        n = len(ds)
        if n == 0:
            continue
        n_text_avail  = sum(1 for r in ds.records if r.text_available)
        n_audio_avail = sum(1 for r in ds.records if r.audio_available)
        if n_text_avail == 0:
            log.warning(
                "Split '%s': ALL %d records have text_available=False. "
                "The model will train without text signal.",
                split_name, n,
            )
        if n_audio_avail == 0:
            log.warning(
                "Split '%s': ALL %d records have audio_available=False. "
                "The model will train without audio embeddings.",
                split_name, n,
            )
        log.info(
            "Split '%s' modality availability: text=%d/%d  audio=%d/%d",
            split_name, n_text_avail, n, n_audio_avail, n,
        )
