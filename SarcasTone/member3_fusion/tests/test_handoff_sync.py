from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from shared.constants import PROSODY_FEATURES, QUALITY_FEATURES
from shared.splits import speaker_group
from src.training.comparison import ComparisonClassifier, build_representations
from src.training.dataset import FusionDataset, collate_fn, load_mustard_fusion_records


def _write_handoffs(tmp_path, *, omit_audio_id: str | None = None, duplicate_audio_id=False):
    ids = [f"u{i}" for i in range(12)]
    mustard = {
        uid: {
            "utterance": f"target {uid}",
            "speaker": f"speaker-{index // 2}",
            "context": [] if index == 0 else [f"context {uid}"],
            "context_speakers": [],
            "sarcasm": bool(index % 2),
            "show": "show",
        }
        for index, uid in enumerate(ids)
    }
    mustard_path = tmp_path / "mustard.json"
    mustard_path.write_text(json.dumps(mustard), encoding="utf-8")

    text_ids = [f" {uid.upper()} " for uid in ids[::-1]]
    np.savez(
        tmp_path / "text.npz",
        ids=np.asarray(text_ids),
        utterance=np.stack([
            np.full(768, int(uid.strip()[1:]), dtype=np.float32) for uid in text_ids
        ]),
        context=np.stack([
            np.full(768, int(uid.strip()[1:]) + 100, dtype=np.float32) for uid in text_ids
        ]),
    )
    np.savez(
        tmp_path / "audio.npz",
        utterance_ids=np.asarray([uid.upper() for uid in ids]),
        embeddings=np.stack([
            np.full(768, index, dtype=np.float32) for index in range(len(ids))
        ]),
    )

    acoustic = pd.DataFrame({
        "utterance_id": [f" {uid.upper()} " for uid in ids],
        **{
            f"acoustic_{column}": [index + column for index in range(len(ids))]
            for column in range(116)
        },
    })
    if omit_audio_id is not None:
        acoustic = acoustic[
            acoustic["utterance_id"].str.strip().str.casefold() != omit_audio_id
        ]
    if duplicate_audio_id:
        acoustic = pd.concat([acoustic, acoustic.iloc[[0]]], ignore_index=True)
    acoustic.to_csv(tmp_path / "acoustic.csv", index=False)

    prosody_ids = [uid.upper() for uid in ids[::2] + ids[1::2]]
    pd.DataFrame({
        "utterance_id": prosody_ids,
        **{
            feature: [int(uid[1:]) + column for uid in prosody_ids]
            for column, feature in enumerate(PROSODY_FEATURES)
        },
    }).to_csv(tmp_path / "prosody.csv", index=False)
    quality_ids = [uid.upper() for uid in ids[1:] + ids[:1]]
    quality_values = {
        "snr_db_estimate": [10.0] * len(ids),
        "clipping_ratio": [0.0] * len(ids),
        "silence_ratio": [1.0 if uid.casefold() == "u0" else 0.1 for uid in quality_ids],
        "duration_in_range": [1.0] * len(ids),
        "rms": [0.1] * len(ids),
        "spectral_flatness": [0.02] * len(ids),
    }
    pd.DataFrame({
        "utterance_id": quality_ids,
        **{feature: quality_values[feature] for feature in QUALITY_FEATURES},
    }).to_csv(tmp_path / "quality.csv", index=False)
    return {
        "mustard_json_path": mustard_path,
        "handoff_csv_path": tmp_path / "acoustic.csv",
        "quality_csv_path": tmp_path / "quality.csv",
        "wav2vec2_npz_path": tmp_path / "audio.npz",
        "text_embeddings_npz_path": tmp_path / "text.npz",
        "prosody_csv_path": tmp_path / "prosody.csv",
    }


def test_loader_joins_all_sources_by_normalized_id_and_preserves_context(tmp_path):
    paths = _write_handoffs(tmp_path)
    report_path = tmp_path / "alignment.json"
    splits, acoustic_dim = load_mustard_fusion_records(
        **paths, alignment_report_path=report_path, seed=7
    )
    records = {record.example_id: record for rows in splits.values() for record in rows}

    assert acoustic_dim == 116
    assert set(records) == {f"u{i}" for i in range(12)}
    assert records["u3"].text_embedding.shape == (768,)
    assert records["u3"].context_embedding.shape == (768,)
    assert records["u3"].context_embedding[0] == 103
    assert records["u3"].has_context
    assert records["u0"].has_context is False
    assert records["u0"].audio_available is False
    assert records["u1"].audio_available is True
    assert records["u3"].prosody_vector[0] == 3
    assert records["u3"].acoustic_features_full[0] == 3
    group_sets = {
        name: {speaker_group(record) for record in rows}
        for name, rows in splits.items()
    }
    assert group_sets["train"].isdisjoint(group_sets["val"])
    assert group_sets["train"].isdisjoint(group_sets["test"])
    assert group_sets["val"].isdisjoint(group_sets["test"])

    dataset = FusionDataset([records["u3"]], acoustic_dim=acoustic_dim)
    batch = collate_fn([dataset[0]])
    assert batch["context_embedding"].shape == (1, 768)
    assert batch["has_context"].tolist() == [True]
    assert json.loads(report_path.read_text(encoding="utf-8"))["common_valid_labelled_count"] == 12


def test_loader_reports_small_unmatched_set_and_fails_on_large_drop(tmp_path):
    paths = _write_handoffs(tmp_path, omit_audio_id="u0")
    report_path = tmp_path / "alignment.json"
    splits, _ = load_mustard_fusion_records(
        **paths, alignment_report_path=report_path, max_alignment_drop_ratio=0.10
    )
    assert sum(map(len, splits.values())) == 11
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["unmatched_ids"]["acoustic"]["missing_from_modality_examples"] == ["u0"]

    _write_handoffs(tmp_path, omit_audio_id="u0")
    acoustic = pd.read_csv(paths["handoff_csv_path"])
    audio_ids = acoustic["utterance_id"].str.strip().str.casefold()
    acoustic = acoustic[~audio_ids.isin(["u0", "u1"])]
    acoustic.to_csv(paths["handoff_csv_path"], index=False)
    with pytest.raises(ValueError, match="labelled utterances aligned"):
        load_mustard_fusion_records(
            **paths,
            alignment_report_path=report_path,
            max_alignment_drop_ratio=0.10,
        )


def test_duplicate_ids_fail_instead_of_joining_ambiguously(tmp_path):
    paths = _write_handoffs(tmp_path, duplicate_audio_id=True)
    with pytest.raises(ValueError, match="duplicate normalized IDs"):
        load_mustard_fusion_records(**paths)


def test_nonfinite_features_are_reported_and_excluded(tmp_path):
    paths = _write_handoffs(tmp_path)
    acoustic = pd.read_csv(paths["handoff_csv_path"])
    acoustic.loc[acoustic.index[0], "acoustic_0"] = np.nan
    acoustic.to_csv(paths["handoff_csv_path"], index=False)
    report_path = tmp_path / "alignment.json"
    splits, _ = load_mustard_fusion_records(
        **paths, alignment_report_path=report_path
    )

    assert sum(map(len, splits.values())) == 11
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["invalid_feature_ids"]["acoustic"]["count"] == 1
    assert report["invalid_feature_ids"]["acoustic"]["examples"] == ["u0"]


def test_psid_uses_normalized_trainable_projections_and_all_audio_features(tmp_path):
    paths = _write_handoffs(tmp_path)
    splits, _ = load_mustard_fusion_records(**paths, seed=11)
    records = splits["train"]
    text, audio, _, _ = build_representations(records)
    assert text.shape[1] == 1537
    assert audio.shape[1] == 923
    assert audio[:, :768].std() > 0

    model = ComparisonClassifier("psid_concat", text.shape[1], audio.shape[1], dropout=0)
    z_text = torch.nn.functional.normalize(model.text_projection(torch.tensor(text)), dim=-1)
    z_audio = torch.nn.functional.normalize(model.audio_projection(torch.tensor(audio)), dim=-1)
    mismatch = torch.abs(z_text - z_audio)
    assert torch.allclose(torch.linalg.vector_norm(z_text, dim=-1), torch.ones(len(text)), atol=1e-5)
    assert mismatch.shape == (len(text), 128)
    model(torch.tensor(text), torch.tensor(audio)).sum().backward()
    assert model.text_projection.weight.grad is not None
    assert model.audio_projection.weight.grad is not None
