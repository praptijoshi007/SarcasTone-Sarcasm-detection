"""Verification script to test 4-way synchronization across Member 1, Member 2, Member 3, and Member 4.

Run with:
    python3.11 verify_all_members_sync.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml
from shared.constants import (
    AUDIO_HIDDEN,
    PROSODY_DIM,
    PROSODY_FEATURES,
    QUALITY_DIM,
    QUALITY_FEATURES,
    TARGET_SAMPLE_RATE,
    TEXT_HIDDEN,
)
from shared.sync import sync_utterance, verify_corpus_sync


def verify_sync() -> bool:
    print("=" * 70)
    print(" SARCAS TONE: 4-WAY MULTIMODAL SYNCHRONIZATION REPORT")
    print(" (M1: Text | M2: Audio | M3: Fusion | M4: XAI, API & Deployment)")
    print("=" * 70)

    all_passed = True

    # ---------------------------------------------------------
    # 1. Contract & Configuration Check
    # ---------------------------------------------------------
    print("\n[1] INTERFACE CONTRACT & DIMENSION CHECK")
    contract_path = REPO_ROOT / "configs" / "interface_contract.yaml"
    m3_cfg_path = REPO_ROOT / "member3_fusion" / "configs" / "fusion_config.yaml"
    m4_cfg_path = REPO_ROOT / "member4" / "configs" / "system_config.yaml"

    with open(contract_path, "r", encoding="utf-8") as f:
        contract = yaml.safe_load(f)

    with open(m3_cfg_path, "r", encoding="utf-8") as f:
        m3_cfg = yaml.safe_load(f)["interface"]

    with open(m4_cfg_path, "r", encoding="utf-8") as f:
        m4_contract = yaml.safe_load(f)["interface_contract"]

    contract_text_dim = contract["text_embeddings"][-1]
    contract_audio_dim = contract["audio_embeddings"][-1]
    contract_prosody_dim = contract["prosody"][-1]
    contract_quality_dim = contract["quality"][-1]
    contract_sr = contract["sample_rate_hz"]

    dims_to_check = [
        ("Text Hidden Dimension", TEXT_HIDDEN, contract_text_dim, m3_cfg["text_hidden"], m4_contract["text_hidden"]),
        ("Audio Hidden Dimension", AUDIO_HIDDEN, contract_audio_dim, m3_cfg["audio_hidden"], m4_contract["audio_hidden"]),
        ("Prosody Vector Dimension", PROSODY_DIM, contract_prosody_dim, m3_cfg["prosody_dim"], m4_contract["prosody_dim"]),
        ("Quality Vector Dimension", QUALITY_DIM, contract_quality_dim, m3_cfg["quality_dim"], m4_contract["quality_dim"]),
        ("Sample Rate (Hz)", TARGET_SAMPLE_RATE, contract_sr, 16000, m4_contract["sample_rate_hz"]),
    ]

    for name, const_val, contract_val, m3_val, m4_val in dims_to_check:
        match = (const_val == contract_val == m3_val == m4_val)
        status = "PASSED" if match else "FAILED"
        if not match:
            all_passed = False
        print(f"  - {name:<26}: {status} (Shared: {const_val}, M3: {m3_val}, M4: {m4_val})")

    # ---------------------------------------------------------
    # 2. Member 1 & Member 2 Alignment (Corpus Check)
    # ---------------------------------------------------------
    print("\n[2] MEMBER 1 (TEXT) & MEMBER 2 (AUDIO) CORPUS ALIGNMENT")
    alignment = verify_corpus_sync()
    print(f"  - Total Member 1 Text Records : {alignment['total_member1_records']}")
    print(f"  - Total Member 2 Audio Records: {alignment['total_member2_records']}")
    print(f"  - Perfectly Aligned Records   : {alignment['aligned_records']}")
    print(f"  - Speaker Mismatches          : {len(alignment['speaker_mismatches'])}")
    print(f"  - Label Mismatches            : {len(alignment['label_mismatches'])}")
    print(f"  - M1/M2 Synchronized Status   : {'PASSED' if alignment['is_synced'] else 'FAILED'}")
    if not alignment['is_synced']:
        all_passed = False

    # ---------------------------------------------------------
    # 3. Member 3 Fusion Dataset Integration Check
    # ---------------------------------------------------------
    print("\n[3] MEMBER 3 (FUSION) DATASET & HANDOFF INTEGRATION")
    try:
        from member3_fusion.src.training.dataset import load_mustard_fusion_records

        d_cfg = yaml.safe_load(m3_cfg_path.read_text(encoding="utf-8"))["data"]
        splits, acoustic_dim = load_mustard_fusion_records(
            mustard_json_path=REPO_ROOT / d_cfg["mustard_json"],
            handoff_csv_path=REPO_ROOT / d_cfg["audio_handoff_csv"],
            quality_csv_path=REPO_ROOT / d_cfg["audio_quality_csv"],
            predictions_csv_path=REPO_ROOT / d_cfg["audio_predictions_csv"],
            split_strategy="speaker",
        )
        total_m3_records = sum(len(v) for v in splits.values())
        print(f"  - Handoff CSVs Loaded Correctly : True")
        print(f"  - Total M3 Records Processed    : {total_m3_records} / 690")
        print(f"  - Speaker-Aware Train Split     : {len(splits['train'])} utterances")
        print(f"  - Speaker-Aware Val Split       : {len(splits['val'])} utterances")
        print(f"  - Speaker-Aware Test Split      : {len(splits['test'])} utterances")
        print(f"  - Acoustic Feature Dimensions   : {acoustic_dim} features")
        m3_dataset_ok = (total_m3_records == 690 and len(splits['train']) > 0)
        print(f"  - Member 3 Dataset Status       : {'PASSED' if m3_dataset_ok else 'FAILED'}")
        if not m3_dataset_ok:
            all_passed = False
    except Exception as e:
        print(f"  - Member 3 Dataset Status       : FAILED ({e})")
        all_passed = False

    # ---------------------------------------------------------
    # 4. Member 4 Multimodal Pipeline, PSID & XAI Integration
    # ---------------------------------------------------------
    print("\n[4] MEMBER 4 SYSTEM, PIPELINE & EXPLAINABILITY (XAI)")
    try:
        from member4.backend.pipeline import MultimodalPipeline
        from member4.backend.main import app

        m4_pipe = MultimodalPipeline()
        sample_wav = REPO_ROOT / "member4" / "fixtures" / "preset_sheldon_sarcastic.wav"
        if not sample_wav.is_file():
            sample_wav = REPO_ROOT / "member2_audio" / "tests" / "fixtures" / "test_sample.wav"

        m4_res = m4_pipe.predict(
            audio_path=sample_wav if sample_wav.is_file() else None,
            text="It's just a privilege to watch your mind at work.",
            context=["I think I solved the puzzle."],
            speaker="SHELDON",
        )

        print(f"  - Pipeline Prediction Result    : {m4_res.label.upper()} (Confidence: {m4_res.confidence*100:.1f}%)")
        print(f"  - PSID Incongruence Score       : {m4_res.psid_dissonance_score:.3f}")
        print(f"  - Modality Gating Weights       : Text {m4_res.modality_weights['text']*100:.1f}% | Audio {m4_res.modality_weights['audio']*100:.1f}%")
        print(f"  - Predictive Entropy            : {m4_res.entropy:.3f} nats (Uncertain: {m4_res.uncertain})")
        print(f"  - Token Saliency (Captum/IG)    : {[t.token for t in m4_res.explanation.top_text_tokens[:3]]}")
        print(f"  - Acoustic Features Attributed  : {[a.feature_name for a in m4_res.explanation.top_acoustic_features[:2]]}")
        print(f"  - REST API Routes Initialized   : {[r.path for r in getattr(app, 'routes', [])]}")

        m4_ok = (m4_res.label == "sarcastic" and m4_res.confidence > 0.70 and len(m4_res.explanation.top_text_tokens) > 0)
        print(f"  - Member 4 System Status        : {'PASSED' if m4_ok else 'FAILED'}")
        if not m4_ok:
            all_passed = False
    except Exception as e:
        print(f"  - Member 4 System Status        : FAILED ({e})")
        all_passed = False

    # ---------------------------------------------------------
    # Final Summary
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    if all_passed:
        print(" OVERALL STATUS: ALL 4 MEMBERS ARE FULLY SYNCHRONIZED & OPERATIONAL! [PASSED]")
    else:
        print(" OVERALL STATUS: SYNCHRONIZATION ISSUES DETECTED! [FAILED]")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = verify_sync()
    sys.exit(0 if success else 1)
