# Member 3 — Multimodal Fusion

Implements the cross-modal fusion layer for **SarcasTone**: a sarcasm-detection
system that fuses text (member 1) and audio (member 2) representations.

---

## Architecture

```
Text  (member1 SpanPoolingModel → [B, 768])  ──┐
                                                ├──→ CrossModalAttention ──→ PSID ──→ ConfidenceGate ──→ Classifier
Audio (member2 Wav2Vec2 → [B, 768])          ──┘
Prosody / Acoustic (member2 handoff CSV → [B, 32+])
Quality (member2 → [B, 6])
```

### Fusion strategies (ablation)

| Strategy | Key |
|---|---|
| Early (raw concat) | `early` |
| Late (average predictions) | `late` |
| Concat cross-attended | `concat` |
| **Full bidirectional cross-attention + PSID + gating** | `cross_attention` ✅ default |

---

## Directory layout

```
member3_fusion/
├── configs/
│   └── fusion_config.yaml        ← all hyper-parameters
├── src/
│   ├── attention/cross_attention.py  ← CrossModalAttentionBlock, StackedCrossModalAttention
│   ├── context/context_encoder.py
│   ├── fusion/
│   │   ├── config.py              ← load_config()
│   │   ├── encoders.py            ← TextEncoderInterface, AudioEncoderInterface
│   │   └── prosody_proj.py
│   ├── gating/confidence_gate.py
│   ├── incongruence/psid.py
│   ├── inference/predict_multimodal.py  ← predict_multimodal()
│   ├── models/fusion_model.py     ← SarcasToneFusionModel, build_fusion_model()
│   ├── training/
│   │   ├── dataset.py             ← FusionDataset, load_mustard_fusion_records()
│   │   └── trainer.py             ← FusionTrainer
│   └── uncertainty/calibration.py ← TemperatureScaler, fit_temperature()
├── tests/
│   └── test_architecture.py
├── train_fusion.py                ← training entry point
├── evaluate_fusion.py             ← evaluation / ablation entry point
├── compare_fusion.py              ← four controlled text/audio comparisons
├── predict.py                     ← inference CLI
└── requirements.txt
```

---

## Quick start

Run the commands below from the `SarcasTone/` project directory.

### 1. Install dependencies

```bash
pip install -r member3_fusion/requirements.txt
```

### 2. Train

```bash
# Full model (cross-attention + PSID + adaptive gating)
python member3_fusion/train_fusion.py --experiment full_model

# Text-only ablation
python member3_fusion/train_fusion.py --experiment text_only --no-text-encoder

# Cached Member 1 embeddings are selected by default when configured.
python member3_fusion/train_fusion.py --experiment full_model

# Current text handoff lacks OOF provenance; this explicitly labels output exploratory.
python member3_fusion/compare_fusion.py --allow-unverified-text-provenance
```

Use `--epochs`, `--batch-size`, `--seed`, and `--bootstrap` to override comparison
settings. `--live-text-encoder` explicitly overrides the cached Member 1 embeddings
for the legacy fusion trainer; it is not used by the comparison command.

### 3. Evaluate

```bash
# Single experiment
python member3_fusion/evaluate_fusion.py --experiment full_model --split test

# Full ablation table
python member3_fusion/evaluate_fusion.py --ablation
```

### 4. Inference

```bash
python member3_fusion/predict.py path/to/audio.wav
python member3_fusion/predict.py path/to/audio.wav --text "Oh, brilliant."
python member3_fusion/predict.py path/to/audio.wav --verbose
```

Python API:

```python
from member3_fusion.predict import predict_multimodal

result = predict_multimodal("audio.wav", text="It's just great.")
print(result["label"])           # "sarcastic" or "non_sarcastic"
print(result["probabilities"])   # {"non_sarcastic": 0.23, "sarcastic": 0.77}
print(result["uncertain"])       # True if entropy > threshold
print(result["audio_available"]) # bool — was audio extractable?
print(result["text_available"])  # bool — was text encodable?
```

### 5. Run unit tests

```bash
cd SarcasTone
python -m pytest member3_fusion/tests/test_architecture.py -v
```

---

## Data inputs

| File | Required | Source |
|---|---|---|
| `datasets/MUSTARD/sarcasm_data.json` | ✅ | MUStARD dataset |
| `member1_text/handoff/mustard_roberta_embeddings.npz` | ✅ | Member 1 cached utterance/context vectors |
| `../member2_audio/outputs/handoff/mustard_wav2vec2_embeddings.npz` | ✅ | Member 2 cached Wav2Vec2 vectors |
| `../member2_audio/outputs/handoff/mustard_acoustic_features_116.csv` | ✅ | Member 2 acoustic features |
| `../member2_audio/outputs/handoff/mustard_prosody_features_32_praat.csv` | ✅ | Member 2 prosody features |
| `../member2_audio/outputs/handoff/mustard_audio_quality_contract_6d_praat.csv` | ✅ | Member 2 quality features |

---

## Controlled comparison and artifacts

The comparison command trains, with one seed, speaker-aware split, label mapping,
train-only normalization, and classifier/training settings:

1. `text_only`: cached utterance and masked context embeddings.
2. `audio_only`: Wav2Vec2, 116 acoustic, 32 prosody, and quality features.
3. `direct_concat`: `[h_t; h_a]` through the same classifier family.
4. `psid_concat`: `[h_t; h_a; |normalize(W_t h_t)-normalize(W_a h_a)|]`.

The PSID projections are trainable linear layers learned on the sarcasm objective;
they are **not** pretrained sentiment or emotion representations. The quality
contract masks unreliable audio features when `duration_in_range < 0.5`,
`silence_ratio >= 0.95`, or `rms <= 1e-8`, while retaining quality values as
predictors. IDs are normalized and joined by key, never by row position.

Outputs are written to `member3_fusion/outputs/comparison/`: an ID alignment report,
split IDs and speakers, model/config checkpoints, per-epoch histories, test metrics
and confusion matrices, paired per-example probabilities, bootstrap confidence
intervals, and paired bootstrap deltas versus `text_only` and direct concatenation.
Metrics are reported
without interpreting a PSID gain unless the measured test results support one.

**Leakage limitation for the current text handoff:** `mustard_roberta_embeddings.npz`
contains no source-checkpoint or training-ID metadata. The nearby
`member1_text/checkpoints/mustard__sarcasm__roberta-base/trainer_state.pt` records
`fold_index: null` and all 690 training IDs, including the 162 IDs in this test
split; Member 1's README describes its deployment refit as training on all labeled
MUStARD examples. The handoff does not identify whether that exact checkpoint
produced the embeddings, and no fold checkpoints are present here. Therefore these
four computed scores are saved as **exploratory, not leakage-safe generalization
results**. `compare_fusion.py` refuses to train by default; the flag above is an
explicit override. A valid held-out comparison requires per-example out-of-fold
Member 1 embeddings or fold checkpoints with their held-out ID mapping. No Member 1
model was retrained.

The configured Member 1/2 paths are in `member3_fusion/configs/fusion_config.yaml`.
MUStARD++ is disabled until matching text/audio exports exist; the loader refuses
to silently mix it into the MUStARD handoffs.

---

## Modality handling

The model degrades gracefully when modalities are missing:

- **No audio** → `audio_available=False`; audio embeddings zeroed; confidence gate routes to text.
- **No text** → `text_embedding` is zeros; `text_available=False`; model uses audio + acoustic features.
- **Both unavailable** → model still runs (using acoustic/prosody features only).

---

## Research contributions (Member 3)

1. **Bidirectional cross-modal attention** between text and audio sequences.
2. **PSID** (Prosody-Semantic Incongruence Detection) — explicit disagreement signal.
3. **Confidence-gated fusion** — quality-aware dynamic weighting.
4. **Temperature calibration** — post-hoc uncertainty quantification.
5. **Ablation framework** — 8 experiments comparing all fusion strategies.
