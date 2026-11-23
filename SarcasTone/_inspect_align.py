"""Temporary alignment inspection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\prapt\python projs\SarcasTone-Sarcasm-detection")
ST = BASE / "SarcasTone"


def norm(x) -> str:
    return str(x).strip()


mustard = json.loads((ST / "datasets/MUSTARD/sarcasm_data.json").read_text(encoding="utf-8"))
print("json", len(mustard), "sample", list(mustard)[:3])
rec = mustard[list(mustard)[0]]
print("json keys", rec.keys())
print("sarcasm types", {type(v["sarcasm"]).__name__ for v in mustard.values()})
print("label counts", sum(bool(v["sarcasm"]) for v in mustard.values()), len(mustard))
empty_ctx = sum(1 for v in mustard.values() if not v.get("context"))
print("empty context", empty_ctx)
speakers = {str(v.get("speaker", "")) for v in mustard.values()}
print("speakers", len(speakers), sorted(speakers)[:30])

text = np.load(ST / "member1_text/handoff/mustard_roberta_embeddings.npz", allow_pickle=True)
text_ids = [norm(x) for x in text["ids"]]
ctx = text["context"]
utt = text["utterance"]
print("text zero ctx", int((np.linalg.norm(ctx, axis=1) < 1e-6).sum()))
print("text zero utt", int((np.linalg.norm(utt, axis=1) < 1e-6).sum()))
print("prob range", float(text["probability_sarcasm"].min()), float(text["probability_sarcasm"].max()))

wav = np.load(BASE / "member2_audio/outputs/handoff/mustard_wav2vec2_embeddings.npz", allow_pickle=True)
wav_ids = [norm(x) for x in wav["utterance_ids"]]
print("wav zero", int((np.linalg.norm(wav["embeddings"], axis=1) < 1e-6).sum()))

ac = pd.read_csv(BASE / "member2_audio/outputs/handoff/mustard_acoustic_features_116.csv")
pr = pd.read_csv(BASE / "member2_audio/outputs/handoff/mustard_prosody_features_32_praat.csv")
qu = pd.read_csv(BASE / "member2_audio/outputs/handoff/mustard_audio_quality_contract_6d_praat.csv")

sets = {
    "json": set(map(norm, mustard)),
    "text": set(text_ids),
    "wav": set(wav_ids),
    "ac": set(ac["utterance_id"].map(norm)),
    "pr": set(pr["utterance_id"].map(norm)),
    "qu": set(qu["utterance_id"].map(norm)),
}
common = set.intersection(*sets.values())
print("common", len(common))
for name, s in sets.items():
    missing = sets["json"] - s
    extra = s - sets["json"]
    print(name, "n", len(s), "missing_from_json", len(missing), "extra", len(extra))
    if missing:
        print("  missing sample", sorted(missing)[:8])
    if extra:
        print("  extra sample", sorted(extra)[:8])

print("acoustic nulls", int(ac.isna().sum().sum()), "inf", int(np.isinf(ac.select_dtypes("number")).sum().sum()))
print("prosody nulls", int(pr.isna().sum().sum()))
print("quality nulls", int(qu.isna().sum().sum()))
print(qu.describe().T[["min", "max", "mean"]].to_string())
print("duration_in_range value counts")
print(qu["duration_in_range"].value_counts().to_dict())
print("clipping>0.01", int((qu["clipping_ratio"] > 0.01).sum()), "silence>0.8", int((qu["silence_ratio"] > 0.8).sum()), "rms==0", int((qu["rms"] <= 0).sum()))

# text id order vs json
print("text order equals json", text_ids == list(map(norm, mustard)))
print("wav order equals text", wav_ids == text_ids)
print("ac order equals text", ac["utterance_id"].map(norm).tolist() == text_ids)
print("pr order equals text", pr["utterance_id"].map(norm).tolist() == text_ids)
