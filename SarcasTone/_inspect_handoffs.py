"""Temporary inspection of member 1/2 handoffs. Delete after use."""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\prapt\python projs\SarcasTone-Sarcasm-detection")


def show_npz(p: Path) -> None:
    print("=== NPZ", p)
    archive = np.load(p, allow_pickle=True)
    print("keys", list(archive.keys()))
    for key in archive.keys():
        value = archive[key]
        print(f"  {key}: shape={getattr(value, 'shape', None)} dtype={getattr(value, 'dtype', None)}")
        if getattr(value, "ndim", 0) == 1 and str(value.dtype).startswith("<U") or (
            getattr(value, "ndim", 0) == 1 and value.dtype.kind in "USO"
        ):
            print("    sample", [str(x) for x in value[:3]])
            ids = [str(x) for x in value]
            print("    n", len(ids), "nunique", len(set(ids)))


def show_csv(p: Path) -> None:
    print("=== CSV", p.name)
    df = pd.read_csv(p)
    print("shape", df.shape)
    print("cols", list(df.columns))
    id_col = None
    for cand in ("utterance_id", "ids", "id", "example_id", "KEY", "clip_id"):
        if cand in df.columns:
            id_col = cand
            break
    if id_col is None:
        id_col = df.columns[0]
        print("id_col guessed", id_col)
    else:
        print("id_col", id_col)
    ids = df[id_col].astype(str)
    print("nunique", ids.nunique(), "dupes", int(ids.duplicated().sum()), "nulls", int(df[id_col].isna().sum()))
    print("sample ids", ids.head(3).tolist())
    if "sarcasm" in df.columns:
        print("sarcasm values", df["sarcasm"].value_counts(dropna=False).to_dict())
    if "speaker" in df.columns:
        print("speakers", df["speaker"].nunique())


for path in [
    BASE / "SarcasTone/member1_text/handoff/mustard_roberta_embeddings.npz",
    BASE / "SarcasTone/member1_text/outputs/handoff/mustard_text_embeddings.npz",
    BASE / "member2_audio/outputs/handoff/mustard_wav2vec2_embeddings.npz",
]:
    show_npz(path)

for name in [
    "mustard_acoustic_features_116.csv",
    "mustard_prosody_features_32_praat.csv",
    "mustard_audio_quality_contract_6d_praat.csv",
]:
    show_csv(BASE / "member2_audio/outputs/handoff" / name)

# nested older handoff
old = BASE / "SarcasTone/member3_fusion/member2_audio/outputs/handoff/mustard_audio_handoff.csv"
if old.is_file():
    show_csv(old)
    q = old.parent / "mustard_audio_quality.csv"
    show_csv(q)
