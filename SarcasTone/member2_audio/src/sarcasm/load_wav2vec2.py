
import os

if "HF_HOME" not in os.environ:
    from pathlib import Path
    os.environ["HF_HOME"] = str(Path(__file__).resolve().parents[2] / ".cache")
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

MODEL_NAME = "facebook/wav2vec2-base"

print("=" * 60)
print("SARCAS TONE: LOADING WAV2VEC2")
print("=" * 60)

torch.set_num_threads(2)

print("Device: CPU")
print("Model:", MODEL_NAME)
print("Loading feature extractor...")

processor = Wav2Vec2FeatureExtractor.from_pretrained(
    MODEL_NAME
)

print("Loading pretrained speech model...")

model = Wav2Vec2Model.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True
)

model.eval()

parameters = sum(
    p.numel() for p in model.parameters()
)

print("\nMODEL LOADED SUCCESSFULLY")
print("Parameters:", f"{parameters:,}")
print("Hidden dimension:", model.config.hidden_size)
print("Sampling rate:", processor.sampling_rate)
print("Device:", next(model.parameters()).device)

print("\nWav2Vec2 is ready for embedding extraction!")
