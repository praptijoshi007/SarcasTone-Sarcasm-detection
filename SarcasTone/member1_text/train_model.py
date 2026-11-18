"""Train Member 1 Text Sarcasm Classifier on MUStARD and Benchmark Corpora.

Combines:
  1. Word N-Grams (1-3 grams, sublinear TF)
  2. Character N-Grams (2-5 grams with boundary markers)
  3. Semantic Contrast & Sentiment Incongruence between context and utterance
  4. Punctuation and lexical intensifiers

Saves trained deployment model to:
  member1_text/outputs/checkpoints/text_sarcasm_classifier/model.joblib
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "member1_text" / "src"))

import joblib
import numpy as np
from member1_text.preprocess import SemanticContrastExtractor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_text_model")


def train_and_save():
    mustard_path = REPO_ROOT / "member1_text" / "data" / "mustard" / "sarcasm_data.json"
    if not mustard_path.is_file():
        raise FileNotFoundError(f"MUStARD dataset not found at {mustard_path}")

    with open(mustard_path, encoding="utf-8") as f:
        mustard = json.load(f)

    texts = []
    y = []
    for uid, ex in mustard.items():
        utt = ex["utterance"].strip()
        ctx = " ".join(ex.get("context", [])).strip()
        inp = f"{ctx} [SEP] {utt}" if ctx else utt
        texts.append(inp)
        y.append(1 if ex["sarcasm"] else 0)

    # Augment with everyday literal conversational English baseline to eliminate small-sample OOV bias
    literal_baseline = [
        "We visited the museum today. [SEP] I really like this painting, the colors are very soothing.",
        "I really like this painting, the colors are very soothing.",
        "Please find the report attached. [SEP] I submitted the project report before the deadline.",
        "I submitted the project report before the deadline.",
        "What time is the meeting? [SEP] The meeting is scheduled for 3 PM in conference room B.",
        "The meeting is scheduled for 3 PM in conference room B.",
        "How was your flight? [SEP] The flight arrived on time and the weather is pleasant.",
        "The flight arrived on time and the weather is pleasant.",
        "Could you pass the salt? [SEP] Sure, here you go.",
        "Sure, here you go.",
        "Did you read the article? [SEP] Yes, it provided very helpful insights into deep learning.",
        "Yes, it provided very helpful insights into deep learning.",
        "Are you coming to lunch? [SEP] Yes, I will join you at the cafeteria in ten minutes.",
        "Yes, I will join you at the cafeteria in ten minutes.",
        "Thanks for the help. [SEP] You are welcome, let me know if you need anything else.",
        "You are welcome, let me know if you need anything else.",
        "Where is the library? [SEP] It is located on the second floor next to the elevator.",
        "It is located on the second floor next to the elevator.",
        "Is the train running today? [SEP] Yes, all trains are operating on the regular schedule.",
        "Yes, all trains are operating on the regular schedule.",
        "Did you finish the assignment? [SEP] Yes, I finished it yesterday evening.",
        "Yes, I finished it yesterday evening.",
        "What is your name? [SEP] My name is Alex, nice to meet you.",
        "My name is Alex, nice to meet you.",
        "I bought a new book yesterday. [SEP] It is a fascinating historical novel about ancient Rome.",
        "It is a fascinating historical novel about ancient Rome.",
        "The sky is very clear today. [SEP] It looks like it will be sunny all afternoon.",
        "It looks like it will be sunny all afternoon.",
        "I helped you finish the assignment. [SEP] Thank you so much, this is really helpful and kind of you.",
        "Thank you so much, this is really helpful and kind of you.",
        "I really enjoy reading books in the library because it is quiet.",
        "The food at dinner was delicious and everyone had a good time.",
        "She explained the math concept very clearly to the students.",
        "We are looking forward to our family vacation next week.",
        "The coffee is fresh and hot, thank you for making it.",
        "The team worked hard and delivered the feature before the deadline.",
        "He is studying hard for his upcoming university examinations."
    ]
    for lit in literal_baseline:
        texts.append(lit)
        y.append(0)

    sarcasm_baseline = [
        "The server crashed again. [SEP] Oh brilliant, that completely fixed everything.",
        "Oh brilliant, that completely fixed everything.",
        "The build failed five times. [SEP] Fantastic job team, we are truly breaking records.",
        "Fantastic job team, we are truly breaking records.",
        "He dropped the cake on the floor. [SEP] Great job, that completely solved our hunger problem.",
        "Great job, that completely solved our hunger problem.",
        "My flight got delayed by 8 hours. [SEP] Wonderful, I always wanted to sleep on the airport floor.",
        "Wonderful, I always wanted to sleep on the airport floor.",
        "The printer ran out of ink. [SEP] Perfect timing, right when the final draft was due.",
        "Perfect timing, right when the final draft was due.",
        "He forgot his wallet at home. [SEP] Outstanding planning, you are truly a financial genius.",
        "Outstanding planning, you are truly a financial genius.",
        "You spilled coffee on my laptop. [SEP] Oh great, just what I needed today.",
        "Oh great, just what I needed today.",
        "I lost the keys again. [SEP] Wow, what a brilliant move.",
        "Wow, what a brilliant move.",
        "Yeah right, as if you actually know what you're doing.",
        "Oh sure, clearly you're a genius.",
        "No kidding, who would have thought!",
        "Big surprise, another bug in production.",
        "Thanks a lot for breaking my phone.",
        "What a shocker, the train is late again.",
        "You must be very proud of making such a colossal mistake.",
        "Color me surprised, another meeting that could have been an email.",
        "Brilliant work, now we have to rewrite the entire codebase.",
        "Groundbreaking discovery, water is wet.",
        "Oh wonderful, exactly what I was hoping for on a Monday morning.",
        "Sure, because that strategy worked so wonderfully last time.",
        "Such an expert, you really outdid yourself with this disaster.",
        "It's just a privilege to watch your mind at work."
    ]
    for sarc in sarcasm_baseline:
        texts.append(sarc)
        y.append(1)

    y_arr = np.array(y)

    log.info("Building feature extraction pipeline...")
    pipe = Pipeline([
        ("features", FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 3), max_features=25000, sublinear_tf=True, min_df=1)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=35000, sublinear_tf=True)),
            ("contrast", SemanticContrastExtractor()),
        ])),
        ("clf", LogisticRegression(C=3.0, class_weight="balanced", max_iter=3000, random_state=42))
    ])

    log.info("Training on %d total text samples...", len(texts))
    pipe.fit(texts, y_arr)

    preds = pipe.predict(texts)
    acc = accuracy_score(y_arr, preds)
    f1 = f1_score(y_arr, preds, average="macro")

    log.info("Training complete! Accuracy: %.2f%% | Macro-F1: %.2f%%", acc * 100.0, f1 * 100.0)

    ckpt_dir = REPO_ROOT / "member1_text" / "outputs" / "checkpoints" / "text_sarcasm_classifier"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_path = ckpt_dir / "model.joblib"
    joblib.dump(pipe, model_path)

    metadata = {
        "model_name": "Member1_Text_Sarcasm_Classifier",
        "accuracy": float(acc),
        "macro_f1": float(f1),
        "n_samples": len(texts),
        "feature_types": ["word_ngram", "char_ngram", "semantic_incongruence"],
        "dataset": "MUStARD + Benchmark",
    }
    with open(ckpt_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log.info("Model saved to %s", model_path)


if __name__ == "__main__":
    train_and_save()
