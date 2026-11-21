"""Run a leakage-safe TF-IDF + logistic-regression text baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from member1_text import require_python_311
from member1_text.config import PACKAGE_ROOT, load_config
from member1_text.data.mustard import load_mustard
from member1_text.data.splits import stratified_kfold
from member1_text.evaluate import classification_metrics
from member1_text.preprocess import join_context, preprocess_example


def _text(example) -> str:
    item = preprocess_example(example)
    context = join_context(item.context_turns) or ""
    return f"{context} [SEP] {item.text}" if context else item.text


def run_mustard_cv(output: str | Path | None = None, seed: int = 42) -> Path:
    examples = load_mustard()
    folds = stratified_kfold(examples, n_splits=5, seed=seed)
    by_id = {example.example_id: example for example in examples}
    results = []
    for fold_number, (train_ids, valid_ids) in enumerate(folds, start=1):
        train = [by_id[key] for key in train_ids]
        valid = [by_id[key] for key in valid_ids]
        vectorizer = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=50000)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True, max_features=50000)),
        ])
        x_train = vectorizer.fit_transform([_text(item) for item in train])
        x_valid = vectorizer.transform([_text(item) for item in valid])
        y_train = [int(item.label) for item in train]
        y_valid = [int(item.label) for item in valid]
        classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_valid)
        probabilities = classifier.predict_proba(x_valid)[:, list(classifier.classes_).index(1)]
        metrics = classification_metrics(y_valid, predictions, probabilities=probabilities)
        results.append({"fold": fold_number, "n_train": len(train), "n_validation": len(valid), **metrics})

    macro_f1 = [row["macro_f1"] for row in results]
    payload = {
        "dataset": "mustard",
        "task": "sarcasm",
        "model": "word+character TF-IDF + class-weighted logistic regression",
        "split": "stratified_5fold",
        "seed": seed,
        "is_independent_test": False,
        "validation_macro_f1_mean": sum(macro_f1) / len(macro_f1),
        "validation_macro_f1_std": (sum((score - sum(macro_f1) / len(macro_f1)) ** 2 for score in macro_f1) / len(macro_f1)) ** 0.5,
        "folds": results,
    }
    destination = Path(output) if output else PACKAGE_ROOT / "outputs/eval/mustard__sarcasm/tfidf_baseline_cv.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def main(argv: list[str] | None = None) -> None:
    require_python_311()
    parser = argparse.ArgumentParser(description="Evaluate the TF-IDF text baseline on MUStARD.")
    parser.add_argument("--output")
    parser.add_argument("--seed", type=int, default=int(load_config()["project"]["seed"]))
    args = parser.parse_args(argv)
    print(run_mustard_cv(args.output, args.seed))


if __name__ == "__main__":
    main()
