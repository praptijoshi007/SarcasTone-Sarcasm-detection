
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT / "outputs" / "metrics" / "sarcasm"
    / "out_of_fold_predictions.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "analysis" / "sarcasm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_NAMES = ["Non-sarcastic", "Sarcastic"]
LABELS = [0, 1]
MODELS = ["RandomForest", "SVM_RBF"]

# --------------------------------------------------
# LOAD AND VALIDATE DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "model", "utterance_id", "speaker", "actual", "predicted"
}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Missing columns: {required_columns - set(df.columns)}"
    )

if not set(df["actual"].unique()).issubset(set(LABELS)):
    raise ValueError("Unexpected values found in actual labels.")

if not set(df["predicted"].unique()).issubset(set(LABELS)):
    raise ValueError("Unexpected values found in predicted labels.")

for model_name in MODELS:
    count = len(df[df["model"] == model_name])

    if count != 690:
        raise ValueError(
            f"{model_name} has {count} predictions instead of 690."
        )

print("=" * 65)
print("SARCAS TONE: AUDIO SARCASM ERROR ANALYSIS")
print("=" * 65)

# --------------------------------------------------
# 1. CONFUSION MATRICES
# --------------------------------------------------

for model_name in MODELS:

    model_df = df[df["model"] == model_name]

    cm = confusion_matrix(
        model_df["actual"],
        model_df["predicted"],
        labels=LABELS
    )

    plt.figure(figsize=(7, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    output_path = (
        OUTPUT_DIR / f"{model_name}_confusion_matrix.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\n{model_name} confusion matrix:")
    print(cm)
    print(f"Saved: {output_path}")

# --------------------------------------------------
# 2. FALSE POSITIVE AND FALSE NEGATIVE ANALYSIS
# --------------------------------------------------

error_summary = []

for model_name in MODELS:

    model_df = df[df["model"] == model_name]

    actual = model_df["actual"]
    predicted = model_df["predicted"]

    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())

    error_summary.append({
        "model": model_name,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "total_errors": fp + fn,
    })

    print(f"\n{model_name}")
    print(f"True Positives: {tp}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"Total Errors: {fp + fn}")

error_df = pd.DataFrame(error_summary)

error_df.to_csv(
    OUTPUT_DIR / "false_positive_negative_summary.csv",
    index=False
)

# --------------------------------------------------
# 3. SPEAKER-WISE SVM PERFORMANCE
# --------------------------------------------------

svm_df = df[df["model"] == "SVM_RBF"].copy()

speaker_results = []

for speaker, group in svm_df.groupby("speaker"):

    y_true = group["actual"]
    y_pred = group["predicted"]

    speaker_results.append({
        "speaker": speaker,
        "samples": len(group),

        "sarcastic_samples": int(
            (y_true == 1).sum()
        ),

        "non_sarcastic_samples": int(
            (y_true == 0).sum()
        ),

        "accuracy": accuracy_score(
            y_true, y_pred
        ),

        # Always evaluate both classes
        "precision_macro": precision_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),

        "recall_macro": recall_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        ),
    })

speaker_df = pd.DataFrame(speaker_results)

speaker_df = speaker_df.sort_values(
    "macro_f1",
    ascending=True
)

speaker_df.to_csv(
    OUTPUT_DIR / "svm_speaker_wise_performance.csv",
    index=False
)

print("\n" + "=" * 65)
print("SVM SPEAKER-WISE PERFORMANCE")
print("=" * 65)

print(
    speaker_df.to_string(
        index=False,
        formatters={
            "accuracy": "{:.3f}".format,
            "precision_macro": "{:.3f}".format,
            "recall_macro": "{:.3f}".format,
            "macro_f1": "{:.3f}".format,
        }
    )
)

# --------------------------------------------------
# 4. EXPORT MISCLASSIFIED UTTERANCES
# --------------------------------------------------

misclassified = df[
    df["actual"] != df["predicted"]
].copy()

misclassified["error_type"] = misclassified.apply(
    lambda row: (
        "False Positive"
        if row["actual"] == 0
        else "False Negative"
    ),
    axis=1
)

misclassified.to_csv(
    OUTPUT_DIR / "misclassified_utterances.csv",
    index=False
)

print(
    f"\nTotal misclassified prediction rows: "
    f"{len(misclassified)}"
)

# --------------------------------------------------
# 5. SPEAKER-WISE SVM F1 VISUALIZATION
# --------------------------------------------------

plt.figure(figsize=(12, 7))

sns.barplot(
    data=speaker_df,
    x="macro_f1",
    y="speaker",
    color="steelblue"
)

plt.xlim(0, 1)
plt.title("SVM Macro-F1 by Speaker")
plt.xlabel("Macro-F1")
plt.ylabel("Speaker")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "svm_speaker_wise_f1.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 6. MODEL ERROR COMPARISON
# --------------------------------------------------

plt.figure(figsize=(8, 5))

error_long = error_df.melt(
    id_vars="model",
    value_vars=[
        "false_positive",
        "false_negative"
    ],
    var_name="error_type",
    value_name="count"
)

sns.barplot(
    data=error_long,
    x="model",
    y="count",
    hue="error_type"
)

plt.title("False Positive vs False Negative Comparison")
plt.xlabel("Model")
plt.ylabel("Number of Errors")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_error_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 7. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("ERROR ANALYSIS COMPLETED")
print("=" * 65)

print(f"Total prediction rows: {len(df)}")
print(f"Unique utterances: {df['utterance_id'].nunique()}")
print(f"Unique speakers: {df['speaker'].nunique()}")

print("\nModel error summary:")
print(error_df.to_string(index=False))

print(f"\nAll analysis files saved in:\n{OUTPUT_DIR}")
