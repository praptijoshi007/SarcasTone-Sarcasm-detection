
from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "checkpoints" / "ravdess_emotion_random_forest.joblib"

VAL_PATH = (
    ROOT / "data" / "processed" /
    "emotion_dataset" / "validation.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "plots"


def analyze():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    features = bundle["feature_columns"]

    validation = pd.read_csv(VAL_PATH)

    X_val = validation[features]
    y_val = validation["emotion"]

    predictions = model.predict(X_val)

    # 1. Confusion matrix
    cm = confusion_matrix(
        y_val,
        predictions,
        labels=model.classes_
    )

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_
    )

    plt.title("RAVDESS Emotion Recognition - Confusion Matrix")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("Actual Emotion")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "emotion_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # 2. Feature importance
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "emotion_feature_importance.csv",
        index=False
    )

    top_features = importance.head(20).sort_values(
        "importance"
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.title("Top 20 Acoustic Features")
    plt.xlabel("Random Forest Feature Importance")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "top_20_acoustic_features.png",
        dpi=300
    )

    plt.close()

    print("\nEMOTION MODEL ANALYSIS COMPLETE")
    print("=" * 45)

    print("\nTop 15 important features:")
    print(importance.head(15).to_string(index=False))

    print("\nPlots saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    analyze()
