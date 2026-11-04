from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from phishing_model import load_dataset, save_model, train_model


def train_and_evaluate(dataset_path: str, model_path: str, metrics_path: str, plot_path: str | None = None) -> dict:
    data = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(
        data["email_text"], data["label"], test_size=0.25, random_state=42, stratify=data["label"]
    )
    model = train_model(__import__("pandas").DataFrame({"email_text": X_train, "label": y_train}))
    predictions = model.predict(X_test)
    labels = ["safe", "phishing"]
    report = classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)
    metrics = {
        "dataset": str(dataset_path), "dataset_rows": int(len(data)),
        "class_counts": {str(k): int(v) for k, v in data["label"].value_counts().items()},
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "classification_report": report,
        "note": "Metrics are from the selected dataset and split; sample dataset results are demonstration-only.",
    }
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_model(model, model_path)
    if plot_path:
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        plt.figure(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.xlabel("Predicted label")
        plt.ylabel("Actual label")
        plt.title("Phishing Detection Confusion Matrix")
        plt.tight_layout()
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=160)
        plt.close()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the phishing email classifier.")
    parser.add_argument("--dataset", default="data/sample_emails.csv")
    parser.add_argument("--model", default="models/phishing_model.joblib")
    parser.add_argument("--metrics", default="models/metrics.json")
    parser.add_argument("--plot", default="models/confusion_matrix.png")
    args = parser.parse_args()
    result = train_and_evaluate(args.dataset, args.model, args.metrics, args.plot)
    print(json.dumps({"accuracy": result["accuracy"], "rows": result["dataset_rows"], "class_counts": result["class_counts"]}, indent=2))
