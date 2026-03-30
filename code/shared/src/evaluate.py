"""Evaluation and plotting helpers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str | Path | None = None) -> None:
    """Plot and optionally save a confusion matrix."""
    matrix = confusion_matrix(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axis,
    )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion Matrix")
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=150)

    plt.close(figure)


def plot_subject_results(results: list[dict], save_path: str | Path | None = None) -> None:
    """Plot per-subject accuracy and kappa."""
    subjects = [entry["subject"] for entry in results]
    accuracies = [entry["test_acc"] for entry in results]
    kappas = [entry["kappa"] for entry in results]

    x_positions = np.arange(len(subjects))
    width = 0.35

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(x_positions - width / 2, accuracies, width, label="Accuracy")
    axis.bar(x_positions + width / 2, kappas, width, label="Kappa")
    axis.set_xlabel("Subject")
    axis.set_ylabel("Score")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(x_positions, [f"S{subject}" for subject in subjects])
    axis.set_title("Per-subject Performance")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()

    if save_path is not None:
        figure.savefig(save_path, dpi=150)

    plt.close(figure)


def evaluate_and_save(
    y_true,
    y_pred,
    class_names,
    result_dir: str | Path,
    *,
    prefix: str = "test",
) -> dict:
    """Save a confusion matrix and classification report."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(y_true, y_pred, class_names, save_path=result_dir / f"{prefix}_confusion_matrix.png")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    with (result_dir / f"{prefix}_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return report


def load_and_evaluate(results_path: str | Path, save_dir: str | Path | None = None) -> list[dict]:
    """Load saved subject results and optionally render a summary plot."""
    with Path(results_path).open("r", encoding="utf-8") as file:
        results = json.load(file)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_subject_results(results, save_path=save_dir / "subject_results.png")

    return results
