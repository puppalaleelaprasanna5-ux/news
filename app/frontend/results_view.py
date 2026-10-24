"""Streamlit-free helpers for the Evaluation tab.

Everything shown comes from JSON files written by the evaluation scripts. Nothing is hard-coded;
a missing or unreadable file yields a (None, message) pair with the command that creates it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"

CROSS_DATASET_FILE = "cross_dataset.json"
TUNING_FILE = "threshold_tuning.json"
COMMANDS = {
    CROSS_DATASET_FILE: "python scripts/cross_dataset_eval.py",
    TUNING_FILE: "python scripts/threshold_tuning.py",
}


def load_result(filename: str, results_dir: Path | None = None) -> tuple[dict[str, Any] | None, str | None]:
    path = Path(results_dir if results_dir is not None else RESULTS_DIR) / filename
    cmd = COMMANDS.get(filename, "the matching script in scripts/")
    if not path.exists():
        return None, f"Not generated yet. Run: {cmd}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, f"{filename} could not be read. Re-run: {cmd}"
    if not isinstance(data, dict):
        return None, f"{filename} has an unexpected format. Re-run: {cmd}"
    return data, None


def pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def num(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "N/A"


def comparison_rows(cross: dict[str, Any]) -> list[dict[str, str]]:
    """ISOT validation (from model metadata) vs WELFake, as display rows."""
    isot = (cross.get("isot_validation_from_metadata") or {}).get("metrics") or {}
    wel = cross.get("metrics") or {}

    def row(name: str, m: dict[str, Any], fake_pct: Any = None) -> dict[str, str]:
        return {"Dataset": name, "Accuracy": pct(m.get("accuracy")), "Precision": pct(m.get("precision")),
                "Recall": pct(m.get("recall")), "F1": pct(m.get("f1")), "ROC-AUC": num(m.get("roc_auc")),
                "Predicted Fake": "N/A" if fake_pct is None else f"{float(fake_pct):.1f}%"}

    return [row("ISOT validation (in-domain)", isot),
            row("WELFake (external)", wel, cross.get("fake_predictions_pct"))]


def accuracy_drop_points(cross: dict[str, Any]) -> float | None:
    try:
        isot = float(cross["isot_validation_from_metadata"]["metrics"]["accuracy"])
        wel = float(cross["metrics"]["accuracy"])
    except (KeyError, TypeError, ValueError):
        return None
    return (isot - wel) * 100.0


def tuning_rows(tuning: dict[str, Any]) -> list[dict[str, str]]:
    labels = {"default_0.5": "Default threshold", "tuned_best_f1": "Tuned (best F1)",
              "tuned_best_balanced_accuracy": "Tuned (best balanced accuracy)"}
    out = []
    for key, label in labels.items():
        m = (tuning.get("test_half") or {}).get(key)
        if not m:
            continue
        out.append({"Setting": label, "Threshold": f"{float(m['threshold']):.2f}",
                    "Accuracy": pct(m.get("accuracy")), "Balanced acc.": pct(m.get("balanced_accuracy")),
                    "Precision": pct(m.get("precision")), "Recall": pct(m.get("recall")),
                    "F1": pct(m.get("f1")), "Predicted Fake": f"{float(m['predicted_fake_pct']):.1f}%"})
    return out


def confusion_rows(matrix: list[list[int]]) -> list[dict[str, Any]]:
    labels = ["Real", "Fake"]
    return [{"True \\ Predicted": labels[i], "Predicted Real": int(matrix[i][0]), "Predicted Fake": int(matrix[i][1])}
            for i in range(2)]
