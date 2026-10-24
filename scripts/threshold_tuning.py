#!/usr/bin/env python3
"""Threshold tuning for the saved ISOT model on WELFake, with a strict tune/test split.

Why: on WELFake the model keeps good ranking (ROC-AUC ~0.92) but its fixed 0.5 cutoff
over-predicts Fake. This script measures how much of the accuracy drop is just a
miscalibrated threshold, WITHOUT retraining.

Protocol (no leakage):
  1. Stratified 50/50 split of WELFake (fixed seed, saved in the JSON).
  2. The threshold is chosen on the TUNING half only.
  3. Every reported metric comes from the TEST half only, at 0.5 and at the tuned threshold.

Same input policy as cross_dataset_eval.py: article BODY only, same cleaning, saved model.
Run scripts/cross_dataset_eval.py --preview-only first to confirm the label mapping.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cross_dataset_eval import (  # noqa: E402
    DEFAULT_DATASET, DEFAULT_MODELS, DEFAULT_OUTPUT,
    _find_label_column, _load_binary_labels, _normalise_columns,
)
from src.models.predict import load_predictor  # noqa: E402
from src.preprocessing.text_cleaning import clean_single_text  # noqa: E402

DEFAULT_SEED = 42
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def score_dataset(csv_path: Path, models_dir: Path, flip_labels: bool, predictor: Any | None = None):
    """Return (y_true, fake_probability) for every scorable row (body only)."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    raw = _normalise_columns(pd.read_csv(csv_path))
    label_col = _find_label_column(raw)
    if "text" not in raw.columns:
        raise ValueError(f"Dataset must have a 'text' column. Found: {list(raw.columns)}")
    raw["_clean"] = raw["text"].apply(clean_single_text)
    keep = raw["_clean"].str.len().gt(0) & raw[label_col].notna()
    df = raw.loc[keep].reset_index(drop=True)
    if df.empty:
        raise ValueError("No usable rows after dropping empty bodies / missing labels.")
    y_true = _load_binary_labels(df[label_col], flip_labels)
    predictor = predictor or load_predictor(models_dir)
    results = predictor.predict_batch(df["text"].tolist())
    idx, probs = [], []
    for i, r in enumerate(results):
        p = r.get("fake_probability")
        if r.get("label_id") in (0, 1) and p is not None:
            idx.append(i)
            probs.append(float(p))
    if not idx:
        raise RuntimeError("The model returned no scorable predictions.")
    return y_true[idx], np.asarray(probs), {"raw_rows": len(raw), "dropped_empty_rows": int((~keep).sum()),
                                           "scorable_rows": len(idx)}


def metrics_at(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) == 2 else None
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "predicted_fake_pct": float(np.mean(y_pred == 1) * 100.0),
        "confusion_matrix": {"labels": ["Real (0)", "Fake (1)"], "matrix": cm.astype(int).tolist()},
    }


def sweep(y_true: np.ndarray, probs: np.ndarray) -> list[dict[str, float]]:
    rows = []
    for t in THRESHOLD_GRID:
        y_pred = (probs >= t).astype(int)
        rows.append({
            "threshold": float(t),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        })
    return rows


def best_threshold(rows: list[dict[str, float]], key: str) -> float:
    # max score; ties broken by the threshold closest to 0.5 (least change from the default)
    best = max(r[key] for r in rows)
    tied = [r["threshold"] for r in rows if abs(r[key] - best) < 1e-12]
    return min(tied, key=lambda t: abs(t - 0.5))


def _plot_sweep(rows, f1_thr, bal_thr, path: Path) -> None:
    t = [r["threshold"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(t, [r["f1"] for r in rows], label="F1")
    ax.plot(t, [r["balanced_accuracy"] for r in rows], label="Balanced accuracy")
    ax.axvline(0.5, color="grey", linestyle=":", label="Default 0.5")
    ax.axvline(f1_thr, color="C0", linestyle="--", label=f"Best-F1 threshold {f1_thr:.2f}")
    ax.set(xlabel="Decision threshold on P(Fake)", ylabel="Score (TUNING half only)",
           title="Threshold sweep on the tuning half")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run(csv_path: Path, models_dir: Path, output_dir: Path, flip_labels: bool = False,
        seed: int = DEFAULT_SEED, predictor: Any | None = None) -> dict[str, Any]:
    y, p, counts = score_dataset(csv_path, models_dir, flip_labels, predictor)
    y_tune, y_test, p_tune, p_test = train_test_split(y, p, test_size=0.5, stratify=y, random_state=seed)

    rows = sweep(y_tune, p_tune)                      # tuning half ONLY
    f1_thr = best_threshold(rows, "f1")
    bal_thr = best_threshold(rows, "balanced_accuracy")

    result = {
        "evaluation": "threshold_tuning",
        "split": {"method": "stratified 50/50", "seed": seed, "tune_rows": int(len(y_tune)),
                  "test_rows": int(len(y_test)), "flip_labels_applied": bool(flip_labels), **counts},
        "chosen_on_tuning_half": {"best_f1_threshold": f1_thr, "best_balanced_accuracy_threshold": bal_thr},
        "test_half": {
            "default_0.5": metrics_at(y_test, p_test, 0.5),
            "tuned_best_f1": metrics_at(y_test, p_test, f1_thr),
            "tuned_best_balanced_accuracy": metrics_at(y_test, p_test, bal_thr),
        },
        "notes": [
            "All metrics in 'test_half' were computed on rows never used to choose the threshold.",
            "Only the decision threshold changed; the model was not retrained.",
            "The tuning and test halves are both WELFake, so this does not show generalization to a third dataset.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_sweep(rows, f1_thr, bal_thr, output_dir / "threshold_sweep.png")
    (output_dir / "threshold_tuning.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_table(result)
    return result


def _print_table(result: dict[str, Any]) -> None:
    print("\nTHRESHOLD TUNING: metrics on the held-out TEST half only")
    print(f"{'Setting':<28} {'Thr':>5} {'Acc':>8} {'BalAcc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Pred Fake %':>12}")
    print("-" * 94)
    for name, key in (("Default threshold", "default_0.5"), ("Tuned (best F1)", "tuned_best_f1"),
                      ("Tuned (best balanced acc)", "tuned_best_balanced_accuracy")):
        m = result["test_half"][key]
        print(f"{name:<28} {m['threshold']:>5.2f} {m['accuracy']:>8.4f} {m['balanced_accuracy']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['predicted_fake_pct']:>11.2f}%")
    print(f"\nThresholds were chosen on {result['split']['tune_rows']} tuning rows; "
          f"reported on {result['split']['test_rows']} test rows (seed {result['split']['seed']}).")
    print("Saved: results/threshold_tuning.json and results/threshold_sweep.png")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-csv", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--flip-labels", action="store_true")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return ap


def main(argv: list[str] | None = None, predictor: Any | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args.input_csv, args.models_dir, args.output_dir, args.flip_labels, args.seed, predictor)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
