#!/usr/bin/env python3
"""Evaluate the saved ISOT-trained model on WELFake without retraining.

The checked-in training/inference code trains on the cleaned article BODY only:
`src/preprocessing/pipeline.py` creates `clean_text` from `text`, and
`src/models/predict.py` documents the body-only inference contract. Therefore,
this evaluation intentionally uses WELFake's `text` column only. The `title`
column is shown in the label-preview for human inspection but is not prepended
at inference time; doing so would change the model input distribution.

WELFake label conventions differ between copies. By default this script maps
raw 0 -> Real and raw 1 -> Fake, matching this repository. Inspect the printed
sample rows first; use --flip-labels if the dataset's raw labels are reversed.
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
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import load_predictor  # noqa: E402
from src.preprocessing.text_cleaning import clean_single_text  # noqa: E402

DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "WELFake_Dataset.csv"
DEFAULT_MODELS = PROJECT_ROOT / "models"
DEFAULT_OUTPUT = PROJECT_ROOT / "results"


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _find_label_column(df: pd.DataFrame) -> str:
    for candidate in ("label", "labels", "class", "target"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Could not find a label column. Expected one of: label, labels, class, target. "
        f"Found columns: {list(df.columns)}"
    )


def _preview_rows(df: pd.DataFrame, label_col: str) -> None:
    print("\nLABEL SANITY CHECK — inspect these examples before trusting the metrics.")
    print("Default mapping: raw label 0 = Real, raw label 1 = Fake.")
    print("If the examples show the opposite, rerun with --flip-labels.\n")
    display_cols = [c for c in ("title", "text", label_col) if c in df.columns]
    for raw_label in sorted(df[label_col].dropna().unique().tolist(), key=str):
        subset = df.loc[df[label_col] == raw_label, display_cols].head(5)
        print(f"--- 5 sample rows for raw label {raw_label!r} (up to 5 available) ---")
        if subset.empty:
            print("(no rows)")
            continue
        for i, (_, row) in enumerate(subset.iterrows(), start=1):
            title = str(row.get("title", ""))[:180].replace("\n", " ")
            body = str(row.get("text", ""))[:280].replace("\n", " ")
            print(f"{i}. title: {title}\n   text: {body}\n   raw_label: {row[label_col]!r}")
        print()


def _load_binary_labels(values: pd.Series, flip_labels: bool) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        bad = values[numeric.isna()].astype(str).unique()[:5].tolist()
        raise ValueError(
            "The dataset label column must contain numeric 0/1 labels. "
            f"Examples of non-numeric labels: {bad}. Verify the CSV's label column and mapping."
        )
    unique = set(numeric.astype(int).unique().tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Expected binary labels 0/1, found: {sorted(unique)}")
    labels = numeric.astype(int).to_numpy()
    return 1 - labels if flip_labels else labels


def _save_confusion_matrix(cm: np.ndarray, output_path: Path, dataset_name: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=[0, 1], yticks=[0, 1],
        xticklabels=["Real (0)", "Fake (1)"],
        yticklabels=["Real (0)", "Fake (1)"],
        ylabel="True label", xlabel="Predicted label",
        title=f"{dataset_name} confusion matrix",
    )
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for row in range(2):
        for col in range(2):
            ax.text(col, row, f"{cm[row, col]:,}", ha="center", va="center",
                    color="white" if cm[row, col] > threshold else "black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _metric_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def evaluate_dataset(
    csv_path: Path,
    models_dir: Path,
    output_dir: Path,
    flip_labels: bool = False,
    preview_only: bool = False,
    predictor: Any | None = None,
) -> dict[str, Any] | None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Place WELFake_Dataset.csv at data/raw/WELFake_Dataset.csv, or pass --input-csv PATH."
        )

    raw = _normalise_columns(pd.read_csv(csv_path))
    label_col = _find_label_column(raw)
    if "text" not in raw.columns:
        raise ValueError(f"Dataset must have a 'text' column. Found columns: {list(raw.columns)}")

    _preview_rows(raw, label_col)
    if preview_only:
        print("Preview only: no model was loaded and no metrics were computed.")
        return None

    raw_rows = len(raw)
    # Match training/inference preprocessing exactly: clean the BODY only.
    # A title-only row is not usable because the model was trained on article bodies.
    raw["_clean_body"] = raw["text"].apply(clean_single_text)
    keep_mask = raw["_clean_body"].str.len().gt(0) & raw[label_col].notna()
    dropped_empty_rows = int((~keep_mask).sum())
    df = raw.loc[keep_mask].copy().reset_index(drop=True)
    if df.empty:
        raise ValueError("No rows remain after dropping empty article bodies or missing labels.")

    y_true = _load_binary_labels(df[label_col], flip_labels)
    if predictor is None:
        predictor = load_predictor(models_dir)

    # Pass original body strings to the predictor; predict.py applies the same
    # clean_single_text function and the saved vectorizer/model without refitting.
    prediction_results = predictor.predict_batch(df["text"].tolist())
    y_pred_all: list[int] = []
    fake_scores_all: list[float | None] = []
    scorable_indices: list[int] = []
    for i, result in enumerate(prediction_results):
        label_id = result.get("label_id")
        fake_probability = result.get("fake_probability")
        if label_id not in (0, 1) or fake_probability is None:
            continue
        scorable_indices.append(i)
        y_pred_all.append(int(label_id))
        fake_scores_all.append(float(fake_probability))

    if not scorable_indices:
        raise RuntimeError(
            "The model returned no scorable predictions. Check text content and vectorizer vocabulary."
        )

    y_eval = y_true[scorable_indices]
    y_pred = np.asarray(y_pred_all, dtype=int)
    fake_scores = np.asarray(fake_scores_all, dtype=float)
    cm = confusion_matrix(y_eval, y_pred, labels=[0, 1])
    auc = None
    if len(np.unique(y_eval)) == 2:
        auc = _metric_or_none(roc_auc_score(y_eval, fake_scores))

    output_dir.mkdir(parents=True, exist_ok=True)
    cm_path = output_dir / "welfake_confusion_matrix.png"
    _save_confusion_matrix(cm, cm_path, "WELFake")

    fake_prediction_pct = float(np.mean(y_pred == 1) * 100.0)
    metrics = {
        "accuracy": _metric_or_none(accuracy_score(y_eval, y_pred)),
        "precision": _metric_or_none(precision_score(y_eval, y_pred, pos_label=1, zero_division=0)),
        "recall": _metric_or_none(recall_score(y_eval, y_pred, pos_label=1, zero_division=0)),
        "f1": _metric_or_none(f1_score(y_eval, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": auc,
    }
    metadata_path = models_dir / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    isot_metrics = metadata.get("validation_metrics", {})
    isot_rows = metadata.get("validation_set_size") or metadata.get("validation_size")

    result = {
        "evaluation": "cross_dataset",
        "model_name": metadata.get("model_name", "Unknown model"),
        "model_training_dataset": metadata.get("training_dataset", "Unknown"),
        "label_convention": {"0": "Real", "1": "Fake"},
        "dataset": {
            "name": "WELFake",
            "path": str(csv_path),
            "raw_rows": int(raw_rows),
            "dropped_empty_rows": dropped_empty_rows,
            "rows_after_drop": int(len(df)),
            "scorable_rows": int(len(scorable_indices)),
            "unscorable_rows": int(len(df) - len(scorable_indices)),
            "raw_label_column": label_col,
            "flip_labels_applied": bool(flip_labels),
            "text_input_policy": "body_only_to_match_training_and_src.models.predict",
            "raw_label_mapping": ({"0": "Fake", "1": "Real"} if flip_labels else {"0": "Real", "1": "Fake"}),
        },
        "metrics": metrics,
        "fake_predictions_pct": fake_prediction_pct,
        "true_class_counts": {
            "real_0": int(np.sum(y_eval == 0)),
            "fake_1": int(np.sum(y_eval == 1)),
        },
        "predicted_class_counts": {
            "real_0": int(np.sum(y_pred == 0)),
            "fake_1": int(np.sum(y_pred == 1)),
        },
        "confusion_matrix": {
            "labels": ["Real (0)", "Fake (1)"],
            "matrix": cm.astype(int).tolist(),
            "png": str(cm_path),
        },
        "isot_validation_from_metadata": {
            "dataset": metadata.get("training_dataset", "ISOT validation"),
            "rows": isot_rows,
            "metrics": {k: _metric_or_none(v) for k, v in isot_metrics.items()},
            "fake_predictions_pct": None,
            "note": "model_metadata.json stores aggregate validation metrics only; it does not contain validation predictions or a confusion matrix.",
        },
        "notes": [
            "This evaluates cross-dataset generalization; WELFake results are not directly comparable to the ISOT validation set because dataset composition and collection artifacts differ.",
            "Model was trained on full article bodies, so short or empty text inputs may be unscorable or less representative.",
            "Check the printed WELFake sample rows to confirm raw label semantics. Use --flip-labels if raw 0 means Fake and raw 1 means Real.",
        ],
    }
    result_path = output_dir / "cross_dataset.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nCROSS-DATASET COMPARISON (Fake is positive class = 1)")
    print(f"{'Dataset':<24} {'Rows':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'Pred Fake %':>12}")
    print("-" * 108)
    _print_metric_row("ISOT validation", isot_rows, isot_metrics, None)
    _print_metric_row("WELFake", len(scorable_indices), metrics, fake_prediction_pct)
    print(f"\nDropped rows: {dropped_empty_rows} of {raw_rows} (empty body or missing label).")
    print(f"Unscorable model outputs: {len(df) - len(scorable_indices)}.")
    print(f"Saved JSON: {result_path}")
    print(f"Saved confusion matrix: {cm_path}")
    if flip_labels:
        print("Label mapping flip applied: raw 0 -> Fake (1), raw 1 -> Real (0).")
    else:
        print("Label mapping used: raw 0 -> Real (0), raw 1 -> Fake (1). Verify this against the sample rows.")
    return result


def _print_metric_row(name: str, rows: Any, metrics: dict[str, Any], fake_pct: float | None) -> None:
    def fmt(value: Any) -> str:
        return "N/A" if value is None else f"{float(value):.4f}"
    row_count = "N/A" if rows is None else str(rows)
    fake_text = "N/A" if fake_pct is None else f"{fake_pct:.2f}%"
    print(
        f"{name:<24} {row_count:>8} {fmt(metrics.get('accuracy')):>10} "
        f"{fmt(metrics.get('precision')):>10} {fmt(metrics.get('recall')):>10} "
        f"{fmt(metrics.get('f1')):>10} {fmt(metrics.get('roc_auc')):>10} {fake_text:>12}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_DATASET,
                        help=f"WELFake CSV path (default: {DEFAULT_DATASET})")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS,
                        help="Directory containing best_model.joblib, tfidf_vectorizer.joblib, and model_metadata.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--flip-labels", action="store_true",
                        help="Flip binary source labels: raw 0 becomes Fake and raw 1 becomes Real")
    parser.add_argument("--preview-only", action="store_true",
                        help="Print 5 sample rows per raw label, then exit without evaluating")
    return parser


def main(argv: list[str] | None = None, predictor: Any | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        evaluate_dataset(
            csv_path=args.input_csv,
            models_dir=args.models_dir,
            output_dir=args.output_dir,
            flip_labels=args.flip_labels,
            preview_only=args.preview_only,
            predictor=predictor,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
