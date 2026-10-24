#!/usr/bin/env python3
"""Threshold tuning for the saved ISOT model on WELFake, with a grouped tune/test split.

Why: on WELFake the model keeps good ranking (ROC-AUC ~0.92) but its fixed 0.5 cutoff
over-predicts Fake. This script measures how much of the accuracy drop is just a
miscalibrated threshold, WITHOUT retraining.

Protocol (duplicate-safe):
  1. Group exact duplicates and conservative near-duplicates, then make a deterministic
     stratified 50/50 group split of WELFake (fixed seed, saved in the JSON).
  2. The threshold is chosen on the TUNING half only.
  3. Every reported metric comes from the held-out TEST groups only, at 0.5 and at the
     tuned threshold.

Near-duplicate candidates use the project's word-trigram SimHash fingerprint and Hamming
distance <= 6. Candidate pairs are transitively grouped to prevent related rows crossing
the split. This is a conservative heuristic, not semantic duplicate verification.

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
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.cross_dataset_eval import (  # noqa: E402
    DEFAULT_DATASET, DEFAULT_MODELS, DEFAULT_OUTPUT,
    _calibration_diagnostics, _find_label_column, _load_binary_labels,
    _normalise_columns, _per_class_metrics,
)
from src.models.predict import load_predictor  # noqa: E402
from src.preprocessing.split_data import SIMHASH_BITS, _simhash  # noqa: E402
from src.preprocessing.text_cleaning import clean_single_text  # noqa: E402

DEFAULT_SEED = 42
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)
NEAR_DUPLICATE_MAX_HAMMING_DISTANCE = 6
_SIMHASH_BAND_WIDTHS = (10, 9, 9, 9, 9, 9, 9)
_MIN_WORDS_FOR_NEAR_DUPLICATE = 12


def _duplicate_group_ids(
    texts: list[str],
    labels: np.ndarray,
    source_rows: list[int] | None = None,
    allow_conflicting_labels: bool = False,
) -> list[str]:
    """Return deterministic connected-component IDs for exact/near-duplicate bodies."""
    if len(texts) != len(labels):
        raise ValueError("Texts and labels must have the same number of rows.")
    if source_rows is None:
        source_rows = list(range(len(texts)))
    if len(source_rows) != len(texts):
        raise ValueError("Source row identities must align with texts and labels.")

    parent = list(range(len(texts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    normalized = [" ".join(str(text).casefold().split()) for text in texts]
    exact_rows: dict[str, int] = {}
    for index, text in enumerate(normalized):
        if text in exact_rows:
            union(index, exact_rows[text])
        else:
            exact_rows[text] = index

    # Seven disjoint bands guarantee that fingerprints within Hamming distance 6 share
    # at least one band; exact hamming verification removes band-collision false positives.
    fingerprints: dict[int, int] = {}
    offset = 0
    for band, width in enumerate(_SIMHASH_BAND_WIDTHS):
        if width <= 0 or offset + width > SIMHASH_BITS:
            raise RuntimeError("Invalid SimHash band configuration.")
        mask = (1 << width) - 1
        candidates_by_band: dict[int, list[tuple[int, int]]] = {}
        for index, text in enumerate(normalized):
            if len(text.split()) < _MIN_WORDS_FOR_NEAR_DUPLICATE:
                continue
            if index not in fingerprints:
                fingerprints[index] = _simhash(text)
            fingerprint = fingerprints[index]
            value = (fingerprint >> offset) & mask
            candidates_by_band.setdefault(value, []).append((index, fingerprint))

        for same_band in candidates_by_band.values():
            for position, (index, fingerprint) in enumerate(same_band):
                for other_index, other_fingerprint in same_band[:position]:
                    if (fingerprint ^ other_fingerprint).bit_count() <= NEAR_DUPLICATE_MAX_HAMMING_DISTANCE:
                        union(index, other_index)
        offset += width
    if offset != SIMHASH_BITS:
        raise RuntimeError("SimHash bands must cover the fingerprint.")

    components: dict[int, list[int]] = {}
    for index in range(len(texts)):
        components.setdefault(find(index), []).append(index)

    conflicting = [
        rows for rows in components.values()
        if len({int(labels[index]) for index in rows}) > 1
    ]
    if conflicting and not allow_conflicting_labels:
        examples = [[rows[0], rows[-1]] for rows in conflicting[:3]]
        source_examples = [[source_rows[row] for row in example] for example in examples]
        raise ValueError(
            f"Found {len(conflicting)} exact/near-duplicate group(s) with conflicting labels; "
            f"review source CSV row positions (zero-based): {source_examples}. "
            "No conflicting examples were dropped."
        )

    return [f"group-{min(components[find(index)])}" for index in range(len(texts))]


def _conflicting_group_audit(
    rows: pd.DataFrame,
    labels: np.ndarray,
    group_ids: list[str],
) -> list[dict[str, Any]]:
    """Describe mixed-label groups without changing or resolving their source labels."""
    members: dict[str, list[dict[str, int]]] = {}
    for source_row, label, group_id in zip(rows.index, labels, group_ids):
        members.setdefault(group_id, []).append(
            {"source_row": int(source_row), "label": int(label)}
        )
    return [
        {"group_id": group_id, "rows": group_rows}
        for group_id, group_rows in sorted(members.items())
        if len({row["label"] for row in group_rows}) > 1
    ]


def _split_grouped_rows(rows: pd.DataFrame, seed: int = DEFAULT_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows reproducibly while keeping each duplicate group wholly in one half."""
    labels = rows["label"].to_numpy(dtype=int)
    groups = rows["group_id"].to_numpy()
    if len(set(groups)) < 2 or any(rows.loc[labels == label, "group_id"].nunique() < 2 for label in (0, 1)):
        raise ValueError("Grouped threshold split requires at least two distinct groups per class.")

    splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    candidates = []
    for tune_idx, test_idx in splitter.split(np.zeros(len(rows)), labels, groups):
        tune_labels, test_labels = labels[tune_idx], labels[test_idx]
        if set(np.unique(tune_labels)) != {0, 1} or set(np.unique(test_labels)) != {0, 1}:
            continue
        test_prevalence = float(test_labels.mean())
        full_prevalence = float(labels.mean())
        balance_error = abs(test_prevalence - full_prevalence)
        size_error = abs(len(test_idx) / len(rows) - 0.5)
        candidates.append((balance_error + size_error, tune_idx, test_idx))
    if not candidates:
        raise ValueError(
            "Could not create a grouped split with both classes represented in tuning and test; "
            "inspect duplicate groups and label distribution."
        )
    _, tune_idx, test_idx = min(candidates, key=lambda candidate: candidate[0])
    tuning, test = rows.iloc[tune_idx].copy(), rows.iloc[test_idx].copy()
    assert set(tuning["group_id"]).isdisjoint(test["group_id"]), (
        "Duplicate group crossed the tuning/test split."
    )
    return tuning, test


def score_dataset(csv_path: Path, models_dir: Path, flip_labels: bool, predictor: Any | None = None):
    """Return aligned source-row, body, label, score and duplicate-group records."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    raw = _normalise_columns(pd.read_csv(csv_path))
    label_col = _find_label_column(raw)
    if "text" not in raw.columns:
        raise ValueError(f"Dataset must have a 'text' column. Found: {list(raw.columns)}")
    raw["_clean"] = raw["text"].apply(clean_single_text)
    keep = raw["_clean"].str.len().gt(0) & raw[label_col].notna()
    df = raw.loc[keep].copy()
    if df.empty:
        raise ValueError("No usable rows after dropping empty bodies / missing labels.")
    y_true = _load_binary_labels(df[label_col], flip_labels)
    clean_texts = df["_clean"].tolist()
    source_rows = [int(row_index) for row_index in df.index]
    group_ids = _duplicate_group_ids(
        clean_texts, y_true, source_rows, allow_conflicting_labels=True
    )
    conflict_exclusions = _conflicting_group_audit(df, y_true, group_ids)
    conflicting_group_ids = {group["group_id"] for group in conflict_exclusions}
    conflict_mask = pd.Series(group_ids, index=df.index).isin(conflicting_group_ids)
    eligible_df = df.loc[~conflict_mask].copy()
    eligible_labels = y_true[~conflict_mask.to_numpy()]
    eligible_group_ids = [
        group_id for group_id in group_ids
        if group_id not in conflicting_group_ids
    ]
    predictor = predictor or load_predictor(models_dir)
    results = predictor.predict_batch(eligible_df["text"].tolist())
    if len(results) != len(eligible_df):
        raise RuntimeError(
            f"The model returned {len(results)} predictions for {len(eligible_df)} eligible rows; "
            "cannot preserve article/label/score alignment."
        )
    records = []
    for i, r in enumerate(results):
        p = r.get("fake_probability")
        if r.get("label_id") in (0, 1) and p is not None:
            if not np.isfinite(float(p)) or not 0.0 <= float(p) <= 1.0:
                continue
            records.append({
                "source_row": int(eligible_df.index[i]),
                "text": str(eligible_df.iloc[i]["text"]),
                "label": int(eligible_labels[i]),
                "fake_probability": float(p),
                "group_id": eligible_group_ids[i],
            })
    if not records:
        raise RuntimeError("The model returned no scorable predictions.")
    rows = pd.DataFrame.from_records(records)
    grouped_usable = df.assign(
        group_id=group_ids,
        normalized_text=[" ".join(str(text).casefold().split()) for text in clean_texts],
    ).groupby("group_id")
    grouped_eligible = eligible_df.assign(group_id=eligible_group_ids).groupby("group_id")
    counts = {
        "raw_rows": len(raw),
        "dropped_empty_rows": int((~keep).sum()),
        "filtered_rows": int(len(df)),
        "conflict_excluded_rows": int(conflict_mask.sum()),
        "eligible_rows": int(len(eligible_df)),
        "scorable_rows": len(rows),
        "model_scorable_rows": int(len(rows)),
        "unscorable_rows": int(len(eligible_df) - len(rows)),
        "duplicate_groups": int(grouped_usable.ngroups),
        "conflict_excluded_groups": len(conflict_exclusions),
        "eligible_groups": int(grouped_eligible.ngroups),
        "scorable_groups": int(rows["group_id"].nunique()),
        "near_duplicate_groups": int(sum(
            group["normalized_text"].nunique() > 1 for _, group in grouped_usable
        )),
        "conflict_exclusions": conflict_exclusions,
    }
    return rows, counts


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
        "per_class_metrics": _per_class_metrics(y_true, y_pred),
        "probability_calibration": _calibration_diagnostics(y_true, probs),
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
    rows, counts = score_dataset(csv_path, models_dir, flip_labels, predictor)
    tune_rows, test_rows = _split_grouped_rows(rows, seed)
    y_tune, p_tune = tune_rows["label"].to_numpy(), tune_rows["fake_probability"].to_numpy()
    y_test, p_test = test_rows["label"].to_numpy(), test_rows["fake_probability"].to_numpy()

    rows = sweep(y_tune, p_tune)                      # tuning half ONLY
    f1_thr = best_threshold(rows, "f1")
    bal_thr = best_threshold(rows, "balanced_accuracy")

    result = {
        "evaluation": "threshold_tuning",
        "split": {
            "method": "stratified grouped 50/50",
            "seed": seed,
            "grouping": "exact normalized body plus word-trigram SimHash, Hamming distance <= 6",
            "tune_rows": int(len(y_tune)),
            "test_rows": int(len(y_test)),
            "tune_groups": int(tune_rows["group_id"].nunique()),
            "test_groups": int(test_rows["group_id"].nunique()),
            "flip_labels_applied": bool(flip_labels),
            "tune_test_groups_disjoint": set(tune_rows["group_id"]).isdisjoint(test_rows["group_id"]),
            **counts,
        },
        "chosen_on_tuning_half": {"best_f1_threshold": f1_thr, "best_balanced_accuracy_threshold": bal_thr},
        "test_half": {
            "default_0.5": metrics_at(y_test, p_test, 0.5),
            "tuned_best_f1": metrics_at(y_test, p_test, f1_thr),
            "tuned_best_balanced_accuracy": metrics_at(y_test, p_test, bal_thr),
        },
        "notes": [
            "All metrics in 'test_half' were computed on duplicate groups never used to choose the threshold.",
            "Near-duplicate grouping is a conservative SimHash heuristic; review its method and limitations before interpreting results.",
            "All rows in conflicting-label duplicate groups are excluded from both threshold halves and recorded in conflict_exclusions; their labels are unresolved, not presumed incorrect.",
            "Only the decision threshold changed; the model was not retrained.",
            "The tuning and test halves are both WELFake, so this does not show generalization to a third dataset.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_sweep(rows, f1_thr, bal_thr, output_dir / "threshold_sweep.png")
    (output_dir / "threshold_tuning.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _print_table(result, output_dir)
    return result


def _print_table(result: dict[str, Any], output_dir: Path) -> None:
    print("\nTHRESHOLD TUNING: metrics on the held-out TEST half only")
    print(f"{'Setting':<28} {'Thr':>5} {'Acc':>8} {'BalAcc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Pred Fake %':>12}")
    print("-" * 94)
    for name, key in (("Default threshold", "default_0.5"), ("Tuned (best F1)", "tuned_best_f1"),
                      ("Tuned (best balanced acc)", "tuned_best_balanced_accuracy")):
        m = result["test_half"][key]
        print(f"{name:<28} {m['threshold']:>5.2f} {m['accuracy']:>8.4f} {m['balanced_accuracy']:>8.4f} "
              f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} {m['predicted_fake_pct']:>11.2f}%")
    print(f"\nThresholds were chosen on {result['split']['tune_rows']} tuning rows "
          f"({result['split']['tune_groups']} groups); reported on {result['split']['test_rows']} "
          f"held-out rows ({result['split']['test_groups']} groups, seed {result['split']['seed']}).")
    print(f"Saved: {output_dir / 'threshold_tuning.json'}")
    print(f"Saved: {output_dir / 'threshold_sweep.png'}")


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
