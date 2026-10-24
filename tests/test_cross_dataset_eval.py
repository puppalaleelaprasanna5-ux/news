import json
from pathlib import Path

import pandas as pd

from scripts import cross_dataset_eval as evaluation


class TinyFakePredictor:
    """Deterministic test double; this test validates evaluation plumbing, not model quality."""

    def predict_batch(self, texts):
        results = []
        for text in texts:
            is_fake = "fabricated" in text.lower() or "miracle" in text.lower()
            results.append({
                "label_id": 1 if is_fake else 0,
                "fake_probability": 0.9 if is_fake else 0.1,
            })
        return results


def test_cross_dataset_script_tiny_csv_writes_expected_json_keys(tmp_path):
    csv_path = tmp_path / "tiny_welfake.csv"
    rows = []
    for i in range(10):
        fake = i % 2 == 1
        rows.append({
            "title": f"Sample headline {i}",
            "text": (
                f"This fabricated miracle report contains enough sample words for evaluation number {i}."
                if fake else
                f"This verified public report contains enough sample words for evaluation number {i}."
            ),
            "label": 1 if fake else 0,
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "model_metadata.json").write_text(json.dumps({
        "model_name": "Test model",
        "training_dataset": "ISOT Fake and Real News Dataset",
        "validation_metrics": {
            "accuracy": 0.9876, "precision": 0.9915, "recall": 0.9809,
            "f1": 0.9862, "roc_auc": 0.9992,
        },
    }), encoding="utf-8")
    output_dir = tmp_path / "results"

    exit_code = evaluation.main([
        "--input-csv", str(csv_path),
        "--models-dir", str(models_dir),
        "--output-dir", str(output_dir),
    ], predictor=TinyFakePredictor())
    assert exit_code == 0

    saved = json.loads((output_dir / "cross_dataset.json").read_text(encoding="utf-8"))
    required_top_level = {
        "evaluation", "model_name", "model_training_dataset", "label_convention",
        "dataset", "metrics", "fake_predictions_pct", "true_class_counts",
        "predicted_class_counts", "confusion_matrix", "isot_validation_from_metadata", "notes",
    }
    assert required_top_level.issubset(saved.keys())
    assert {"accuracy", "precision", "recall", "f1", "roc_auc"}.issubset(saved["metrics"].keys())
    assert saved["dataset"]["raw_rows"] == 10
    assert saved["dataset"]["dropped_empty_rows"] == 0
    assert saved["dataset"]["scorable_rows"] == 10
    assert len(saved["confusion_matrix"]["matrix"]) == 2
    assert (output_dir / "welfake_confusion_matrix.png").exists()
