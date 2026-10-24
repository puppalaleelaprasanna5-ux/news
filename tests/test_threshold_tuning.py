import json

import numpy as np
import pandas as pd

from scripts import threshold_tuning as tt


class ShiftedPredictor:
    """Test double: probabilities are separable but shifted upward, so 0.5 over-predicts Fake.
    Validates the tuning plumbing, not real model quality."""

    def predict_batch(self, texts):
        out = []
        for t in texts:
            n = int(t.split("#")[-1])
            fake = "fake" in t
            base = 0.62 + (n % 10) * 0.03 if fake else 0.35 + (n % 10) * 0.03   # real overlaps >0.5 sometimes
            out.append({"label_id": int(base >= 0.5), "fake_probability": round(base, 4)})
        return out


def _make_csv(path, n=200):
    rows = []
    for i in range(n):
        fake = i % 2 == 1
        rows.append({"title": f"t{i}", "text": f"this is a {'fake' if fake else 'real'} sample article #{i}",
                     "label": int(fake)})
    pd.DataFrame(rows).to_csv(path, index=False)


def _models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return d


def test_threshold_tuning_writes_outputs_and_reports_test_half_only(tmp_path):
    csv = tmp_path / "w.csv"
    _make_csv(csv)
    out = tmp_path / "results"
    code = tt.main(["--input-csv", str(csv), "--models-dir", str(_models_dir(tmp_path)),
                    "--output-dir", str(out)], predictor=ShiftedPredictor())
    assert code == 0
    res = json.loads((out / "threshold_tuning.json").read_text())
    assert (out / "threshold_sweep.png").exists()
    assert res["split"]["tune_rows"] + res["split"]["test_rows"] == 200
    assert res["split"]["seed"] == 42
    assert {"default_0.5", "tuned_best_f1", "tuned_best_balanced_accuracy"} == set(res["test_half"])
    # shifted probabilities => tuning should not make test F1 worse than the default threshold
    assert res["test_half"]["tuned_best_f1"]["f1"] >= res["test_half"]["default_0.5"]["f1"] - 1e-9
    assert res["test_half"]["tuned_best_f1"]["threshold"] > 0.5
    # every test-half confusion matrix must sum to the test size
    for m in res["test_half"].values():
        assert sum(map(sum, m["confusion_matrix"]["matrix"])) == res["split"]["test_rows"]
        assert set(m["per_class_metrics"]) == {"Real", "Fake"}
        assert "brier_score" in m["probability_calibration"]
        assert m["probability_calibration"]["calibration_status"] == "diagnostic_only_not_calibrated"


def test_tuning_and_test_halves_do_not_overlap_and_are_reproducible():
    y = np.array([0, 1] * 50)
    p = np.linspace(0, 1, 100)
    from sklearn.model_selection import train_test_split
    a = train_test_split(np.arange(100), test_size=0.5, stratify=y, random_state=42)
    b = train_test_split(np.arange(100), test_size=0.5, stratify=y, random_state=42)
    assert set(a[0]).isdisjoint(set(a[1]))
    assert (a[0] == b[0]).all()


def test_best_threshold_tie_breaks_toward_default():
    rows = [{"threshold": 0.3, "f1": 0.9, "balanced_accuracy": 0.0},
            {"threshold": 0.6, "f1": 0.9, "balanced_accuracy": 0.0}]
    assert tt.best_threshold(rows, "f1") == 0.6
