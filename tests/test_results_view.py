import json

from app.frontend import results_view as rv
from streamlit.testing.v1 import AppTest

APP_FILE = str(rv.PROJECT_ROOT / "app" / "frontend" / "app.py")

CROSS = {  # structure written by cross_dataset_eval.py; values from a real run
    "metrics": {"accuracy": 0.8474, "precision": 0.7888, "recall": 0.9561, "f1": 0.8644, "roc_auc": 0.9231},
    "fake_predictions_pct": 61.68,
    "confusion_matrix": {"matrix": [[25736, 9291], [1593, 34701]]},
    "isot_validation_from_metadata": {"metrics": {"accuracy": 0.9876, "precision": 0.9915, "recall": 0.9809,
                                                  "f1": 0.9862, "roc_auc": 0.9992}},
}


def _m(t):
    return {"threshold": t, "accuracy": 0.9, "balanced_accuracy": 0.9, "precision": 0.9, "recall": 0.9,
            "f1": 0.9, "predicted_fake_pct": 50.0}


TUNING = {"split": {"tune_rows": 10, "test_rows": 10, "seed": 42},
          "test_half": {"default_0.5": _m(0.5), "tuned_best_f1": _m(0.7), "tuned_best_balanced_accuracy": _m(0.7)}}


def test_missing_file_gives_command_hint(tmp_path):
    data, err = rv.load_result(rv.CROSS_DATASET_FILE, tmp_path)
    assert data is None and "python scripts/cross_dataset_eval.py" in err


def test_corrupt_file_is_handled(tmp_path):
    (tmp_path / rv.TUNING_FILE).write_text("{not json")
    data, err = rv.load_result(rv.TUNING_FILE, tmp_path)
    assert data is None and "python scripts/threshold_tuning.py" in err


def test_comparison_rows_and_drop():
    rows = rv.comparison_rows(CROSS)
    assert rows[0]["Accuracy"] == "98.76%" and rows[1]["Accuracy"] == "84.74%"
    assert rows[1]["Predicted Fake"] == "61.7%" and rows[0]["Predicted Fake"] == "N/A"
    assert round(rv.accuracy_drop_points(CROSS), 2) == 14.02
    assert rv.accuracy_drop_points({}) is None


def test_formatters_tolerate_bad_values():
    assert rv.pct(None) == "N/A" and rv.num("x") == "N/A"
    assert rv.tuning_rows({}) == []


def test_evaluation_tab_empty_state(tmp_path, monkeypatch):
    monkeypatch.setattr(rv, "RESULTS_DIR", tmp_path)
    at = AppTest.from_file(APP_FILE, default_timeout=60).run()
    assert not at.exception
    texts = " ".join(i.value for i in at.info)
    assert "Not generated yet" in texts


def test_evaluation_tab_renders_real_files(tmp_path, monkeypatch):
    (tmp_path / rv.CROSS_DATASET_FILE).write_text(json.dumps(CROSS))
    (tmp_path / rv.TUNING_FILE).write_text(json.dumps(TUNING))
    monkeypatch.setattr(rv, "RESULTS_DIR", tmp_path)
    at = AppTest.from_file(APP_FILE, default_timeout=60).run()
    assert not at.exception
    assert not list(at.info)                       # no "not generated" notices
    assert any("14.0" in m.value for m in at.metric)
