import json

import numpy as np
import pandas as pd
import pytest

from scripts import threshold_tuning as tt


class ShiftedPredictor:
    """Test double: probabilities are separable but shifted upward, so 0.5 over-predicts Fake.
    Validates the tuning plumbing, not real model quality."""

    def predict_batch(self, texts):
        out = []
        for t in texts:
            n = int(t.split("#", 1)[1].split()[0])
            fake = "fake" in t
            base = 0.62 + (n % 10) * 0.03 if fake else 0.35 + (n % 10) * 0.03   # real overlaps >0.5 sometimes
            out.append({"label_id": int(base >= 0.5), "fake_probability": round(base, 4)})
        return out


def _article(fake, n):
    label = "fake" if fake else "real"
    unique_words = " ".join(f"token{n}_{i}" for i in range(24))
    return f"this is a {label} sample article #{n} with distinct details {unique_words}"


def _make_csv(path, n=200):
    rows = []
    for i in range(n):
        fake = i % 2 == 1
        rows.append({"title": f"t{i}", "text": _article(fake, i), "label": int(fake)})
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
    assert res["split"]["method"] == "stratified grouped 50/50"
    assert res["split"]["grouping"].startswith("exact normalized body")
    assert res["split"]["near_duplicate_groups"] == 0
    assert res["split"]["tune_groups"] + res["split"]["test_groups"] == 200
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


def test_exact_duplicates_stay_in_one_group():
    text = "same article body with sufficient repeated words for grouping"
    groups = tt._duplicate_group_ids([text, text, "different article body"], np.array([0, 0, 1]))
    assert groups[0] == groups[1]
    assert groups[0] != groups[2]


def test_near_duplicates_stay_in_one_group():
    words = [f"storyword{i}" for i in range(80)]
    original = " ".join(words)
    changed = " ".join(words[:-3] + ["replacementalpha", "replacementbeta", "replacementgamma"])
    groups = tt._duplicate_group_ids([original, changed], np.array([1, 1]))
    assert groups[0] == groups[1]


def test_conflicting_exact_duplicate_labels_raise_without_dropping_rows():
    text = "same article body with sufficient repeated words for grouping"
    with pytest.raises(ValueError, match="conflicting labels") as error:
        tt._duplicate_group_ids([text, text], np.array([0, 1]), source_rows=[4, 17])
    assert "No conflicting examples were dropped" in str(error.value)
    assert "[[4, 17]]" in str(error.value)


def test_conflicting_near_duplicate_labels_raise_without_dropping_rows():
    words = [f"storyword{i}" for i in range(80)]
    original = " ".join(words)
    changed = " ".join(words[:-3] + ["replacementalpha", "replacementbeta", "replacementgamma"])
    with pytest.raises(ValueError, match="conflicting labels"):
        tt._duplicate_group_ids([original, changed], np.array([0, 1]))


def test_group_split_is_deterministic_and_keeps_groups_disjoint():
    records = []
    for label in (0, 1):
        for group in range(20):
            group_id = f"{label}-{group}"
            records.append({"source_row": len(records), "text": group_id, "label": label,
                            "fake_probability": label * 0.6 + group / 1000, "group_id": group_id})
            if group % 3 == 0:
                records.append({"source_row": len(records), "text": group_id + " duplicate", "label": label,
                                "fake_probability": label * 0.6 + group / 1000, "group_id": group_id})
    rows = pd.DataFrame(records)

    tune_a, test_a = tt._split_grouped_rows(rows, seed=42)
    tune_b, test_b = tt._split_grouped_rows(rows, seed=42)

    assert tune_a["source_row"].tolist() == tune_b["source_row"].tolist()
    assert test_a["source_row"].tolist() == test_b["source_row"].tolist()
    assert set(tune_a["group_id"]).isdisjoint(test_a["group_id"])
    assert set(tune_a["label"].unique()) == set(test_a["label"].unique()) == {0, 1}


def test_run_uses_tuning_rows_for_thresholds_and_test_rows_for_metrics(tmp_path, monkeypatch):
    csv_path = tmp_path / "grouped.csv"
    rows = [
        {"text": _article(bool(index % 2), index), "label": index % 2}
        for index in range(40)
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    predictor = ShiftedPredictor()
    models_dir = _models_dir(tmp_path)
    scored, _ = tt.score_dataset(csv_path, models_dir, False, predictor)
    expected_tune, expected_test = tt._split_grouped_rows(scored, seed=42)

    observed = {"sweep": None, "metrics": []}

    def capture_sweep(y_true, probabilities):
        observed["sweep"] = (y_true.copy(), probabilities.copy())
        return [{"threshold": 0.5, "f1": 0.5, "balanced_accuracy": 0.5}]

    def capture_metrics(y_true, probabilities, threshold):
        observed["metrics"].append((y_true.copy(), probabilities.copy()))
        return {"threshold": threshold}

    monkeypatch.setattr(tt, "sweep", capture_sweep)
    monkeypatch.setattr(tt, "metrics_at", capture_metrics)
    monkeypatch.setattr(tt, "_plot_sweep", lambda *args: None)
    monkeypatch.setattr(tt, "_print_table", lambda *args: None)

    result = tt.run(csv_path, models_dir, tmp_path / "output", predictor=predictor)

    tune_y, tune_p = observed["sweep"]
    np.testing.assert_array_equal(tune_y, expected_tune["label"].to_numpy())
    np.testing.assert_array_equal(tune_p, expected_tune["fake_probability"].to_numpy())
    assert len(observed["metrics"]) == 3
    for test_y, test_p in observed["metrics"]:
        np.testing.assert_array_equal(test_y, expected_test["label"].to_numpy())
        np.testing.assert_array_equal(test_p, expected_test["fake_probability"].to_numpy())
    assert result["split"]["tune_groups"] == expected_tune["group_id"].nunique()
    assert result["split"]["test_groups"] == expected_test["group_id"].nunique()


def test_score_dataset_preserves_source_text_label_score_alignment(tmp_path):
    class AlignedPredictor:
        def predict_batch(self, texts):
            return [
                ({"label_id": None, "fake_probability": None} if "unscorable" in text else
                 {"label_id": 1 if "fake" in text else 0,
                  "fake_probability": 0.75 if "fake" in text else 0.25})
                for text in texts
            ]

    csv_path = tmp_path / "aligned.csv"
    duplicate = _article(True, 10)
    real = _article(False, 11)
    pd.DataFrame([
        {"text": duplicate, "label": 1},
        {"text": "   ", "label": 0},
        {"text": duplicate, "label": 1},
        {"text": _article(False, 12) + " unscorable", "label": 0},
        {"text": real, "label": 0},
    ]).to_csv(csv_path, index=False)

    rows, counts = tt.score_dataset(csv_path, _models_dir(tmp_path), False, AlignedPredictor())

    assert rows["source_row"].tolist() == [0, 2, 4]
    assert rows["text"].tolist() == [duplicate, duplicate, real]
    assert rows["label"].tolist() == [1, 1, 0]
    assert rows["fake_probability"].tolist() == [0.75, 0.75, 0.25]
    assert rows.iloc[0]["group_id"] == rows.iloc[1]["group_id"]
    assert counts["dropped_empty_rows"] == 1
    assert counts["unscorable_rows"] == 1


def test_group_split_requires_both_classes_in_multiple_groups():
    rows = pd.DataFrame([
        {"source_row": 0, "text": "a", "label": 0, "fake_probability": 0.1, "group_id": "real"},
        {"source_row": 1, "text": "b", "label": 1, "fake_probability": 0.9, "group_id": "fake"},
    ])
    with pytest.raises(ValueError, match="at least two distinct groups per class"):
        tt._split_grouped_rows(rows)


def test_best_threshold_tie_breaks_toward_default():
    rows = [{"threshold": 0.3, "f1": 0.9, "balanced_accuracy": 0.0},
            {"threshold": 0.6, "f1": 0.9, "balanced_accuracy": 0.0}]
    assert tt.best_threshold(rows, "f1") == 0.6
