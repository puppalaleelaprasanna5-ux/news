"""
Tests for Phase 4 ML modules (features, training, evaluation, artifact I/O).

IMPORTANT: these tests use small, hand-written FIXTURE text data, NOT the
real 27k-article training set. Their purpose is to verify the ML plumbing
(vectorizer fit/transform discipline, model training shapes, metric
computation, artifact save/load round-tripping) works correctly and fast -
not to reproduce or validate the real model's performance. See
notebooks/03_ml_models.ipynb for evaluation against the actual dataset.

Run with:
    pytest tests/test_ml_models.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.tfidf_features import (
    build_tfidf_vectorizer,
    fit_tfidf_on_train,
    transform_with_tfidf,
    fit_svd_on_train,
    transform_with_svd,
)
from src.models.train_models import (
    train_logistic_regression,
    train_naive_bayes,
    train_linear_svm,
    train_random_forest,
    train_xgboost,
)
from src.models.evaluate_models import (
    compute_metrics,
    get_confusion_matrix,
    get_classification_report,
    get_positive_class_scores,
)
from src.models.model_utils import save_artifact, load_artifact, save_metadata, load_metadata, LABEL_NAMES


# --- Small fixture "corpus" (NOT the real dataset — see module docstring) ---
# Deliberately simple and separable so tiny models can actually learn a pattern
# in a handful of training examples.
FIXTURE_TRAIN_TEXTS = [
    "the government announced a new policy today reuters washington",
    "president said the economy grew this quarter reuters",
    "officials confirmed the election results reuters london",
    "the senate passed the bill on tuesday reuters",
    "shocking secret doctors dont want you to know click here",
    "you wont believe what happened next viral video watch now",
    "breaking outrageous conspiracy exposed share this everywhere",
    "unbelievable miracle cure big pharma hiding from you",
] * 4  # repeat to give TF-IDF/SVD enough rows to work with
FIXTURE_TRAIN_LABELS = np.array(([0, 0, 0, 0, 1, 1, 1, 1] * 4))

FIXTURE_VAL_TEXTS = [
    "reuters reports the government passed new legislation washington",
    "shocking viral secret you wont believe doctors hiding",
]
FIXTURE_VAL_LABELS = np.array([0, 1])


@pytest.fixture
def fitted_tfidf():
    vectorizer = build_tfidf_vectorizer(max_features=200, min_df=1, max_df=1.0)
    X_train = fit_tfidf_on_train(vectorizer, FIXTURE_TRAIN_TEXTS)
    return vectorizer, X_train


# --- TF-IDF feature tests ---------------------------------------------------

def test_tfidf_vectorizer_fits_on_small_fixture(fitted_tfidf):
    vectorizer, X_train = fitted_tfidf
    assert X_train.shape[0] == len(FIXTURE_TRAIN_TEXTS)
    assert X_train.shape[1] > 0
    assert len(vectorizer.vocabulary_) > 0


def test_tfidf_transform_does_not_refit(fitted_tfidf):
    vectorizer, X_train = fitted_tfidf
    vocab_before = dict(vectorizer.vocabulary_)

    X_val = transform_with_tfidf(vectorizer, FIXTURE_VAL_TEXTS)

    # Vocabulary must be UNCHANGED after transforming validation data.
    assert vectorizer.vocabulary_ == vocab_before
    # Validation matrix must have the same number of columns as training.
    assert X_val.shape[1] == X_train.shape[1]
    assert X_val.shape[0] == len(FIXTURE_VAL_TEXTS)


def test_tfidf_transform_before_fit_raises():
    from sklearn.exceptions import NotFittedError
    vectorizer = build_tfidf_vectorizer()
    with pytest.raises(NotFittedError):
        transform_with_tfidf(vectorizer, FIXTURE_VAL_TEXTS)


def test_svd_fit_and_transform_shapes(fitted_tfidf):
    vectorizer, X_train = fitted_tfidf
    n_components = 5  # small, since the fixture vocabulary itself is small
    svd = fit_svd_on_train(X_train, n_components=n_components)
    X_train_svd = transform_with_svd(svd, X_train)

    assert X_train_svd.shape == (X_train.shape[0], n_components)

    X_val = transform_with_tfidf(vectorizer, FIXTURE_VAL_TEXTS)
    X_val_svd = transform_with_svd(svd, X_val)
    assert X_val_svd.shape == (len(FIXTURE_VAL_TEXTS), n_components)


# --- Model training tests ---------------------------------------------------

def test_logistic_regression_trains_and_predicts(fitted_tfidf):
    vectorizer, X_train = fitted_tfidf
    model, elapsed = train_logistic_regression(X_train, FIXTURE_TRAIN_LABELS, max_iter=200)

    assert elapsed >= 0
    X_val = transform_with_tfidf(vectorizer, FIXTURE_VAL_TEXTS)
    preds = model.predict(X_val)
    assert preds.shape == (len(FIXTURE_VAL_TEXTS),)
    assert set(preds).issubset({0, 1})


def test_naive_bayes_trains_and_predicts(fitted_tfidf):
    vectorizer, X_train = fitted_tfidf
    model, elapsed = train_naive_bayes(X_train, FIXTURE_TRAIN_LABELS)

    X_val = transform_with_tfidf(vectorizer, FIXTURE_VAL_TEXTS)
    preds = model.predict(X_val)
    assert preds.shape == (len(FIXTURE_VAL_TEXTS),)


def test_linear_svm_trains_and_predicts(fitted_tfidf):
    vectorizer, X_train = fitted_tfidf
    model, elapsed = train_linear_svm(X_train, FIXTURE_TRAIN_LABELS, max_iter=2000)

    X_val = transform_with_tfidf(vectorizer, FIXTURE_VAL_TEXTS)
    preds = model.predict(X_val)
    assert preds.shape == (len(FIXTURE_VAL_TEXTS),)


def test_random_forest_trains_and_predicts(fitted_tfidf):
    vectorizer, X_train = fitted_tfidf
    svd = fit_svd_on_train(X_train, n_components=5)
    X_train_svd = transform_with_svd(svd, X_train)

    model, elapsed = train_random_forest(X_train_svd, FIXTURE_TRAIN_LABELS, n_estimators=10, max_depth=3)

    X_val_svd = transform_with_svd(svd, transform_with_tfidf(vectorizer, FIXTURE_VAL_TEXTS))
    preds = model.predict(X_val_svd)
    assert preds.shape == (len(FIXTURE_VAL_TEXTS),)


def test_xgboost_trains_and_predicts_or_skips_gracefully(fitted_tfidf):
    vectorizer, X_train = fitted_tfidf
    svd = fit_svd_on_train(X_train, n_components=5)
    X_train_svd = transform_with_svd(svd, X_train)

    model, elapsed = train_xgboost(X_train_svd, FIXTURE_TRAIN_LABELS, n_estimators=10, max_depth=3)

    if model is None:
        pytest.skip("xgboost not installed in this environment - train_xgboost degraded gracefully.")

    X_val_svd = transform_with_svd(svd, transform_with_tfidf(vectorizer, FIXTURE_VAL_TEXTS))
    preds = model.predict(X_val_svd)
    assert preds.shape == (len(FIXTURE_VAL_TEXTS),)


# --- Evaluation tests --------------------------------------------------------

def test_compute_metrics_returns_expected_keys():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_score = np.array([0.1, 0.6, 0.9, 0.8])

    metrics = compute_metrics(y_true, y_pred, y_score)
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_compute_metrics_without_score_gives_nan_auc():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    metrics = compute_metrics(y_true, y_pred, y_score=None)
    assert np.isnan(metrics["roc_auc"])


def test_confusion_matrix_has_fixed_label_order():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    cm = get_confusion_matrix(y_true, y_pred)
    # labels=[0, 1] -> rows/cols ordered [Real, Fake]
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 1  # true Real predicted Real
    assert cm[0, 1] == 1  # true Real predicted Fake (false positive)
    assert cm[1, 1] == 2  # true Fake predicted Fake


def test_classification_report_uses_fake_real_names():
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])
    report = get_classification_report(y_true, y_pred)
    assert "Fake" in report
    assert "Real" in report


def test_get_positive_class_scores_handles_predict_proba(fitted_tfidf):
    vectorizer, X_train = fitted_tfidf
    model, _ = train_logistic_regression(X_train, FIXTURE_TRAIN_LABELS, max_iter=200)
    X_val = transform_with_tfidf(vectorizer, FIXTURE_VAL_TEXTS)

    scores = get_positive_class_scores(model, X_val)
    assert scores.shape == (len(FIXTURE_VAL_TEXTS),)
    assert np.all((scores >= 0) & (scores <= 1))  # predict_proba output


def test_get_positive_class_scores_handles_decision_function(fitted_tfidf):
    vectorizer, X_train = fitted_tfidf
    model, _ = train_linear_svm(X_train, FIXTURE_TRAIN_LABELS, max_iter=2000)
    X_val = transform_with_tfidf(vectorizer, FIXTURE_VAL_TEXTS)

    scores = get_positive_class_scores(model, X_val)
    assert scores.shape == (len(FIXTURE_VAL_TEXTS),)  # decision_function output, unbounded


# --- Artifact save/load tests -------------------------------------------------

def test_save_and_load_artifact_roundtrip(tmp_path, fitted_tfidf):
    vectorizer, X_train = fitted_tfidf
    model, _ = train_logistic_regression(X_train, FIXTURE_TRAIN_LABELS, max_iter=200)

    model_path = tmp_path / "model.joblib"
    save_artifact(model, model_path)
    assert model_path.exists()

    loaded_model = load_artifact(model_path)
    X_val = transform_with_tfidf(vectorizer, FIXTURE_VAL_TEXTS)

    # Loaded model must produce IDENTICAL predictions to the original.
    np.testing.assert_array_equal(model.predict(X_val), loaded_model.predict(X_val))


def test_load_artifact_missing_file_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_artifact(tmp_path / "does_not_exist.joblib")


def test_save_and_load_metadata_roundtrip(tmp_path):
    metadata = {"model_name": "LogisticRegression", "random_state": 42, "ngram_range": [1, 2]}
    meta_path = tmp_path / "metadata.json"

    save_metadata(metadata, meta_path)
    loaded = load_metadata(meta_path)

    assert loaded == metadata


def test_label_names_match_project_convention():
    # Same convention established in Phase 2 (src/preprocessing/load_data.py):
    # 1 = Fake, 0 = Real. This test fails loudly if that convention ever drifts.
    assert LABEL_NAMES[1] == "Fake"
    assert LABEL_NAMES[0] == "Real"
