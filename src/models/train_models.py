"""
Training functions for the Phase 4 candidate ML models.

Each train_* function returns (fitted_model, training_time_seconds) so training
cost is always measured and reported, never guessed at.

random_state=42 is used everywhere randomness is involved, for reproducibility.
"""

import time

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 42


def _timed_fit(model, X_train, y_train):
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    return model, elapsed


def train_logistic_regression(X_train, y_train, C: float = 1.0, max_iter: int = 1000):
    """Logistic Regression on TF-IDF features."""
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=RANDOM_STATE)
    return _timed_fit(model, X_train, y_train)


def train_naive_bayes(X_train, y_train, alpha: float = 1.0):
    """Multinomial Naive Bayes on TF-IDF features (requires non-negative features, which TF-IDF satisfies)."""
    model = MultinomialNB(alpha=alpha)
    return _timed_fit(model, X_train, y_train)


def train_linear_svm(X_train, y_train, C: float = 1.0, max_iter: int = 5000):
    """
    Linear SVM on TF-IDF features.

    Why LinearSVC for this task: TF-IDF vectors are extremely high-dimensional
    (tens of thousands of features) and sparse. A linear kernel is the natural
    fit for that geometry - text classification problems are very often
    linearly separable (or close to it) in a high-dimensional bag-of-words/
    n-gram space, so a non-linear kernel (e.g. RBF) would add a lot of
    computational cost for little to no accuracy benefit, and would struggle
    to scale to this many features/documents anyway. LinearSVC uses an
    optimized solver (liblinear) specifically for this linear+sparse case.
    """
    model = LinearSVC(C=C, max_iter=max_iter, random_state=RANDOM_STATE)
    return _timed_fit(model, X_train, y_train)


def train_random_forest(X_train_dense, y_train, n_estimators: int = 100, max_depth=20):
    """
    Random Forest on dense, dimensionality-reduced features (TF-IDF -> SVD).

    n_estimators/max_depth are kept moderate deliberately: this project runs
    on a single-core, modest-memory environment (documented in README ->
    Phase 4), so hyperparameters were chosen to keep training time reasonable
    rather than to squeeze out maximum possible accuracy.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return _timed_fit(model, X_train_dense, y_train)


def train_xgboost(X_train_dense, y_train, n_estimators: int = 150, max_depth: int = 6,
                   learning_rate: float = 0.3):
    """
    XGBoost on dense, dimensionality-reduced features (TF-IDF -> SVD), for the
    same reason as Random Forest above.

    Returns (None, 0.0) if xgboost is not installed in this environment,
    rather than crashing - callers must check for a None model before using it.
    """
    if not XGBOOST_AVAILABLE:
        return None, 0.0

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",  # fastest histogram-based method, important on limited CPU
        n_jobs=-1,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    return _timed_fit(model, X_train_dense, y_train)
