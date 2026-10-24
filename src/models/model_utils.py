"""
Model artifact persistence utilities: saving/loading trained models,
vectorizers, and dimensionality-reduction transforms, plus the metadata
file describing the final selected model configuration.
"""

import json
from pathlib import Path

import joblib

from src.preprocessing.load_data import LABEL_FAKE, LABEL_REAL

# Human-readable label names, built from the SAME constants the rest of the
# project uses (src/preprocessing/load_data.py) - never redefine the
# fake/real -> 0/1 mapping in a second place.
LABEL_NAMES = {LABEL_FAKE: "Fake", LABEL_REAL: "Real"}


def save_artifact(obj, path) -> None:
    """Save any picklable object (model, vectorizer, SVD transform) with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path):
    """Load a previously-saved artifact. Raises FileNotFoundError with a clear message if missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find model artifact at '{path}'. "
            f"Run notebooks/03_ml_models.ipynb first to train and save it."
        )
    return joblib.load(path)


def save_metadata(metadata: dict, path) -> None:
    """Save model metadata (config, not data) as pretty-printed JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_metadata(path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Could not find metadata file at '{path}'.")
    with open(path) as f:
        return json.load(f)
