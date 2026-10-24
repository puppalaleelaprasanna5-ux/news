"""
Inference module (Phase 5): turn a raw news article into a readable prediction
using the artifacts saved in Phase 4. No retraining happens here.

    raw article text
        -> clean_single_text()          (same cleaning function Phase 2 used to build `clean_text`)
        -> fitted TF-IDF vectorizer     (models/tfidf_vectorizer.joblib, transform-only)
        -> Logistic Regression model    (models/best_model.joblib)
        -> readable prediction dict

WHY THIS MATCHES TRAINING:
    Phase 4 trained on `train["clean_text"]`, and `clean_text` was produced by
    `clean_single_text(text)` (src/preprocessing/text_cleaning.py). The model was
    trained on the article BODY only - titles were never used. So the input to
    this module is the article body text, and we reuse the very same cleaning
    function rather than re-implementing it.

Label convention (project-wide, from src/preprocessing/load_data.py):
    1 = Fake, 0 = Real

Quick use from Python:
    from src.models.predict import FakeNewsPredictor
    predictor = FakeNewsPredictor()
    result = predictor.predict("WASHINGTON (Reuters) - The Senate on Tuesday ...")
    print(result["label"], result["fake_probability"])

Quick use from the command line (from the project root):
    python -m src.models.predict --text "Article text here ..."
    python -m src.models.predict --file article.txt --explain
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

# Allow `python src/models/predict.py` as well as `python -m src.models.predict`.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.tfidf_features import transform_with_tfidf
from src.models.evaluate_models import get_positive_class_scores
from src.models.model_utils import LABEL_NAMES, load_artifact, load_metadata
from src.preprocessing.load_data import LABEL_FAKE, LABEL_REAL
from src.preprocessing.quality_checks import VERY_SHORT_ARTICLE_WORD_THRESHOLD
from src.preprocessing.text_cleaning import clean_single_text

DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "best_model.joblib"
VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"
METADATA_FILENAME = "model_metadata.json"

# Display bands for the confidence of the predicted class. These are simple
# presentation thresholds, NOT calibrated probabilities - Logistic Regression
# scores have not been calibration-tested in this project.
HIGH_CONFIDENCE = 0.90
MODERATE_CONFIDENCE = 0.70

UNDETERMINED = "Undetermined"

DISCLAIMER = (
    "This is a statistical text-pattern score from a model trained on the ISOT dataset. "
    "It is not a fact-check and not proof that an article is true or false. The model "
    "partly relies on source-formatting patterns (see model_metadata.json)."
)


class FakeNewsPredictor:
    """Loads the saved artifacts once and predicts on new article text."""

    def __init__(self, models_dir=DEFAULT_MODELS_DIR):
        models_dir = Path(models_dir)

        # sklearn emits InconsistentVersionWarning when an artifact was saved with a
        # different scikit-learn version. We capture it so it can be shown to the
        # user (see model_info()) instead of being silently lost.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.model = load_artifact(models_dir / MODEL_FILENAME)
            self.vectorizer = load_artifact(models_dir / VECTORIZER_FILENAME)
        self.metadata = load_metadata(models_dir / METADATA_FILENAME)
        self.load_warnings = [str(w.message) for w in caught]

        self._verify_artifacts()
        self._coefficients = self.model.coef_[0]
        self._feature_names = self.vectorizer.get_feature_names_out()

    # ------------------------------------------------------------------ checks
    def _verify_artifacts(self):
        """Fail loudly if the model, vectorizer and metadata don't belong together."""
        n_vocab = len(self.vectorizer.vocabulary_)
        n_coef = self.model.coef_.shape[1]
        problems = []

        if n_vocab != n_coef:
            problems.append(f"vectorizer has {n_vocab} features but model expects {n_coef}")
        if n_vocab != self.metadata.get("vocabulary_size"):
            problems.append(
                f"metadata vocabulary_size={self.metadata.get('vocabulary_size')} but vectorizer has {n_vocab}"
            )
        if list(self.vectorizer.ngram_range) != self.metadata.get("ngram_range"):
            problems.append("vectorizer ngram_range does not match metadata")
        if type(self.model).__name__ != self.metadata.get("model_class"):
            problems.append(
                f"model is {type(self.model).__name__} but metadata says {self.metadata.get('model_class')}"
            )
        if sorted(self.model.classes_) != [LABEL_REAL, LABEL_FAKE]:
            problems.append(f"unexpected model classes: {list(self.model.classes_)}")
        if self.metadata.get("label_convention") != {str(k): v for k, v in LABEL_NAMES.items()}:
            problems.append("metadata label_convention does not match the project's LABEL_NAMES")

        if problems:
            raise RuntimeError("Model artifacts are inconsistent: " + "; ".join(problems))

    # -------------------------------------------------------------- prediction
    def predict(self, text, explain: bool = False, top_n: int = 5) -> dict:
        """
        Predict for ONE article body. Returns a JSON-friendly dict.

        Raises ValueError if `text` is not a string or is empty after cleaning.
        Set explain=True to also get the terms that pushed the score most.
        """
        return self.predict_batch([text], explain=explain, top_n=top_n)[0]

    def predict_batch(self, texts, explain: bool = False, top_n: int = 5) -> list:
        """Predict for a list of article bodies (one vectorizer call for the whole list)."""
        cleaned = [self._clean_and_validate(t, i) for i, t in enumerate(texts)]

        X = transform_with_tfidf(self.vectorizer, cleaned)      # transform-only, never refit
        fake_probs = get_positive_class_scores(self.model, X)   # P(Fake) from predict_proba

        results = []
        for i, clean in enumerate(cleaned):
            row = X[i]
            results.append(self._build_result(clean, row, float(fake_probs[i]), explain, top_n))
        return results

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _clean_and_validate(text, index) -> str:
        if not isinstance(text, str):
            raise ValueError(f"Input #{index} must be a string, got {type(text).__name__}.")
        clean = clean_single_text(text)
        if not clean:
            raise ValueError(f"Input #{index} is empty after cleaning - please provide article text.")
        return clean

    def _build_result(self, clean_text, row, fake_prob, explain, top_n) -> dict:
        n_words = len(clean_text.split())
        n_known_terms = int(row.nnz)  # how many of the model's 30,000 terms appear in this text
        warnings_list = []

        if n_words < VERY_SHORT_ARTICLE_WORD_THRESHOLD:
            warnings_list.append(
                f"Very short input ({n_words} words). The model was trained on full-length "
                f"articles, so treat this result with extra caution."
            )

        if n_known_terms == 0:
            # No known vocabulary -> all-zero feature vector -> the score would come only from the
            # model's intercept, which says nothing about this text. Do not pretend it does.
            warnings_list.append(
                "None of the words in this text are in the model's vocabulary (e.g. non-English "
                "text), so no prediction can be made."
            )
            result = {
                "label": UNDETERMINED,
                "label_id": None,
                "fake_probability": None,
                "real_probability": None,
                "probabilities_calibrated": None,
                "confidence": None,
                "confidence_level": None,
            }
        else:
            label_id = LABEL_FAKE if fake_prob >= 0.5 else LABEL_REAL  # same rule as model.predict
            real_prob = 1.0 - fake_prob
            confidence = max(fake_prob, real_prob)
            result = {
                "label": LABEL_NAMES[label_id],
                "label_id": label_id,
                "fake_probability": round(fake_prob, 4),
                "real_probability": round(real_prob, 4),
                "probabilities_calibrated": False,
                "confidence": round(confidence, 4),
                "confidence_level": self._confidence_level(confidence),
            }

        result.update({
            "n_words": n_words,
            "n_known_terms": n_known_terms,
            "warnings": warnings_list,
            "model_name": self.metadata["model_name"],
            "disclaimer": DISCLAIMER,
        })
        if explain:
            result["top_terms"] = self._top_terms(row, top_n)
        return result

    @staticmethod
    def _confidence_level(confidence: float) -> str:
        if confidence >= HIGH_CONFIDENCE:
            return "high"
        if confidence >= MODERATE_CONFIDENCE:
            return "moderate"
        return "low"

    def _top_terms(self, row, top_n) -> dict:
        """
        Terms in THIS text that pushed the score hardest. For Logistic Regression the
        log-odds are  intercept + sum(tfidf_value * coefficient),  so each term's
        contribution is simply tfidf_value * coefficient (positive -> toward Fake).
        """
        idx = row.indices
        contributions = row.data * self._coefficients[idx]
        order = contributions.argsort()
        toward_real = [(self._feature_names[idx[j]], float(contributions[j])) for j in order[:top_n] if contributions[j] < 0]
        toward_fake = [(self._feature_names[idx[j]], float(contributions[j])) for j in order[::-1][:top_n] if contributions[j] > 0]
        return {
            "toward_fake": [{"term": t, "contribution": round(c, 4)} for t, c in toward_fake],
            "toward_real": [{"term": t, "contribution": round(c, 4)} for t, c in toward_real],
        }

    def model_info(self) -> dict:
        """Everything a UI/API might want to show about the loaded model."""
        keys = ["model_name", "model_class", "feature_type", "ngram_range", "vocabulary_size",
                "training_dataset", "training_set_size", "label_convention",
                "validation_metrics", "known_limitations"]
        info = {k: self.metadata.get(k) for k in keys}
        info["load_warnings"] = self.load_warnings
        return info


def load_predictor(models_dir=DEFAULT_MODELS_DIR) -> FakeNewsPredictor:
    """Convenience wrapper: load artifacts once (e.g. at app start-up) and reuse the object."""
    return FakeNewsPredictor(models_dir)


def _percent(value: float) -> str:
    """Format a probability as a percentage, never showing a misleading 100.0% / 0.0%."""
    if value >= 0.999:
        return ">99.9%"
    if value <= 0.001:
        return "<0.1%"
    return f"{value:.1%}"


def format_prediction(result: dict) -> str:
    """Turn a prediction dict into a short human-readable report."""
    lines = []
    if result["label"] == UNDETERMINED:
        lines.append("Prediction : Undetermined (not enough known vocabulary)")
    else:
        lines.append(f"Prediction : {result['label']}  "
                     f"(uncalibrated score band: {result['confidence_level']})")
        lines.append(f"Fake score : {_percent(result['fake_probability'])}    "
                     f"Real score: {_percent(result['real_probability'])} (not calibrated)")
    lines.append(f"Text size  : {result['n_words']} words, {result['n_known_terms']} known model terms")
    for w in result["warnings"]:
        lines.append(f"Warning    : {w}")
    if "top_terms" in result:
        fake_terms = ", ".join(f"{t['term']} ({t['contribution']:+.2f})" for t in result["top_terms"]["toward_fake"])
        real_terms = ", ".join(f"{t['term']} ({t['contribution']:+.2f})" for t in result["top_terms"]["toward_real"])
        lines.append(f"Toward Fake: {fake_terms or '-'}")
        lines.append(f"Toward Real: {real_terms or '-'}")
    lines.append(f"Note       : {result['disclaimer']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predict whether a news article looks Fake or Real.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Article text to classify.")
    source.add_argument("--file", help="Path to a UTF-8 text file containing the article.")
    parser.add_argument("--explain", action="store_true", help="Also show the terms that drove the score.")
    parser.add_argument("--json", action="store_true", help="Print the raw result as JSON.")
    args = parser.parse_args(argv)

    text = args.text if args.text is not None else Path(args.file).read_text(encoding="utf-8")

    try:
        predictor = load_predictor()
        result = predictor.predict(text, explain=args.explain)
    except (FileNotFoundError, ValueError, RuntimeError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    for w in predictor.load_warnings:
        print(f"Artifact warning: {w}", file=sys.stderr)
    print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else format_prediction(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
