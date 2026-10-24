"""
Regression guards run alongside Phase 7: they confirm earlier behaviour (saved model artifacts,
Phase 5 prediction, dependency pins) is unchanged by the RAG additions.

    python -m pytest tests/test_phase7_regression.py -v
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import DEFAULT_MODELS_DIR, FakeNewsPredictor

ARTIFACTS_PRESENT = all(
    (DEFAULT_MODELS_DIR / name).exists()
    for name in ("best_model.joblib", "tfidf_vectorizer.joblib", "model_metadata.json")
)


class TestPhase7DoesNotBreakEarlierPhases(unittest.TestCase):
    @unittest.skipUnless(ARTIFACTS_PRESENT, "model artifacts not present")
    def test_saved_model_artifacts_still_load_and_are_consistent(self):
        predictor = FakeNewsPredictor()
        self.assertEqual(type(predictor.model).__name__, "LogisticRegression")
        self.assertEqual(len(predictor.vectorizer.vocabulary_), predictor.model.coef_.shape[1])

    @unittest.skipUnless(ARTIFACTS_PRESENT, "model artifacts not present")
    def test_phase5_prediction_still_works(self):
        result = FakeNewsPredictor().predict(
            "WASHINGTON (Reuters) - The U.S. Senate on Tuesday passed a spending bill that funds the "
            "government through September, officials said."
        )
        self.assertIn("label", result)
        self.assertIn("fake_probability", result)
        self.assertGreaterEqual(result["fake_probability"], 0.0)
        self.assertLessEqual(result["fake_probability"], 1.0)

    def test_existing_dependency_pins_were_not_changed(self):
        text = (PROJECT_ROOT / "requirements.txt").read_text()
        for pin in ("scikit-learn==1.8.0", "numpy==1.26.4", "pandas==2.2.3", "faiss-cpu==1.9.0", "requests==2.32.3"):
            self.assertIn(pin, text)

    def test_isot_training_data_is_not_configured_as_an_evidence_source(self):
        from src.rag.config import DEFAULT_DOCS_DIR, SAMPLE_DOCS_DIR
        for path in (DEFAULT_DOCS_DIR, SAMPLE_DOCS_DIR):
            self.assertNotIn("raw", path.parts)
            self.assertNotIn("processed", path.parts)


if __name__ == "__main__":
    unittest.main()
