"""
Tests for the Phase 5 inference module (src/models/predict.py).

Unlike tests/test_ml_models.py (which uses tiny fixture data), these tests load the
REAL saved artifacts in models/ - they check that the saved model can actually be
used for inference. They do not need the dataset. If the artifacts are missing
(e.g. a fresh clone), the tests are skipped rather than failed.

Written with unittest (which pytest also collects), so either works:
    python -m unittest tests.test_inference -v
    pytest tests/test_inference.py -v
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.predict import (
    DEFAULT_MODELS_DIR, FakeNewsPredictor, UNDETERMINED, format_prediction,
)
from src.preprocessing.text_cleaning import add_clean_text_column

ARTIFACTS_PRESENT = all(
    (DEFAULT_MODELS_DIR / name).exists()
    for name in ("best_model.joblib", "tfidf_vectorizer.joblib", "model_metadata.json")
)

# Hand-written sample texts (NOT taken from the dataset).
REAL_STYLE = (
    "WASHINGTON (Reuters) - The U.S. Senate on Tuesday passed a spending bill that funds the "
    "government through September, officials said, sending the measure to the House of "
    "Representatives, which is expected to vote on Wednesday."
)
FAKE_STYLE = (
    "You wont believe what Obama just did to America! Watch the video here and read more "
    "below. Featured image via Getty Images. Share this with everyone you know before they "
    "delete it!"
)
# Messy on purpose: HTML entity, tag, curly quotes, non-breaking space, repeated whitespace.
MESSY_TEXT = (
    "<p>WASHINGTON (Reuters) -   The senator said\u00a0on Tuesday that the \u201cbill\u201d "
    "was ready &amp; that talks would\n\ncontinue on Wednesday.</p>"
)


@unittest.skipUnless(ARTIFACTS_PRESENT, "Saved artifacts not found in models/ - run notebooks/03_ml_models.ipynb first.")
class TestFakeNewsPredictor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = FakeNewsPredictor()

    # --- loading / consistency ---------------------------------------------
    def test_artifacts_load_and_are_consistent(self):
        p = self.predictor
        self.assertEqual(len(p.vectorizer.vocabulary_), p.model.coef_.shape[1])
        self.assertEqual(p.metadata["model_name"], "Logistic Regression")

    def test_loaded_coefficients_match_notebook_03_analysis(self):
        # Values printed in notebooks/03_ml_models.ipynb Section 12 - confirms the file on
        # disk is the same model that was analysed when the model was selected.
        vocab = self.predictor.vectorizer.vocabulary_
        coef = self.predictor.model.coef_[0]
        self.assertAlmostEqual(coef[vocab["reuters"]], -16.792, places=2)
        self.assertAlmostEqual(coef[vocab["via"]], 8.584, places=2)

    def test_mismatched_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("best_model.joblib", "tfidf_vectorizer.joblib", "model_metadata.json"):
                shutil.copy(DEFAULT_MODELS_DIR / name, tmp)
            meta_path = Path(tmp) / "model_metadata.json"
            meta = json.loads(meta_path.read_text())
            meta["vocabulary_size"] = 123
            meta_path.write_text(json.dumps(meta))
            with self.assertRaises(RuntimeError):
                FakeNewsPredictor(tmp)

    def test_missing_artifacts_give_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                FakeNewsPredictor(tmp)

    # --- output format --------------------------------------------------------
    def test_result_has_expected_fields_and_is_json_serialisable(self):
        result = self.predictor.predict(REAL_STYLE)
        for key in ("label", "label_id", "fake_probability", "real_probability", "confidence",
                    "confidence_level", "n_words", "n_known_terms", "warnings", "model_name", "disclaimer"):
            self.assertIn(key, result)
        json.dumps(result)  # must not raise (no numpy types left in the dict)
        self.assertAlmostEqual(result["fake_probability"] + result["real_probability"], 1.0, places=3)
        self.assertIn(result["label"], ("Fake", "Real"))

    def test_format_prediction_returns_readable_text(self):
        text = format_prediction(self.predictor.predict(FAKE_STYLE, explain=True))
        self.assertIn("Prediction", text)
        self.assertIn("Toward Fake", text)

    # --- preprocessing consistency with training -----------------------------
    def test_inference_preprocessing_matches_training_pipeline(self):
        # Training path: Phase 2's add_clean_text_column -> vectorizer.transform -> model.
        # Inference path: FakeNewsPredictor.predict. Both must give the same probability.
        texts = [REAL_STYLE, FAKE_STYLE, MESSY_TEXT]
        df = add_clean_text_column(pd.DataFrame({"text": texts}))
        X = self.predictor.vectorizer.transform(df["clean_text"].fillna(""))
        training_path_probs = self.predictor.model.predict_proba(X)[:, 1]
        for text, expected in zip(texts, training_path_probs):
            self.assertAlmostEqual(self.predictor.predict(text)["fake_probability"], expected, places=4)

    def test_label_agrees_with_model_predict(self):
        for text in (REAL_STYLE, FAKE_STYLE, MESSY_TEXT):
            clean_df = add_clean_text_column(pd.DataFrame({"text": [text]}))
            X = self.predictor.vectorizer.transform(clean_df["clean_text"])
            self.assertEqual(self.predictor.predict(text)["label_id"], int(self.predictor.model.predict(X)[0]))

    def test_batch_matches_single(self):
        texts = [REAL_STYLE, FAKE_STYLE, MESSY_TEXT]
        batch = self.predictor.predict_batch(texts)
        for text, from_batch in zip(texts, batch):
            self.assertEqual(from_batch["fake_probability"], self.predictor.predict(text)["fake_probability"])

    # --- explanation ------------------------------------------------------------
    def test_explain_terms_have_correct_signs(self):
        terms = self.predictor.predict(FAKE_STYLE, explain=True, top_n=5)["top_terms"]
        self.assertTrue(all(t["contribution"] > 0 for t in terms["toward_fake"]))
        self.assertTrue(all(t["contribution"] < 0 for t in terms["toward_real"]))

    # --- bad / unusual input ------------------------------------------------------
    def test_invalid_inputs_raise_value_error(self):
        for bad in ("", "   \n\t ", None, 123, ["a list"]):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    self.predictor.predict(bad)

    def test_text_with_no_known_vocabulary_is_undetermined(self):
        result = self.predictor.predict("zzqx qwxz vvkk xxjq")
        self.assertEqual(result["label"], UNDETERMINED)
        self.assertIsNone(result["fake_probability"])
        self.assertEqual(result["n_known_terms"], 0)
        self.assertTrue(result["warnings"])

    def test_very_short_text_gets_a_warning(self):
        result = self.predictor.predict("Reuters said on Tuesday the bill passed.")
        self.assertTrue(any("short" in w.lower() for w in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
