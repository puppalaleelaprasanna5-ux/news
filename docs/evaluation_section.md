# Evaluation and Limitations (report section)

> Numbers marked ⟦…⟧ come from `results/threshold_tuning.json` after you run
> `python scripts/threshold_tuning.py`. Replace them with your real values. Do not guess them.

## 1. Cross-dataset generalization

The ISOT-trained TF-IDF + Logistic Regression model reaches 98.76% accuracy on its in-domain
validation set. That figure partly reflects dataset-specific artifacts (Reuters formatting,
image-credit tokens) found in our error analysis, so we tested the saved model, without
retraining, on 71,321 scorable WELFake articles (body text only, matching the training input;
783 rows with empty bodies were dropped and 30 were unscorable).

| Dataset | Accuracy | Precision | Recall | F1 | ROC-AUC | Predicted Fake |
|---|---|---|---|---|---|---|
| ISOT validation (in-domain) | 98.76% | 99.15% | 98.09% | 98.62% | 0.9992 | n/a |
| WELFake (external) | 84.74% | 78.88% | 95.61% | 86.44% | 0.9231 | 61.7% (true rate 50.9%) |

Accuracy fell by about 14 points. The WELFake confusion matrix has 9,291 false positives
(26.5% of real articles flagged Fake) and 1,593 false negatives (4.4% of fake articles missed).
The model therefore over-predicts Fake on unfamiliar data. A high ROC-AUC (0.923) with weaker
thresholded accuracy indicates that ranking transfers reasonably but the fixed 0.5 cutoff does not.

## 2. Threshold tuning (no retraining)

To test that explanation, WELFake was split 50/50 (stratified, seed 42). A decision threshold
was chosen on the tuning half only and every metric below was computed on the held-out test half.

| Setting | Threshold | Accuracy | Balanced acc. | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Default | 0.50 | ⟦…⟧ | ⟦…⟧ | ⟦…⟧ | ⟦…⟧ | ⟦…⟧ |
| Tuned (best F1) | ⟦…⟧ | ⟦…⟧ | ⟦…⟧ | ⟦…⟧ | ⟦…⟧ | ⟦…⟧ |

Interpretation guide (write only the sentence that matches your numbers):
- If tuned accuracy is clearly higher: *part of the drop is miscalibration of the decision threshold,
  which a small amount of target-domain data can correct.*
- If it barely changes: *the drop is not mainly a threshold problem; the model's learned features
  transfer imperfectly, which motivates a transformer model and out-of-domain training data.*

## 3. Limitations

- WELFake is a compilation of several public sources and may partly overlap with ISOT-style
  sources, so the reported accuracy may overestimate true out-of-domain performance.
- WELFake labels are noisy (for example, opinion pieces appear under "Real").
- The tuning and test halves are both WELFake, so tuned results do not show generalization to a third dataset.
- One external dataset does not establish general robustness.
- The model classifies writing style and wording, not factual truth.

## 4. Future work

Probability calibration, a fine-tuned transformer classifier, evaluation on further datasets
(for example LIAR), and NLI-based evidence verification against a real knowledge base.

## Viva questions

**Why did accuracy drop from 98.8% to 84.7%?**
Distribution shift. The model learned ISOT-specific patterns (source formatting, phrasing) as well
as genuine cues, and those are less reliable on WELFake. It is a generalization finding, not a bug.

**If accuracy dropped, why is AUC still 0.92?**
AUC measures ranking, independent of threshold. The ranking mostly transfers, but the fixed 0.5
cutoff does not fit the new data, which is why too many real articles are flagged Fake.

**How do you know the threshold gain is not overfitting?**
The threshold was chosen on one half and evaluated on the other, which was never used for selection,
with a fixed seed so the split is reproducible.

**How do you know the labels weren't flipped?**
I printed sample rows per raw label and read them before evaluating: label 0 was wire-service news,
label 1 was clickbait and propaganda. The confusion matrix and predicted-Fake percentage were also
sensible rather than collapsed to one class.

**Does this prove the model works in general?**
No. It is one external dataset with possible source overlap. It shows the in-domain score was
optimistic and quantifies by how much.
