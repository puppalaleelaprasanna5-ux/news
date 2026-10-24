# NewsCred model reliability evaluation report

## Executive summary

The saved ISOT-trained TF-IDF + Logistic Regression model was preserved. Existing recorded
results show a large drop from ISOT validation to WELFake and an elevated Fake prediction rate
on WELFake. Tuning a decision threshold on one WELFake half gives only a small improvement on
the held-out half. This is evidence of domain shift, not evidence that threshold adjustment
solves generalization.

This report transcribes the committed evaluation artifacts and repository-recorded ISOT
metrics. It does **not** claim a fresh run: this checkout has no raw ISOT/WELFake CSVs or
processed split CSVs. The JSON's recorded WELFake input path (`C:\fake-news-detector\data\raw\WELFake_Dataset.csv`)
is from the original evaluation environment and is not a path in this checkout. The artifacts
contain no run timestamp or dataset checksum.

## Model and evaluation protocol

- Saved inference artifacts: `models/best_model.joblib`,
  `models/tfidf_vectorizer.joblib`, and `models/model_metadata.json`.
- Model: Logistic Regression, TF-IDF word unigrams/bigrams, 30,000 features.
- Training dataset: ISOT Fake and Real News Dataset; model metadata records 27,050
  training articles.
- Labels: `0 = Real`, `1 = Fake`.
- Input: article body only. Training and inference both call the shared
  `clean_single_text` preprocessing function; inference transforms with the saved
  vectorizer and does not refit.
- The validation split is used for model selection. Repository documentation records the
  one-time final ISOT test result below; the test set is not used by the threshold-tuning
  script.
- Subject and date are excluded from model features. Existing model metadata and coefficient
  analysis document formatting/source artifacts (including Reuters boilerplate and image
  credit/URL patterns).

## Recorded ISOT results

| Split | Rows | Accuracy | Precision (Fake) | Recall (Fake) | F1 (Fake) | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Validation | Not stored in metadata | 98.76% | 99.15% | 98.09% | 98.62% | 0.9992 |
| Final test | Not available in this checkout | 98.79% | 99.27% | 98.05% | 98.65% | 0.9982 |

Validation aggregates are from `models/model_metadata.json`. The final-test aggregates are
recorded in the repository's Phase 4 evaluation documentation and notebook. The committed
artifacts do not provide the ISOT test confusion matrix or per-class metrics; those values are
not inferred here.

## Recorded external WELFake results

The committed `results/cross_dataset.json` records evaluation of the saved model without
retraining. It records 72,134 input rows, 783 dropped rows (empty body or missing label),
71,351 rows after dropping, 71,321 scorable predictions, and 30 unscorable predictions.
Evaluation input was the body text only. The recorded raw-label convention is `0 = Real`,
`1 = Fake`, with no label flip.

| Metric | WELFake |
|---|---:|
| Accuracy | 84.74% |
| Precision (Fake) | 78.88% |
| Recall (Fake) | 95.61% |
| F1 (Fake) | 86.44% |
| ROC-AUC | 0.9231 |
| Predicted Fake | 61.68% |
| Actual Fake | 50.89% |

Confusion matrix (actual rows, predicted columns; `[Real, Fake]`):

| Actual \ Predicted | Real | Fake |
|---|---:|---:|
| Real | 25,736 | 9,291 |
| Fake | 1,593 | 34,701 |

Per-class metrics derived directly from that recorded matrix:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Real | 94.17% | 73.47% | 82.55% | 35,027 |
| Fake | 78.88% | 95.61% | 86.44% | 36,294 |

The model flagged 9,291 of 35,027 real articles as Fake (26.52%) and missed 1,593 of
36,294 fake articles (4.39%). The weaker Real recall and higher predicted-Fake share are
important alongside the aggregate accuracy.

## Threshold protocol and recorded held-out results

`results/threshold_tuning.json` records the historical, row-stratified 50/50 WELFake split
with seed 42:
35,660 tuning rows and 35,661 held-out test rows. The best-F1 threshold (0.56) and
best-balanced-accuracy threshold (0.65) were selected on the tuning half only. All metrics
below are from the held-out test half. These committed values predate duplicate-aware splitting
and have not been regenerated because the WELFake CSV is absent from this checkout; they are
not results from the current grouped protocol.

The current `scripts/threshold_tuning.py` groups exact normalized article bodies and
conservative near-duplicate candidates before splitting. Near-duplicates of at least 12 words
use word-trigram SimHash (Hamming distance <= 6), with seven fingerprint bands so pairs within
that distance are candidates for comparison; connected candidates remain in the same group.
Shorter bodies are grouped only on exact normalized text. This is a heuristic, not semantic
verification, and its false-positive grouping may reduce the effective sample size. If a group
contains conflicting labels, evaluation stops and reports source row positions rather than
silently dropping examples. Tuning thresholds use only tuning groups; all reported test
metrics use held-out groups.

| Setting | Threshold | Accuracy | Balanced accuracy | Precision (Fake) | Recall (Fake) | F1 (Fake) |
|---|---:|---:|---:|---:|---:|---:|
| Default | 0.50 | 84.56% | 84.36% | 78.68% | 95.54% | 86.30% |
| Best F1 | 0.56 | 84.89% | 84.72% | 79.77% | 94.19% | 86.38% |
| Best balanced accuracy | 0.65 | 85.19% | 85.07% | 81.69% | 91.36% | 86.26% |

These settings trade recall for modestly better accuracy/precision. The model's deployed 0.5
decision threshold was not changed. The threshold-test half remains WELFake, not a third
independent dataset.

## Leakage, duplicates, and preprocessing audit

- Existing preprocessing removes empty rows, cross-class exact-text conflicts, and exact
  duplicate bodies before splitting. The split helper hard-fails on article-ID and raw-text
  overlap.
- The preprocessing pipeline also reports exact `clean_text` overlap. The current code now
  additionally reports conservative cross-split near-duplicate **candidates** using
  word-trigram SimHash (Hamming distance <= 6); it does not delete or silently reassign those rows. Candidate counts
  must be generated from the original ISOT data by rerunning preprocessing.
- The saved processed splits are absent here, so no current-checkout duplicate or
  near-duplicate count can be asserted.
- The WELFake source CSV is also absent, so the new duplicate-grouped threshold evaluation
  cannot be run here. The recorded row-split metrics above remain historical and do not
  establish that duplicate groups were separated.
- The dataset's `subject` and `date` fields are excluded by design; model documentation
  separately identifies formatting/source cues in the learned text features. These
  artifacts can still correlate with labels even when structured columns are excluded.
- TF-IDF fits on training text only and inference uses the saved transform-only vectorizer.

## Probability calibration

The selected Logistic Regression exposes `predict_proba`, and inference returns its Fake and
Real class probabilities. The project has **not** fitted or validated a probability
calibration method; display confidence bands are thresholds over raw model outputs, not
calibrated confidence. Therefore the API marks probabilities as uncalibrated.

The evaluation scripts now emit Brier score and 10-bin expected calibration error as
diagnostics when rerun against the source dataset. These diagnostics do not fit a calibrator
and should not be treated as proof of calibration. The committed evaluation JSON predates
these diagnostics, and the raw dataset is absent, so no calibration metric is reported here.

## Limitations and recommendations

1. WELFake aggregates public sources and may overlap with ISOT source/content patterns;
   possible overlap can overstate transfer performance.
2. WELFake labels may be noisy, and only one external collection is represented.
3. There are no source-held-out or topic-held-out evaluations. A random row split cannot
   establish generalization to unfamiliar publishers or subjects.
4. The classifier detects text-pattern similarity, not factual truth. Retrieved passage
   similarity is likewise not evidence of support or contradiction.
5. Re-run the reported experiments from verified local source datasets; record dataset
   checksums, versions, label validation, and output artifacts.
6. Add publisher/source and topic-grouped holdouts, plus temporal evaluation, before
   considering retraining. Report per-class metrics and calibration diagnostics by subgroup.
7. If probability values will be interpreted quantitatively, fit and evaluate calibration
   (for example, sigmoid or isotonic calibration) on a representative calibration set that
   is separate from final testing.

No retraining is recommended from the artifacts alone. Revisit that decision only after a
reproducible, leakage-audited evaluation shows that a candidate improves relevant
out-of-source and out-of-topic performance.
