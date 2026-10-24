# Evaluation results and limitations

The current evaluation record, metric provenance, threshold protocol, calibration status,
limitations, and recommendations are in [evaluation_report.md](./evaluation_report.md).

The committed `results/cross_dataset.json` and `results/threshold_tuning.json` contain the
recorded WELFake metrics. The source CSV and processed ISOT splits are not present in this
checkout, so the recorded figures cannot be regenerated here. Do not treat them as a fresh
evaluation or use them to justify retraining without first reproducing them from the source
datasets.

From the repository root, after placing `WELFake_Dataset.csv` under `data/raw/` (or passing
`--input-csv PATH`), inspect label examples and run:

```powershell
python scripts/cross_dataset_eval.py --preview-only
python scripts/cross_dataset_eval.py
python scripts/threshold_tuning.py
```

The evaluation scripts use the existing saved model and vectorizer. Threshold selection uses
only the tuning half; all thresholded metrics are reported on the held-out half.
