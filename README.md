# AI-Based Fake News Detection and News Credibility Analysis Using Machine Learning and Evidence Retrieval

Final-year B.Tech (Computer Science - Data Science) major project.

## Status

Phase 1 (Project Setup) complete.
Phase 2 (Dataset and Preprocessing) complete.
Phase 3 (Exploratory Data Analysis) complete.
Phase 4 (Machine Learning Baseline) complete.
Phase 5 (Inference Pipeline & Application Readiness) complete (Dependency issue fixed: scikit-learn==1.8.0).
Phase 6 (Llama 3.1 8B integration) was **removed**: it needs more memory than the target machine has. See "Phase 6 (removed)" below.
Phase 7 (FAISS + RAG Evidence Retrieval) complete. The real-dependency checks are in `tests/test_rag_real_dependencies.py`; see the current verification record below.
Phase 8 (FastAPI backend + Streamlit dashboard) implemented. See the current verification record below for exact test scope and results.

## Tech Stack

- Python **3.11 or 3.12** (see "Supported Python version" below)
- scikit-learn (1.8.0), XGBoost (ML models)
- NLTK (NLP preprocessing)
- FAISS (RAG vector store) and Sentence Transformers (local text embeddings) — used in Phase 7
- FastAPI (backend API)
- Streamlit (frontend dashboard)
- SQLite (storage)

## Supported Python version

**Python 3.11 or 3.12.** Python 3.13 and newer are **not supported** by the pinned dependencies.

| Constraint | Source |
|---|---|
| Python >= 3.11 | `scikit-learn==1.8.0` (its minimum supported Python) |
| Python <= 3.12 | `numpy==1.26.4` (supports Python 3.9 - 3.12; it has no Python 3.13 build) |

Both pins are kept as-is because the saved models in `models/` were created with scikit-learn 1.8.0.
Python 3.10 and older are also unsupported (scikit-learn 1.8.0 requires 3.11+).

**Verification status:** the supported range above is derived from what the pinned packages declare (scikit-learn
1.8.0 needs Python >= 3.11; numpy 1.26.4 has no Python 3.13 build). The project's code and saved models have only ever
been run on **Python 3.12.3**; Python 3.11 has **not** been tested.

**Open item (Python 3.13):** the project author reports running Phase 7 on **Python 3.13.14**. The pins above cannot be installed unchanged there (`numpy==1.26.4` has no Python 3.13 build), so the author's environment must contain some newer package versions than `requirements.txt` lists. The assistant has not seen that environment's `pip freeze`, has not tested any code on Python 3.13, and has not changed any pin. Saved models trained with scikit-learn 1.8.0 may print version warnings if a different scikit-learn is installed (the predictor reports these in `/health` under `ml_model.load_warnings`). Run `pip freeze > installed_versions.txt` on your machine and compare it with `requirements.txt`; if you want the pins updated to match what works for you, share that file.

A clean-environment `pip install -r requirements.txt` followed by `pip check` has **not been verified for the current
`requirements.txt`**. An earlier revision of this README stated that it had; that claim could not be confirmed and
predates the Phase 7 packages (`sentence-transformers`, `pypdf`), so it was removed. Please run both commands yourself
(see "Phase 7 -> Dependency check").

## Project Structure

```
fake-news-detector/
│
├── data/
│   ├── raw/                # original ISOT dataset files you download (not committed)
│   ├── processed/          # cleaned/split data written by the preprocessing step (not committed)
│   └── rag/
│       ├── sample_reference_docs/   # three FICTIONAL demonstration documents (committed)
│       └── reference_docs/          # put your own .txt / .pdf evidence documents here
│
├── notebooks/               # Jupyter notebooks for each analysis stage
│
├── models/                  # saved trained models (not committed)
│
├── src/
│   ├── preprocessing/        # text cleaning, tokenization
│   ├── features/             # TF-IDF, n-gram feature builders
│   ├── models/                # training/inference wrappers for ML models
│   ├── rag/                   # document loading, chunking, embeddings, FAISS index, retrieval (Phase 7)
│   └── credibility/            # credibility scoring engine
│
├── app/
│   ├── backend/               # FastAPI app
│   └── frontend/               # NewsCred Streamlit app (app.py, render.py, helpers.py, styles.css)
│
├── scripts/                 # rag_cli.py (build/search the FAISS index), cross_dataset_eval.py and threshold_tuning.py (evaluation)
│
├── tests/
│
├── .streamlit/              # NewsCred theme (config.toml)
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── main.py
```

## Windows Setup Instructions

Open **Command Prompt** or **PowerShell** in the folder where you extracted this project.

### 1. Verify Python

```bat
python --version
```

You should see `Python 3.11.x` or `Python 3.12.x`. If you see 3.13 or newer (or 3.10 or older), install Python 3.11 or 3.12 from python.org (check "Add Python to PATH" during install); the pinned `numpy==1.26.4` / `scikit-learn==1.8.0` will not install correctly otherwise.

### 2. Create the virtual environment

```bat
python -m venv .venv
```

### 3. Activate the virtual environment

Command Prompt:
```bat
.venv\Scripts\activate
```

PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks the script, run this once, then try activating again:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

You'll know activation worked when you see `(.venv)` at the start of your command line.

### 4. Upgrade pip

```bat
python -m pip install --upgrade pip
```

### 5. Install dependencies

```bat
python -m pip install -r requirements.txt
```

This may take a few minutes, especially for `faiss-cpu`, `xgboost`, and `sentence-transformers`.
`pyarrow==18.1.0` is pinned because Streamlit depends on PyArrow; PyArrow 26.0.0 failed to
import with the project's `numpy==1.26.4` pin, requiring NumPy 2.0 or newer.
For an already-created environment that has the incompatible PyArrow version, apply the pin
without changing NumPy:

```bat
python -m pip install pyarrow==18.1.0
```

### 6. Verify installation

```bat
python -c "import torch; import numpy, pandas, pyarrow, sklearn, xgboost, fastapi, streamlit, faiss, sentence_transformers, transformers, pypdf, dotenv; print('numpy', numpy.__version__, '| pandas', pandas.__version__, '| pyarrow', pyarrow.__version__, '| scikit-learn', sklearn.__version__, '| streamlit', streamlit.__version__); print('All packages imported successfully')"
python -m pip check
```

Import PyTorch first to initialize its Windows native libraries before the other ML packages.
The version output should include NumPy `1.26.4` and PyArrow `18.1.0`; the import command and
`pip check` should both complete successfully.

### 7. Set up environment variables

```bat
copy .env.example .env
```

Then open `.env` in a text editor. There are no secrets or API keys to fill in. On macOS/Linux use `cp .env.example .env` and `source .venv/bin/activate` instead of the Windows commands above.

### 8. Verify the base environment

```bat
python main.py
```

Expected output:
```
==================================================
Fake News Detection System - Environment Ready
==================================================
```

## Running the dashboard and API

The Phase 1 setup-test pages were replaced in Phase 8. See **Phase 8 -> Running the application** below for the exact commands.

Stop it with `Ctrl+C` in the terminal.

## Git Setup

```bat
git init
git add .
git commit -m "Phase 1: project setup and skeleton"
```

To connect to GitHub afterward:

1. Create a new **empty** repository on GitHub (do not initialize it with a README).
2. Run:

```bat
git remote add origin https://github.com/yourusername/fake-news-detector.git
git branch -M main
git push -u origin main
```

## Phase 2 — Dataset and Preprocessing

### Dataset used

**ISOT Fake and Real News Dataset.** Two CSV files:
- `Fake.csv` — fake news articles
- `True.csv` — real news articles

Each file contains: `title`, `text`, `subject`, `date`.

### Dataset acquisition

The dataset is **not included** in this repository or ZIP (licensing/size reasons — see
`.gitignore`, which excludes `data/raw/*`).

To obtain it:
1. Go to Kaggle and search for "Fake and Real News Dataset" (also listed as
   `clmentbisaillon/fake-and-real-news-dataset`), or use your own copy of the ISOT dataset.
2. Download `Fake.csv` and `True.csv`.
3. Place both files directly inside:
   ```
   data/raw/Fake.csv
   data/raw/True.csv
   ```
4. Do not rename the files — the loading code looks for these exact filenames.

### Label definition

| Label | Meaning |
|---|---|
| `1` | Fake news |
| `0` | Real news |

This convention (`LABEL_FAKE = 1`, `LABEL_REAL = 0`) is defined once in
`src/preprocessing/load_data.py` and used consistently throughout the project.

### Preprocessing steps

1. **Load & validate** (`src/preprocessing/load_data.py`): reads both CSVs, checks that
   `title`, `text`, `subject`, `date` are present, adds the `label` column, combines both
   files, and adds a unique `article_id` (MD5 hash of the article text) used later to detect
   duplicates and verify there's no overlap between data splits.
2. **Data quality check — before cleaning** (`src/preprocessing/quality_checks.py`): counts
   total/fake/real records, missing values, duplicate articles, empty articles, very short
   articles (<20 words), and article length statistics (words and characters).
3. **Cleaning, done in three explicit, separately-reported stages** (this order matters —
   see "Cross-class conflict handling" below for why):
   - **3a. Remove empty/missing text** (`remove_missing_or_empty_text`): drops rows whose
     text is missing (`NaN`) or empty/whitespace-only once normalized. The real ISOT dataset
     represents some missing article bodies as a single space `" "` rather than a true empty
     string, so this check normalizes whitespace before deciding something is "empty."
   - **3b. Remove cross-class conflicts** (`remove_cross_class_duplicates`): see below.
   - **3c. Remove ordinary duplicates** (`remove_duplicate_text`): drops exact-text repeats
     that share the *same* label (keeps the first occurrence). This uses **exact** string
     matching, not the normalized comparison used in 3a/3b — see "Why ordinary duplicate
     removal uses exact matching" below for why that distinction matters.
   - Does **not** drop "very short" articles automatically — that's a judgment call, so the
     report simply flags them for awareness.
4. **Light text cleaning** (`src/preprocessing/text_cleaning.py`): adds a new `clean_text`
   column with only genuine-noise removal — HTML tag/entity stripping, unicode
   normalization (NFKC), control-character removal, and whitespace collapsing. The original
   `text` column is **never modified**. We deliberately do **not** lowercase, strip
   punctuation, or remove stopwords here — those are model-specific choices tested in
   Phase 4, not blind defaults applied to a dataset we haven't explored yet.
5. **Data quality check — after cleaning**: same report, run again, to see the effect of
   step 3.

### Cross-class conflict handling

Before splitting, we explicitly check whether the same article text is labeled **fake in one
row and real in another** — a genuine labeling conflict we cannot resolve by guessing, so
every row carrying that ambiguous text is removed (from both classes), rather than
arbitrarily keeping one label.

This check:
- Runs in `remove_cross_class_duplicates()` (`src/preprocessing/quality_checks.py`), as its
  own step, separate from ordinary duplicate removal.
- Compares text using **normalized** matching (whitespace collapsed, casefolded) — safe here
  specifically because a genuine label conflict is a rare, high-stakes event worth catching
  even if it only differs by trivial whitespace/case.
- Runs **after** empty-text removal, so a whitespace-only "article" is never mistaken for a
  real conflict. This matters concretely: the real ISOT dataset's *one* raw cross-class
  overlap (a single-space `" "` body appearing in 626 fake rows and 1 real row) is pure
  filler, not a real article, and is correctly caught and removed as *empty text* in step 3a,
  before conflict detection ever runs — so `remove_cross_class_duplicates()` correctly
  reports **0 genuine conflicts** on this dataset.
- Reports two numbers: how many *rows* were removed, and how many *distinct* conflicting text
  values caused it (useful to distinguish "one conflicting article copied many times" from
  "many different conflicting articles").

### Why ordinary duplicate removal uses exact matching

While building this, normalized (whitespace/case-insensitive) matching was tested for
ordinary duplicate removal too, since it seemed like a natural extension. It was rejected
after testing showed it incorrectly merges legitimately **different** articles on the real
dataset — several short "video embed" articles share a near-empty body (e.g. literally just
`"Watch:"`) as a placeholder for embedded video content, with completely different headlines
and stories. Normalized matching would have deleted up to **9 rows** of genuinely distinct
articles as if they were duplicates of each other. Exact-text matching avoids this, at the
cost of occasionally missing a true near-duplicate that differs only by trivial whitespace
(see the `clean_text` note below).

### Data splitting strategy

- 70% train / 15% validation / 15% test.
- Stratified on `label` so the fake/real ratio is preserved in every split.
- Fixed random seed (`RANDOM_SEED = 42` in `src/preprocessing/split_data.py`) for
  reproducibility.

### Data leakage prevention

- Deduplication (including cross-class conflict removal) happens **before** the split, so
  the same article can never land in two different splits.
- `assert_no_overlap()` checks `article_id` across all three splits immediately after
  splitting — a **hard** check (raises an error if it fails).
- `assert_no_text_overlap_across_splits()` independently checks the raw `text` column too
  (not just the `article_id` hash) — also a **hard** check.
- `count_text_overlap_across_splits()` additionally checks `clean_text` — but as a **reported
  count, not a hard failure**. Why the difference: light cleaning (whitespace collapsing) can
  occasionally make two genuinely different raw rows collapse into identical `clean_text` if
  they differed only by trivial whitespace. On the real ISOT dataset this affects exactly
  **3 pairs (6 rows) out of 38,644** — all same-label, near-duplicate articles that are
  almost certainly the same story scraped/republished twice (e.g. one copy of an article
  ending in `"...Via: TMZ"` and another ending in `"...Via: TMZ "` with a trailing space).
  Because catching these via normalized matching was shown above to also risk deleting
  genuinely distinct short articles, this is surfaced as a diagnostic for awareness (and
  something to keep in mind in Phase 4, since a TF-IDF vectorizer would see these 6 rows as
  producing identical feature vectors) rather than silently forced into either "duplicate" or
  "not duplicate" without your input.
- The preprocessing pipeline also reports conservative cross-split near-duplicate candidates
  using word-trigram SimHash (Hamming distance <= 6). This is an audit heuristic: candidates
  are not automatically dropped, and zero candidates is not proof that no similar articles
  exist. Review candidates and record the source dataset/version when running on the real data.
- No parameters are "learned" from the data in Phase 2 (no vectorizer, no scaler yet), so
  there's nothing to accidentally fit on the wrong split at this stage. This matters starting
  Phase 4: any future TF-IDF vectorizer (or similar) must be fit **only** on `train.csv`,
  then applied transform-only to `validation.csv` and `test.csv`.

### Location of raw and processed data

```
data/raw/            <- put Fake.csv and True.csv here yourself (not committed)
data/processed/      <- train.csv, validation.csv, test.csv (generated, not committed)
```

### How to run Phase 2

1. Place `Fake.csv` and `True.csv` in `data/raw/` (see "Dataset acquisition" above).
2. From the project root, with the virtual environment active:
   ```bat
   python -m src.preprocessing.pipeline
   ```
   This prints data-quality reports before/after cleaning, the split sizes, and saves
   `train.csv`, `validation.csv`, `test.csv` to `data/processed/`.
3. Alternatively, open and run `notebooks/01_data_analysis.ipynb` for the same steps with
   more inspection (column types, missing values, class distribution, article length stats)
   along the way.
4. Run the tests:
   ```bat
   pytest tests/test_preprocessing.py -v
   ```

## Phase 3 — Exploratory Data Analysis

### What was analyzed

`notebooks/02_eda.ipynb` explores `data/processed/train.csv` (27,050 articles) in depth. Validation and
test sets are loaded **only** for the duplicate/overlap diagnostics section, to re-confirm Phase 2's
leakage guarantees on the actual saved files — never to explore patterns or influence feature/model choices.

### Major EDA categories

Dataset overview and class distribution; missing-value checks; article length (character/word/sentence
count); title analysis (length, capitalization, punctuation); subject/category analysis; date analysis;
word-frequency analysis (overall and per-class); class-specific word analysis (log-frequency ratio);
linguistic style comparison; outlier analysis (IQR-based); and duplicate/overlap diagnostics.

### Visualizations created

Class distribution bar chart; word-count histogram and box plot (fake vs. real); title word-count box
plot; subject-frequency count plot by class; article-count-over-time line chart (parsed dates only); top-20
word frequency bar charts (overall, fake, real); word-count and title-length outlier box plots. All charts
use Matplotlib/Seaborn with titles and axis labels, and a consistent fake=red / real=blue color convention.

### Important findings

- **Class balance:** 45.2% fake / 54.8% real in the training set — moderate imbalance, not severe.
- **Title style differs more than body style:** fake titles average ~14 words vs. ~10 for real titles,
  with a much higher uppercase-letter ratio (~35% vs. ~7%) and far more frequent exclamation marks (~11% of
  fake titles vs. ~0.1% of real titles).
- **Article body length** is only weakly different between classes (fake slightly longer on average), with
  heavily overlapping distributions.
- **`subject` is a perfect class separator** in this dataset — fake and real articles use completely
  non-overlapping subject taxonomies (confirmed via crosstab). This is a dataset-collection artifact, not a
  property of "fakeness," and is why `subject` is excluded as a model feature.
- **Dates parse successfully for 99.99%** of training articles (27,048/27,050); the 2 failures are URLs
  mistakenly stored in the `date` column in the original source data. Fake and real articles also span
  different, only partially overlapping date ranges — another reason `date` is excluded as a feature.
- **Embedded metadata artifacts:** the words most disproportionately common in fake articles are largely
  URL/image-credit fragments (`https`, `flickr`, `wikimedia`, `youtu`, Getty photographer surnames), not
  emotionally-charged language. Words most common in real articles are largely foreign political figures'
  names plus a Reuters-internal short-link fragment (`tmsnrt`) — both point to *data provenance* differences
  between the two source collections, not general linguistic properties of misinformation. This is the
  single most important caution to carry into Phase 4.
- **Duplicate diagnostics reconfirm Phase 2:** zero `article_id` or raw-text overlap between any splits; a
  small, already-documented `clean_text` overlap remains (trivial whitespace-only near-duplicates), not new
  leakage.
- **Outliers:** almost no unusually short articles remain (Phase 2 already filtered those), but a long tail
  of unusually long articles exists in both classes.

### Potential ML features (for Phase 4)

**Likely useful:** TF-IDF on `clean_text` (word n-grams), character n-grams, title-derived features (length,
uppercase ratio, punctuation counts), article length, body punctuation-density features.

**To avoid or use only with explicit caution:** `subject` and `date` (both act as near-perfect but
artifactual class separators — see findings above), raw URL/image-credit tokens surfaced in the
class-specific word analysis, and `article_id` (a dataset-internal bookkeeping identifier with no
relationship to content).

### Known dataset limitations

- Fake and real articles come from sources with non-overlapping `subject` categories and only partially
  overlapping date ranges — likely reflecting different collection processes, not inherent properties of
  fake vs. real news.
- Some fake articles retain embedded image-credit/URL text (a scraping artifact) that a naive model could
  learn to key on instead of genuine content signals.
- Article-length and sentence-structure differences between classes are real but modest; they are not, on
  their own, reliable fake-news indicators.
- All findings are specific to this dataset and should not be generalized as universal fake-news
  characteristics.

### How to run Phase 3

```bat
jupyter notebook notebooks/02_eda.ipynb
```
Requires `data/processed/train.csv`, `validation.csv`, and `test.csv` to already exist (see Phase 2).

## Phase 4 — Machine Learning Baseline

### Models implemented

All five candidate models were implemented and **successfully trained and evaluated** in this environment
(single CPU core, ~4 GB RAM):

| Model | Ran successfully? | Feature representation |
|---|---|---|
| Logistic Regression | Yes | TF-IDF (direct) |
| Multinomial Naive Bayes | Yes | TF-IDF (direct) |
| Linear SVM | Yes | TF-IDF (direct) |
| Random Forest | Yes | TF-IDF → TruncatedSVD (200 components) |
| XGBoost | Yes | TF-IDF → TruncatedSVD (200 components) |

Random Forest and XGBoost use a reduced representation (TruncatedSVD) rather than the raw ~27,050 × 30,000
sparse TF-IDF matrix converted to dense, which would need several GB just for the training matrix — not
practical on this hardware. The actual explained variance retained by 200 SVD components is reported
directly in `notebooks/03_ml_models.ipynb` (~15%), rather than assumed to be sufficient.

### Features

- **TF-IDF** on `clean_text`, word-level, `ngram_range=(1,2)`, `max_features=30000`, `min_df=3`, `max_df=0.9`
  (`src/features/tfidf_features.py`). Fit **only** on `train.csv`; `validation.csv`/`test.csv` are
  transform-only, enforced by the module's function design (separate `fit_*` vs. `transform_*` functions).
- **TruncatedSVD** (200 components), fit only on the training TF-IDF matrix, used solely for the two
  tree-based candidates above — not part of the final selected model's pipeline.
- `subject`, `date`, and `article_id` are **not** used as features (see "Dataset limitations" below).

### Evaluation

- **Train** → fit TF-IDF/SVD, train all five candidate models.
- **Validation** → model comparison, hyperparameter choices, error analysis, and final model **selection**.
- **Test** → touched exactly once, in `notebooks/04_model_evaluation.ipynb`, strictly *after* model
  selection was already finalized. No result from the test set was used to change the model, the
  vectorizer, or any hyperparameter.

**Validation results** (see `notebooks/03_ml_models.ipynb` Section 8 for the full table):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Train Time |
|---|---:|---:|---:|---:|---:|---:|
| Linear SVM | 0.9959 | 0.9966 | 0.9943 | 0.9954 | 0.9999 | 0.5s |
| Logistic Regression | 0.9876 | 0.9915 | 0.9809 | 0.9862 | 0.9992 | 0.6s |
| XGBoost (SVD) | 0.9838 | 0.9858 | 0.9782 | 0.9820 | 0.9984 | 7.2s |
| Random Forest (SVD) | 0.9646 | 0.9715 | 0.9496 | 0.9604 | 0.9940 | 43.2s |
| Naive Bayes | 0.9558 | 0.9563 | 0.9454 | 0.9508 | 0.9906 | 0.04s |

(Positive class = Fake (1) throughout, per `src/models/evaluate_models.py`.)

**Model selection:** Logistic Regression was selected — not the single highest scorer (Linear SVM scores
marginally higher), but the best balance of near-top validation performance, a **native `predict_proba`**
output (needed for the future credibility-scoring stage in Phase 8, whereas LinearSVC only offers an
uncalibrated `decision_function`), and direct coefficient interpretability (which is what made the dataset-
artifact finding below possible to demonstrate concretely). Full reasoning: `notebooks/03_ml_models.ipynb`
Section 14.

**Final test-set evaluation** (one-time, `notebooks/04_model_evaluation.ipynb`): Accuracy 0.9879, Precision
0.9927, Recall 0.9805, F1 0.9865, ROC-AUC 0.9982 — closely matching the validation numbers above, indicating
consistent generalization between these two held-out splits of the same dataset.

### Dataset limitations (carried forward from Phase 3, now confirmed in the trained model itself)

Inspecting the selected model's actual coefficients (`notebooks/03_ml_models.ipynb` Section 12) confirms
Phase 3's EDA concerns directly:
- Strongest **Fake**-associated terms are largely embedded image-credit/link boilerplate (`via`, `read
  more`, `image`, `featured image`, `com`, `getty`) — a scraping artifact, not sensational language.
- Strongest **Real**-associated terms are dominated by Reuters wire-service formatting (`reuters`,
  `washington reuters`, `said`, `said on`, day-of-week phrases) — a single publisher's house style, not a
  general marker of truthful reporting.

**This model's high accuracy should not be read as proof of general-purpose fake-news detection ability.**
It reflects, in part, this specific dataset's collection artifacts. This is exactly why the project's overall
architecture (per the original brief) treats this ML component as one input to a larger hybrid system —
combined with LLM-based contextual analysis and evidence retrieval in later phases — rather than as a
standalone verdict.

### Model artifacts

Saved to `models/` (the trained artifacts are included in the project ZIP; they can be regenerated by running
the notebooks if you have the dataset):
- `best_model.joblib` — the selected Logistic Regression model
- `tfidf_vectorizer.joblib` — the fitted TF-IDF vectorizer (fit on training data only)
- `model_metadata.json` — configuration, validation metrics, excluded features and why, and known
  limitations, all in one place

### Reusable modules

- `src/features/tfidf_features.py` — TF-IDF and SVD fit/transform functions (fit-only-on-train enforced by
  API design)
- `src/models/train_models.py` — training functions for all five candidates, each returning `(model,
  training_time)`
- `src/models/evaluate_models.py` — metrics, confusion matrix, classification report, and a
  `predict_proba`/`decision_function`-agnostic scoring helper for ROC-AUC
- `src/models/model_utils.py` — artifact save/load (joblib + JSON), reusing the project's existing
  `LABEL_FAKE`/`LABEL_REAL` convention from Phase 2

### How to run Phase 4

```bat
jupyter notebook notebooks/03_ml_models.ipynb
```
Trains all five models, compares them on validation, selects the final model, and saves artifacts to
`models/`. Then:
```bat
jupyter notebook notebooks/04_model_evaluation.ipynb
```
Loads the saved artifacts and reports the one-time final test-set evaluation.
```bat
pytest tests/test_ml_models.py -v
```
Runs the fast unit tests (small fixtures, not the full dataset).

## Phase 5 — Inference Pipeline & Application Readiness

### Final model

**Logistic Regression + TF-IDF** (validation F1 0.9862, ROC-AUC 0.9992; one-time test F1 0.9865, ROC-AUC 0.9982).

Linear SVM scored higher on every validation metric (F1 0.9954; it would miss roughly 15 Fake articles vs 50 for
Logistic Regression on the validation set). Logistic Regression was kept because it gives a native probability
(needed for Phase 8 credibility scoring), is already trained/saved/test-evaluated, and is interpretable. Full
reasoning and numbers: `notebooks/05_inference_pipeline.ipynb` Section 4. Switching to Linear SVM (+ calibration)
is possible but would require the dataset and a retrain.

### Saved artifacts (in `models/`, created in Phase 4, verified in Phase 5)

| File | Contents |
|---|---|
| `best_model.joblib` | fitted Logistic Regression |
| `tfidf_vectorizer.joblib` | TF-IDF vectorizer fitted on the training set only |
| `model_metadata.json` | configuration, validation metrics, limitations |

All three are loaded together; the loader refuses to start if they disagree (vocabulary size, n-gram range,
model class, label convention).

### Input requirements

- A single string: the article **body** text (the model was trained on `clean_text` of the body; titles were not used).
- Best results with full-length articles. Inputs under 20 words get a warning. Text with no words in the model's
  vocabulary (e.g. non-English) returns `Undetermined`.
- Empty/blank/non-string input raises `ValueError`.

### Prediction process

```
article text -> clean_single_text() -> TF-IDF transform -> Logistic Regression -> prediction dict
```

The cleaning function is the same one Phase 2 used to build `clean_text`, so inference matches training
(verified by `tests/test_inference.py`).

### Using it

```python
from src.models.predict import load_predictor

predictor = load_predictor()                       # load once
result = predictor.predict(article_text, explain=True)
print(result["label"], result["fake_probability"], result["confidence_level"])
```

```bat
python -m src.models.predict --file article.txt --explain
python -m src.models.predict --text "Article text here" --json
```

Result fields: `label` (Fake / Real / Undetermined), `label_id`, `fake_probability`, `real_probability`,
`confidence`, `confidence_level` (display band only, not calibrated), `n_words`, `n_known_terms`, `warnings`,
`model_name`, `disclaimer`, and (with `explain=True`) `top_terms`. The dict is JSON-serialisable, so it can be
returned directly from a future API endpoint. `predictor.model_info()` returns model details for a UI.

### Example (hand-written test article, not from the dataset)

Input: `WASHINGTON (Reuters) - The U.S. Senate on Tuesday passed a spending bill ...` -> **Real**, P(Fake) = 0.0006;
strongest terms toward Real: `reuters`, `washington reuters`, `on tuesday`.

### Testing

- `notebooks/05_inference_pipeline.ipynb` — artifact inspection, model selection, pipeline walkthrough, 8 test
  inputs, edge cases, simulated API request/response.
- `python -m unittest tests.test_inference -v` (or `pytest tests/test_inference.py -v`) — 13 tests against the real
  saved artifacts. They skip automatically if `models/` is empty.

### Limitations seen during Phase 5 testing

On hand-written test articles, typical-style examples matched the intended label (4 of 4), but two adversarial
examples did not: an invented false claim written in Reuters style was classed **Real despite a raw model output of
99.8% toward Real**, and a factual weather report carrying an image credit was classed Fake. That 99.8% was not
calibrated and was not a probability that the claim was true. This agrees with the Phase 4 finding that the
model partly reads source formatting rather than truthfulness. Two cases are illustrative, not a statistical test.
Probabilities are not calibration-tested, and very short inputs can still produce confident output (a warning is
attached).

### Environment note & Phase 5 Fix

The saved artifacts (`models/best_model.joblib` and `models/tfidf_vectorizer.joblib`) were pickled with scikit-learn **1.8.0**, whereas `requirements.txt` previously pinned **1.5.2**.
This dependency mismatch was resolved by updating `requirements.txt` to the exact pin **`scikit-learn==1.8.0`**. Compatibility was verified by successfully loading the saved model and vectorizer with zero version mismatch warnings, running inference predictions, and passing the automated test suites on Python 3.12.3 (Python 3.11 was not tested).

**Open risk (unverified) - numpy version vs. the saved vectorizer.** The saved TF-IDF vectorizer was pickled under
**numpy 2.x**: its `vocabulary_` stores numpy integers through `numpy._core.multiarray`, a module path that only exists in
numpy 2. `requirements.txt` pins `numpy==1.26.4`. The saved model and vectorizer have been verified to load and predict
only in an environment running numpy 2.4.4 with scikit-learn 1.8.0, **not** under the pinned numpy 1.26.4 (that version
could not be installed offline). Check this right after installing the requirements:

```bash
python -c "import joblib; joblib.load('models/tfidf_vectorizer.joblib'); print('vectorizer loads OK')"
python -m pytest tests/test_inference.py -v
```

If it fails with `No module named 'numpy._core'`, either (a) move to a numpy 2.x pin after confirming that `faiss-cpu`
and the other pins accept it (and update the pin assertion in `tests/test_phase7_regression.py` deliberately), or (b) re-run
`notebooks/03_ml_models.ipynb` locally to regenerate the artifacts under your installed versions. The pin was left
unchanged because neither option could be tested here and the pin is asserted by an existing test.

## Phase 6 (removed) — Llama 3.1 8B integration

Phase 6 originally added a local **Llama 3.1 8B** model (served by Ollama) that summarised articles, extracted factual
claims and flagged sensational language. It has been **removed from the project** (code, API field, dashboard card,
tests and configuration) because the model does not fit in the memory of the machine used for development and demos.

What changed as a result:

* `src/llama/`, `scripts/test_ollama.py`, `tests/test_llama_analysis.py` and `notebooks/05_llama_analysis.ipynb` were deleted, and the `OLLAMA_*` settings are gone from `.env.example`.
* **Claims for evidence retrieval** now come from the request (`claims`), or, when none are given, from a simple **rule-based sentence selector** (`src/rag/claim_selector.py`: sentences with figures, capitalised names or assertion verbs). It is a heuristic, not language understanding, and can pick poor sentences.
* **The overall assessment** now rests on the ML prediction alone. With a single component it can only say `warning_signs_detected` (ML predicted Fake) or `insufficient_information`; a "Real" prediction never produces a reassuring signal. This is deliberate, because the cross-dataset evaluation below shows the classifier loses accuracy on unfamiliar sources.
* `POST /analyze` no longer accepts `run_llama`, `/health` no longer reports `ollama`, and the response has no `llama_analysis` component.

Mentions of Llama in the older "verification status" notes below describe the project as it was when those checks were run.

## Evaluation (cross-dataset results)

The ISOT-trained TF-IDF + Logistic Regression model was evaluated, **without retraining**, on 71,321 scorable articles from
WELFake (body text only, matching the training input; 783 empty rows dropped, 30 unscorable). Reproduce with
`python scripts/cross_dataset_eval.py` (check labels first with `--preview-only`). The dashboard's **Evaluation** tab shows the saved results.
The complete provenance, per-class metrics, leakage/calibration status, limitations, and recommendations are in
[docs/evaluation_report.md](docs/evaluation_report.md). The checked-in JSON represents a prior run; this checkout does
not include the raw WELFake CSV or processed ISOT splits, so these results cannot currently be regenerated here.

From the repository root, after obtaining WELFake and placing it at `data/raw/WELFake_Dataset.csv` (or supplying
`--input-csv PATH`), run the reproducible evaluation commands:

```powershell
python scripts/cross_dataset_eval.py --preview-only
python scripts/cross_dataset_eval.py
python scripts/threshold_tuning.py
```

The scripts keep the saved classifier unchanged, report confusion and per-class metrics, and add Brier/ECE
probability diagnostics. Those diagnostics do not calibrate the model. To audit ISOT duplicates and splits, place
`Fake.csv` and `True.csv` under `data/raw/`, then run `python -m src.preprocessing.pipeline`.

| Dataset | Accuracy | Precision | Recall | F1 | ROC-AUC | Predicted Fake |
|---|---|---|---|---|---|---|
| ISOT validation (in-domain) | 98.76% | 99.15% | 98.09% | 98.62% | 0.9992 | n/a |
| WELFake (external) | 84.74% | 78.88% | 95.61% | 86.44% | 0.9231 | 61.7% (true rate 50.9%) |

Accuracy fell by about 14 points. On WELFake the model flagged 9,291 real articles as Fake (26.5% of real articles) and missed 1,593 fake ones (4.4%).

**Threshold tuning does not fix it** (`python scripts/threshold_tuning.py`). The decision threshold was chosen on one half of
WELFake and all numbers below come from the other, held-out half (35,661 rows, seed 42):

| Setting | Threshold | Accuracy | Balanced acc. | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Default | 0.50 | 84.56% | 84.36% | 78.68% | 95.54% | 86.30% |
| Tuned (best F1) | 0.56 | 84.89% | 84.72% | 79.77% | 94.19% | 86.38% |
| Tuned (best balanced accuracy) | 0.65 | 85.19% | 85.07% | 81.69% | 91.36% | 86.26% |

Tuning recovers only 0.3 to 0.6 accuracy points, so the drop comes from features that transfer imperfectly, not from the cutoff.

**Limitations of this evaluation:** WELFake is a compilation of public sources and may partly overlap with ISOT-style sources, so the figures may overestimate true out-of-domain performance; WELFake labels are noisy; both tuning halves are WELFake, so tuned results say nothing about a third dataset; and one external dataset does not establish general robustness. The classifier recognises writing style and wording, not factual truth.

## Phase 7 — FAISS + RAG Evidence Retrieval

### Objective

Given a claim (supplied by the user, or selected from the text by simple rules), retrieve the most relevant passages from a collection of **reference documents you supply**, and return them with their source information so a later phase can use them.

**What Phase 7 does NOT do:** it does not decide whether a claim is true or false, does not verify anything, produces no credibility score and no combined verdict (that is Phase 8 and later). **A retrieval score is a cosine similarity between text embeddings. It is not a probability of truth, and a highly similar passage may support, contradict or merely mention the claim.** The ISOT training dataset is deliberately *not* used as an evidence source: training examples are not independent verification evidence.

### Architecture

```
reference docs (.txt / .pdf)                          claim (from Phase 6, or typed in manually)
        |                                                         |
  document_loader.py   -> pages + metadata                        |
  text_chunker.py      -> overlapping chunks (+ metadata)         |
  embeddings.py        -> normalized vectors  <------ same model ------> embeddings.py
  faiss_index.py       -> FAISS IndexFlatIP + chunks.jsonl + manifest.json
                                      |                           |
                                      '------ retriever.py <------'
                                                |
                          top-k passages (text, source, page, chunk_id, rank, score)
                                                |
                          evidence_pipeline.py: article -> claims -> evidence
                          (ML prediction and evidence are returned SEPARATELY)
```

| File | Role |
|---|---|
| `src/rag/config.py` | defaults (paths, chunk size/overlap, top-k, similarity threshold, model name), overridable via `.env` |
| `src/rag/document_loader.py` | reads `.txt` and `.pdf`; keeps filename, source id, page numbers (PDF), content hash; skips empty/unreadable/unsupported files with a reason |
| `src/rag/text_chunker.py` | sentence-aware overlapping chunks; never crosses PDF pages; configurable size/overlap |
| `src/rag/embeddings.py` | `SentenceTransformerEmbedder` (local model, L2-normalized output); `HashingEmbedder` (word-overlap, tests/demos only) |
| `src/rag/faiss_index.py` | build / save / load / validate / incrementally rebuild the index |
| `src/rag/retriever.py` | `EvidenceRetriever`: top-k, threshold, de-duplication, per-claim result and status |
| `src/rag/evidence_pipeline.py` | `gather_evidence_for_article`: Phase 6 claims -> evidence, outputs kept separate |
| `scripts/rag_cli.py` | `build` and `search` commands |
| `data/rag/sample_reference_docs/` | **synthetic, fictional** demo documents (not real evidence) |
| `notebooks/06_rag_evidence_retrieval.ipynb` | step-by-step demonstration |

### Installation

```bash
pip install -r requirements.txt
```

Phase 7 adds `sentence-transformers==3.3.1` (which installs PyTorch and `transformers`) and `pypdf==5.9.0`; `faiss-cpu==1.9.0` was already pinned. Existing pins (`numpy==1.26.4`, `scikit-learn==1.8.0`, ...) were not changed. On a machine without a GPU you can avoid the large CUDA download by installing the CPU build of PyTorch first, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cpu`, then `pip install -r requirements.txt`.

> **Verification status of the dependency pins:** `sentence-transformers==3.3.1` and `pypdf==5.9.0` were chosen for Python 3.11/3.12 compatibility, but the Phase 7 environment used to write this code had **no internet access**, so a clean `pip install -r requirements.txt` and `pip check` were **not** run for the new packages. Please run them on your machine; if `pip` reports a conflict, tell me the message and the pin can be adjusted.

### Dependency check

```bash
python -m pip check
python -c "import torch; import numpy, pyarrow, pandas, streamlit, sklearn, faiss, sentence_transformers, transformers; print('numpy', numpy.__version__, '| pyarrow', pyarrow.__version__, '| pandas', pandas.__version__, '| streamlit', streamlit.__version__, '| scikit-learn', sklearn.__version__, '| faiss', faiss.__version__, '| sentence-transformers', sentence_transformers.__version__, '| torch', torch.__version__, '| transformers', transformers.__version__)"
python -m pytest tests/test_rag_real_dependencies.py -v -rs
```

The RAG integration tests use the actual `faiss-cpu` library and actual Sentence Transformers model. The
first model use may download the configured model from Hugging Face; later runs can use the local cache.
They skip if those dependencies or model files are unavailable; a skip is unverified, not a pass.

### Embedding model setup

Default model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional, about 90 MB, Apache-2.0 licence, runs on CPU). No API key and no paid service is involved.

* **First run needs internet once** to download the model from the Hugging Face Hub into your local cache (`~/.cache/huggingface`). After that it works fully offline. To forbid network access afterwards, set `HF_HUB_OFFLINE=1` or pass `--offline` to the CLI.
* Pre-download explicitly: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"`
* Use another local model with `RAG_EMBEDDING_MODEL` in `.env`. The index records the model name and **refuses to load for a different model**, so rebuild after switching.

### Adding reference documents

Put `.txt` and `.pdf` files in `data/rag/reference_docs/` (sub-folders are included). Page numbers are recorded for PDFs (the physical page position in the file, not the printed page label). Empty files, scanned PDFs without a text layer, encrypted or corrupted PDFs, and unsupported file types are skipped and reported, not fatal. Use documents you trust: retrieval can only be as reliable as the collection.

To try the pipeline without your own documents, use `data/rag/sample_reference_docs/` (synthetic; every file says so at the top).

### Building and rebuilding the index

```bash
python scripts/rag_cli.py build                         # data/rag/reference_docs -> data/rag/index
python scripts/rag_cli.py build --docs data/rag/sample_reference_docs --index data/rag/demo_index
python scripts/rag_cli.py build --force                 # re-embed everything
```

or from Python:

```python
from src.rag import SentenceTransformerEmbedder, build_or_update_index
result = build_or_update_index("data/rag/reference_docs", "data/rag/index", SentenceTransformerEmbedder())
print(result.status, result.index.size, result.load_report.skipped)
```

Behaviour: no index -> `created`; unchanged documents/settings/model -> `unchanged` (nothing re-embedded); anything changed -> `rebuilt` (embeddings of unchanged chunks are reused, only new/changed chunks are embedded); a corrupted existing index is rebuilt with a warning. A rebuild writes to a temporary folder and swaps it in, so a failed rebuild leaves the previous index intact. If the documents folder does not exist the call fails and the existing index is untouched. A folder with no usable text produces an empty index (with a warning), which retrieval handles without crashing.

### Searching for evidence

```bash
python scripts/rag_cli.py search "The Alder River levee was built in 1990." --top-k 3
python scripts/rag_cli.py search "..." --json
```

```python
from src.rag import EvidenceRetriever, SentenceTransformerEmbedder
retriever = EvidenceRetriever.from_directory("data/rag/index", SentenceTransformerEmbedder(), top_k=5, min_score=0.35)
result = retriever.retrieve("Some claim")          # one claim
results = retriever.retrieve_many(["claim 1", "claim 2"])   # independent per claim
```

Result for each claim:

```json
{
  "claim": "...",
  "retrieved_evidence": [
    {"text": "<stored passage>", "source": "file.txt", "source_id": "file.txt", "page": null,
     "chunk_id": 3, "rank": 1, "score": 0.0, "score_type": "cosine_similarity"}
  ],
  "retrieval_status": "evidence_retrieved",
  "limitations": ["Retrieval relevance does not establish factual correctness.", "..."],
  "parameters": {"top_k": 5, "min_score": 0.35, "score_type": "cosine_similarity"}
}
```

(`score` above is a placeholder; real values are computed per query.) `retrieval_status` is one of:

| Status | Meaning |
|---|---|
| `evidence_retrieved` | at least one passage met the similarity threshold |
| `no_relevant_evidence` | nothing reached the threshold (or the collection is empty); says nothing about whether the claim is true |
| `retrieval_unavailable` | no/corrupted/mismatched index, or embedding failed; the reason is in `error` |
| `invalid_claim` | the claim was empty or not a string |

`page` is `null` for `.txt` sources. Source, page and chunk id come from stored metadata only; no URLs, dates or page numbers are ever invented. Near-identical passages (word-set Jaccard >= 0.90) are returned once. **The default `min_score=0.35` is an uncalibrated starting point for `all-MiniLM-L6-v2`: tune it on your own documents.**

### From a news article to claims and evidence

```python
from src.rag import gather_evidence_for_article
from src.models.predict import load_predictor         # optional

output = gather_evidence_for_article(article_text, retriever, predictor=load_predictor(),
                                     article_metadata={"title": "...", "source_name": "..."})
# or with your own claims:  gather_evidence_for_article(article_text, retriever, manual_claims=["claim"])
```

The result keeps `article` (text + the metadata you supplied, unchanged), `ml_prediction` and `claims` (evidence per claim) as **separate** entries and never merges them into one prediction. Claims come from `manual_claims` when given; otherwise a rule-based sentence selector picks candidate sentences (`claim_source` is then `"heuristic"`). No component is asked to write or fill in evidence.

### Saving and reloading the index

An index is a folder with three plain files (no pickle): `index.faiss` (exact inner-product FAISS index), `chunks.jsonl` (text and metadata for each vector; line *N* belongs to vector *N*) and `manifest.json` (model name, dimension, chunk settings, per-document hashes, SHA-256 checksums). `build_or_update_index` saves automatically; load manually with `EvidenceIndex.load(dir, expected_model_name=...)`. Loading verifies checksums, vector/chunk counts, dimension and the vector-ID mapping, and raises `IndexNotFoundError`, `IndexCorruptedError` or `IndexMismatchError`. Checksums catch accidental damage, not a deliberate attacker who can edit both files; FAISS's reader is not hardened against malicious files, so **only load indexes you built yourself**.

### Running the tests

```bash
python -m pytest tests/ -v -rs                   # everything (Phases 2-7); -rs prints the reason for every skip
python -m pytest tests/test_rag_real_dependencies.py -v -rs   # real FAISS + real embedding model only
```

Three kinds of Phase 7 test exist, and it matters which one you are looking at:

* `tests/test_rag_ingestion.py`, `test_rag_retrieval.py`, `test_rag_pipeline.py` use the real **FAISS** library but a small
  deterministic *lexical* stand-in embedder (`HashingEmbedder`) and mocked predictor objects. They need no model
  download. Both retrieval/pipeline modules **skip entirely if `faiss-cpu` is not installed**.
* `tests/test_rag_real_dependencies.py` (13 tests) uses the real FAISS **and the real `all-MiniLM-L6-v2` model**: embedding
  shape/normalisation/determinism, semantic ranking on the sample documents, vector-ID-to-chunk mapping, save/reload,
  empty/missing/corrupted/mismatched indexes, rebuild on document change, an unrelated claim returning no evidence, and the
  documented CLI commands including `--offline`. Without `faiss`, `sentence-transformers` or the model, each test is skipped
  and labelled **UNVERIFIED**.
* The old `test_ml_models.py` XGBoost test skips when XGBoost is not installed.

**Expected result on a machine with everything installed and internet on the first run:** 161 tests, 0 skipped (or 1 skipped
if XGBoost is not installed). **Any other skip means something is still unverified.** If the unrelated-claim test fails, its
message prints the best score it saw - that means `RAG_MIN_SCORE` needs tuning for your documents, not that retrieval is broken.

### Verification status (what was actually run)

*Historical log: written before Llama/Ollama was removed, so mentions of them describe the project as it was at the time.*

Both verification passes so far ran in sandboxes with **no internet access**: PyPI and the Hugging Face Hub returned HTTP 403, so
`faiss-cpu`, `sentence-transformers`, `torch` and `pytest` could not be installed and no embedding model was available.
Honest summary of the most recent pass:

| Check | Status |
|---|---|
| Full suite, real installed libraries (Phase 1-6 tests, `pypdf` and Phase 7 parts not needing FAISS) | **101 passed, 0 failed**, 60 skipped (run through a local pytest-compatible runner; real `pytest` was unavailable) |
| The 60 skips | 46 existing RAG tests that need `faiss` (34 retrieval + 12 pipeline), 13 new real-library tests, 1 XGBoost-optional test |
| Phase 5 saved model + vectorizer load and predict; Phase 6 analyzer; ML / Llama / evidence outputs stay separate | **Verified** (Phase 6 notebook also executed cleanly offline). The Phase 6 -> Phase 7 handoff was exercised with the real predictor, the real analyzer validation and the real retriever code (Ollama HTTP mocked): manual claims, blank/duplicate claims, Ollama down, malformed Llama JSON, missing index and no retriever all degrade to a status instead of raising - 17/17 checks |
| README CLI commands | **Verified:** every documented command parses against the real argument parser, and `build` / `search` fail with a clear message (exit code 1 / status `retrieval_unavailable`) when FAISS is missing |
| Notebooks | All 7 are structurally valid and every code cell compiles. `05_llama_analysis` ran 5/5 cells without error. `06_rag_evidence_retrieval` runs through document loading, chunking and its labelled lexical-embedder fallback, then stops at index creation with `faiss-cpu is not installed` (expected here) |
| **Real FAISS** index build / add / search / save / reload / empty / corrupted / rebuild / score meaning | **UNVERIFIED.** Written to the real FAISS API and covered by 46 existing tests plus the new file, but never executed against the real library. An earlier pass ran the 46 tests against a NumPy stand-in, which checks this project's logic, not FAISS |
| **Real embedding model** (`all-MiniLM-L6-v2`) load, 384 dimensions, normalisation, retrieval quality, `RAG_MIN_SCORE=0.35` default | **UNVERIFIED.** The 0.35 threshold is a heuristic that has never been tried on real embeddings |
| Model works offline after first download (`--offline`) | **UNVERIFIED** (covered by the CLI test in the new file) |
| Clean `pip install -r requirements.txt` + `pip check` (including `faiss-cpu==1.9.0` + `sentence-transformers==3.3.1` + `numpy==1.26.4` together) | **UNVERIFIED** |
| Saved vectorizer loads under the pinned `numpy==1.26.4` | **UNVERIFIED - open risk**, see "Phase 5 -> Environment note" |
| Live Ollama / Llama call; real `pytest` | **Not run** |
| Python 3.11 | **Not tested** |

The suite was run on numpy 2.4.4 and pandas 3.0.2 (what the sandbox had), **not** the pinned 1.26.4 / 2.2.3; scikit-learn was
1.8.0 as pinned.

**To finish verification on your machine** (needs internet once, for the model download):

```bash
pip install -r requirements.txt
pip check
python -m pytest tests/ -v -rs
python scripts/rag_cli.py build --docs data/rag/sample_reference_docs --index data/rag/demo_index
python scripts/rag_cli.py search "The Alder River levee was built in 1990." --index data/rag/demo_index --top-k 3
```

then re-run `notebooks/06_rag_evidence_retrieval.ipynb` (its outputs are cleared on purpose). If something fails against real FAISS,
the most likely place is `src/rag/faiss_index.py` (the serialize / deserialize / reconstruct calls).

### Known limitations

* Similarity is not support: a passage can contradict a claim and still score high. Judging support/contradiction is a later step.
* Retrieval only knows your reference documents; a claim with no coverage returns `no_relevant_evidence`, which does not mean the claim is true or false.
* Chunking is character-based and sentence-aware, not perfect: abbreviations can occasionally split sentences, and overlap fragments may start mid-sentence.
* Scanned PDFs need OCR first (not included). Only `.txt` and `.pdf` are supported.
* Exact (flat) search is ideal for small/medium collections; very large collections would need an approximate FAISS index.
* The index is not encrypted or signed (see the loading warning above).
* The similarity threshold and de-duplication limit are heuristics.
* Phase 7 itself adds no frontend, API endpoint, credibility score or verdict (the Phase 8 API/dashboard only *display* retrieved passages).

## Phase 8 — Final Integration (FastAPI backend + Streamlit dashboard)

### What it does

One request runs two independent components and returns their outputs **separately**, followed by a cautious summary:

| Component | Source | Output |
|---|---|---|
| ML prediction | saved TF-IDF + Logistic Regression (`models/`, Phases 4-5) | label (Fake/Real), uncalibrated Fake/Real scores and score band, influential words |
| Evidence retrieval | FAISS index + `all-MiniLM-L6-v2` (Phase 7) | for each claim, the most similar reference passages with source, passage id, cosine similarity |
| Final cautious assessment | rule-based, in `src/credibility/assessment.py` | one of four qualitative signals + observations + what cannot be concluded |

**What it deliberately does NOT do:** no credibility percentage, no "true/false/verified" label for any claim, no merging of the outputs into one score. Retrieved passages are *similar text*, not proof, and they never change the overall signal. The synthesis only reads the ML label; because only one component feeds it, it can never give a reassuring signal (a "Real" prediction yields `insufficient_information`).

### Architecture

```
Streamlit dashboard (app/frontend/app.py)  --HTTP-->  FastAPI (app/backend/main.py)
  app/frontend/helpers.py                                   |
                                              AnalysisService (src/credibility/service.py)
                                   .---------------+----------------+
                                   |               |                |
                       FakeNewsPredictor   EvidenceRetriever   build_final_assessment
                       (src/models, P5)    (src/rag, P7)        (src/credibility)
```

* `src/credibility/service.py`: framework-independent orchestration (this is where the logic and most tests live).
* `app/backend/main.py`: thin HTTP layer, `create_app(service)` factory, concise error handlers (no stack traces, submitted input never echoed back).
* Components load **once** and are reused. A failed load is retried on the next request, and the FAISS index is reloaded automatically if it is rebuilt, so fixing a problem does not require restarting the API.
* Each component can fail independently; the others still run (**partial result**, `"partial": true`). Raw exception text goes to the server log only.
* Order per request: ML -> retrieval (claims come from the request or the rule-based sentence selector).

### Prerequisites

* Everything in `requirements.txt` (FastAPI, Uvicorn, Streamlit, httpx are pinned there; no paid service or API key is used).
* The trained model files in `models/` (`best_model.joblib`, `tfidf_vectorizer.joblib`, `model_metadata.json`; included in the project ZIP).
* A FAISS index (see below). The embedding model `sentence-transformers/all-MiniLM-L6-v2` must have been downloaded once (the first index build or search does it; afterwards it is cached and works offline).
* Python: see "Supported Python version" above, including the Python 3.13 open item.

### RAG index creation

The reference documents are the project's **synthetic, fictional** demo documents (`data/rag/sample_reference_docs/`). Every passage retrieved from them is labelled as fictional demonstration data in the dashboard.

```bash
# from the project root, in Git Bash, with the virtual environment active
python scripts/rag_cli.py build --docs data/rag/sample_reference_docs --index data/rag/demo_index
```

Tell the API where the index is, either way:

* put `RAG_INDEX_DIR=./data/rag/demo_index` in your `.env` (copy `.env.example` to `.env` first), **or**
* leave `RAG_INDEX_DIR` unset entirely: if `data/rag/index` does not exist but `data/rag/demo_index` does, the API uses the demo index (`/health` shows which index and whether it is fictional demo data).

For your own real reference documents see "Phase 7 -> Adding reference documents" (index them into `data/rag/index`).

### Running the application (Git Bash on Windows)

Open **two** Git Bash windows and, in each, go to the project folder (adjust the path):

```bash
cd /c/path/to/fake-news-detector
source .venv/Scripts/activate        # your virtual environment
```

**Terminal 1: API**

```bash
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2: dashboard**

```bash
python -m streamlit run app/frontend/app.py
```

Streamlit opens `http://localhost:8501`. The dashboard needs the API from Terminal 1 to be running; if it is not, the page says so and offers a **Retry connection** button. API docs: `http://127.0.0.1:8000/docs`. Stop either server with `Ctrl+C`. The first analysis is slow (the embedding model is loaded on first use).

### The NewsCred dashboard (Streamlit frontend)

The Streamlit page is branded **NewsCred** and has three sections: **Analyze News**, **How It Works** and **About**.

**Analyze News** workflow: paste a headline or article (or pick one of the three built-in examples), then press **Analyze News**. Optional settings are collapsed under **Advanced options**: retrieve evidence passages, passages per claim (1-5) and optional claims to search for. The **Evaluation** tab shows the saved cross-dataset and threshold-tuning results from `results/`. The defaults work without configuring anything.

Results appear as three separate cards, each filled only from what `POST /analyze` returned: **A** machine learning prediction (label, uncalibrated scores and score band), **B** selected claims and retrieved passages with source/page/similarity metadata plus an explicit noncommittal evidence interpretation, and **C** overall assessment (the cautious, non-verdict summary from the API). The API's additive `credibility_assessment` object keeps machine prediction, claims/evidence, interpretations, and the overall summary in separate predictable fields. A score or label is never derived or invented by the page; a missing value is shown as `n/a`.

**API address:** the page reads `BACKEND_URL` (default `http://127.0.0.1:8000`, or built from `BACKEND_HOST`/`BACKEND_PORT`), from the environment or `.env`. It calls only `GET /health` (service status line, on page load and on Retry) and `POST /analyze`. If the API cannot be reached the page shows a plain-language message, a **Retry connection** button and a **Troubleshooting details** section with the address it tried, the start command and the technical error. It never shows demo results in place of a failed call.

Frontend files: `app/frontend/app.py` (page), `render.py` (HTML builders; all API/LLM text is HTML-escaped there), `helpers.py` (API client), `styles.css` (theme), and `.streamlit/config.toml` (colours; Streamlit must be started from the project root so it is picked up). Tests: `tests/test_newscred_frontend.py`.

### API testing examples

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The Alder River levee in Brightwater was completed in 1990."}'

# Search for your own claims:
curl -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" \
  -d '{"text": "Some headline", "claims": ["Library membership is free for all residents."], "top_k": 2}'
```

Request fields: `text` (required, max 20,000 characters), `claims` (optional list), `run_rag` (default true), `top_k` (1-10, default 3).

`GET /health` returns `{"api": "ok", "status": "ok" | "degraded" | "unavailable", "components": {"ml_model": ..., "rag": ...}}` (always HTTP 200 while the API itself is up).

`POST /analyze` returns:

```json
{
  "input": {"char_count": 0, "word_count": 0, "input_kind": "headline_or_short_text | article"},
  "components": {
    "ml_prediction":  {"status": "ok|unavailable|error", "result": {"label": "...", "fake_probability": 0.0, "...": "..."}, "error": null, "notice": "..."},
    "evidence":       {"status": "ok|unavailable|error|skipped", "claim_source": "manual|input_text|heuristic|none",
                       "claims": [{"claim": "...", "retrieval_status": "evidence_retrieved|no_relevant_evidence|retrieval_unavailable|invalid_claim",
                                   "retrieved_evidence": [{"text": "...", "source": "file.txt", "page": null, "chunk_id": 0, "rank": 1,
                                                           "score": 0.0, "score_type": "cosine_similarity", "is_synthetic_demo": true}]}]}
  },
  "component_status": {"ml_prediction": "ok", "evidence": "ok"},
  "partial": false, "warnings": [],
  "final_assessment": {"overall_signal": "...", "headline": "...", "observations": [], "cannot_conclude": [], "suggested_next_steps": [], "is_verdict": false},
  "credibility_assessment": {
    "machine_learning": {"status": "ok", "predicted_label": "...", "fake_probability": 0.0,
                         "real_probability": 0.0, "probabilities_calibrated": false, "explanation": "..."},
    "article_claims": {"status": "ok", "claim_source": "heuristic",
                       "items": [{"claim": "...", "retrieval_status": "...", "passages": [
                         {"source_name": "file.txt", "text": "...", "metadata": {"page": null}}
                       ], "evidence_interpretation": {"classification": "unclassified", "reason": "..."}}],
                       "empty_state": null},
    "overall": {"summary": "...", "model_observation": "...", "evidence_observation": "...",
                "limitations": [], "limitation_statement": "...", "is_verdict": false}
  },
  "limitations": [], "disclaimer": "..."
}
```

(Values above are placeholders showing the shape only; real values are computed per request.) Errors: empty/too-long input -> HTTP 422 `{"error": "invalid_input", "message": "..."}`; malformed request -> 422 `{"error": "invalid_request", "problems": [...]}`; unexpected server failure -> 500 `{"error": "internal_error", ...}` with no details.

`credibility_assessment` is additive; the existing `components` and `final_assessment` fields remain for compatibility.
Retrieved passages have source names and only metadata available in the index. Evidence is `unclassified` when passages
exist because similarity alone does not determine support or contradiction; missing/empty/unavailable results are
`inconclusive`. Model probabilities are explicitly uncalibrated.

**Which claims are searched:** claims supplied in the request; otherwise, for a short headline (<= 40 words), the headline itself (labelled `input_text`); otherwise up to 5 sentences chosen by a simple rule-based selector (labelled `heuristic`); if no sentence qualifies, retrieval is skipped. Claims are never invented.

### Running the tests

```bash
python -m pytest tests/ -v -rs
python -m pytest tests/test_phase8_service.py tests/test_phase8_assessment.py tests/test_phase8_frontend_helpers.py tests/test_phase8_api.py tests/test_newscred_frontend.py -v -rs
```

Use `-rs` and read the skip reasons: a skipped test is **unverified**, not passed. Tests that need a real FAISS index (or `streamlit`, `fastapi`, `httpx`) skip when that package is missing. With real pytest a module-level skip (for example `test_rag_retrieval.py` without `faiss`) is reported as **one** skip for the whole module, so totals differ between environments; compare passed/failed/skipped, not the grand total. Tests that check the *error messages* for missing FAISS and for each index failure run in every environment. Phase 8 tests use the real saved ML model, fake embedding components (no model download), and (for retrieval) a real FAISS index built with a tiny test embedder. `tests/test_rag_real_dependencies.py` is the Phase 7 check against real `faiss-cpu` + the real embedding model (needs the model downloaded; it skips otherwise).

On Windows, if the real RAG tests raise a PyTorch `c10.dll` initialization error when run in the same pytest process as the other native-library tests, run the suite in two processes:

```powershell
python -m pytest tests\ --ignore=tests\test_rag_real_dependencies.py -q -rs
python -m pytest tests\test_rag_real_dependencies.py -q -rs
```

This still executes the entire test set; keep the real-dependency run separate and inspect its result rather than treating it as skipped or passed based on the other command. In this environment, setting `KMP_DUPLICATE_LIB_OK=TRUE` did not resolve the combined-process failure.

### Final fix: missing FAISS vs missing index (what was actually run)

*Historical log: written before Llama/Ollama was removed, so mentions of them describe the project as it was at the time.*

**Reported failure:** `test_missing_rag_index_gives_unavailable_evidence_with_other_components_ok` expected the message to mention `scripts/rag_cli.py build`, but in an environment without FAISS the message said the index "could not be used (is faiss-cpu installed?)".

**Root cause (reproduced):** two genuine problems, not one.

1. *Application:* `EvidenceIndex.load()` checks that `faiss` imports **before** it checks the index directory, so without FAISS a missing index can never raise `IndexNotFoundError`. The failure was a plain `EvidenceIndexError`, which the service could not tell apart from any other index problem, so it showed a guess. The check order itself is correct (a user without FAISS cannot build an index either), but the message was vague and the condition was not identifiable by type.
2. *Test:* the test needed an importable `faiss` to reach the missing-index code path but did not declare that, unlike every other FAISS-dependent test in the file.

**Why 226 and 144 are both "right":** the test passes wherever `import faiss` succeeds (the earlier redesign pass states that it ran with real `faiss-cpu` installed) and fails wherever it does not (the independent review). It was environment-dependent, not flaky.

**What changed (4 files):**

* `src/rag/faiss_index.py`: new `FaissNotInstalledError`, a subclass of `EvidenceIndexError` (existing handlers still work) with the original message unchanged. Also exported from `src/rag/__init__.py`.
* `src/credibility/service.py`: the service now reports **three different conditions with three different, actionable messages**: FAISS missing (install it), index missing (build it), index damaged / built with another model / otherwise unloadable (rebuild it); plus a generic "could not be prepared" for unexpected errors. No file paths or raw exception text are shown to the user. `/health` uses the same messages.
* `tests/test_phase8_service.py`: the original assertions are **unchanged**; the test now declares `pytest.importorskip("faiss")`. Four new tests run in every environment (they simulate a failed `import faiss` or patch `EvidenceIndex.load`; no real FAISS needed): FAISS missing is distinct from index missing; FAISS is checked before the directory; `/health` reports it; each of six index failures yields its own message with no path or raw error leaked. Against the *original* `service.py` three of these fail; against the fix they pass.

**Measured in this environment** (Python 3.12.3, no network; real `pytest` was not available, so a small local pytest-compatible runner was used and counts are per test): **243 tests: 138 passed, 0 failed, 105 skipped.** The 105 skips are exactly: 23 `test_newscred_frontend` (no `streamlit`), 9 `test_phase8_api` (no `fastapi`), 46 `test_rag_pipeline` + `test_rag_retrieval` (no `faiss`), 13 `test_phase8_service` (no `faiss`), 13 `test_rag_real_dependencies` (no `faiss` / embedding model), 1 XGBoost-optional. **Skipped tests are not passes.**

Also run for real here: the saved model and vectorizer are byte-identical to the previous ZIP and load and predict with no version warnings; all seven notebooks are valid and every code cell compiles; every Python file compiles; the real `AnalysisService` (real model; real Ollama client with nothing listening) was driven through the project's real frontend client against a throwaway **stdlib** HTTP server standing in for FastAPI: `/health` degraded with the FAISS message, `/analyze` returned partial results (ML ok, Llama unavailable), a manual claim produced `evidence: unavailable` with the FAISS message, empty input was rejected, and a stopped backend produced the "Cannot reach the analysis API ... start it with ..." message.

**Not run here (so unverified by this pass):** the real FastAPI app and `tests/test_phase8_api.py`; the Streamlit app start-up and the 23 rendering/escaping tests; any retrieval against real FAISS or the real embedding model; a live Ollama request; a clean `pip install -r requirements.txt` on any Python version (including 3.13). **Expected, not observed:** with every package installed and the embedding model not yet downloaded, 230 passed and 13 skipped; in an environment like the reviewer's (no FAISS), the earlier failure becomes a skip and the four new tests pass.

### Verification status for the NewsCred frontend redesign (what was actually run)

*Historical log: written before Llama/Ollama was removed, so mentions of them describe the project as it was at the time.*

Run in a sandbox with Python 3.12.3 and the real packages (`fastapi`, `uvicorn`, `streamlit`, `httpx`, `faiss-cpu`, `numpy==1.26.4`, `scikit-learn==1.8.0`):

* The full test suite: 226 passed, 13 skipped. The 13 skipped are `tests/test_rag_real_dependencies.py`, which need the `all-MiniLM-L6-v2` model downloaded from Hugging Face (not reachable in the sandbox), so they are **unverified**.
* The real API (`python -m uvicorn app.backend.main:app`) started and answered `GET /health` and `POST /analyze` (real saved ML model; empty input returned HTTP 422 with the documented message). The "Cannot reach the analysis API" message seen before was simply the API not running; no backend code change was needed.
* The real dashboard page was executed with Streamlit's `AppTest` against that live API (ML card rendered from the real model output; Llama and evidence cards showed the API's own "unavailable" messages because Ollama and a RAG index were not present), against a dead port (friendly message, Retry button and troubleshooting section), and `python -m streamlit run app/frontend/app.py` started and served the page (HTTP 200).
* **Not verified:** how the page looks in a real browser (no browser was available, so colours, spacing and responsiveness were not visually checked), a live Ollama/Llama response, and retrieval with the real embedding model. Check these on your machine before the demo.

The older Phase 8 accounting that follows is kept for history; its "never run" items for the real FastAPI server and Streamlit app are superseded by the list above.

### Verification status for Phase 8 (what was actually run)

*Historical log: written before Llama/Ollama was removed, so mentions of them describe the project as it was at the time.*

The sandbox where Phase 8 was written had **no internet access**: `fastapi`, `uvicorn`, `streamlit`, `httpx`, `pytest`, `faiss-cpu` and `sentence-transformers` could not be installed there (Python 3.12.3). Honest accounting:

* **Run for real:** the Phase 8 service and assessment logic (39 tests) and dashboard helpers (7 tests) against the **real saved ML model and vectorizer**; all earlier Phase 2-7 tests that can run without the missing packages; every Python file compiles; all notebooks are valid JSON; the saved model artifacts are byte-identical to the Phase 6 upload. The tests were executed with a minimal pytest-compatible runner (real `pytest` was not available), and retrieval tests ran against a NumPy stand-in for FAISS.
* **Smoke-checked only with throwaway stand-ins, not the real frameworks:** the FastAPI route/handler code (called directly with a fake `fastapi`/`pydantic`) and the Streamlit page (executed with a fake `streamlit` across empty, full, partial and all-failed results, to catch crashes).
* **Written but NOT run:** `tests/test_phase8_api.py` (9 tests using FastAPI's `TestClient`; it needs `fastapi` + `httpx`) and the real-library tests in `tests/test_rag_real_dependencies.py` (13, skipped).
* **Never run:** the real FastAPI server, the real Streamlit app, a live Ollama/Llama request, real-embedding retrieval quality through the API, and the start-up commands above. **Run the commands in this README and the tests with `-rs` on your machine before the demo**, and report any failure; the first suspects would be `app/backend/main.py` and `app/frontend/app.py`, since those are the files that could not be executed against the real frameworks.

### Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| Dashboard: "The analysis service is not running" | API not running. Start Terminal 2 and keep it open, then press **Retry connection**. Open **Troubleshooting details** to see the address the page uses and make sure it matches `BACKEND_URL`/port. |
| `rag` unavailable "FAISS (the faiss-cpu package) is not installed or could not be imported" | A **dependency** problem, reported before anything about the index (without FAISS an index can be neither loaded nor built). With the virtual environment active run `pip install -r requirements.txt`, check with `python -c "import faiss; print(faiss.__version__)"`, then restart the backend. If `pip` finds no matching `faiss-cpu` for your Python version, that Python is probably too new for the pinned wheel; use a Python version listed under Prerequisites. |
| `rag` unavailable "No evidence index was found" | Build it (see "RAG index creation") and set `RAG_INDEX_DIR` (or unset it so the demo index is auto-detected). A `.env` copied from `.env.example` sets `RAG_INDEX_DIR=./data/rag/index`, which does not exist until you build it. |
| `rag` unavailable "The evidence index could not be loaded" | An index problem that is not one of the cases above or below. The details are in the backend log. Rebuild with `python scripts/rag_cli.py build --force`. |
| `rag` "built with a different embedding model" / "damaged" | `python scripts/rag_cli.py build --docs ... --index ... --force`. |
| Evidence: "embedding model could not be loaded" | First run needs internet once to download `all-MiniLM-L6-v2`; afterwards it is cached. |
| `ml_model` unavailable | `models/` is missing files; run `notebooks/03_ml_models.ipynb` (needs the ISOT dataset). |
| Scikit-learn version warnings in `load_warnings` | Installed scikit-learn differs from the 1.8.0 the model was saved with; see the Python 3.13 open item. |
| Port 8000 already in use | `--port 8001` for uvicorn, and set `BACKEND_URL=http://127.0.0.1:8001` before starting Streamlit. |
| `ModuleNotFoundError: app` or `src` | Run the commands from the project root with the virtual environment active. |

### Known limitations (mention these in your demonstration)

* **Not a fact-checker.** Nothing is verified. ML predictions and retrieved passages can each be wrong.
* **The ML model** was trained on the ISOT dataset (2016-2017 US-centric political news). The earlier analysis found it partly relies on formatting artifacts (e.g. Reuters boilerplate), so high accuracy there does not mean it generalises (on WELFake accuracy dropped to 84.7%; see Evaluation). It is unreliable on headlines, non-English text and other domains.
* **Retrieval** only knows the indexed documents. The sample collection is **fictional demonstration data**, so retrieved passages must not be presented as real evidence. A high similarity score does not mean support (a passage can contradict a claim), and the default similarity threshold (0.35) is an uncalibrated heuristic.
* **The final assessment** is a simple rule-based summary, not a calibrated credibility score.
* Single user / local use: no authentication, no request queue, no history storage. 
* English text only in practice; maximum input 20,000 characters.

## Phase Checklist

- [x] Phase 1: Project Setup
- [x] Phase 2: Dataset and Preprocessing
- [x] Phase 3: Exploratory Data Analysis
- [x] Phase 4: Machine Learning Baseline
- [x] Phase 5: Inference Pipeline & Application Readiness
- [x] ~~Phase 6: Llama Integration & Language Analysis~~ (removed; see "Phase 6 (removed)")
- [x] Cross-dataset evaluation and threshold tuning (WELFake)
- [ ] (Unscheduled) Advanced NLP/Transformer Model
- [x] Phase 7: Evidence/RAG Component (author-reported demo index built on their machine; see "Phase 7 -> Verification status")
- [x] Phase 8: Final Integration (FastAPI backend + Streamlit dashboard; implemented, real-framework runs still to be done on your machine - see "Phase 8 -> Verification status")
- [ ] Phase 9: not started

## Important Notes

- This system is a **fake-news detection and credibility analysis tool**, not an absolute truth detector. It never claims "100% fake" or "100% true" without authoritative evidence.
- ML predictions and retrieved evidence are always presented as clearly separate, labeled sections.

## Current verification record (2026-10-24)

This record supersedes the older environment/test summaries above where they conflict.

- Tested in the repository `.venv` on Python 3.12.10. Key versions: NumPy 1.26.4,
  PyArrow 18.1.0, Pandas 2.2.3, Streamlit 1.40.1, scikit-learn 1.8.0,
  faiss-cpu 1.9.0, sentence-transformers 3.3.1, PyTorch 2.14.1+cpu, and
  Transformers 4.57.6. `python -m pip check` reported no broken requirements.
- PyArrow is pinned directly in `requirements.txt`: Streamlit depends on PyArrow but
  otherwise allows a newer release that failed to import with this project's NumPy 1.26.4.
- Import verification passed for the versions above. Use the project environment's Python,
  e.g. `.venv\Scripts\python.exe` in PowerShell; a separately installed global Python is not
  the environment used for these checks.
- Focused reliability/API/frontend tests: **99 passed**, 15 warnings. Streamlit AppTest files:
  **29 passed**, 1 warning. Real Sentence Transformers + FAISS integration file:
  **13 passed**, no skips.
- Complete suite in one pytest process: **not clean** on this Windows environment; 13 real
  RAG integration cases failed while PyTorch loaded `c10.dll` after other native-library tests.
  The documented split-process runs cover all tests: **216 passed** outside the real RAG file,
  then **13 passed** in that file. Do not combine these into a claim that the one-process run passed.
- FastAPI `/health` and `/analyze` were exercised through the real FastAPI app with the saved
  classifier, actual Sentence Transformers model, and an actual FAISS index. The index was
  built offline at the configured `data/rag/index` path from the project's fictional sample
  documents (three sources, five chunks); the API marked retrieved evidence unclassified and
  did not return a verdict.
- Evaluation numbers in [docs/evaluation_report.md](docs/evaluation_report.md) remain recorded
  results from committed artifacts, not newly regenerated metrics: raw ISOT/WELFake CSVs and
  processed ISOT splits are absent from this checkout. No independent dataset was downloaded.
