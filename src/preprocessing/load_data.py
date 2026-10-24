"""
Data loading module for the ISOT Fake and Real News dataset.

LABEL CONVENTION (used throughout this whole project):
    1 = FAKE news
    0 = REAL news

Expected raw files (NOT included in this repository — see README.md
"Phase 2 - Dataset acquisition" for download instructions):
    data/raw/Fake.csv
    data/raw/True.csv

Each file is expected to contain at least these columns:
    title, text, subject, date
"""

import hashlib
from pathlib import Path

import pandas as pd

# --- Label convention -------------------------------------------------------
LABEL_FAKE = 1
LABEL_REAL = 0

# --- Columns every raw file must have ---------------------------------------
REQUIRED_COLUMNS = ["title", "text", "subject", "date"]


def _validate_columns(df: pd.DataFrame, source_name: str) -> None:
    """Raise a clear, beginner-friendly error if a required column is missing."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"'{source_name}' is missing required column(s): {missing}. "
            f"Expected columns: {REQUIRED_COLUMNS}. "
            f"Found columns: {list(df.columns)}"
        )


def load_raw_csv(file_path, source_name: str) -> pd.DataFrame:
    """
    Load a single raw CSV file (Fake.csv or True.csv) and validate its columns.

    Parameters
    ----------
    file_path : str or Path
        Path to the CSV file.
    source_name : str
        Human-readable name used in error messages (e.g. "Fake.csv").

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the file is missing one of the required columns.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Could not find '{file_path}'.\n"
            f"Please download the ISOT Fake and Real News dataset and place "
            f"'{file_path.name}' inside the 'data/raw/' folder.\n"
            f"See README.md -> Phase 2 -> Dataset acquisition for exact instructions."
        )

    df = pd.read_csv(file_path)

    # Standardize column names: strip whitespace, lowercase.
    # This protects us from small inconsistencies between dataset versions
    # (e.g. "Title" vs "title", or a trailing space in a column name).
    df.columns = [str(c).strip().lower() for c in df.columns]

    _validate_columns(df, source_name)
    return df


def load_isot_dataset(raw_data_dir) -> pd.DataFrame:
    """
    Load and combine the ISOT Fake and Real News dataset.

    Expects two files inside `raw_data_dir`:
        Fake.csv
        True.csv

    Returns
    -------
    pd.DataFrame with columns:
        title, text, subject, date, label
    where label = 1 (fake) or 0 (real), as defined by LABEL_FAKE / LABEL_REAL above.
    """
    raw_data_dir = Path(raw_data_dir)

    fake_df = load_raw_csv(raw_data_dir / "Fake.csv", "Fake.csv")
    real_df = load_raw_csv(raw_data_dir / "True.csv", "True.csv")

    # Keep only the columns we actually need, in a consistent order.
    fake_df = fake_df[REQUIRED_COLUMNS].copy()
    real_df = real_df[REQUIRED_COLUMNS].copy()

    fake_df["label"] = LABEL_FAKE
    real_df["label"] = LABEL_REAL

    combined_df = pd.concat([fake_df, real_df], ignore_index=True)

    # Shuffle so fake/real articles aren't sitting in two big blocks.
    # This does NOT affect the train/val/test split later (that uses its
    # own fixed random seed), it just makes a quick df.head() more useful.
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return combined_df


def add_article_id(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Add a stable, unique 'article_id' column based on an MD5 hash of the
    article text.

    Why: this gives us a reliable way to check for duplicate articles and
    to verify that no article ends up in more than one train/val/test split.
    """
    df = df.copy()
    # fillna first: astype(str) alone can still hand NaN (a float) through to
    # the hashing function on some pandas/numpy versions, which crashes.
    safe_text = df[text_col].fillna("").astype(str)
    df["article_id"] = safe_text.apply(
        lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()
    )
    return df
