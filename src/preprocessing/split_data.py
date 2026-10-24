"""
Train / validation / test split utilities.

Split ratios: 70% train, 15% validation, 15% test.
Stratified on the label column so the fake/real ratio is preserved in
each split. A fixed random seed is used so the split is reproducible.

DATA LEAKAGE PREVENTION - read this before using in later phases:

    1. Deduplication must happen BEFORE this split, not after, so the same
       article can never appear in more than one split. (Phase 2's pipeline
       already does this — see pipeline.py.)

    2. Any preprocessing step that "learns" parameters from data (for
       example, a TF-IDF vectorizer's vocabulary in Phase 4, or a scaler's
       mean/variance) must be FIT only on the training split returned here,
       then applied with transform-only (never re-fit) to validation/test.

    3. The test set should not be looked at again until final evaluation.
       Use the validation set for model selection and hyperparameter tuning.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


def split_dataset(df: pd.DataFrame, label_col: str = "label",
                   train_size: float = TRAIN_SIZE,
                   validation_size: float = VALIDATION_SIZE,
                   test_size: float = TEST_SIZE,
                   random_state: int = RANDOM_SEED):
    """
    Split `df` into stratified train / validation / test DataFrames.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    if abs((train_size + validation_size + test_size) - 1.0) > 1e-9:
        raise ValueError("train_size + validation_size + test_size must equal 1.0")

    # Step 1: split off the training set from everything else.
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=df[label_col],
        random_state=random_state,
    )

    # Step 2: split the remainder into validation and test,
    # preserving their relative proportions (15% / 15% of the original ->
    # 50% / 50% of what's left).
    relative_val_size = validation_size / (validation_size + test_size)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        stratify=temp_df[label_col],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, val_df, test_df


def assert_no_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame,
                       test_df: pd.DataFrame, id_col: str = "article_id") -> None:
    """
    Raise an AssertionError if any article ID appears in more than one split.

    Call this immediately after splitting, every time, as a safety net
    against data leakage.
    """
    train_ids = set(train_df[id_col])
    val_ids = set(val_df[id_col])
    test_ids = set(test_df[id_col])

    assert train_ids.isdisjoint(val_ids), "Overlap found between train and validation sets!"
    assert train_ids.isdisjoint(test_ids), "Overlap found between train and test sets!"
    assert val_ids.isdisjoint(test_ids), "Overlap found between validation and test sets!"


def assert_no_text_overlap_across_splits(train_df: pd.DataFrame, val_df: pd.DataFrame,
                                          test_df: pd.DataFrame, text_col: str = "text") -> None:
    """
    Raise an AssertionError if the same article text appears in more than
    one split. This checks the actual text content directly (not just
    article_id), as an independent safety net: article_id is a hash of the
    text, so this check should always agree with assert_no_overlap() — but
    checking the raw values too protects against any future bug in how
    article_id is computed. Reusable for both the 'text' and 'clean_text'
    columns.
    """
    train_texts = set(train_df[text_col])
    val_texts = set(val_df[text_col])
    test_texts = set(test_df[text_col])

    assert train_texts.isdisjoint(val_texts), (
        f"Duplicate '{text_col}' found between train and validation sets!"
    )
    assert train_texts.isdisjoint(test_texts), (
        f"Duplicate '{text_col}' found between train and test sets!"
    )
    assert val_texts.isdisjoint(test_texts), (
        f"Duplicate '{text_col}' found between validation and test sets!"
    )


def count_text_overlap_across_splits(train_df: pd.DataFrame, val_df: pd.DataFrame,
                                      test_df: pd.DataFrame, text_col: str = "text") -> int:
    """
    Non-asserting version of assert_no_text_overlap_across_splits(): returns
    the number of DISTINCT text values that appear in more than one split,
    instead of raising.

    Why this exists: article_id and raw 'text' should NEVER overlap across
    splits (that would be genuine data leakage, and is enforced with a hard
    assert). 'clean_text' is a different story — light cleaning (whitespace
    collapsing, unicode normalization) can occasionally make two DIFFERENT
    raw articles collapse into identical clean_text if they differed only by
    trivial whitespace. That is a real but narrow modeling consideration for
    later phases, not a Phase 2 correctness bug, so it is reported as a
    count here rather than treated as a hard failure.
    """
    train_texts = set(train_df[text_col])
    val_texts = set(val_df[text_col])
    test_texts = set(test_df[text_col])

    overlap = (train_texts & val_texts) | (train_texts & test_texts) | (val_texts & test_texts)
    return len(overlap)
