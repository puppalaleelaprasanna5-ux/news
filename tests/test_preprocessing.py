"""
Tests for Phase 2 preprocessing modules.

IMPORTANT — about the data used in these tests:
    These tests use a small, hand-written FIXTURE dataset created on the
    fly (see FAKE_ROWS / REAL_ROWS / _write_fixture_csvs below). They do
    NOT use the real ISOT dataset. Their purpose is to check that the
    preprocessing FUNCTIONS behave correctly (loading, labeling, cleaning,
    splitting) in isolation — this is standard unit-testing practice, not
    a substitute for running the pipeline on the real data.

    To verify the pipeline against the actual ISOT dataset, place Fake.csv
    and True.csv in data/raw/ (see README.md) and run:
        python -m src.preprocessing.pipeline
    or run notebooks/01_data_analysis.ipynb.

Run these tests with:
    pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.load_data import (
    load_isot_dataset,
    load_raw_csv,
    add_article_id,
    LABEL_FAKE,
    LABEL_REAL,
)
from src.preprocessing.text_cleaning import clean_single_text, add_clean_text_column
from src.preprocessing.quality_checks import (
    compute_data_quality_report,
    remove_low_quality_rows,
    remove_missing_or_empty_text,
    remove_cross_class_duplicates,
    remove_duplicate_text,
    clean_dataset,
)
from src.preprocessing.split_data import (
    split_dataset,
    assert_no_overlap,
    assert_no_text_overlap_across_splits,
)


# --- Fixture data (NOT the real dataset — see module docstring) ------------

FAKE_ROWS = [
    {"title": "Shocking claim goes viral", "text": "This is a fake article about something shocking.", "subject": "News", "date": "January 1, 2020"},
    {"title": "Another wild story", "text": "Another fake article with a wild unverified claim.", "subject": "News", "date": "January 2, 2020"},
    {"title": "Duplicate story", "text": "This exact text appears twice in the fake set.", "subject": "News", "date": "January 3, 2020"},
    {"title": "Duplicate story", "text": "This exact text appears twice in the fake set.", "subject": "News", "date": "January 3, 2020"},
    {"title": "Empty body story", "text": "", "subject": "News", "date": "January 4, 2020"},
    {"title": "Missing body story", "text": None, "subject": "News", "date": "January 5, 2020"},
]

REAL_ROWS = [
    {"title": "Government announces policy", "text": "WASHINGTON (Reuters) - The government announced a new policy today.", "subject": "politicsNews", "date": "January 1, 2020"},
    {"title": "Markets close higher", "text": "NEW YORK (Reuters) - Stock markets closed higher on Monday.", "subject": "worldnews", "date": "January 2, 2020"},
    {"title": "Election results confirmed", "text": "LONDON (Reuters) - Election officials confirmed the results.", "subject": "worldnews", "date": "January 3, 2020"},
]


def _write_fixture_csvs(tmp_path: Path) -> Path:
    """Write a small fixture Fake.csv / True.csv into tmp_path and return the dir."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    pd.DataFrame(FAKE_ROWS).to_csv(raw_dir / "Fake.csv", index=False)
    pd.DataFrame(REAL_ROWS).to_csv(raw_dir / "True.csv", index=False)

    return raw_dir


# --- load_data.py tests -----------------------------------------------------

def test_load_raw_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_csv(tmp_path / "does_not_exist.csv", "does_not_exist.csv")


def test_load_raw_csv_missing_column_raises(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"title": ["a"], "text": ["b"]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        load_raw_csv(bad_csv, "bad.csv")


def test_load_isot_dataset_combines_and_labels(tmp_path):
    raw_dir = _write_fixture_csvs(tmp_path)
    df = load_isot_dataset(raw_dir)

    assert len(df) == len(FAKE_ROWS) + len(REAL_ROWS)
    assert set(df.columns) >= {"title", "text", "subject", "date", "label"}

    fake_count = (df["label"] == LABEL_FAKE).sum()
    real_count = (df["label"] == LABEL_REAL).sum()
    assert fake_count == len(FAKE_ROWS)
    assert real_count == len(REAL_ROWS)


def test_add_article_id_is_stable_and_unique_per_text(tmp_path):
    raw_dir = _write_fixture_csvs(tmp_path)
    df = load_isot_dataset(raw_dir)
    df = add_article_id(df)

    assert "article_id" in df.columns
    # The two identical duplicate rows should get the exact same ID.
    duplicate_texts = df[df["text"] == "This exact text appears twice in the fake set."]
    assert duplicate_texts["article_id"].nunique() == 1


# --- quality_checks.py tests -------------------------------------------------

def test_missing_text_and_duplicates_removed(tmp_path):
    raw_dir = _write_fixture_csvs(tmp_path)
    df = load_isot_dataset(raw_dir)
    df = add_article_id(df)

    before_count = len(df)
    cleaned_df = remove_low_quality_rows(df, text_col="text")

    # Missing / empty text rows should be gone.
    assert cleaned_df["text"].isnull().sum() == 0
    assert (cleaned_df["text"].astype(str).str.strip() == "").sum() == 0

    # Duplicate should be reduced to exactly one copy.
    assert cleaned_df["text"].duplicated().sum() == 0
    assert len(cleaned_df) < before_count


def test_data_quality_report_counts_are_correct(tmp_path):
    raw_dir = _write_fixture_csvs(tmp_path)
    df = load_isot_dataset(raw_dir)

    report = compute_data_quality_report(df)

    assert report.total_records == len(FAKE_ROWS) + len(REAL_ROWS)
    assert report.fake_count == len(FAKE_ROWS)
    assert report.real_count == len(REAL_ROWS)
    assert report.empty_articles >= 1   # the empty-text fixture row
    assert report.duplicate_records >= 1  # the duplicate fixture rows


# --- text_cleaning.py tests --------------------------------------------------

def test_clean_single_text_strips_html_and_normalizes_whitespace():
    dirty = "<p>Hello&nbsp;&nbsp;World</p>\n\n\tExtra   spaces"
    cleaned = clean_single_text(dirty)

    assert "<p>" not in cleaned
    assert "</p>" not in cleaned
    assert "  " not in cleaned  # no double spaces left
    assert "Hello" in cleaned and "World" in cleaned


def test_clean_single_text_handles_non_string_input():
    assert clean_single_text(None) == ""
    assert clean_single_text(float("nan")) == ""


def test_add_clean_text_column_preserves_original_text(tmp_path):
    raw_dir = _write_fixture_csvs(tmp_path)
    df = load_isot_dataset(raw_dir)
    df = remove_low_quality_rows(df, text_col="text")

    original_texts = df["text"].copy()
    df = add_clean_text_column(df, source_col="text", target_col="clean_text")

    assert "clean_text" in df.columns
    # The original 'text' column must be completely untouched.
    assert df["text"].equals(original_texts)


# --- split_data.py tests ------------------------------------------------------

def test_split_dataset_produces_expected_proportions():
    # A larger synthetic label-only dataframe used purely to test the
    # SPLIT MATH and stratification, not real article content.
    df = pd.DataFrame({
        "text": [f"article number {i}" for i in range(100)],
        "label": [0, 1] * 50,
    })
    df = add_article_id(df)

    train_df, val_df, test_df = split_dataset(df)

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Stratification check: label balance should be roughly preserved.
    for split_df in (train_df, val_df, test_df):
        fake_ratio = split_df["label"].mean()
        assert 0.3 < fake_ratio < 0.7


def test_split_dataset_has_no_overlap():
    df = pd.DataFrame({
        "text": [f"article number {i}" for i in range(200)],
        "label": [0, 1] * 100,
    })
    df = add_article_id(df)

    train_df, val_df, test_df = split_dataset(df)

    # Should not raise.
    assert_no_overlap(train_df, val_df, test_df)

    train_ids = set(train_df["article_id"])
    val_ids = set(val_df["article_id"])
    test_ids = set(test_df["article_id"])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_split_dataset_has_no_text_overlap():
    # Independent check on raw text content, not just article_id.
    df = pd.DataFrame({
        "text": [f"article number {i}" for i in range(200)],
        "label": [0, 1] * 100,
    })
    df = add_article_id(df)

    train_df, val_df, test_df = split_dataset(df)

    # Should not raise.
    assert_no_text_overlap_across_splits(train_df, val_df, test_df, text_col="text")


# --- Cross-class conflict handling tests (new) -------------------------------

def test_same_text_same_label_is_ordinary_duplicate():
    """Same text, same label -> handled by ordinary duplicate removal, not flagged as a conflict."""
    df = pd.DataFrame({
        "text": ["identical article body", "identical article body", "unique article"],
        "label": [1, 1, 0],
    })

    clean_df, removed, distinct = remove_cross_class_duplicates(df, text_col="text", label_col="label")
    assert removed == 0
    assert distinct == 0
    assert len(clean_df) == 3  # nothing removed at the conflict-detection stage

    deduped_df, dup_removed = remove_duplicate_text(clean_df, text_col="text")
    assert dup_removed == 1
    assert len(deduped_df) == 2


def test_same_text_different_labels_is_identified_as_conflict_and_removed():
    """Same text, different labels -> flagged as a conflict and ALL copies removed."""
    df = pd.DataFrame({
        "text": ["disputed article body", "disputed article body", "unique fake", "unique real"],
        "label": [1, 0, 1, 0],
    })

    clean_df, removed, distinct = remove_cross_class_duplicates(df, text_col="text", label_col="label")

    assert removed == 2
    assert distinct == 1
    assert "disputed article body" not in set(clean_df["text"])
    assert len(clean_df) == 2
    assert set(clean_df["text"]) == {"unique fake", "unique real"}


def test_whitespace_only_text_removed_as_empty_before_conflict_checking():
    """
    Whitespace-only text appearing under both labels (like the real ISOT
    dataset's single-space rows) must be removed by empty-text filtering
    BEFORE cross-class conflict detection ever sees it — it should NOT be
    reported as a cross-class conflict.
    """
    df = pd.DataFrame({
        "text": [" ", " ", "  ", "real article body", "fake article body"],
        "label": [1, 1, 0, 0, 1],
    })

    no_empty_df, empty_removed = remove_missing_or_empty_text(df, text_col="text")
    assert empty_removed == 3
    assert len(no_empty_df) == 2

    # Now run conflict detection on what's left: no conflicts should be found,
    # because the whitespace rows (which spanned both labels) are already gone.
    clean_df, conflicts_removed, distinct_conflicts = remove_cross_class_duplicates(
        no_empty_df, text_col="text", label_col="label"
    )
    assert conflicts_removed == 0
    assert distinct_conflicts == 0
    assert len(clean_df) == 2


def test_no_conflicts_pipeline_continues_normally():
    """When there are no cross-class conflicts at all, clean_dataset() should
    report zero conflicts removed and behave like ordinary cleaning."""
    df = pd.DataFrame({
        "text": ["fake one", "fake one", "fake two", "real one", "real two", ""],
        "label": [1, 1, 1, 0, 0, 0],
    })

    clean_df, stats = clean_dataset(df, text_col="text", label_col="label")

    assert stats["empty_removed"] == 1
    assert stats["cross_class_removed"] == 0
    assert stats["cross_class_distinct_texts"] == 0
    assert stats["duplicate_removed"] == 1  # "fake one" appeared twice
    assert stats["final_count"] == len(clean_df) == 4


def test_clean_dataset_full_sequence_counts_are_correct_and_ordered():
    """
    Integration test covering all four required scenarios at once:
    empty text, ordinary same-label duplicates, and a genuine cross-class
    conflict, all in one dataset — verifying the counts AND the final content.
    """
    df = pd.DataFrame({
        "text": ["", "dup", "dup", "conflict", "conflict", "unique"],
        "label": [1, 0, 0, 1, 0, 1],
    })

    clean_df, stats = clean_dataset(df, text_col="text", label_col="label")

    assert stats["empty_removed"] == 1
    assert stats["cross_class_removed"] == 2
    assert stats["cross_class_distinct_texts"] == 1
    assert stats["duplicate_removed"] == 1  # the second "dup" row (same label both times)
    assert stats["final_count"] == 2

    remaining_texts = set(clean_df["text"])
    assert remaining_texts == {"dup", "unique"}
    assert "conflict" not in remaining_texts
