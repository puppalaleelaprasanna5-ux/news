"""
Data quality checks for the fake-news dataset.

`compute_data_quality_report` is purely diagnostic (read-only) — it does not
modify the dataset. Use `remove_low_quality_rows` separately if you want to
actually drop empty/duplicate articles.
"""

from dataclasses import dataclass
import re

import pandas as pd

# An article shorter than this many words is flagged as "very short".
# This is a diagnostic threshold only — it does NOT cause rows to be
# dropped automatically. That decision is left to you after seeing the report.
VERY_SHORT_ARTICLE_WORD_THRESHOLD = 20


@dataclass
class DataQualityReport:
    total_records: int
    fake_count: int
    real_count: int
    missing_values: dict
    duplicate_records: int
    empty_articles: int
    very_short_articles: int
    length_stats_words: dict
    length_stats_chars: dict


def compute_data_quality_report(df: pd.DataFrame, text_col: str = "text",
                                 label_col: str = "label") -> DataQualityReport:
    """
    Compute a snapshot of dataset quality: counts, missing values,
    duplicates, empty/short articles, and length statistics.
    """
    total_records = len(df)
    fake_count = int((df[label_col] == 1).sum())
    real_count = int((df[label_col] == 0).sum())

    missing_values = df.isnull().sum().to_dict()

    duplicate_records = int(df.duplicated(subset=[text_col]).sum())

    text_series = df[text_col].fillna("").astype(str)
    empty_articles = int((text_series.str.strip() == "").sum())

    word_counts = text_series.str.split().apply(len)
    char_counts = text_series.str.len()

    very_short_articles = int((word_counts < VERY_SHORT_ARTICLE_WORD_THRESHOLD).sum())

    length_stats_words = {
        "mean": float(word_counts.mean()) if total_records else 0.0,
        "median": float(word_counts.median()) if total_records else 0.0,
        "min": int(word_counts.min()) if total_records else 0,
        "max": int(word_counts.max()) if total_records else 0,
    }
    length_stats_chars = {
        "mean": float(char_counts.mean()) if total_records else 0.0,
        "median": float(char_counts.median()) if total_records else 0.0,
        "min": int(char_counts.min()) if total_records else 0,
        "max": int(char_counts.max()) if total_records else 0,
    }

    return DataQualityReport(
        total_records=total_records,
        fake_count=fake_count,
        real_count=real_count,
        missing_values=missing_values,
        duplicate_records=duplicate_records,
        empty_articles=empty_articles,
        very_short_articles=very_short_articles,
        length_stats_words=length_stats_words,
        length_stats_chars=length_stats_chars,
    )


def print_data_quality_report(report: DataQualityReport) -> None:
    """Pretty-print a DataQualityReport to the console/notebook output."""
    print("=" * 55)
    print("DATA QUALITY SUMMARY")
    print("=" * 55)
    print(f"Total records        : {report.total_records}")
    print(f"Fake articles (1)    : {report.fake_count}")
    print(f"Real articles (0)    : {report.real_count}")
    print(f"Duplicate records    : {report.duplicate_records}")
    print(f"Empty articles       : {report.empty_articles}")
    print(f"Very short articles (<{VERY_SHORT_ARTICLE_WORD_THRESHOLD} words): {report.very_short_articles}")
    print("-" * 55)
    print("Missing values per column:")
    for col, count in report.missing_values.items():
        print(f"  {col:15s}: {count}")
    print("-" * 55)
    print("Article length in words:", report.length_stats_words)
    print("Article length in characters:", report.length_stats_chars)
    print("=" * 55)


_MULTI_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_for_comparison(text) -> str:
    """
    Light, COMPARISON-ONLY text normalization.

    This is used purely to decide whether two rows should be treated as
    "the same article" for empty/duplicate/cross-class-conflict detection.
    It does NOT touch the real 'text' or 'clean_text' columns and is never
    saved anywhere — it exists only inside the functions below.

    Normalization applied: collapse all whitespace to single spaces, strip
    leading/trailing whitespace, and casefold (so e.g. trailing spaces or
    a different capitalization don't hide an otherwise-identical article).
    """
    if not isinstance(text, str):
        return ""
    return _MULTI_WHITESPACE_RE.sub(" ", text).strip().casefold()


def remove_missing_or_empty_text(df: pd.DataFrame, text_col: str = "text"):
    """
    Remove rows whose text is missing (NaN) or empty/whitespace-only once
    normalized (so a single space " " counts as empty, matching how the
    ISOT dataset actually represents missing article bodies).

    Returns
    -------
    (df_without_empty_rows, num_rows_removed)
    """
    df = df.copy()
    df[text_col] = df[text_col].fillna("")

    normalized = df[text_col].astype(str).apply(_normalize_for_comparison)
    is_nonempty = normalized != ""

    num_removed = int((~is_nonempty).sum())
    return df[is_nonempty].reset_index(drop=True), num_removed


def remove_cross_class_duplicates(df: pd.DataFrame, text_col: str = "text",
                                   label_col: str = "label"):
    """
    Detect and remove CROSS-CLASS label conflicts: articles whose (normalized)
    text appears under BOTH label=1 (fake) and label=0 (real).

    We cannot trust either label when the identical article text is claimed
    to be both fake and real, so ALL rows carrying that ambiguous text are
    removed — from both classes — rather than arbitrarily keeping one.

    This must run BEFORE ordinary duplicate removal, and only on text that
    has already passed the empty-text filter (see remove_missing_or_empty_text),
    so that e.g. whitespace-only text is never mistaken for a real conflict.

    Returns
    -------
    (df_without_conflicts, num_rows_removed, num_distinct_conflicting_texts)
    """
    df = df.copy()
    df["_cmp_text"] = df[text_col].astype(str).apply(_normalize_for_comparison)

    labels_per_text = df.groupby("_cmp_text")[label_col].nunique()
    conflicting_keys = set(labels_per_text[labels_per_text > 1].index)

    is_conflicting = df["_cmp_text"].isin(conflicting_keys)
    num_removed = int(is_conflicting.sum())
    num_distinct_conflicts = len(conflicting_keys)

    df_clean = df[~is_conflicting].drop(columns=["_cmp_text"]).reset_index(drop=True)
    return df_clean, num_removed, num_distinct_conflicts


def remove_duplicate_text(df: pd.DataFrame, text_col: str = "text"):
    """
    Remove ORDINARY duplicate articles: rows whose text EXACTLY matches
    another row that shares the SAME label (kept first occurrence, drop
    the rest). By the time this runs, any cross-class conflicts have
    already been removed by remove_cross_class_duplicates().

    IMPORTANT — why this uses EXACT matching, not the normalized/casefolded
    comparison used elsewhere in this module:

    On the real ISOT dataset, several very short "video embed" articles
    (e.g. body text just "Watch:") are legitimately DIFFERENT articles
    (different headlines, different stories) that merely share the same
    trivial placeholder body text once whitespace/case differences are
    ignored. Using normalized matching here was tested and found to
    incorrectly merge up to 9 such rows as if they were duplicates of each
    other. Exact matching avoids that false-merge risk. (Whitespace-only
    "empty" text is still caught separately by remove_missing_or_empty_text,
    and genuine cross-class label conflicts are still caught by
    remove_cross_class_duplicates() using normalized comparison, where the
    stricter check is actually needed and safe.)

    Returns
    -------
    (df_without_duplicates, num_rows_removed)
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates(subset=[text_col])
    num_removed = before - len(df)

    df = df.reset_index(drop=True)
    return df, num_removed


def clean_dataset(df: pd.DataFrame, text_col: str = "text", label_col: str = "label"):
    """
    Full Phase 2 cleaning sequence, in this exact order:
        1. Remove missing/empty (including whitespace-only) text
        2. Remove cross-class duplicates (same text, conflicting labels)
        3. Remove remaining ordinary (same-class) duplicate text

    This ordering matters: step 1 must run before step 2, or a whitespace-only
    "article" could be misread as a genuine label conflict; step 2 must run
    before step 3, or ordinary deduplication could silently keep an arbitrary
    label for a conflicting article instead of removing it outright.

    Returns
    -------
    (clean_df, stats) where stats is a dict with keys:
        empty_removed, cross_class_removed, cross_class_distinct_texts,
        duplicate_removed, final_count
    """
    stats = {}

    df, stats["empty_removed"] = remove_missing_or_empty_text(df, text_col)

    df, stats["cross_class_removed"], stats["cross_class_distinct_texts"] = (
        remove_cross_class_duplicates(df, text_col, label_col)
    )

    df, stats["duplicate_removed"] = remove_duplicate_text(df, text_col)

    stats["final_count"] = len(df)

    return df, stats


def remove_low_quality_rows(df: pd.DataFrame, text_col: str = "text",
                             label_col: str = "label") -> pd.DataFrame:
    """
    Backward-compatible convenience wrapper around clean_dataset() that
    returns just the cleaned DataFrame (no stats). Prefer calling
    clean_dataset() directly when you want the breakdown of counts.

    Runs, in order: empty-text removal -> cross-class conflict removal ->
    ordinary duplicate removal.
    """
    clean_df, _ = clean_dataset(df, text_col=text_col, label_col=label_col)
    return clean_df
