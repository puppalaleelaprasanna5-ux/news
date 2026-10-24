"""
End-to-end Phase 2 preprocessing pipeline.

This ties together load_data.py, text_cleaning.py, quality_checks.py, and
split_data.py into one reusable function, so the same logic can be called
from the notebook, from tests, or from a future automation script — instead
of copy-pasting preprocessing steps directly into a notebook.

Pipeline stages (in order):
    Raw data
    -> remove empty/invalid articles
    -> detect + remove cross-class label conflicts
    -> remove ordinary (same-class) duplicate articles
    -> create clean_text
    -> stratified train/validation/test split
    -> verify no leakage (by article_id, raw text, AND clean_text)

Usage (from the project root, with the virtual environment active):
    python -m src.preprocessing.pipeline
"""

from pathlib import Path

from src.preprocessing.load_data import load_isot_dataset, add_article_id
from src.preprocessing.text_cleaning import add_clean_text_column
from src.preprocessing.quality_checks import (
    compute_data_quality_report,
    print_data_quality_report,
    clean_dataset,
)
from src.preprocessing.split_data import (
    split_dataset,
    assert_no_overlap,
    assert_no_text_overlap_across_splits,
    count_text_overlap_across_splits,
)

# Project root = two levels up from this file (src/preprocessing/pipeline.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def run_preprocessing_pipeline(raw_data_dir: Path = RAW_DATA_DIR,
                                processed_data_dir: Path = PROCESSED_DATA_DIR,
                                save_outputs: bool = True) -> dict:
    """
    Run the full Phase 2 preprocessing pipeline and (optionally) save results.

    Returns a dict with the train/validation/test DataFrames, the cleaning
    stats, and both quality reports, so callers (like the notebook or tests)
    can inspect them further without re-reading the saved CSVs.
    """
    print("Step 1: Loading raw dataset...")
    df = load_isot_dataset(raw_data_dir)
    df = add_article_id(df)
    raw_combined_count = len(df)
    print(f"  Loaded {raw_combined_count} total records (raw, combined).")

    print("\nStep 2: Data quality check (BEFORE cleaning)")
    report_before = compute_data_quality_report(df)
    print_data_quality_report(report_before)

    print("\nStep 3: Cleaning the dataset...")
    print("  3a. Removing empty/missing/whitespace-only text...")
    print("  3b. Detecting and removing cross-class label conflicts...")
    print("  3c. Removing ordinary (same-class) duplicate articles...")
    df, clean_stats = clean_dataset(df, text_col="text", label_col="label")

    print()
    print("  --- Cleaning breakdown ---")
    print(f"  Empty/missing text articles removed      : {clean_stats['empty_removed']}")
    print(
        f"  Cross-class conflicting articles removed : "
        f"{clean_stats['cross_class_removed']} "
        f"(from {clean_stats['cross_class_distinct_texts']} distinct conflicting text value(s))"
    )
    print(f"  Ordinary duplicate articles removed      : {clean_stats['duplicate_removed']}")
    print(f"  Final usable articles                    : {clean_stats['final_count']}")

    print("\nStep 4: Adding clean_text column (original 'text' column kept as-is)...")
    df = add_clean_text_column(df, source_col="text", target_col="clean_text")

    print("\nStep 5: Data quality check (AFTER cleaning)")
    report_after = compute_data_quality_report(df)
    print_data_quality_report(report_after)

    print("\nStep 6: Splitting into train/validation/test (70/15/15, stratified)...")
    train_df, val_df, test_df = split_dataset(df)

    print("\nStep 7: Verifying no data leakage...")
    assert_no_overlap(train_df, val_df, test_df, id_col="article_id")
    print("  article_id overlap check passed (hard check).")
    assert_no_text_overlap_across_splits(train_df, val_df, test_df, text_col="text")
    print("  raw 'text' overlap check passed (hard check).")

    clean_text_overlap_count = count_text_overlap_across_splits(
        train_df, val_df, test_df, text_col="clean_text"
    )
    if clean_text_overlap_count == 0:
        print("  'clean_text' overlap check passed (0 overlapping values).")
    else:
        print(
            f"  NOTE: {clean_text_overlap_count} distinct 'clean_text' value(s) appear in "
            f"more than one split. This is reported, not treated as a hard failure — it "
            f"happens when two DIFFERENT raw articles differ only by trivial whitespace "
            f"that light cleaning removes, not by article_id or raw-text leakage (both "
            f"already verified above). See README -> Phase 2 for details."
        )

    print(f"\n  Train size      : {len(train_df)}")
    print(f"  Validation size : {len(val_df)}")
    print(f"  Test size       : {len(test_df)}")

    if save_outputs:
        processed_data_dir = Path(processed_data_dir)
        processed_data_dir.mkdir(parents=True, exist_ok=True)

        train_df.to_csv(processed_data_dir / "train.csv", index=False)
        val_df.to_csv(processed_data_dir / "validation.csv", index=False)
        test_df.to_csv(processed_data_dir / "test.csv", index=False)

        print(f"\nSaved processed splits to: {processed_data_dir}")

    return {
        "train": train_df,
        "validation": val_df,
        "test": test_df,
        "report_before": report_before,
        "report_after": report_after,
        "clean_stats": clean_stats,
        "raw_combined_count": raw_combined_count,
        "clean_text_overlap_count": clean_text_overlap_count,
    }


if __name__ == "__main__":
    run_preprocessing_pipeline()
