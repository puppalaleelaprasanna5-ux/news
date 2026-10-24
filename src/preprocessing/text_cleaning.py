"""
Text cleaning utilities for news articles.

DESIGN PHILOSOPHY - why cleaning is deliberately LIGHT at this stage:

    Fake-news detection can depend on signals like capitalization
    (e.g. ALL CAPS HEADLINES), punctuation (e.g. excessive "!!!" or "???"),
    and exact wording. Aggressively lowercasing everything, stripping all
    punctuation, or removing stopwords at this stage could destroy useful
    signal before we've even tested whether it helps or hurts.

    This module only removes genuine NOISE, not information:
        - HTML tags / entities (rare in this dataset, handled defensively)
        - Non-printable / control characters
        - Redundant whitespace
        - Unicode inconsistencies (curly quotes, non-breaking spaces, etc.)

    Heavier, model-specific normalization (lowercasing, stopword removal,
    stemming) belongs in Phase 4 (feature engineering), where each choice
    can be tested and compared rather than applied blindly up front.

The ORIGINAL 'text' column is never modified. This module only ADDS a new
'clean_text' column, so the raw article text stays available for later use
(e.g. showing the original text in the final dashboard, or trying different
cleaning strategies later without reloading raw data).
"""

import html
import re
import unicodedata

import pandas as pd

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_WHITESPACE_RE = re.compile(r"\s+")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]")


def clean_single_text(text) -> str:
    """
    Apply light, safe cleaning to a single piece of text.

    Steps:
        1. Handle missing/non-string input -> return empty string
        2. Unescape HTML entities (e.g. "&amp;" -> "&")
        3. Strip any HTML tags (defensive; this dataset is plain text,
           but scraped news text sometimes carries stray tags)
        4. Normalize unicode (NFKC) - fixes curly quotes, odd spacing, etc.
        5. Remove control characters
        6. Collapse repeated whitespace into single spaces
        7. Strip leading/trailing whitespace

    Note: does NOT lowercase, remove punctuation, or remove stopwords.
    """
    if not isinstance(text, str):
        return ""

    text = html.unescape(text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub(" ", text)
    text = _MULTI_WHITESPACE_RE.sub(" ", text)
    text = text.strip()

    return text


def add_clean_text_column(df: pd.DataFrame, source_col: str = "text",
                           target_col: str = "clean_text") -> pd.DataFrame:
    """
    Add a cleaned version of `source_col` as a new `target_col`.

    The original `source_col` is left completely untouched — we return a
    new DataFrame with an additional column rather than overwriting anything.
    """
    df = df.copy()
    df[target_col] = df[source_col].apply(clean_single_text)
    return df
