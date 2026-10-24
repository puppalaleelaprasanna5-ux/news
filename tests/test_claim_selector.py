"""Tests for the rule-based claim/sentence selector (replaces LLM claim extraction)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.claim_selector import MAX_WORDS, MIN_WORDS, select_claims

TEXT = ("The city council said on Monday it will repair the Alder River bridge at a cost of 4 million dollars. "
        "It was a nice day. Why does this keep happening? Mayor Jane Smith confirmed the work begins in March 2027. "
        "Residents were happy about it overall and went home.")


def test_picks_sentences_with_figures_names_or_assertion_verbs_in_article_order():
    out = select_claims(TEXT)
    assert out == ["The city council said on Monday it will repair the Alder River bridge at a cost of 4 million dollars.",
                   "Mayor Jane Smith confirmed the work begins in March 2027."]


def test_selected_claims_are_verbatim_sentences_from_the_text():
    for claim in select_claims(TEXT):
        assert claim in TEXT


def test_questions_and_very_short_or_long_sentences_are_skipped():
    assert select_claims("Is the Senate going to vote on 5 bills today?") == []
    assert select_claims("Senate voted 5 times.") == []                                    # shorter than MIN_WORDS
    assert select_claims(" ".join(["Senate voted 5 times today"] * 40) + ".") == []        # longer than MAX_WORDS
    assert MIN_WORDS < MAX_WORDS


def test_respects_max_claims_and_handles_bad_input():
    many = " ".join(f"The council said that project number {i} will cost {i} million dollars in total." for i in range(1, 9))
    assert len(select_claims(many, max_claims=3)) == 3
    assert select_claims(many, max_claims=0) == []
    for bad in ("", "   ", None, 42, ["x"]):
        assert select_claims(bad) == []


def test_plain_text_with_nothing_checkable_returns_nothing():
    assert select_claims("just plain lowercase words " * 6) == []


def test_is_deterministic():
    assert select_claims(TEXT) == select_claims(TEXT)
