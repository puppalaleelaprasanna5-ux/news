"""Rule-based selection of sentences to search evidence for (replaces LLM claim extraction).

This is a simple heuristic, NOT claim verification and NOT natural-language understanding.
It picks sentences that look like checkable statements: they contain figures, named things
(capitalised words), or reporting/assertion verbs. It can pick poor sentences, miss real
claims, and cannot tell whether a sentence is true. Output order follows the article.
"""
import re
from typing import List

MIN_WORDS = 6
MAX_WORDS = 60
DEFAULT_MAX_CLAIMS = 5

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_DIGIT = re.compile(r"\d")
_ASSERTION_WORDS = {
    "said", "says", "announced", "announce", "announces", "claimed", "claims", "reported", "reports",
    "confirmed", "confirms", "stated", "states", "according", "voted", "approved", "rejected", "signed",
    "ordered", "plans", "planned", "will", "won", "lost", "killed", "arrested", "found", "revealed",
    "increased", "decreased", "rose", "fell", "passed", "banned", "introduced", "denied",
}


def _score(sentence: str) -> int:
    words = sentence.split()
    score = 0
    if _DIGIT.search(sentence):
        score += 2
    # capitalised words that are not the first word: a rough proxy for names/places/organisations
    score += min(3, sum(1 for w in words[1:] if w[:1].isupper()))
    lowered = {re.sub(r"\W+", "", w.lower()) for w in words}
    if lowered & _ASSERTION_WORDS:
        score += 2
    if sentence.rstrip().endswith("?"):
        score -= 3
    return score


def select_claims(text: str, max_claims: int = DEFAULT_MAX_CLAIMS) -> List[str]:
    """Return up to `max_claims` candidate sentences (highest heuristic score), in article order."""
    if not isinstance(text, str) or not text.strip() or max_claims <= 0:
        return []
    candidates = []
    for position, raw in enumerate(_SENTENCE_SPLIT.split(" ".join(text.split()))):
        sentence = raw.strip()
        n_words = len(sentence.split())
        if MIN_WORDS <= n_words <= MAX_WORDS:
            candidates.append((position, sentence, _score(sentence)))
    # keep only sentences with a positive score; best first, earlier sentence wins ties
    ranked = sorted((c for c in candidates if c[2] > 0), key=lambda c: (-c[2], c[0]))[:max_claims]
    return [sentence for _, sentence, _ in sorted(ranked, key=lambda c: c[0])]
