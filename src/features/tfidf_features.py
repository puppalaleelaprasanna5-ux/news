"""
TF-IDF feature extraction for the fake-news classifier.

DATA LEAKAGE RULE (see Phase 2/3 docs for the same principle applied elsewhere):
    The vectorizer (and any dimensionality reduction built on top of it) must be
    FIT only on the training set. Validation and test data are only ever
    TRANSFORMED with the already-fitted vectorizer, never used to refit it.
    The functions below are split into fit_* and transform_* specifically so
    this can't be done by accident - there is no single "fit_transform on
    whatever you hand it" convenience function here on purpose.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Defaults chosen for a ~27k-document training set on modest hardware
# (see README -> Phase 4 for the reasoning and measured vocabulary size).
DEFAULT_NGRAM_RANGE = (1, 2)
DEFAULT_MAX_FEATURES = 30000
DEFAULT_MIN_DF = 3          # ignore extremely rare terms (likely typos/noise)
DEFAULT_MAX_DF = 0.9        # ignore terms that appear in >90% of documents (near-stopwords)


def build_tfidf_vectorizer(ngram_range=DEFAULT_NGRAM_RANGE,
                            max_features=DEFAULT_MAX_FEATURES,
                            min_df=DEFAULT_MIN_DF,
                            max_df=DEFAULT_MAX_DF) -> TfidfVectorizer:
    """Construct an (unfitted) TF-IDF vectorizer with project-standard defaults."""
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        lowercase=True,
        strip_accents="unicode",
        sublinear_tf=True,  # log-scale term frequency, standard for text classification
    )


def fit_tfidf_on_train(vectorizer: TfidfVectorizer, train_texts):
    """
    Fit the vectorizer on TRAINING text only, and return the transformed
    training matrix. This is the ONLY function in this module that calls
    vectorizer.fit(...) - everything else is transform-only.
    """
    X_train = vectorizer.fit_transform(train_texts)
    return X_train


def transform_with_tfidf(vectorizer: TfidfVectorizer, texts):
    """
    Transform already-fitted-vectorizer text (validation/test/new data).
    Raises sklearn's NotFittedError if the vectorizer hasn't been fit yet -
    that's intentional, it's a safety net against accidentally skipping
    fit_tfidf_on_train().
    """
    return vectorizer.transform(texts)


def fit_svd_on_train(X_train_tfidf, n_components: int, random_state: int = 42) -> TruncatedSVD:
    """
    Fit TruncatedSVD (dimensionality reduction) on the TRAINING TF-IDF matrix
    only. Used for tree-based models (Random Forest, XGBoost) that don't
    scale well to a huge sparse TF-IDF matrix directly - see README -> Phase 4
    for why this architecture (TF-IDF -> SVD -> tree model) was chosen.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    svd.fit(X_train_tfidf)
    return svd


def transform_with_svd(svd: TruncatedSVD, X_tfidf):
    """Transform-only: project already-fitted-SVD TF-IDF vectors into the reduced space."""
    return svd.transform(X_tfidf)
