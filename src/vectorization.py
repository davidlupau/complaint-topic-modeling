"""TF-IDF and CountVectorizer pipelines for the LSA/LDA topic models, with reporting and persistence."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — OUTPUT LOCATION
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
VECTOR_DIR = PROJECT_ROOT / "data" / "vectorized"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — VECTORIZATION STEPS
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_vectorizer(series: pd.Series) -> tuple[TfidfVectorizer, spmatrix]:
    """Fit a TfidfVectorizer on cleaned complaint text for the LSA pipeline.

    Uses TF-IDF weights (use_idf/smooth_idf) since LSA's singular value
    decomposition operates on term importance rather than raw counts.
    min_df/max_df/ngram_range/max_features shape the vocabulary identically
    to build_count_vectorizer, so both pipelines model the same terms.

    Parameters:
        series (pd.Series): Cleaned complaint text to vectorize, one document
            per row.

    Returns:
        tuple[TfidfVectorizer, scipy.sparse.spmatrix]: The fitted vectorizer
            and the resulting document-term matrix.
    """
    vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.5,
        ngram_range=(1, 1),
        max_features=None,
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
    )
    matrix = vectorizer.fit_transform(series)
    print(f"Fitted TfidfVectorizer on {len(series)} documents")
    return vectorizer, matrix


def build_count_vectorizer(series: pd.Series) -> tuple[CountVectorizer, spmatrix]:
    """Fit a CountVectorizer on cleaned complaint text for the LDA pipeline.

    Uses raw term counts with no IDF weighting, since LDA models documents as
    a mixture of topics generated from a Dirichlet-multinomial process over
    word counts. Shares identical min_df/max_df/ngram_range/max_features with
    build_tfidf_vectorizer so both pipelines model the same vocabulary.

    Parameters:
        series (pd.Series): Cleaned complaint text to vectorize, one document
            per row.

    Returns:
        tuple[CountVectorizer, scipy.sparse.spmatrix]: The fitted vectorizer
            and the resulting document-term matrix.
    """
    vectorizer = CountVectorizer(
        min_df=5,
        max_df=0.5,
        ngram_range=(1, 1),
        max_features=None,
    )
    matrix = vectorizer.fit_transform(series)
    print(f"Fitted CountVectorizer on {len(series)} documents")
    return vectorizer, matrix


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — VECTORIZATION REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def summarize_vectorizer(vectorizer: TfidfVectorizer | CountVectorizer,
                         matrix: spmatrix) -> dict:
    """Print vocabulary size, matrix shape and sparsity for a fitted vectorizer.

    Parameters:
        vectorizer (TfidfVectorizer | CountVectorizer): A fitted scikit-learn
            vectorizer.
        matrix (scipy.sparse.spmatrix): The document-term matrix produced by
            vectorizer.fit_transform().

    Returns:
        dict: "vocab_size", "shape" and "sparsity" (fraction of zero entries).
    """
    vocab_size = len(vectorizer.vocabulary_)
    shape = matrix.shape
    sparsity = 1 - (matrix.nnz / (shape[0] * shape[1]))

    print(f"Vocabulary size: {vocab_size}")
    print(f"Matrix shape: {shape[0]} documents x {shape[1]} terms")
    print(f"Sparsity: {sparsity:.4%}")

    return {"vocab_size": vocab_size, "shape": shape, "sparsity": sparsity}


def top_terms_by_document_frequency(vectorizer: TfidfVectorizer | CountVectorizer,
                                    matrix: spmatrix, top_n: int = 20) -> pd.Series:
    """Return the top_n vocabulary terms ranked by document frequency.

    Document frequency counts how many documents contain each term at least
    once, regardless of raw term counts or TF-IDF weight. This is the quantity
    min_df/max_df filtering actually operates on, so it is the right lens for
    checking whether that filtering behaved as intended.

    Parameters:
        vectorizer (TfidfVectorizer | CountVectorizer): A fitted scikit-learn
            vectorizer exposing get_feature_names_out().
        matrix (scipy.sparse.spmatrix): The document-term matrix produced by
            vectorizer.fit_transform().
        top_n (int): Number of top terms to return. Defaults to 20.

    Returns:
        pd.Series: The top_n terms indexed by term name, with their document
            frequency counts, ordered from most to least frequent.
    """
    doc_freq = np.asarray((matrix > 0).sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    freq_series = pd.Series(doc_freq, index=terms, name="doc_frequency")
    freq_series.index.name = "term"
    return freq_series.sort_values(ascending=False).head(top_n)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — SAVING ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

def save_vectorization_artifacts(vectorizer: TfidfVectorizer | CountVectorizer,
                                 matrix: spmatrix, name: str) -> None:
    """Persist a fitted vectorizer and its document-term matrix to data/vectorized/.

    The vectorizer is saved with joblib and the sparse matrix with scipy's npz
    format, creating the folder if needed, so later notebooks can load both
    without refitting.

    Parameters:
        vectorizer (TfidfVectorizer | CountVectorizer): A fitted scikit-learn
            vectorizer to persist.
        matrix (scipy.sparse.spmatrix): The document-term matrix to persist.
        name (str): Base name shared by both artifacts, e.g. "tfidf" produces
            tfidf_vectorizer.joblib and tfidf_matrix.npz.

    Returns:
        None
    """
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    vectorizer_path = VECTOR_DIR / f"{name}_vectorizer.joblib"
    matrix_path = VECTOR_DIR / f"{name}_matrix.npz"

    joblib.dump(vectorizer, vectorizer_path)
    sparse.save_npz(matrix_path, matrix)

    print(f"Saved vectorizer to: {vectorizer_path}")
    print(f"Saved matrix to: {matrix_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_vectorization(df: pd.DataFrame, text_column: str = "cleaned_complaint") -> dict:
    """Run both vectorization pipelines over the cleaned corpus and save artifacts.

    Fits Pipeline A (TF-IDF, for LSA) and Pipeline B (CountVectorizer, for
    LDA) on the same text column, reports vocabulary size, matrix shape,
    sparsity and the top terms by document frequency for each, and persists
    both fitted vectorizers and matrices to data/vectorized/.

    Parameters:
        df (pd.DataFrame): The processed complaints DataFrame.
        text_column (str): Name of the cleaned text column to vectorize.
            Defaults to "cleaned_complaint".

    Returns:
        dict: "tfidf_vectorizer", "tfidf_matrix", "count_vectorizer" and
            "count_matrix" holding the fitted vectorizers and matrices for
            reuse without reloading from disk.
    """
    print("\nRunning vectorization pipelines...\n")

    print("--- Pipeline A: TF-IDF ---")
    tfidf_vectorizer, tfidf_matrix = build_tfidf_vectorizer(df[text_column])
    summarize_vectorizer(tfidf_vectorizer, tfidf_matrix)
    print(top_terms_by_document_frequency(tfidf_vectorizer, tfidf_matrix))
    save_vectorization_artifacts(tfidf_vectorizer, tfidf_matrix, "tfidf")

    print("\n--- Pipeline B: CountVectorizer ---")
    count_vectorizer, count_matrix = build_count_vectorizer(df[text_column])
    summarize_vectorizer(count_vectorizer, count_matrix)
    print(top_terms_by_document_frequency(count_vectorizer, count_matrix))
    save_vectorization_artifacts(count_vectorizer, count_matrix, "count")

    return {
        "tfidf_vectorizer": tfidf_vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "count_vectorizer": count_vectorizer,
        "count_matrix": count_matrix,
    }
