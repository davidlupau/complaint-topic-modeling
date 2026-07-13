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
# SECTION 2 — VECTORIZATION REPORTING
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
# SECTION 3 — SAVING ARTIFACTS
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
