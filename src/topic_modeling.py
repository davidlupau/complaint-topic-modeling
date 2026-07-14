"""LSA/LDA topic modeling: k-selection, fitting, topic inspection and pipeline comparison."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.matutils import Sparse2Corpus
from gensim.models import CoherenceModel
from scipy.sparse import spmatrix
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — OUTPUT LOCATION
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analysis_output"


def _output_path(filename: str) -> Path:
    """Return the full path for a plot inside analysis_output/, creating the folder.

    Parameters:
        filename (str): Name of the PNG file to save inside analysis_output/.

    Returns:
        Path: Absolute path where the plot should be written.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / filename


def _slugify(title: str) -> str:
    """Turn a plot title into a filesystem-safe filename stem.

    Parameters:
        title (str): The plot title to convert.

    Returns:
        str: Lowercase, underscore-separated version of title.
    """
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")


def _top_word_indices(component: np.ndarray, n_words: int) -> np.ndarray:
    """Return the indices of the n_words highest-weighted terms in a component.

    Parameters:
        component (np.ndarray): One row of a model's components_ matrix
            (a topic's weight over the vocabulary).
        n_words (int): Number of top-weighted terms to return.

    Returns:
        np.ndarray: Indices of the top n_words terms, descending by weight.
    """
    return component.argsort()[::-1][:n_words]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — K SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_lsa_k(tfidf_matrix: spmatrix, k_range: range = range(3, 11),
                   random_state: int = 42) -> pd.DataFrame:
    """Fit TruncatedSVD across a range of k and report total explained variance.

    Parameters:
        tfidf_matrix (scipy.sparse.spmatrix): The TF-IDF document-term matrix.
        k_range (range): Candidate numbers of components to try. Defaults to
            range(3, 11).
        random_state (int): Random seed for TruncatedSVD. Defaults to 42.

    Returns:
        pd.DataFrame: Columns "k" and "explained_variance" (total explained
            variance ratio summed across components), one row per k.
    """
    print("\nEvaluating LSA explained variance across k...\n")

    records = []
    for k in k_range:
        svd = TruncatedSVD(n_components=k, random_state=random_state)
        svd.fit(tfidf_matrix)
        explained_variance = svd.explained_variance_ratio_.sum()

        print(f"k={k}: explained variance = {explained_variance:.4f}")
        records.append({"k": k, "explained_variance": explained_variance})

    return pd.DataFrame(records)


def evaluate_lda_k(count_matrix: spmatrix, vectorizer: CountVectorizer,
                   k_range: range = range(3, 11), random_state: int = 42) -> pd.DataFrame:
    """Fit LatentDirichletAllocation across a range of k and report u_mass coherence.

    The gensim Dictionary and BoW corpus are built directly from the fitted
    CountVectorizer's vocabulary and the count matrix (via Sparse2Corpus and
    Dictionary.from_corpus), so the original text never needs to be
    re-tokenized for coherence scoring.

    Parameters:
        count_matrix (scipy.sparse.spmatrix): The raw-count document-term
            matrix.
        vectorizer (CountVectorizer): The fitted CountVectorizer that produced
            count_matrix.
        k_range (range): Candidate numbers of topics to try. Defaults to
            range(3, 11).
        random_state (int): Random seed for LatentDirichletAllocation.
            Defaults to 42.

    Returns:
        pd.DataFrame: Columns "k" and "coherence" (u_mass coherence score),
            one row per k.
    """
    print("\nEvaluating LDA u_mass coherence across k...\n")

    feature_names = vectorizer.get_feature_names_out()
    id2word = {idx: term for term, idx in vectorizer.vocabulary_.items()}
    corpus = Sparse2Corpus(count_matrix, documents_columns=False)
    dictionary = Dictionary.from_corpus(corpus, id2word=id2word)

    records = []
    for k in k_range:
        lda = LatentDirichletAllocation(n_components=k, random_state=random_state)
        lda.fit(count_matrix)

        topics = [
            [feature_names[i] for i in _top_word_indices(component, 10)]
            for component in lda.components_
        ]
        coherence = CoherenceModel(
            topics=topics, corpus=corpus, dictionary=dictionary, coherence="u_mass"
        ).get_coherence()

        print(f"k={k}: u_mass coherence = {coherence:.4f}")
        records.append({"k": k, "coherence": coherence})

    return pd.DataFrame(records)


def plot_k_selection(results_df: pd.DataFrame, metric_col: str, title: str) -> None:
    """Plot a k-selection metric across k and save it to analysis_output/.

    Parameters:
        results_df (pd.DataFrame): Output of evaluate_lsa_k() or
            evaluate_lda_k(), with a "k" column and a metric_col column.
        metric_col (str): Name of the column to plot on the y-axis.
        title (str): Plot title, also used to derive the saved filename.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["k"], results_df[metric_col], marker="o", color="#FF6B6B")
    ax.set_title(title)
    ax.set_xlabel("Number of topics (k)")
    ax.set_ylabel(metric_col.replace("_", " ").title())
    ax.set_xticks(results_df["k"])
    fig.tight_layout()

    save_path = _output_path(f"{_slugify(title)}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — FINAL MODEL FITTING
# ─────────────────────────────────────────────────────────────────────────────

def fit_lsa(tfidf_matrix: spmatrix, n_components: int, random_state: int = 42) -> TruncatedSVD:
    """Fit the final TruncatedSVD (LSA) model at a chosen number of topics.

    Parameters:
        tfidf_matrix (scipy.sparse.spmatrix): The TF-IDF document-term matrix.
        n_components (int): Chosen number of topics.
        random_state (int): Random seed. Defaults to 42.

    Returns:
        TruncatedSVD: The fitted LSA model.
    """
    model = TruncatedSVD(n_components=n_components, random_state=random_state)
    model.fit(tfidf_matrix)
    print(f"Fitted LSA with {n_components} components "
          f"(explained variance: {model.explained_variance_ratio_.sum():.4f})")
    return model


def fit_lda(count_matrix: spmatrix, n_components: int,
           random_state: int = 42) -> LatentDirichletAllocation:
    """Fit the final LatentDirichletAllocation (LDA) model at a chosen number of topics.

    Parameters:
        count_matrix (scipy.sparse.spmatrix): The raw-count document-term
            matrix.
        n_components (int): Chosen number of topics.
        random_state (int): Random seed. Defaults to 42.

    Returns:
        LatentDirichletAllocation: The fitted LDA model.
    """
    model = LatentDirichletAllocation(n_components=n_components, random_state=random_state)
    model.fit(count_matrix)
    print(f"Fitted LDA with {n_components} components")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — TOPIC INSPECTION
# ─────────────────────────────────────────────────────────────────────────────

def get_top_words(model: TruncatedSVD | LatentDirichletAllocation,
                  vectorizer: TfidfVectorizer | CountVectorizer,
                  n_topics: int, n_words: int = 10) -> dict:
    """Return the top n_words terms for each topic of a fitted model.

    Works for either model type: both TruncatedSVD (LSA) and
    LatentDirichletAllocation (LDA) expose a components_ matrix of shape
    (n_topics, n_features), so the same argsort-based extraction applies to
    both.

    Parameters:
        model (TruncatedSVD | LatentDirichletAllocation): A fitted topic model.
        vectorizer (TfidfVectorizer | CountVectorizer): The fitted vectorizer
            that produced the matrix model was trained on (supplies term
            names via get_feature_names_out()).
        n_topics (int): Number of topics to extract (model.n_components).
        n_words (int): Number of top words per topic. Defaults to 10.

    Returns:
        dict: {topic_idx: [top words]}, one entry per topic.
    """
    feature_names = vectorizer.get_feature_names_out()

    top_words = {}
    for topic_idx in range(n_topics):
        indices = _top_word_indices(model.components_[topic_idx], n_words)
        words = [feature_names[i] for i in indices]
        top_words[topic_idx] = words
        print(f"Topic {topic_idx}: {', '.join(words)}")

    return top_words


def plot_top_words(top_words_dict: dict, title: str) -> None:
    """Plot the top words per topic as one horizontal bar chart per topic.

    Bar length reflects each word's rank within its topic (most important at
    the top), since get_top_words() only returns ordered words, not weights.

    Parameters:
        top_words_dict (dict): Output of get_top_words(), {topic_idx: [words]}.
        title (str): Plot title, also used to derive the saved filename.

    Returns:
        None
    """
    n_topics = len(top_words_dict)
    fig, axes = plt.subplots(1, n_topics, figsize=(4 * n_topics, 6), squeeze=False)
    axes = axes[0]

    for topic_idx, words in top_words_dict.items():
        ax = axes[topic_idx]
        ranks = list(range(len(words), 0, -1))
        ax.barh(words[::-1], ranks[::-1], color="#FF6B6B", edgecolor="white")
        ax.set_title(f"Topic {topic_idx}")
        ax.set_xlabel("Rank")

    fig.suptitle(title)
    fig.tight_layout()

    save_path = _output_path(f"{_slugify(title)}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PIPELINE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_topic_overlap(lsa_top_words: dict, lda_top_words: dict) -> pd.DataFrame:
    """Compute pairwise Jaccard similarity between LSA and LDA topic word sets.

    Parameters:
        lsa_top_words (dict): Output of get_top_words() for the LSA model.
        lda_top_words (dict): Output of get_top_words() for the LDA model.

    Returns:
        pd.DataFrame: LSA topics as rows, LDA topics as columns, each cell the
            Jaccard similarity between the two topics' word sets.
    """
    print("\nComparing topic overlap (Jaccard similarity) between LSA and LDA...\n")

    rows = {}
    for lsa_idx, lsa_words in lsa_top_words.items():
        lsa_set = set(lsa_words)
        row = {}
        for lda_idx, lda_words in lda_top_words.items():
            lda_set = set(lda_words)
            union = lsa_set | lda_set
            row[f"LDA Topic {lda_idx}"] = len(lsa_set & lda_set) / len(union) if union else 0.0
        rows[f"LSA Topic {lsa_idx}"] = row

    overlap_df = pd.DataFrame(rows).T
    print(overlap_df.round(3).to_string())
    return overlap_df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — DOCUMENT-LEVEL ASSIGNMENT & VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def assign_dominant_topic(doc_topic_matrix: np.ndarray) -> np.ndarray:
    """Assign each document its single highest-weighted topic.

    Works for either pipeline's document-topic matrix (LSA's transform()
    output or LDA's transform() output), since both are dense arrays of
    shape (n_documents, n_topics).

    Parameters:
        doc_topic_matrix (np.ndarray): Document-topic weights, shape
            (n_documents, n_topics).

    Returns:
        np.ndarray: The argmax topic index per document, shape (n_documents,).
    """
    dominant = np.asarray(doc_topic_matrix).argmax(axis=1)
    print(f"Assigned dominant topics for {dominant.shape[0]} documents")
    return dominant


def plot_topic_vs_product(dominant_topics: np.ndarray, product_labels: pd.Series,
                          title: str) -> None:
    """Plot a Product x dominant-topic cross-tab heatmap and save it.

    This is a post-hoc validation check only: Product is never used as model
    input, so any alignment between dominant topics and Product categories
    reflects genuine thematic clustering, not information leakage.

    Parameters:
        dominant_topics (np.ndarray): Per-document dominant topic index, from
            assign_dominant_topic().
        product_labels (pd.Series): The Product column, aligned positionally
            with dominant_topics.
        title (str): Plot title, also used to derive the saved filename.

    Returns:
        None
    """
    crosstab = pd.crosstab(
        pd.Series(product_labels).reset_index(drop=True),
        pd.Series(dominant_topics, name="Dominant topic"),
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(crosstab.values, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_xticklabels(crosstab.columns)
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_yticklabels(crosstab.index)
    ax.set_xlabel("Dominant topic")
    ax.set_ylabel("Product")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Number of complaints")

    for i in range(crosstab.shape[0]):
        for j in range(crosstab.shape[1]):
            ax.text(j, i, crosstab.values[i, j], ha="center", va="center",
                    color="black", fontsize=8)

    fig.tight_layout()
    save_path = _output_path(f"{_slugify(title)}.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

def run_topic_modeling(df: pd.DataFrame, vectorization_results: dict) -> dict:
    """Run the full topic modeling pipeline end to end and save its outputs.

    Selects the number of topics (k) automatically as the k with the highest
    LDA u_mass coherence over the default k_range, and reuses that same k for
    LSA so the two pipelines' topics are directly comparable. This is a
    headless default for non-interactive runs (e.g. via main.py); the
    notebook instead lets k be chosen interactively from the k-selection
    plots this function also produces.

    Parameters:
        df (pd.DataFrame): The processed complaints DataFrame, including the
            Product column used for post-hoc validation.
        vectorization_results (dict): Output of run_vectorization(), holding
            "tfidf_vectorizer", "tfidf_matrix", "count_vectorizer" and
            "count_matrix".

    Returns:
        dict: "chosen_k", "lsa_model", "lda_model", "lsa_top_words",
            "lda_top_words", "topic_overlap", "lsa_dominant_topics" and
            "lda_dominant_topics".
    """
    print("\nRunning topic modeling pipelines...\n")

    tfidf_vectorizer = vectorization_results["tfidf_vectorizer"]
    tfidf_matrix = vectorization_results["tfidf_matrix"]
    count_vectorizer = vectorization_results["count_vectorizer"]
    count_matrix = vectorization_results["count_matrix"]

    print("--- Selecting k for LSA ---")
    lsa_k_results = evaluate_lsa_k(tfidf_matrix)
    plot_k_selection(lsa_k_results, "explained_variance", "LSA explained variance by k")

    print("\n--- Selecting k for LDA ---")
    lda_k_results = evaluate_lda_k(count_matrix, count_vectorizer)
    plot_k_selection(lda_k_results, "coherence", "LDA u_mass coherence by k")

    chosen_k = int(lda_k_results.loc[lda_k_results["coherence"].idxmax(), "k"])
    print(f"\nChosen k (max LDA coherence): {chosen_k}")

    print("\n--- Fitting final models ---")
    lsa_model = fit_lsa(tfidf_matrix, chosen_k)
    lda_model = fit_lda(count_matrix, chosen_k)

    print("\n--- LSA top words ---")
    lsa_top_words = get_top_words(lsa_model, tfidf_vectorizer, chosen_k)
    print("\n--- LDA top words ---")
    lda_top_words = get_top_words(lda_model, count_vectorizer, chosen_k)

    plot_top_words(lsa_top_words, "Top words per LSA topic")
    plot_top_words(lda_top_words, "Top words per LDA topic")

    topic_overlap = compare_topic_overlap(lsa_top_words, lda_top_words)

    lsa_dominant_topics = assign_dominant_topic(lsa_model.transform(tfidf_matrix))
    lda_dominant_topics = assign_dominant_topic(lda_model.transform(count_matrix))

    plot_topic_vs_product(lsa_dominant_topics, df["Product"], "LSA dominant topic vs Product")
    plot_topic_vs_product(lda_dominant_topics, df["Product"], "LDA dominant topic vs Product")

    return {
        "chosen_k": chosen_k,
        "lsa_model": lsa_model,
        "lda_model": lda_model,
        "lsa_top_words": lsa_top_words,
        "lda_top_words": lda_top_words,
        "topic_overlap": topic_overlap,
        "lsa_dominant_topics": lsa_dominant_topics,
        "lda_dominant_topics": lda_dominant_topics,
    }
