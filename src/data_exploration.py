"""Exploratory data analysis checks on the raw complaints corpus."""

import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — OUTPUT LOCATION
# ─────────────────────────────────────────────────────────────────────────────

# Resolve the project root the same way utils.py locates data/ (go up from src/)
# so plots always land in <project_root>/analysis_output/ regardless of the
# current working directory.
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "analysis_output"

TEXT_COL = "Consumer_complaint"
CLASS_COL = "Product"


def _output_path(file_name: str) -> Path:
    """Return the full path for a plot inside analysis_output/, creating the folder.

    Parameters:
        file_name (str): Name of the PNG file to save inside analysis_output/.

    Returns:
        Path: Absolute path where the plot should be written.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    return OUTPUT_DIR / file_name


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — EXPLORATORY DATA ANALYSIS CHECKS
# ─────────────────────────────────────────────────────────────────────────────

def check_shape_and_missing(df: pd.DataFrame) -> dict:
    """Report the shape, column dtypes and missing value counts of the dataset.

    Prints a summary to stdout with a focus on the free-text complaint column,
    so we know whether any narratives are missing before further analysis.

    Parameters:
        df (pd.DataFrame): The complaints DataFrame to inspect.

    Returns:
        dict: Contains "shape" (tuple), "dtypes" (Series) and "missing" (Series
            of missing counts per column).
    """
    print("\n=== Shape & missing values ===\n")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

    print("Column dtypes:")
    print(df.dtypes.to_string())

    missing = df.isna().sum()
    print("\nMissing values per column:")
    print(missing.to_string())

    text_missing = int(df[TEXT_COL].isna().sum())
    print(f"\nMissing '{TEXT_COL}' narratives: {text_missing}")

    return {"shape": df.shape, "dtypes": df.dtypes, "missing": missing}


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Count duplicated complaint narratives and show a few examples.

    Duplicate narratives can bias topic frequencies, so we surface how many exist
    and preview a few of them before deciding whether to drop them.

    Parameters:
        df (pd.DataFrame): The complaints DataFrame to inspect.

    Returns:
        pd.DataFrame: The rows whose narrative in the complaint column is
            duplicated (all occurrences), sorted so duplicates sit together.
    """
    print("\n=== Duplicate narratives ===\n")

    duplicate_mask = df[TEXT_COL].duplicated(keep=False)
    duplicates = df[duplicate_mask].sort_values(TEXT_COL)

    n_duplicated_rows = int(duplicate_mask.sum())
    n_extra_rows = int(df[TEXT_COL].duplicated().sum())
    print(f"Rows sharing a duplicated narrative: {n_duplicated_rows}")
    print(f"Redundant rows (duplicates beyond the first): {n_extra_rows}")

    if n_duplicated_rows:
        print("\nExamples of duplicated narratives:")
        examples = duplicates[TEXT_COL].drop_duplicates().head(3)
        for i, narrative in enumerate(examples, start=1):
            preview = narrative[:200].replace("\n", " ")
            print(f"\n[{i}] {preview}...")

    return duplicates


def check_class_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Report value counts and percentages for the Product class column.

    Class imbalance tells us whether some product categories dominate the corpus,
    which informs how we interpret and weight the topics discovered later.

    Parameters:
        df (pd.DataFrame): The complaints DataFrame to inspect.

    Returns:
        pd.DataFrame: One row per product category with "count" and "percentage"
            columns, sorted by count descending.
    """
    print("\n=== Class balance (Product) ===\n")

    counts = df[CLASS_COL].value_counts()
    percentages = (counts / len(df) * 100).round(2)
    balance = pd.DataFrame({"count": counts, "percentage": percentages})

    print(f"Number of categories: {balance.shape[0]}\n")
    print(balance.to_string())

    return balance


def check_text_length(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise word- and character-length distributions of the narratives.

    Prints min/max/mean/median for both measures and saves a histogram. Length
    statistics guide preprocessing choices such as minimum document length and
    the maximum sequence length used during vectorization.

    Parameters:
        df (pd.DataFrame): The complaints DataFrame to inspect.

    Returns:
        pd.DataFrame: A copy of the narratives with added "word_count" and
            "char_count" columns.
    """
    print("\n=== Text length distribution ===\n")

    text = df[TEXT_COL].fillna("").astype(str)
    lengths = pd.DataFrame({
        TEXT_COL: text,
        "word_count": text.str.split().str.len(),
        "char_count": text.str.len(),
    })

    for measure in ("word_count", "char_count"):
        stats = lengths[measure]
        print(f"{measure}: min={stats.min()}, max={stats.max()}, "
              f"mean={stats.mean():.1f}, median={stats.median():.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(lengths["word_count"], bins=50, color="#FF6B6B", edgecolor="white")
    axes[0].set_title("Word count distribution")
    axes[0].set_xlabel("Words per complaint")
    axes[0].set_ylabel("Number of complaints")
    axes[1].hist(lengths["char_count"], bins=50, color="#4D96FF", edgecolor="white")
    axes[1].set_title("Character count distribution")
    axes[1].set_xlabel("Characters per complaint")
    axes[1].set_ylabel("Number of complaints")
    fig.tight_layout()

    save_path = _output_path("text_length_distribution.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {save_path}")
    plt.show()

    return lengths


def check_frequent_tokens(df: pd.DataFrame, column: str = TEXT_COL, top_n: int = 30,
                          filename: str = "top_raw_tokens.png") -> pd.Series:
    """Return the most frequent whitespace-split tokens in a text column.

    Tokenization here is deliberately naive so that raw noise is exposed, such as
    the XXXX redaction placeholders and punctuation. Run on the raw narrative
    column, this reveals which stopwords, redaction tokens and symbols will need
    removing during preprocessing; run on a cleaned column, it instead serves as
    a post-cleaning sanity check.

    Parameters:
        df (pd.DataFrame): The complaints DataFrame to inspect.
        column (str): Name of the text column to tokenize. Defaults to
            TEXT_COL ("Consumer_complaint").
        top_n (int): Number of most frequent tokens to report. Defaults to 30.
        filename (str): Name of the PNG file to save inside analysis_output/.
            Defaults to "top_raw_tokens.png".

    Returns:
        pd.Series: The top_n tokens indexed by token, with their counts,
            ordered from most to least frequent.
    """
    print(f"\n=== Top {top_n} tokens in '{column}' ===\n")

    text = df[column].fillna("").astype(str).str.cat(sep=" ")
    # Split on whitespace but keep punctuation attached to words so that noise
    # (e.g. "account." or standalone symbols) stays visible.
    tokens = re.split(r"\s+", text.strip())
    counts = Counter(t for t in tokens if t)

    top_tokens = pd.Series(dict(counts.most_common(top_n)), name="count")
    top_tokens.index.name = "token"
    print(top_tokens.to_string())

    fig, ax = plt.subplots(figsize=(10, 8))
    ordered = top_tokens.iloc[::-1]  # smallest at top for horizontal bar chart
    ax.barh(ordered.index, ordered.values, color="#FF6B6B", edgecolor="white")
    ax.set_title(f"Top {top_n} most frequent tokens in '{column}'")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Token")
    fig.tight_layout()

    save_path = _output_path(filename)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {save_path}")
    plt.show()

    return top_tokens


def show_sample_complaints(df: pd.DataFrame, n: int = 5,
                           random_state: int = 42) -> pd.Series:
    """Print n randomly sampled full narratives for manual inspection.

    Reading whole complaints end-to-end reveals structure, tone and noise that
    aggregate statistics miss, helping shape preprocessing rules. The random_state
    keeps the sample reproducible across runs.

    Parameters:
        df (pd.DataFrame): The complaints DataFrame to inspect.
        n (int): Number of narratives to sample. Defaults to 5.
        random_state (int): Seed for reproducible sampling. Defaults to 42.

    Returns:
        pd.Series: The sampled narratives from the complaint column.
    """
    print(f"\n=== {n} sample complaints (random_state={random_state}) ===\n")

    sample = df[TEXT_COL].sample(n=n, random_state=random_state)
    for position, (idx, narrative) in enumerate(sample.items(), start=1):
        print(f"[{position}] (row {idx})")
        print(f"{narrative}\n")
        print("-" * 80)

    return sample
