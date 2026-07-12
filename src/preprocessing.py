import re
import string
from pathlib import Path

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — NLTK DATA
# ─────────────────────────────────────────────────────────────────────────────

_REQUIRED_NLTK_DATA = {
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def _ensure_nltk_data() -> None:
    """Download required NLTK corpora/models if they aren't already present.

    Checks each resource before downloading so repeated imports stay silent
    and fast once the data is cached locally.
    """
    for package, resource_path in _REQUIRED_NLTK_DATA.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_data()

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

TEXT_COL = "Consumer_complaint"
CLASS_COL = "Product"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PREPROCESSING STEPS
# ─────────────────────────────────────────────────────────────────────────────

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns needed for topic modeling.

    Sentiment and Priority are structured metadata excluded from topic
    extraction per the project's conception phase. Product is retained only
    as a later external validation signal, never as model input.

    Parameters:
        df (pd.DataFrame): The raw complaints DataFrame.

    Returns:
        pd.DataFrame: A copy containing only Consumer_complaint and Product.
    """
    selected = df[[TEXT_COL, CLASS_COL]].copy()
    print(f"Selected columns: {list(selected.columns)} ({selected.shape[0]} rows)")
    return selected


def clean_text(text: str) -> str:
    """Lowercase and strip noise from a single complaint narrative.

    Removes the dataset's XXXX anonymization tokens, punctuation and digits,
    then collapses repeated whitespace left behind by the removals. The source
    narratives pre-split some contractions/clitics with a leading space (e.g.
    "would n't", "account 's"), so "n't" is reunited with its preceding word
    while "'s"/"'re"/"'m"/"'ve"/"'d"/"'ll" fragments are dropped entirely rather
    than concatenated, since e.g. "account 's" -> "accounts" would wrongly
    collide with the genuine plural "accounts" already in the vocabulary.

    Parameters:
        text (str): Raw complaint narrative.

    Returns:
        str: The cleaned narrative.
    """
    text = text.lower()
    text = re.sub(r"\bx{2,}\b", " ", text)
    text = re.sub(r"(\w+) n['’]t\b", r"\1nt", text)
    text = re.sub(r" ['’](?:s|re|m|ve|d|ll)\b", "", text)
    text = text.replace("'", "").replace("’", "")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_and_remove_stopwords(text: str) -> list[str]:
    """Tokenize a cleaned narrative and drop standard English stopwords.

    Parameters:
        text (str): Cleaned complaint narrative (see clean_text).

    Returns:
        list[str]: Tokens with stopwords removed.
    """
    tokens = word_tokenize(text)
    return [token for token in tokens if token not in STOPWORDS]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatize a list of tokens using the default noun part of speech.

    Parameters:
        tokens (list[str]): Tokens to lemmatize.

    Returns:
        list[str]: Lemmatized tokens.
    """
    return [LEMMATIZER.lemmatize(token) for token in tokens]


def preprocess_corpus(df: pd.DataFrame, min_tokens: int = 5) -> pd.DataFrame:
    """Run the full preprocessing pipeline over the complaints DataFrame.

    Selects the relevant columns, cleans each narrative, tokenizes and removes
    stopwords, lemmatizes the remaining tokens, and rejoins them into a
    space-separated "cleaned_complaint" column. Rows whose cleaned token count
    falls below min_tokens are dropped, since a topic model cannot draw
    meaningful signal from near-empty documents.

    Parameters:
        df (pd.DataFrame): The raw complaints DataFrame.
        min_tokens (int): Minimum number of cleaned tokens a row must have to
            be kept. Defaults to 5.

    Returns:
        pd.DataFrame: Columns Consumer_complaint, Product and
            cleaned_complaint, with short documents removed.
    """
    print("\nRunning preprocessing pipeline...\n")

    selected = select_columns(df)

    cleaned_tokens = (
        selected[TEXT_COL]
        .astype(str)
        .apply(clean_text)
        .apply(tokenize_and_remove_stopwords)
        .apply(lemmatize_tokens)
    )

    selected["cleaned_complaint"] = cleaned_tokens.apply(" ".join)
    print(f"Cleaned and lemmatized {selected.shape[0]} narratives")

    token_counts = cleaned_tokens.apply(len)
    keep_mask = token_counts >= min_tokens
    n_dropped = int((~keep_mask).sum())
    result = selected[keep_mask].reset_index(drop=True)

    print(f"Dropped {n_dropped} rows with fewer than {min_tokens} cleaned tokens")
    print(f"Resulting row count: {result.shape[0]}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — SAVING OUTPUT
# ─────────────────────────────────────────────────────────────────────────────

def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Save the processed DataFrame to data/processed/, creating it if needed.

    Parameters:
        df (pd.DataFrame): The processed DataFrame to save.
        filename (str): Name of the CSV file to write inside data/processed/.

    Returns:
        None
    """
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")
