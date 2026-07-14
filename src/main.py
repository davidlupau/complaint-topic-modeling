"""End-to-end pipeline: load, explore, preprocess, vectorize and topic-model the complaints corpus."""

from utils import load_dataset
from data_exploration import (
    check_shape_and_missing,
    check_duplicates,
    check_class_balance,
    check_text_length,
    check_frequent_tokens,
    show_sample_complaints,
)
from preprocessing import preprocess_corpus, save_processed_data
from vectorization import run_vectorization
from topic_modeling import run_topic_modeling

def main ():
    # Loading dataset
    df = load_dataset("Customer_Complaints_Sentiment_and_Priority_Dataset.csv")
    if df is None:
        return

    # Exploratory data analysis pass
    check_shape_and_missing(df)
    check_duplicates(df)
    check_class_balance(df)
    check_text_length(df)
    check_frequent_tokens(df)
    show_sample_complaints(df)

    # Preprocessing pass
    processed_df = preprocess_corpus(df)
    save_processed_data(processed_df, "complaints_clean.csv")

    # Vectorization pass
    vectorization_results = run_vectorization(processed_df)

    # Topic modeling pass
    topic_modeling_results = run_topic_modeling(processed_df, vectorization_results)


if __name__ == "__main__":
    main()
