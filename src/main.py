from utils import load_dataset
from data_exploration import (
    check_shape_and_missing,
    check_duplicates,
    check_class_balance,
    check_text_length,
    check_frequent_tokens,
    show_sample_complaints,
)

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


if __name__ == "__main__":
    main()
