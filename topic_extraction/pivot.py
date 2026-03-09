import argparse
import pandas as pd

def pivot_topics(df_path: str) -> pd.DataFrame:
    """
    Pivot the DataFrame to have topics as columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing topic data.
    index_cols (list): List of columns to set as index.
    topic_col (str): The column name that contains topic identifiers.
    value_col (str): The column name that contains values associated with topics.

    Returns:
    pd.DataFrame: A pivoted DataFrame with topics as columns.
    """
    df = pd.read_csv(df_path)
    pivoted_df = df.pivot_table(index='subtopic_name', values='subtopic', aggfunc='count')
    # sort descending by count
    pivoted_df = pivoted_df.sort_values(by='subtopic', ascending=False)
    print(pivoted_df)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    args = parser.parse_args()
    pivot_topics(args.input)