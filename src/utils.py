import pandas as pd


def load_dataframe(file_path: str) -> pd.DataFrame:
    """Loads a DataFrame from a CSV file."""
    return pd.read_csv(file_path)


def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """Saves a DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)
