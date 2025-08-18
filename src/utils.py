import joblib
import pandas as pd


def load_dataframe(file_path: str) -> pd.DataFrame:
    """Loads a DataFrame from a CSV file."""
    return pd.read_csv(file_path)


def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    """Saves a DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)


def save_object_to_file(input_object, file_path) -> None:
    """Saves the preprocessing and modeling objects to a pickle file."""
    if input_object is None:
        raise ValueError("The object has not been fitted yet. \
                         Run for training data first")
    joblib.dump(input_object, file_path)
    print(f"Object saved to {file_path}")

