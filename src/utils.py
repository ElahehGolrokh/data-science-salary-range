import joblib
import os
import pandas as pd

from pathlib import Path
from sklearn.base import RegressorMixin


def load_dataframe(file_path: str, dir_path: str= None) -> pd.DataFrame:
    """Loads a DataFrame from a CSV file."""
    if dir_path is not None:
        file_path = os.path.join(dir_path, file_path)
    return pd.read_csv(file_path)


def save_dataframe(df: pd.DataFrame,
                   file_path: str,
                   dir_path: str) -> None:
    """Saves a DataFrame to a CSV file."""
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_path = os.path.join(dir_path, file_path)
    df.to_csv(file_path, index=False)


def save_object_to_file(input_object,
                        file_name: str,
                        dir_path: str) -> None:
    """Saves the preprocessing and modeling objects to a pickle file."""
    if input_object is None:
        raise ValueError("The object has not been fitted yet. \
                         Run for training data first")
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_path = os.path.join(dir_path, file_name)
    joblib.dump(input_object, file_path)
    print(f"Object saved to {file_path}")


def load_object_from_file(file_path: str) -> None:
    """Loads a pickled object from a file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")
    return joblib.load(file_path)


def build_models_from_config(config_dict: dict,
                             model_map: dict) -> dict:
    """Build regression models from a configuration dictionary."""
    models = {}
    for name, params in config_dict.items():
        models[name] = model_map[name](**params)
    return models


def get_default_model(config_dict: dict,
                      model_map: dict) -> RegressorMixin:
    """Get the default model specified in the config."""
    model_name = config_dict["training"]["default_model"]["name"]
    model_params = config_dict["training"]["default_model"].get("params", {})
    return model_map[model_name](**model_params)


def get_root() -> Path:
    """Get the root directory of the project."""
    root = Path(__file__).resolve().parent
    while not (root / ".git").exists() and root != root.parent:
        root = root.parent
    return root
