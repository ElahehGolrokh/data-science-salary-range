import logging
import os
import pandas as pd

from .utils import load_dataframe, save_dataframe


class Splitter:
    """
    A utility class to split a DataFrame into train and test sets.

    Example
    -------
    >>> splitter = Splitter("data/feature_engineered/df_features.csv")
    >>> train_df, test_df = splitter.split()

    Parameters
    ----------
    input_path : str
        Path to the input dataset (CSV).
    train_size : float, default=0.8
        Proportion of the dataset to include in the train split. Must be between 0 and 1.
    random_state : int, default=42
        Random seed for reproducibility.
    save_flag : bool, default=True
        Whether to save the resulting train and test splits to disk.
    dir_path : str, default="data/preprocessed"
        Directory to save the processed data.
    train_path : str, default="train.csv"
        Filename for the training set.
    test_path : str, default="test.csv"
        Filename for the test set.

    Attributes
    ----------
    train_df_ : pd.DataFrame | None
        Training set after split.
    test_df_ : pd.DataFrame | None
        Test set after split.
    """
    def __init__(self, input_path: str,
                 train_size: float = 0.8,
                 random_state: int = 42,
                 save_flag: bool = True,
                 dir_path: str = "data/preprocessed",
                 train_path: str = "train.csv",
                 test_path: str = "test.csv"):
        self.input_path = input_path
        self.train_size = train_size
        self.random_state = random_state
        self.save_flag = save_flag
        self.dir_path = dir_path
        self.train_path = train_path
        self.test_path = test_path

        # placeholders
        self.train_df_: pd.DataFrame | None = None
        self.test_df_: pd.DataFrame | None = None

        # logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # validations
        self._handle_errors()

    def _handle_errors(self):
        """
        Validates the parameters and checks for potential errors.
        """
        if not (0 < self.train_size < 1):
            raise ValueError("train_size must be between 0 and 1 (exclusive).")

        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the data into train and test sets"""
        input_df = load_dataframe(self.input_path)
        self.train_df_ = input_df.sample(frac=self.train_size,
                                   random_state=self.random_state)
        self.test_df_ = input_df.drop(self.train_df_.index)

        self.logger.info("Train shape: %s, Test shape: %s",
                         self.train_df_.shape, self.test_df_.shape)
        if self.save_flag:
            self._save_splits()
        return self.train_df_, self.test_df_

    def _save_splits(self) -> None:
        """Saves the train and test splits to disk."""
        self.logger.info("Saving train and test splits to disk.")
        # create directory if it doesn't exist
        os.makedirs(self.dir_path, exist_ok=True)
        train_path = os.path.join(self.dir_path, self.train_path)
        test_path = os.path.join(self.dir_path, self.test_path)
        if os.path.exists(train_path) or os.path.exists(test_path):
            self.logger.warning("Train/Test files already exist and will be overwritten.")
        save_dataframe(self.train_df_, train_path)
        save_dataframe(self.test_df_, test_path)


