import logging
import os
import pandas as pd

from .utils import load_dataframe, save_dataframe


class Splitter:
    """
    A utility class to split a DataFrame into train and test sets.

    Example
    -------
    >>> splitter = Splitter("data/df_feature_engineered.csv")
    >>> train_df, test_df = splitter.split()

    Parameters
    ----------
    input_path : str, default="data/df_feature_engineered.csv"
        Path to the input dataset (CSV).
    train_size : float, default=0.8
        Proportion of the dataset to include in the train split. Must be between 0 and 1.
    random_state : int, default=42
        Random seed for reproducibility.
    save_flag : bool, default=True
        Whether to save the resulting train and test splits to disk.
    train_path : str, default="data/train_df.csv"
        Filename for the training set.
    test_path : str, default="data/test_df.csv"
        Filename for the test set.

    Attributes
    ----------
    train_df_ : pd.DataFrame | None
        Training set after split.
    test_df_ : pd.DataFrame | None
        Test set after split.
    """
    def __init__(self, input_path: str,
                 train_path: str,
                 test_path: str,
                 train_size: float,
                 random_state: int,
                 save_flag: bool = True,):
        self.input_path = input_path
        self.train_size = train_size
        self.random_state = random_state
        self.save_flag = save_flag
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
        save_dataframe(self.train_df_, self.train_path)
        save_dataframe(self.test_df_, self.test_path)


class Preprocessor:
    def __init__(self, input_df: pd.DataFrame,
                 one_hot_encoder_path: str,
                 mlb_path: str,
                 scaler_path: str,
                 columns_to_drop: list=None,
                 src_df: pd.DataFrame=None,
                 save_objects: bool=True,
                 ):
        self.input_df = input_df
        self.columns_to_drop = columns_to_drop
        self.src_df = src_df
        self.one_hot_encoder_path = one_hot_encoder_path
        self.mlb_path = mlb_path
        self.scaler_path = scaler_path
        self.save_objects = save_objects

        # Learned attributes (set after pipeline run)
        self.one_hot_encoder_ = None
        self.mlb_ = None
        self.scaler_ = None
    
    def _drop_useless_features(self):
        """Drops useless features"""
        if not self.columns_to_drop:
            self.columns_to_drop = ['min_salary',  # Not informative
                                    'max_salary',  # Not informative
                                    'revenue',  # Large number of missing values
                                    'company',  # Not informative
                                    'job_title',  # High frequency of dominant category
                                    ]
        self.input_df.drop(columns=self.columns_to_drop, inplace=True)
    
    def _impute_missing_values(self):
        """
        Imputes missing values with the mode for categorical columns
        of the train data
        """
        for col in ['seniority_level', 'status', 'ownership']:
            if not self.src_df:
                self.input_df[col].fillna(self.input_df[col].mode()[0], inplace=True)
            else:
                self.input_df[col].fillna(self.src_df[col].mode()[0], inplace=True)

        # Impute missing values with the median for company_size because based
        # on describe stats, this column is skewed & so median is a better choice
        # for imputation than the mean
        if not self.src_df:
            self.input_df['company_size'].fillna(self.input_df['company_size'].median(),
                                                 inplace=True)
        else:
            self.input_df['company_size'].fillna(self.src_df['company_size'].median(),
                                                 inplace=True)