import ast
import joblib
import logging
import os
import pandas as pd

from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .utils import load_dataframe, save_dataframe, save_object_to_file


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
                 path_to_save: str,
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
        self.path_to_save = path_to_save
        self.save_objects = save_objects

        # Learned attributes (set after pipeline run)
        self.one_hot_encoder_ = None
        self.mlb_ = None
        self.scaler_ = None
    
    def run(self):
        self._drop_useless_features()
        self._impute_missing_values()
        self._remove_outliers(q=.99)
        self._one_hot_encode_categorical()
        self._process_skills()
        self._standardize()
        save_dataframe(self.input_df, self.path_to_save)
        return self.input_df
    
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
    
    def _remove_outliers(self, q: float=.99, right_skewed: bool=True) -> None:
        """Removes outliers based on the passed quantile"""
        cols = ['company_size', 'mean_salary']
        for col in cols:
            if right_skewed:
                self.input_df = self.input_df[self.input_df[col] <= self.input_df[col].quantile(q)]
            else:
                self.input_df = self.input_df[self.input_df[col] >= self.input_df[col].quantile(q)]

        print(f'train_df shape after removing outliers: {self.input_df.shape}')
    
    def _one_hot_encode_categorical(self) -> None:
        """
        One-hot encodes categorical columns
        """
        categorical_columns = ['seniority_level', 'status', 'location',
                               'headquarter', 'industry', 'ownership']

        if not self.src_df:
            # Initialize OneHotEncoder with drop='first' to avoid multicollinearity
            # and handle_unknown='ignore' to handle potential unseen categories in the test set
            self.one_hot_encoder_ = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
            # Fit and transform on the training data
            df_encoded = self.one_hot_encoder_.fit_transform(self.input_df[categorical_columns])
            if self.save_objects:
                save_object_to_file(self.one_hot_encoder_, self.one_hot_encoder_path)
        else:
            # Load the encoder from the file
            self.one_hot_encoder_ = joblib.load(self.one_hot_encoder_path)
            # Transform the test data
            df_encoded = self.one_hot_encoder_.transform(self.input_df[categorical_columns])

        # Convert the encoded arrays back to DataFrames
        df_encoded = pd.DataFrame(df_encoded,
                                  columns=self.one_hot_encoder_.get_feature_names_out(categorical_columns),
                                  index=self.input_df.index)
        # Drop the original categorical columns from self.input_df
        self.input_df = self.input_df.drop(columns=categorical_columns)
        # Concatenate the encoded DataFrames with the remaining columns
        self.input_df = pd.concat([self.input_df, df_encoded], axis=1)
        print(f'input_df shape after one-hot encoding: {self.input_df.shape}')
    
    def _process_skills(self):
        """Processes skills"""
        self.input_df['skills'] = self.input_df['skills'].apply(self._parse_skills)
        self.input_df['skills'] = self.input_df['skills'].apply(self._normalize_skills)
        self._handle_high_cardinality()
        self._fit_multilabel_binarizer()

    @staticmethod
    def _parse_skills(x):
        """Ensures skills are lists"""
        if isinstance(x, str):
            try:
                return ast.literal_eval(x)
            except (ValueError, SyntaxError):
                return []
        elif isinstance(x, list):
            return x
        return []

    @staticmethod
    def _normalize_skills(skills):
        """Normalizes skill names"""
        return [s.strip().lower() for s in skills if isinstance(s, str)]

    def _handle_high_cardinality(self):
        """Handles high cardinality: keep only top N skills"""
        N_TOP_SKILLS = 50
        if not self.src_df:
            all_skills = [skill for sublist in self.input_df['skills'] for skill in sublist]
        else:
            all_skills = [skill for sublist in self.src_df['skills'] for skill in sublist]

        top_skills = {s for s, _ in Counter(all_skills).most_common(N_TOP_SKILLS)}
        self.input_df['skills'] = self.input_df['skills'].apply(lambda skills: [s if s in top_skills else 'other' for s in skills])

    def _fit_multilabel_binarizer(self):
        """Fits MultiLabelBinarizer on train, transform both"""
        if not self.src_df:
            mlb = MultiLabelBinarizer()
            skills_encoded = mlb.fit_transform(self.input_df['skills'])
            if self.save_objects:
                save_object_to_file(mlb, self.mlb_path)
        else:
            mlb = joblib.load(self.mlb_path)
            skills_encoded = mlb.transform(self.input_df['skills'])

        # Convert to DataFrames & merge
        skills_df = pd.DataFrame(skills_encoded,
                                 columns=[f"skill_{s}" for s in mlb.classes_],
                                 index=self.input_df.index)
        self.input_df = pd.concat([self.input_df.drop(columns=['skills']), skills_df], axis=1)
        print(f'input_df shape after processing skills: {self.input_df.shape}')
    
    def _standardize(self):
        """Standardizes numerical features"""
        if not self.src_df:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(self.input_df[['mean_salary', 'company_size']])
            if self.save_objects:
                save_object_to_file(scaler, self.scaler_path)
        else:
            scaler = joblib.load(self.scaler_path)
            scaled = scaler.transform(self.input_df[['mean_salary', 'company_size']])
        scaled = pd.DataFrame(scaled, columns=['mean_salary', 'company_size'], index=self.input_df.index)
        self.input_df = pd.concat([scaled, self.input_df.drop(columns=['mean_salary', 'company_size'])], axis=1)
        print(f'input_df shape after standardizing: {self.input_df.shape}')
