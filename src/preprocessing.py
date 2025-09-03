import ast
import logging
import numpy as np
import os
import pandas as pd

from collections import Counter
from omegaconf import OmegaConf
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .utils import load_dataframe, load_object, save_dataframe, save_object


class Splitter:
    """
    A utility class to split a DataFrame into train and test sets.

    Example
    -------
    >>> splitter = Splitter(config)
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
    def __init__(self,
                 config: OmegaConf,
                 save_flag: bool):
        # Store init params
        self.save_flag = save_flag

        # Parameters from config
        self.dir_path = config.dirs.data
        self.input_path = config.files.feature_engineered
        self.train_size = config.preprocessing.train_size
        self.random_state = config.preprocessing.random_state
        self.train_path = config.files.train_data
        self.test_path = config.files.test_data

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

        if not os.path.exists(os.path.join(self.dir_path, self.input_path)):
            raise FileNotFoundError(f"Input file not found: {self.input_path}")

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits the data into train and test sets"""
        input_df = load_dataframe(dir_path=self.dir_path,
                                  file_path=self.input_path)
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
        save_dataframe(self.train_df_, self.train_path, self.dir_path)
        save_dataframe(self.test_df_, self.test_path, self.dir_path)


class Preprocessor:
    def __init__(self,
                 config: OmegaConf,
                 save_flag: bool,
                 one_hot_encoder_: OneHotEncoder = None,
                 mlb_: MultiLabelBinarizer = None,
                 scaler_: StandardScaler = None):
        # Store init params
        self.config = config
        self.save_flag = save_flag

        # Parameters from config
        self.columns_to_drop = config.preprocessing.columns_to_drop
        self.data_dir_path = config.dirs.data
        self.artifacts_dir_path = config.dirs.artifacts
        self.one_hot_encoder_name = config.files.one_hot_encoder
        self.mlb_name = config.files.mlb
        self.scaler_name = config.files.scaler
        self.numerical_features = config.preprocessing.numerical_features
        self.categorical_features = config.preprocessing.categorical_features
        self.target = config.preprocessing.target[0]

        # Learned attributes (set after pipeline run)
        self.one_hot_encoder_ = one_hot_encoder_
        self.mlb_ = mlb_
        self.scaler_ = scaler_

    def run(self,
            input_df: pd.DataFrame,
            src_df: pd.DataFrame=None,
            phase: str=None,
            preprocessed_path: str=None,
            transform_target: bool = None) -> pd.DataFrame:
        self._handle_errors(input_df, phase)
        if phase != "inference":
            input_df = self._drop_useless_features(input_df)
            input_df = self._impute_missing_values(input_df, src_df)
            # input_df = self._remove_outliers(input_df, src_df, q=.99)
        # 🚨 Ensure target is removed in inference
        if self.target in input_df.columns and phase == "inference":
            input_df = input_df.drop(columns=[self.target], axis=1)
        input_df = self._one_hot_encode_categorical(input_df, src_df)
        input_df = self._process_skills(input_df, src_df)
        input_df = self._standardize(input_df, src_df)
        input_df = self._ordinal_encode_features(input_df)
        if transform_target:
            input_df = self._log_transform_target(input_df)
        if self.save_flag:
            save_dataframe(input_df, preprocessed_path, self.data_dir_path)
        return input_df

    def _handle_errors(self,
                       input_df: pd.DataFrame,
                       phase: str):
        """Handle errors that may occur during preprocessing"""
        if phase != "inference":
            columns = self.config.preprocessing.all_features
        else:
            columns = [feat for feat in self.config.preprocessing.all_features
                        if feat not in self.config.preprocessing.columns_to_drop
                        and feat not in self.config.preprocessing.target]
        for col in columns:
            if col not in input_df.columns:
                raise ValueError(f"Column not found in input_df: {col}")


    def _drop_useless_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Drops useless features"""
        input_df = input_df.drop(columns=self.columns_to_drop)
        return input_df

    def _impute_missing_values(self,
                               input_df: pd.DataFrame,
                               src_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Imputes missing values with the mode for categorical columns
        of the train data
        """
        for col in ['seniority_level', 'status', 'ownership']:
            if src_df is None:
                input_df[col].fillna(input_df[col].mode()[0], inplace=True)
            else:
                input_df[col].fillna(src_df[col].mode()[0], inplace=True)

        # Impute missing values with the median for company_size because based
        # on describe stats, this column is skewed & so median is a better choice
        # for imputation than the mean
        if src_df is None:
            input_df['company_size'].fillna(input_df['company_size'].median(),
                                                 inplace=True)
        else:
            input_df['company_size'].fillna(src_df['company_size'].median(),
                                                 inplace=True)
        return input_df

    def _remove_outliers(self,
                         input_df: pd.DataFrame,
                         src_df: pd.DataFrame = None,
                         q: float=.99,
                         right_skewed: bool=True) -> pd.DataFrame:
        """
        Removes outliers from training data based on the passed quantile

        Note: Since that based on EDA plots we are going to remove outliers
        from both features & target columns, we consider them all as a single
        group for outlier removal & create a cols list including all of them.
        """
        if src_df is None:
            cols = self.numerical_features + [self.target]
            for col in cols:
                if right_skewed:
                    input_df = input_df[input_df[col] <= input_df[col].quantile(q)]
                else:
                    input_df = input_df[input_df[col] >= input_df[col].quantile(q)]

            print(f'train_df shape after removing outliers: {input_df.shape}')
        return input_df
    
    def _one_hot_encode_categorical(self,
                                    input_df: pd.DataFrame,
                                    src_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        One-hot encodes categorical columns
        """
        if src_df is None:
            # Initialize OneHotEncoder with drop='first' to avoid multicollinearity
            # and handle_unknown='ignore' to handle potential unseen categories in the test set
            self.one_hot_encoder_ = OneHotEncoder(drop='first',
                                                  handle_unknown='ignore',
                                                  sparse_output=False)
            # Fit and transform on the training data
            df_encoded = self.one_hot_encoder_.fit_transform(input_df[self.categorical_features])
            if self.save_flag:
                save_object(self.one_hot_encoder_,
                            self.one_hot_encoder_name,
                            self.artifacts_dir_path)
        else:
            if self.one_hot_encoder_ is None:
                # Load the encoder from the file
                try:
                    self.one_hot_encoder_ = load_object(self.one_hot_encoder_name,
                                                        self.artifacts_dir_path)
                except Exception as e:
                    raise ValueError(f"Error loading one-hot encoder: {e}")
            # Transform the test data
            df_encoded = self.one_hot_encoder_.transform(input_df[self.categorical_features])

        # Convert the encoded arrays back to DataFrames
        df_encoded = pd.DataFrame(df_encoded,
                                  columns=self.one_hot_encoder_.get_feature_names_out(self.categorical_features),
                                  index=input_df.index)
        # Drop the original categorical columns from input_df
        input_df = input_df.drop(columns=self.categorical_features)
        # Concatenate the encoded DataFrames with the remaining columns
        input_df = pd.concat([input_df, df_encoded], axis=1)
        print(f'input_df shape after one-hot encoding: {input_df.shape}')
        return input_df

    def _process_skills(self,
                        input_df: pd.DataFrame,
                        src_df: pd.DataFrame = None) -> pd.DataFrame:
        """Processes skills"""
        input_df['skills'] = input_df['skills'].apply(self._parse_skills)
        input_df['skills'] = input_df['skills'].apply(self._normalize_skills)
        input_df = self._handle_high_cardinality(input_df, src_df)
        input_df = self._fit_multilabel_binarizer(input_df, src_df)
        return input_df

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

    @staticmethod
    def _handle_high_cardinality(input_df: pd.DataFrame,
                                 src_df: pd.DataFrame = None) -> pd.DataFrame:
        """Handles high cardinality: keep only top N skills"""
        N_TOP_SKILLS = 50
        if src_df is None:
            all_skills = [skill for sublist in input_df['skills'] for skill in sublist]
        else:
            all_skills = [skill for sublist in src_df['skills'] for skill in sublist]

        top_skills = {s for s, _ in Counter(all_skills).most_common(N_TOP_SKILLS)}
        input_df['skills'] = input_df['skills'].apply(lambda skills: [s if s in top_skills else 'other' for s in skills])
        return input_df

    def _fit_multilabel_binarizer(self,
                                  input_df: pd.DataFrame,
                                  src_df: pd.DataFrame = None) -> pd.DataFrame:
        """Fits MultiLabelBinarizer on train, transform both"""
        if src_df is None:
            self.mlb_ = MultiLabelBinarizer()
            skills_encoded = self.mlb_.fit_transform(input_df['skills'])
            if self.save_flag:
                save_object(self.mlb_,
                            self.mlb_name,
                            self.artifacts_dir_path)
        else:
            if self.mlb_ is None:
                try:
                    # Load the encoder from the file
                    self.mlb_ = load_object(self.mlb_name,
                                            self.artifacts_dir_path)
                except Exception as e:
                    raise ValueError(f"Error loading MultiLabelBinarizer: {e}")
            skills_encoded = self.mlb_.transform(input_df['skills'])

        # Convert to DataFrames & merge
        skills_df = pd.DataFrame(skills_encoded,
                                 columns=[f"skill_{s}" for s in self.mlb_.classes_],
                                 index=input_df.index)
        input_df = pd.concat([input_df.drop(columns=['skills']), skills_df], axis=1)
        return input_df

    def _standardize(self,
                     input_df: pd.DataFrame,
                     src_df: pd.DataFrame = None) -> pd.DataFrame:
        """Standardizes numerical features"""
        if src_df is None:
            self.scaler_ = StandardScaler()
            scaled = self.scaler_.fit_transform(input_df[self.numerical_features])
            if self.save_flag:
                save_object(self.scaler_,
                            self.scaler_name,
                            self.artifacts_dir_path)
        else:
            if self.scaler_ is None:
                try:
                    self.scaler_ = load_object(self.scaler_name,
                                               self.artifacts_dir_path)
                except Exception as e:
                    raise ValueError(f"Error loading scaler: {e}")
            scaled = self.scaler_.transform(input_df[self.numerical_features])
        scaled = pd.DataFrame(scaled, columns=self.numerical_features, index=input_df.index)
        input_df = pd.concat([scaled,
                              input_df.drop(columns=self.numerical_features)],
                              axis=1)
        print(f'input_df shape after standardizing: {input_df.shape}')
        return input_df

    def _log_transform_target(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies log transformation to the target variable.
        """
        input_df[self.target] = np.log1p(input_df[self.target])
        return input_df

    def _log_transform_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies log transformation to the specified features.
        """
        for col in self.numerical_features:
            input_df[col] = np.log1p(input_df[col])
        return input_df

    def _ordinal_encode_features(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies ordinal encoding to the specified features.
        """
        if "seniority_level" not in input_df.columns:
            raise ValueError("seniority_level column missing during ordinal encoding")
        seniority_map = {
            "junior": 1,
            "midlevel": 2,
            "senior": 3,
            "lead": 4,
        }
        input_df["seniority_level"] = input_df["seniority_level"].map(seniority_map)

        # Handle missing/unknown
        input_df["seniority_level"].fillna(-1, inplace=True)
        return input_df
