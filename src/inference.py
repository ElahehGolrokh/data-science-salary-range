import numpy as np
import pandas as pd

from omegaconf import OmegaConf
from sklearn.base import RegressorMixin

from .preprocessing import Preprocessor
from .utils import load_object_from_file


class InferencePipeline:
    def __init__(self,
                 config: OmegaConf,
                 model: RegressorMixin,
                 input_df: pd.DataFrame,
                 src_df: pd.DataFrame):
        self.config = config
        self.model = model
        self.feature_selection = config.inference.feature_selection
        self.input_df = input_df
        self.src_df = src_df

    def run(self) -> pd.DataFrame:
        """Runs the inference pipeline on the input DataFrame."""
        # Preprocess the input data
        preprocessed_df = self._preprocess()

        # Make predictions
        predictions = self.model.predict(preprocessed_df)

        # Post-process the predictions
        result = self._postprocess(predictions)

        return np.round(result)

    def _preprocess(self) -> pd.DataFrame:
        """Preprocesses the input DataFrame."""
        # Implement preprocessing steps (e.g., feature engineering, scaling)
        preprocessor = Preprocessor(self.config, save_flag=False)
        preprocessed_df = preprocessor.run(input_df=self.input_df,
                                           src_df=self.src_df,
                                           phase='inference')
        if self.feature_selection:
            preprocessed_df = self._select_features(preprocessed_df)
        return preprocessed_df

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selects features from the preprocessed DataFrame."""
        columns_to_keep = load_object_from_file(self.config.training.selector_path)
        df = df[columns_to_keep]
        return df

    def _postprocess(self, predictions: np.ndarray) -> float:
        """Post-processes the predictions."""
        # Implement postprocessing steps (e.g., inverse scaling)
        if self.config.preprocessing.transform_target:
            predictions = np.expm1(predictions)

        return float(predictions)
