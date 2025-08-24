import numpy as np
import pandas as pd

from omegaconf import OmegaConf
from sklearn.base import RegressorMixin

from .utils import load_object


class InferencePipeline:
    def __init__(self,
                 config: OmegaConf,
                 model: RegressorMixin,
                 input_df: pd.DataFrame,
                 src_df: pd.DataFrame,
                 transform_target: bool = None,
                 columns_to_keep: list[str] = None):
        self.config = config
        self.model = model
        self.feature_selection = config.inference.feature_selection
        self.input_df = input_df
        self.src_df = src_df
        self.transform_target = transform_target if transform_target is not None else config.preprocessing.transform_target
        self.columns_to_keep = columns_to_keep

    def run(self) -> pd.DataFrame:
        """Runs the inference pipeline on the input DataFrame."""
        # Preprocess the input data
        self._preprocess()

        # Make predictions
        predictions = self.model.predict(self.input_df)

        # Post-process the predictions
        if self.transform_target:
            predictions = self.postprocess(predictions)

        return np.round(predictions)

    def _preprocess(self) -> pd.DataFrame:
        """Preprocesses the input DataFrame."""
        if self.feature_selection:
            self.input_df = self._select_features(self.input_df)

    def _select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Selects features from the preprocessed DataFrame."""
        if self.columns_to_keep is None:
            self.columns_to_keep = load_object(self.config.files.selected_features,
                                               self.config.dirs.artifacts)
        # Reindex ensures missing cols are added with 0, extra cols are dropped
        df = df.reindex(columns=self.columns_to_keep, fill_value=0)
        print(f"✅ Selected and aligned features ({len(self.columns_to_keep)} features)")
        return df

    def postprocess(self, input_vector: np.ndarray) -> float:
        """Post-processes the input_vector."""
        # Implement postprocessing steps (e.g., inverse scaling)
        input_vector = np.expm1(input_vector).astype(float)

        return input_vector
