import numpy as np
import pandas as pd

from omegaconf import OmegaConf
from sklearn.base import RegressorMixin

from .utils import load_object, select_features, postprocess_target


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
        if self.feature_selection:
            self._get_columns_to_keep()
            self.input_df = select_features(self.input_df, self.columns_to_keep)

        # Make predictions
        predictions = self.model.predict(self.input_df)

        # Post-process the predictions
        if self.transform_target:
            predictions = postprocess_target(predictions)

        return np.round(predictions)

    def _get_columns_to_keep(self):
        if self.columns_to_keep is None:
            self.columns_to_keep = load_object(
                self.config.files.selected_features,
                self.config.dirs.artifacts
            )
