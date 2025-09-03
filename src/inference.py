import numpy as np
import pandas as pd

from omegaconf import OmegaConf
from sklearn.base import RegressorMixin

from .utils import load_object, select_features, postprocess_target


class InferencePipeline:
    """
    Pipeline for running inference using a trained regression model.

    Handles feature selection, preprocessing, prediction, and optional
    target transformation.

    Parameters
    ----------
    config : OmegaConf
        Hydra/OmegaConf configuration with preprocessing/inference settings.
    model : RegressorMixin
        Trained regression model implementing `predict()`.
    input_df : pd.DataFrame
        Input features for prediction.
    src_df : pd.DataFrame
        Original dataset (used for feature alignment and transformations).
    transform_target : bool, optional
        Whether to apply inverse transformation to predictions. If None,
        defaults to `config.preprocessing.transform_target`.
    columns_to_keep : list of str, optional
        Subset of features to use for prediction. If None, will be loaded
        from artifacts when feature selection is enabled.

    Public Methods
    --------------
    run() -> pd.DataFrame
        Executes the inference pipeline and returns np.ndarray of predicted
        values (rounded).

    """
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
        self._handle_errors()
        # Preprocess the input data
        if self.feature_selection:
            self._get_columns_to_keep()
            print(f'-----------Inference with {len(self.columns_to_keep)} features')
            self.input_df = select_features(self.input_df,
                                            self.columns_to_keep)

        try:
            # Make predictions
            predictions = self.model.predict(self.input_df)

        except Exception as e:
            raise ValueError(f"❌ Error during prediction: {e}")

        # Post-process the predictions
        if self.transform_target:
            predictions = postprocess_target(predictions)

        return np.round(predictions)

    def _handle_errors(self):
        if self.input_df is None or self.input_df.empty:
            raise ValueError("Input DataFrame is empty or not defined.")

    def _get_columns_to_keep(self):
        if self.columns_to_keep is None:
            self.columns_to_keep = load_object(
                self.config.files.selected_features,
                self.config.dirs.artifacts
            )
