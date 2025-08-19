import logging
import pandas as pd

from abc import ABC, abstractmethod
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from xgboost import XGBRegressor


from .utils import build_models_from_config

MODEL_MAP = {
    "LinearRegression": LinearRegression,
    "Lasso": Lasso,
    "Ridge": Ridge,
    "RandomForest": RandomForestRegressor,
    "GradientBoosting": GradientBoostingRegressor,
    "XGB": XGBRegressor,
}


class BaseModelingPipeline(ABC):
    """
    Base class for modeling pipelines, providing common setup tasks such as
    data validation, logging configuration, and model registry creation.

    Parameters
    ----------
    config : dict
        Configuration object (e.g., loaded from YAML/JSON) containing training parameters.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target values.
    selected_features_ : list of str, optional
        Pre-selected feature names. If None, will be determined by a child class.

    Attributes
    ----------
    config : dict
        Full configuration for the pipeline.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    model_name_path : str
        Path to save the final model.
    selector_path : str
        Path to save selected features.
    save_flag : bool
        Whether to save the trained model and selector.
    logging_flag : bool
        Whether to enable logging.
    final_models_ : dict[str, RegressorMixin]
        Candidate models to be used for selection/training.
    selected_features_ : list[str] or None
        Features retained after feature selection.
    best_model_ : RegressorMixin or None
        Best-performing model after training/selection.
    best_model_name_ : str or None
        Name of the best-performing model.
    logger : logging.Logger
        Logger for the pipeline class.

    Notes
    -----
    This class is not intended to be used directly. Subclasses should implement
    specific tasks such as feature selection or model training.
    """
    def __init__(self,
                 config: dict,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 selected_features_: list[str] = None):

        # Store init params
        self.config = config
        self.X_train = X_train
        self.y_train = y_train

        # Parameters from config
        self.model_name_path = self.config.training.model_name_path
        self.selector_path = self.config.training.selector_path
        self.save_flag = self.config.training.save_flag
        self.logging_flag = self.config.training.logging_flag

        # Learned attributes (set after pipeline run)
        self.final_models_: dict[str, RegressorMixin] = None
        self.selected_features_ = selected_features_
        self.best_model_: RegressorMixin | None = None
        self.best_model_name_: str | None = None

        self.logger = None
        self._handle_shape_error()
        self._setup_logging()
        self._setup_final_models()

    def _handle_shape_error(self):
        """
        Raises a ValueError if the shapes of the input data do not match.
        """
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples.")

    def _setup_logging(self):
        if self.logging_flag:
            logging.basicConfig(level=logging.INFO,
                                format="%(asctime)s - %(levelname)s - %(message)s")
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logging.getLogger("dummy")
            self.logger.disabled = True

    def _setup_final_models(self):
        """
        Sets up the final models for the pipeline.
        """
        self.final_models_ = build_models_from_config(self.config.training.final_models,
                                                      MODEL_MAP)
    
    @abstractmethod
    def run(self):
        """
        Abstract method that must be implemented by subclasses.
        Defines the execution flow of the pipeline (e.g., selection or training).
        """
        pass
