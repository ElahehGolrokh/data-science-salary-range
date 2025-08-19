import logging
import numpy as np
import pandas as pd

from kneed import KneeLocator
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor

from .utils import build_models_from_config, \
    get_default_model, save_object_to_file, \
    load_object_from_file

MODEL_MAP = {
    "LinearRegression": LinearRegression,
    "Lasso": Lasso,
    "Ridge": Ridge,
    "RandomForest": RandomForestRegressor,
    "GradientBoosting": GradientBoostingRegressor,
    "XGB": XGBRegressor,
}


class ModelSelector:
    """
    Pipeline for feature selection and model comparison in regression tasks.


    This class performs Recursive Feature Elimination (RFE) to select
    informative features, compares candidate regression models via
    cross-validation, and identifies the best model based on performance.


    Parameters
    ----------
    config : dict
    Configuration object containing model and training parameters.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target values.
    selected_features_ : list of str, default=None
        Predefined list of selected features. If None, RFE is performed.
    feature_selection_flag : bool, default=True
        Whether to perform feature selection.
    compare_models_flag : bool, default=False
        Whether to compare models and select the best one.


    Attributes
    ----------
    final_models_ : dict of {str: RegressorMixin}
        Candidate models built from configuration.
    selected_features_ : list of str
        Features selected by RFE.
    best_feature_counts_ : int
        Optimal number of features determined by cross-validation and elbow method.
    best_model_ : RegressorMixin
        Best-performing model after comparison.
    best_model_name_ : str
        Name of the best-performing model.


    Methods
    -------
    run()
    Executes feature selection and/or model comparison.


    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> import pandas as pd
    >>> X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
    >>> X = pd.DataFrame(X)
    >>> y = pd.Series(y)
    >>> selector = ModelSelector(config={}, X_train=X, y_train=y)
    >>> selector.run()
    >>> selector.best_model_name_
    'RandomForest'
    """
    def __init__(self,
                 config: dict,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 selected_features_: list[str] = None,
                 feature_selection_flag: bool = True,
                 compare_models_flag: bool = False):

        # Store init params
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        self.feature_selection_flag = feature_selection_flag
        self.compare_models_flag = compare_models_flag

        # Parameters from config
        self.feature_counts = self.config.training.feature_counts
        self.random_state = self.config.training.random_state
        self.model = get_default_model(self.config, MODEL_MAP)
        self.cv_splits = self.config.training.cv_splits
        self.shuffle = self.config.training.shuffle
        self.scoring = self.config.training.scoring
        self.rfe_step = self.config.training.rfe_step
        self.model_name_path = self.config.training.model_name_path
        self.selector_path = self.config.training.selector_path
        self.save_flag = self.config.training.save_flag
        self.logging_flag = self.config.training.logging_flag

        # Learned attributes (set after pipeline run)
        self.final_models_: dict[str, RegressorMixin] = None
        self.selected_features_ = selected_features_
        self.best_feature_counts_: int | None = None
        self.best_model_: RegressorMixin | None = None
        self.best_model_name_: str | None = None

    # initialization methods
    def _handle_shape_error(self):
        """
        Raises a ValueError if the shapes of the input data do not match.
        """
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples.")

    def _initial_settings(self) -> None:
        """
        Sets the final models to choose from.
        """
        self._handle_shape_error()
        # Configure logging to output to the console
        if self.logging_flag:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
            self.logger.info("Logging configured.")
        self.final_models_ = build_models_from_config(self.config.training.final_models,
                                                      MODEL_MAP)

    def run(self) -> None:
        """
        Runs the entire pipeline.
        """
        self._initial_settings()
        # Feature selection step
        if self.feature_selection_flag:
            self._find_best_feature_counts()
            self._refit_rfe()
            if self.save_flag:
                save_object_to_file(self.selected_features_,
                                    self.selector_path,
                                    self.config.paths.artifacts_dir)
        else:
            if self.selected_features_ is None:
                try:
                    self.selected_features_ = load_object_from_file(self.selector_path)
                    self.logger.info("Loaded selected features: %s", self.selected_features_)
                except Exception as e:
                    raise ValueError("selected_features_ is None. You can either " +
                                     "set feature_selection_flag=True to perform feature " +
                                     "selection or give selected_features_ a default list.")
        # Model comparison step
        if self.compare_models_flag:
            self._get_best_model()
            if self.save_flag:
                # Save only the name of the best model
                with open(self.model_name_path, "w") as f:
                    f.write(self.best_model_name_)
                self.logger.info("Saved best model name to %s", self.model_name_path)
        else:
            try:
                with open(self.model_name_path, "r") as f:
                    self.best_model_name_ = f.read().strip()
                self.best_model_ = self.final_models_[self.best_model_name_]
            except Exception as e:
                raise ValueError("best_model_name_ is None")
        return self.best_model_name_, self.selected_features_

    # feature selection
    def _find_best_feature_counts(self,) -> int:
        """
        Finds the best number of features to preserve.
        """
        if self.logging_flag:
            self.logger.info('Finding the optimum number of features...')
        if self.model is None:
            self.model = RandomForestRegressor()
        mean_scores = []
        for n_features in self.feature_counts:
            cv = KFold(n_splits=self.cv_splits, shuffle=self.shuffle, random_state=self.random_state)
            rfe = RFE(estimator=self.model,
                      n_features_to_select=n_features,
                      step=self.rfe_step,
                      verbose=0)
            pipeline = Pipeline([
                ('rfe', rfe),
                ('model', self.model)
            ])
            scores = cross_val_score(pipeline, self.X_train, self.y_train, cv=cv, scoring=self.scoring)
            mean_scores.append(np.mean(scores))
            self.logger.info("CV score for %s features: %.4f", n_features, np.mean(scores))
        self.best_feature_counts_ = self._choose_features_from_elbow(mean_scores, self.feature_counts)
        if len(self.X_train.columns) < self.best_feature_counts_:
            raise ValueError(f"X_train must have at least {self.best_feature_counts_} features.")

    def _refit_rfe(self) -> list[str]:
        """
        Refits RFE on the full training set with the best number of features.
        """
        if self.logging_flag:
            self.logger.info(f'Refitting RFE with {self.best_feature_counts_} features...')
        rfe = RFE(estimator=self.model,
                  n_features_to_select=self.best_feature_counts_,
                  step=self.rfe_step,
                  verbose=0)
        rfe.fit(self.X_train, self.y_train)
        self.selected_features_ = self.X_train.columns[rfe.support_].tolist()

    # model selection
    def _compare_models(self) -> pd.DataFrame:
        """
        Trains a range of models on the selected features.
        """
        comparison_df = pd.DataFrame(data=None,
                                     columns=self.final_models_.keys(),
                                     index=['Scores'])
        for name, model in self.final_models_.items():
            kfold = KFold(n_splits=self.cv_splits,
                          shuffle=self.shuffle,
                          random_state=self.random_state)
            cv_score = np.mean(cross_val_score(model,
                                               self.X_train[self.selected_features_],
                                               self.y_train,
                                               cv=kfold,
                                               scoring=self.scoring))
            self.logger.info("CV mean score for %s: %.4f", name, cv_score)
            comparison_df.loc['Scores', name] = cv_score

        return comparison_df

    def _get_best_model(self) -> tuple[RegressorMixin, str]:
        """
        Returns the best model based on mean cross-validation score.
        """
        if self.logging_flag:
            self.logger.info('Getting the best trained model...')
        comparison_df = self._compare_models()
        self.best_model_name_ = comparison_df.idxmax(axis=1)[0]
        self.best_model_ = self.final_models_[self.best_model_name_]
        self.logger.info("The best model based on mean cross-validation score is: %s", self.best_model_name_)

    @staticmethod
    def _choose_features_from_elbow(mean_scores: list[float], feature_counts: list[int],
                                   curve: str = 'concave', direction: str = 'increasing') -> int:
        """
        Given a list of mean CV scores and corresponding feature counts,
        find the elbow point (best number of features) without plotting.

        Parameters
        ----------
        mean_scores : list of float
            Mean CV scores in the same order as feature_counts.
        feature_counts : list of int
            Number of features tested.
        curve : str
            'concave' (most common for increasing scores) or 'convex'.
        direction : str
            'increasing' if higher score is better, else 'decreasing'.

        Returns
        -------
        int
            Best number of features based on elbow detection.
        """
        # Ensure numpy arrays for KneeLocator
        feature_counts = np.array(feature_counts)
        mean_scores = np.array(mean_scores)

        # Detect knee point
        kl = KneeLocator(feature_counts, mean_scores, curve=curve, direction=direction)

        if kl.knee is None:
            # Fallback: choose feature count with max score
            return feature_counts[np.argmax(mean_scores)]

        return int(kl.knee)


class ModelTrainer:
    """
    Train the best regression model on selected features.


    This class loads previously selected features and the chosen model
    (identified by ``ModelSelector``), fits it on the full training set,
    and optionally saves the final model.


    Parameters
    ----------
    config : dict
        Configuration object containing model and training parameters.
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target values.
    selected_features_ : list of str, default=None
        List of selected features. If None, features are loaded from file.
    best_model_name_ : str, default=None
        Name of the best model. If None, it is loaded from file.


    Attributes
    ----------
    final_models_ : dict of {str: RegressorMixin}
        Candidate models built from configuration.
    selected_features_ : list of str
        Features used for training.
    best_model_name_ : str
        Name of the chosen model.
    best_model_ : RegressorMixin
        Fitted regression model.


    Methods
    -------
    run()
        Executes final training and saves the fitted model if required.


    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> import pandas as pd
    >>> X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
    >>> X = pd.DataFrame(X)
    >>> y = pd.Series(y)
    >>> trainer = ModelTrainer(config={}, X_train=X, y_train=y)
    >>> trainer.run()
    """
    def __init__(self,
                 config: dict,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 selected_features_: list[str] = None,
                 best_model_name_: str = None):

        # Store init params
        self.config = config
        self.X_train = X_train
        self.y_train = y_train

        # Parameters from config
        self.model_name_path = self.config.training.model_name_path
        self.model_path = self.config.training.model_path
        self.selector_path = self.config.training.selector_path
        self.save_flag = self.config.training.save_flag
        self.logging_flag = self.config.training.logging_flag

        # Learned attributes (set after pipeline run)
        self.final_models_: dict[str, RegressorMixin] = None
        self.selected_features_ = selected_features_
        self.best_model_name_ = best_model_name_
        self.best_model_: RegressorMixin | None = None

    # initialization methods
    def _handle_shape_error(self):
        """
        Raises a ValueError if the shapes of the input data do not match.
        """
        if self.X_train.shape[0] != self.y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of samples.")

    def _initial_settings(self) -> None:
        """
        Sets the final models to choose from.
        """
        self._handle_shape_error()
        # Configure logging to output to the console
        if self.logging_flag:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
            self.logger.info("Logging configured.")
        self.final_models_ = build_models_from_config(self.config.training.final_models,
                                                      MODEL_MAP)

    def run(self) -> None:
        """
        Runs the entire pipeline.
        """
        self._initial_settings()
        if self.selected_features_ is None:
            try:
                self.selected_features_ = load_object_from_file(self.selector_path)
                self.logger.info("Loaded selected features: %s", self.selected_features_)
            except Exception as e:
                raise ValueError("selected_features_ is None. You can either " +
                                 "run `ModelSelector.run()` to perform feature " +
                                 "selection or give selected_features_ a default list.")
        if self.best_model_name_ is None:
            try:
                with open(self.model_name_path, "r") as f:
                    self.best_model_name_ = f.read().strip()
            except Exception as e:
                raise ValueError("best_model_name_ is None")

        self.best_model_ = self.final_models_[self.best_model_name_]

        # Final fitting step
        self._fit()
        if self.save_flag:
            save_object_to_file(self.best_model_,
                                self.model_path,
                                self.config.paths.artifacts_dir)
            self.logger.info("Saved best model to %s", self.model_path)
        self.logger.info('Pipeline execution completed successfully.')

    def _fit(self) -> None:
        """
        Fits the best model on the full training set.
        """
        if self.logging_flag:
            self.logger.info('Fitting the best model on the full training set...')
        self.best_model_.fit(self.X_train[self.selected_features_], self.y_train)
