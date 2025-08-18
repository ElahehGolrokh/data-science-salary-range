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


class ModelingPipeline:
    """
    End-to-end regression pipeline with RFE-based feature selection,
    model comparison, final training, and artifact saving.

    Workflow
    --------
    1) Tune the number of features with RFE inside cross-validation.
    2) Refit RFE on the full training data with the chosen feature count.
    3) Compare candidate models via cross-validation on selected features.
    4) Fit the best model on the full training data.
    6) Optionally persist artifacts (model + selected features).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test target.
    feature_counts : list[int] or None, default=None
        Candidate RFE feature counts to test. Defaults to [5, 10, 15, 20, 30, 40].
    random_state : int, default=42
        Random seed for reproducibility.
    default_model : sklearn.base.RegressorMixin or None, default=None
        Base model used for RFE. Defaults to RandomForestRegressor.
    cv_splits : int, default=5
        Number of cross-validation folds.
    shuffle : bool, default=True
        Whether to shuffle data during CV splits.
    scoring : str, default="neg_mean_squared_error"
        Scoring metric for cross-validation.
    final_models_ : dict[str, sklearn.base.RegressorMixin] or None, default=None
        Candidate models for final comparison. If None, defaults are set.
    rfe_step : int, default=2
        Step size for RFE elimination.
    model_path : str, default="final_model.pkl"
        Path to save the fitted model.
    selector_path : str, default="final_selector.pkl"
        Path to save the selected feature names.
    save_flag : bool, default=True
        Whether to save the artifacts.
    logging_flag : bool, default=True
        Whether to use logging for messages.

    Attributes
    ----------
    selected_features_ : list of str
        Feature names chosen by RFE, available after :meth:`run_pipeline`.
    best_feature_counts_ : int
        Best number of features selected, available after :meth:`run_pipeline`.
    best_model_ : sklearn.base.RegressorMixin
        The fitted best model, available after :meth:`run_pipeline`.
    best_model_name_ : str
        Name of the best model, available after :meth:`run_pipeline`.

    Public Methods
    -------
    run_pipeline()
        Execute the full workflow end-to-end.

    Examples
    --------
    >>> pipe = ModelingPipeline(
    ...     X_train, y_train, X_test, y_test,
    ...     config=config)
    >>> pipe.run_pipeline()
    >>> pipe.best_model_name_
    'RandomForest'
    >>> pipe.selected_features_[:5]
    ['skill_python', 'yoe', 'level_L5', 'location_US_CA', 'industry_SWE']
    """
    def __init__(self,
                 config: dict,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 X_test: pd.DataFrame,
                 y_test: pd.Series,
                 selected_features_: list[str] = None,
                 feature_selection_flag: bool = True,
                 compare_models_flag: bool = False,
                 train_flag: bool = True):

        # Store init params
        self.config = config
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_selection_flag = feature_selection_flag
        self.compare_models_flag = compare_models_flag
        self.train_flag = train_flag

        # Parameters from config
        self.feature_counts = self.config.training.feature_counts
        self.random_state = self.config.training.random_state
        self.model = get_default_model(self.config, MODEL_MAP)
        self.cv_splits = self.config.training.cv_splits
        self.shuffle = self.config.training.shuffle
        self.scoring = self.config.training.scoring
        self.rfe_step = self.config.training.rfe_step
        self.model_path = self.config.training.model_path
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

        if self.X_test.shape[0] != self.y_test.shape[0]:
            raise ValueError("X_test and y_test must have the same number of samples.")

        if self.X_train.shape[1] != self.X_test.shape[1]:
            raise ValueError("X_train and X_test must have the same number of features.")

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

    def run_pipeline(self) -> None:
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
                save_object_to_file(self.best_model_,
                                    self.model_path,
                                    self.config.paths.artifacts_dir)
                self.logger.info("Saved best model to %s", self.model_path)
        else:
            self.best_model_name_ = self.config.training.default_model.name
            self.best_model_ = self.final_models_[self.best_model_name_]
            self.logger.warning("Using default model: %s as the best model for training",
                                self.best_model_name_)
        # Final fitting step
        if self.train_flag:
            self._final_fit()
            if self.save_flag:
                save_object_to_file(self.best_model_,
                                    self.model_path,
                                    self.config.paths.artifacts_dir)
                self.logger.info("Saved best model to %s", self.model_path)
        self.logger.info('Pipeline execution completed successfully.')

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

    # model training
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

    def _final_fit(self) -> None:
        """
        Fits the best model on the full training set.
        """
        if self.logging_flag:
            self.logger.info('Fitting the best model on the full training set...')
        self.best_model_.fit(self.X_train[self.selected_features_], self.y_train)

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
