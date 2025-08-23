import numpy as np
import pandas as pd

from kneed import KneeLocator
from sklearn.base import RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import RFE

from .utils import get_default_model, save_object, \
                   load_object, save_text, \
                   load_text
from .base import BaseModelingPipeline, MODEL_MAP


class ModelSelector(BaseModelingPipeline):
    """
    Pipeline for feature selection and model comparison in regression tasks.

    Responsibilities
    ----------------
    - Evaluates feature subsets using RFE and cross-validation.
    - Compares multiple candidate models.
    - Selects the best model based on mean CV score.
    - Stores the selected features and chosen model.

    Inherits from
    -------------
    BaseModelingPipeline : Provides logging, error handling, and model registry.

    Parameters
    ----------
    feature_selection_flag : bool, default=True
        Whether to perform feature selection.
    compare_models_flag : bool, default=False
        Whether to compare models and select the best one.
    feature_counts : list of int, default=None
        List of feature counts to consider during selection.
    random_state : int, default=None
        Random seed for reproducibility.
    model : RegressorMixin, default=None
        The regression model to use. If None, a default model will be selected.
    cv_splits : int, default=5
        Number of cross-validation splits.
    shuffle : bool, default=True
        Whether to shuffle the data before splitting.
    scoring : str, default='neg_mean_squared_error'
        Scoring metric for model evaluation.
    rfe_step : int, default=1
        Number of features to remove at each step of RFE.

    Attributes
    ----------
    best_feature_counts_ : int
        Optimal number of features determined during selection.
    best_model_name_ : str
        Name of the best-performing model.

    Public Methods
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
        super().__init__(config, X_train, y_train, selected_features_)
        # Store init params
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

        # Learned attributes (set after pipeline run)
        self.best_feature_counts_: int | None = None
        self.best_model_name_: str | None = None

    def run(self) -> None:
        """
        Runs the entire pipeline.
        """
        # Feature selection step
        if self.feature_selection_flag:
            self._find_best_feature_counts()
            self._refit_rfe()
            if self.save_flag:
                save_object(self.selected_features_,
                                    self.selected_features_file,
                                    self.artifacts_dir_path)
        else:
            if self.selected_features_ is None:
                try:
                    self.selected_features_ = load_object(self.selected_features_file,
                                                                    dir_path=self.artifacts_dir_path)
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
                save_text(self.best_model_name_,
                                  self.best_model_name_file,
                                  self.artifacts_dir_path)
                self.logger.info("Saved best model name to %s", self.best_model_name_file)
        else:
            try:
                self.best_model_name_ = load_text(self.best_model_name_file,
                                                            dir_path=self.artifacts_dir_path)
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


class ModelTrainer(BaseModelingPipeline):
    """
    Train the best regression model on selected features.


    This class loads previously selected features and the chosen model
    (identified by ``ModelSelector``), fits it on the full training set,
    and optionally saves the final model.

    Inherits from
    -------------
    BaseModelingPipeline : Provides logging, error handling, and model registry.

    Parameters
    ----------
    model_path : str
        Path to save the trained model file.

    Attributes
    ----------
    best_model_name_ : str, default=None
        Name of the best model. If None, it is loaded from file.
    best_model_ : RegressorMixin
        Fitted regression model.


    Public Methods
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

        super().__init__(config, X_train, y_train, selected_features_)
        self.best_model_name_ = best_model_name_
        self.best_model_ = None

        # Parameters from config
        self.final_model_file = self.config.files.final_model

    def run(self) -> None:
        """
        Runs the entire pipeline.
        """
        if self.selected_features_ is None:
            try:
                self.selected_features_ = load_object(self.selected_features_file,
                                                                dir_path=self.artifacts_dir_path)
                self.logger.info("Loaded selected features: %s", self.selected_features_)
            except Exception as e:
                raise ValueError("selected_features_ is None. You can either " +
                                 "run `ModelSelector.run()` to perform feature " +
                                 "selection or give selected_features_ a default list.")
        if self.best_model_name_ is None:
            try:
                self.best_model_name_ = load_text(self.best_model_name_file,
                                                            dir_path=self.artifacts_dir_path)
            except Exception as e:
                raise ValueError("best_model_name_ is None")

        self.best_model_ = self.final_models_[self.best_model_name_]

        # Final fitting step
        self._fit()
        if self.save_flag:
            save_object(self.best_model_,
                                self.final_model_file,
                                self.artifacts_dir_path)
            self.logger.info("Saved best model to %s", self.final_model_file)
        self.logger.info('Pipeline execution completed successfully.')

    def _fit(self) -> None:
        """
        Fits the best model on the full training set.
        """
        if self.logging_flag:
            self.logger.info('Fitting the best model on the full training set...')
        self.best_model_.fit(self.X_train[self.selected_features_], self.y_train)
