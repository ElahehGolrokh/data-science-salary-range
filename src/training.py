import numpy as np
import os
import pandas as pd

from kneed import KneeLocator
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import RFE

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from typing import Dict, Callable
import warnings

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
    feature_selection : bool, default=True
        Whether to perform feature selection.
    compare_models : bool, default=False
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
    feature_count_ : int, default=None
        Number of features to select.

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
    'RandomForestRegressor'
    """
    def __init__(self,
                 config: dict,
                 X_train: pd.DataFrame,
                 y_train: pd.Series,
                 selected_features_: list[str] = None,
                 compare_models: bool = False,
                 feature_count_: int = None):
        super().__init__(config, X_train, y_train, selected_features_)
        # Store init params
        self.compare_models = compare_models
        self.feature_count_ = feature_count_

        # Parameters from config
        self.feature_counts = self.config.training.feature_counts
        self.random_state = self.config.training.random_state
        self.model = get_default_model(self.config, MODEL_MAP)
        self.cv_splits = self.config.training.cv_splits
        self.shuffle = self.config.training.shuffle
        self.scoring = self.config.training.scoring
        self.rfe_step = self.config.training.rfe_step

        # Learned attributes (set after pipeline run)
        self.best_model_name_: str | None = None

    def run(self) -> None:
        """
        Runs the entire pipeline.
        """
        # Feature selection step
        if self.feature_selection:
            if self.feature_count_ is None:
                if self.logging_flag:
                    self.logger.info('Finding the optimum number of features...')
                    self.feature_count_ = self._find_best_feature_counts()
            else:
                if self.logging_flag:
                    self.logger.info('Using predefined feature count: %s', self.feature_count_)

            self.selected_features_ = self._refit_rfe()

        else:
            # Use all features
            self.selected_features_ = self.X_train.columns.tolist()
        
        if self.save_flag:
            save_object(self.selected_features_,
                        self.selected_features_file,
                        self.artifacts_dir_path)

        # Model comparison step
        if self.compare_models:
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
                self.best_model_name_ = None
                self.logger.warning("best_model_name_ is None. run train.py -cm to"
                                    " compare models & find the best performing one.")
        return self.best_model_name_, self.selected_features_

    # feature selection
    def _find_best_feature_counts(self,) -> int:
        """
        Finds the best number of features to preserve.
        """
        if self.model is None:
            self.model = RandomForestRegressor()
        mean_scores = []
        for n_features in self.feature_counts:
            if n_features > self.X_train.shape[1]:
                raise ValueError(f"Cannot select {n_features} features; "
                                 f"only {self.X_train.shape[1]} features available.")
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
        best_feature_count = self._choose_features_from_elbow(mean_scores, self.feature_counts)
        return best_feature_count

    def _refit_rfe(self) -> list[str]:
        """
        Refits RFE on the full training set with the best number of features.
        """
        if self.logging_flag:
            self.logger.info(f'Refitting RFE with {self.feature_count_} features...')
        rfe = RFE(estimator=self.model,
                  n_features_to_select=self.feature_count_,
                  step=self.rfe_step,
                  verbose=0)
        rfe.fit(self.X_train, self.y_train)
        selected_features_ = self.X_train.columns[rfe.support_].tolist()
        # Force certain features to be included
        force_to_keep = ['seniority_level']
        selected_features_ = selected_features_ + [f for f in force_to_keep if f not in selected_features_]

        return list(selected_features_)

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

    def _get_best_model(self) -> None:
        """
        Sets the best model based on mean cross-validation score.
        """
        if self.logging_flag:
            self.logger.info('Getting the best trained model...')
        comparison_df = self._compare_models()
        self.best_model_name_ = comparison_df.idxmax(axis=1)[0]
        self.logger.info("The best model based on mean cross-validation score is: %s",
                         self.best_model_name_)

    @staticmethod
    def _choose_features_from_elbow(mean_scores: list[float],
                                    feature_counts: list[int],
                                    curve: str = 'concave',
                                    direction: str = 'increasing') -> int:
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
    model_name_ : str, default=None
        Name of the model. If None, it is loaded from file.
    model_ : RegressorMixin
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
                 model_name: str = None):

        super().__init__(config, X_train, y_train, selected_features_)
        self.model_name = model_name
        self.model_ = None

        # Parameters from config
        self.final_model_file = self.config.files.final_model
        self.best_params_file = self.config.files.best_params

    def run(self) -> None:
        """
        Runs the entire pipeline.
        """
        if self.feature_selection and self.selected_features_ is None:
            try:
                self.selected_features_ = load_object(self.selected_features_file,
                                                      dir_path=self.artifacts_dir_path)
                self.logger.info("Loaded selected features: %s", self.selected_features_)
            except Exception as e:
                raise ValueError("No selected features found. Either set `feature_selection=True`"
                                 " in the config & run `ModelSelector.run()`, or provide "
                                 "`selected_features_`.")
        if self.model_name is None:
            try:
                self.model_name = load_text(self.best_model_name_file,
                                            dir_path=self.artifacts_dir_path)
            except Exception as e:
                raise ValueError("model_name_ is None. Either pass a model name "
                                 "or run `ModelSelector.run()` with `compare_models = True`"
                                 " to perform model selection.")

        if isinstance(self.model_name , str):
            if self.model_name == "RandomForestRegressor":
                self.model_ = self._load_model()
            else:
                # Load models other than RandomForestRegressor from final_models_ dict
                self.model_ = self.final_models_[self.model_name]
                self.logger.info("Loaded model: %s", self.model_name)
                self.logger.info("This model is fitted with default hyperparameters.")
        elif isinstance(self.model_name, RandomForestRegressor):
            # Then fine-tunning is already run & the best hyperparameters are loaded
            self.model_ = self.model_name
            self.logger.info("Loaded RandomForestRegressor model after fine-tuning with the \
                             best hyperparameters.")
        else:
            raise ValueError(f"Invalid model name: {self.model_name}")

        # Final fitting step
        self._fit()
        if self.save_flag:
            save_object(self.model_,
                        self.final_model_file,
                        self.artifacts_dir_path)
            self.logger.info("Saved best model to %s", self.final_model_file)
        self.logger.info('Pipeline execution completed successfully with %s features. \n' \
                         'selected_features: %s', len(self.selected_features_), self.selected_features_)

    def _load_model(self) -> RandomForestRegressor:
        """
        Load RandomForestRegressor with either fine-tuned best params (if available)
        or default params from config.yaml.
        """
        best_params_path = os.path.join(self.artifacts_dir_path,
                                        self.best_params_file)

        if os.path.exists(best_params_path):
            # ✅ Fine-tuned params found
            best_params = load_object(self.best_params_file,
                                      self.artifacts_dir_path)
            self.logger.info(f"⚡ Loading fine-tuned RandomForestRegressor with params \
                             from {self.best_params_file}")
            return RandomForestRegressor(**best_params)
        else:
            # ❌ No fine-tuned params → load defaults
            self.logger.warning("⚠️ No fine-tuned params found. Loading default \
                                RandomForestRegressor from config.")
            default_model_cfg = self.config.training.default_model
            model_name = default_model_cfg["name"]
            model_params = default_model_cfg.get("params", {})
            return MODEL_MAP[model_name](**model_params)

    def _fit(self) -> None:
        """
        Fits the best model on the full training set.
        """
        if self.logging_flag:
            self.logger.info('Fitting the best model on the full training set...')
        if self.feature_selection:
            self.X_train = self.X_train[self.selected_features_]
        self.model_.fit(self.X_train, self.y_train)


class RandomForestFineTuner:
    """
    Fine-tunes a RandomForestRegressor using grid search or random search.

    This class wraps scikit-learn's GridSearchCV or RandomizedSearchCV
    for hyperparameter optimization of RandomForestRegressor. It can
    automatically save the best parameters, reload them, and evaluate
    the fitted model with custom metrics.

    Parameters
    ----------
    config : dict
        Configuration object (e.g., loaded from YAML/JSON) containing
        training and preprocessing parameters.
    search : {"grid", "random"}, default="grid"
        Type of hyperparameter search to perform.
    cv : int, default=5
        Number of cross-validation folds.
    n_iter : int, default=30
        Number of parameter settings sampled in randomized search.
        Ignored if ``search="grid"``.
    scoring : str, default="r2"
        Scoring metric for optimization (must be valid sklearn scorer).
    n_jobs : int, default=-1
        Number of parallel jobs.

    Attributes
    ----------
    param_grid : dict
        Candidate hyperparameter values.
    best_model : RandomForestRegressor or None
        Best fitted model after search.
    best_params : dict or None
        Best hyperparameter set found.
    searcher : GridSearchCV or RandomizedSearchCV or None
        The underlying search object after fitting.
    random_state : int
        Random seed for reproducibility.
    save_flag : bool
        Whether to save the best parameters to disk.

    >>> Example
    >>> config = {...}
    >>> tuner = RandomForestFineTuner(config)
    >>> best_model = tuner.fit(X_train, y_train)
    >>> tuner.evaluate(X_test, y_test, {'r2_score': r2_score,
                                        'mean_absolute_error': mean_absolute_error})
    """
    def __init__(self,
                 config: dict,
                 search: str = "grid",
                 cv: int = 5,
                 n_iter: int = 30,
                 scoring: str = "r2",
                 n_jobs: int = -1):
        """
        search: "grid" or "random"
        cv: number of cross-validation folds
        n_iter: number of parameter settings for RandomizedSearchCV
        scoring: metric for optimization ("r2", "neg_mean_absolute_error", etc.)
        n_jobs: parallel jobs
        """
        self.config = config
        self.search = search
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.random_state = self.config.preprocessing.random_state
        self.save_flag = self.config.training.save_flag
        self.best_model = None
        self.best_params = None
        self.searcher = None

        # Define parameter grid
        self.param_grid = {
            "n_estimators": [200, 400, 800, 1200],
            "max_depth": [None, 10, 20, 30, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 5, 10],
            "max_features": ["auto", "sqrt", 0.3, 0.5, 0.7],
            "bootstrap": [True, False]
        }

    def fit(self, X, y):
        base_model = RandomForestRegressor(random_state=self.random_state,
                                           n_jobs=self.n_jobs)

        if self.search == "grid":
            self.searcher = GridSearchCV(
                estimator=base_model,
                param_grid=self.param_grid,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                verbose=2,
            )
        else:  # random search
            self.searcher = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                scoring=self.scoring,
                cv=self.cv,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=2,
            )

        self.searcher.fit(X, y)
        self.best_model = self.searcher.best_estimator_
        self.best_params = self.searcher.best_params_
        if self.save_flag:
            save_object(self.best_params,
                        self.config.files.best_params,
                        self.config.dirs.artifacts)
        return self.best_model

    def evaluate(self, X, y_true, custom_metrics: Dict[str, Callable]):
        """
        Evaluate the fitted model on given data using custom metrics.
        It is flexible enough to support any scikit-learn compatible
        metrics (e.g., `mean_absolute_error`, `r2_score`) or custom functions 
        following the signature `func(y_true, y_pred) -> float`.

        Parameters
        ----------
        X : ArrayLike or pd.DataFrame
            Input features for prediction.
        y_true : ArrayLike
            True target values.
        custom_metrics : Dict[str, Callable]
            Dictionary of metric functions with names as keys.
            Each function must take (y_true, y_pred) and return a float.

        Returns
        -------
        Dict[str, float]
            Metric names mapped to their computed values.

        Raises
        ------
        ValueError
            If the model is not yet fitted.
        """
        if self.best_model is None:
            raise ValueError("Model not yet fitted. Call fit() first.")

        y_pred = self.best_model.predict(X)
        results = dict()
        for metric_name, metric_func in custom_metrics.items():
            try:
                results[metric_name] = metric_func(y_true, y_pred)
            except Exception as e:
                results[metric_name] = np.nan
                warnings.warn(f"Metric {metric_name} failed: {e}")
        return results
