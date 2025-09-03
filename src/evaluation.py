import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score, 
    mean_absolute_percentage_error
)
from sklearn.base import RegressorMixin
from omegaconf import OmegaConf
import warnings


from .utils import load_object, save_dataframe, save_text, select_features
from .inference import InferencePipeline

warnings.filterwarnings('ignore')


class Evaluator:
    """
    Evaluates the performance of the trained model on the test set.

    >>> Example usage:

    >>> Load data
    loader = DataLoader(config,
                        file_path=config.files.preprocessed_test)
    X_test, y_test = loader.load()
    src_df = load_dataframe(config.files.train_data,
                            config.dirs.data)

    >>> Initialize evaluator
    evaluator = Evaluator(config,
                          X_test,
                          y_test,
                          src_df,
                          save_results)

    >>> Run evaluation
    evaluator.run(model=your_trained_model)

    >>> Print summary
    evaluator.print_summary()
    """
    def __init__(self,
                 config: OmegaConf,
                 X_test: pd.DataFrame,
                 y_test: pd.DataFrame,
                 src_df: pd.DataFrame,
                 save_results: bool = True,
                 name_prefix: str = None,
                 model: RegressorMixin = None):
        self.config = config
        self.feature_selection = config.inference.feature_selection
        self.X_test = X_test
        self.y_test = y_test
        self.src_df = src_df
        self.save_results = save_results
        self.name_prefix = name_prefix
        self.model = model
        self.predictions_df_ = None
        self.metrics_ = {}
        self.report_ = None

    def run(self, model=None):
        """
        Complete evaluation pipeline
        """
        self._handle_errors()
        # Use provided model or load from config
        if model is None:
            try:
                self.model = load_object(self.config.files.final_model,
                                         self.config.dirs.artifacts)
            except Exception as e:
                raise FileNotFoundError(f"Error loading model: {e}")

        # Generate predictions
        self.predictions_df_ = self._predict()
        
        # Calculate metrics
        self.metrics_ = self._calculate_metrics()

        # Generate evaluation report
        self.report_ = self._generate_report()
        
        # Create visualizations
        self.evaluation_figure = self._create_visualizations()

        # Create feature importance plot
        self.feature_importance_figure = self._get_feature_importance()

        if self.save_results:
            self._save_results()
        
        # return {
        #     'metrics': self.metrics_,
        #     'predictions': self.predictions_df_,
        #     'report': report
        # }

    def _handle_errors(self):
        """Handle errors that may occur during evaluation"""
        if self.X_test.shape[0] == 0 or self.y_test.shape[0] == 0:
            raise ValueError("Test set is empty.")
        if self.y_test.shape[0] != self.X_test.shape[0]:
            raise ValueError("Test set and labels must have the same number of samples.")

    def _predict(self) -> pd.DataFrame:
        """Generate predictions and create comparison dataframe"""
        inference_pipeline = InferencePipeline(self.config,
                                               self.model,
                                               input_df=self.X_test,
                                               src_df=self.src_df)
        y_pred = inference_pipeline.run()
        y_test_processed = self.y_test.copy()

        # Handle both Series and DataFrame for y_test
        if isinstance(y_test_processed, pd.DataFrame):
            actual_values = y_test_processed.iloc[:, 0].values
        else:
            actual_values = y_test_processed.values

        predictions_df = pd.DataFrame({
            "Actual": actual_values,
            "Predicted": y_pred,
            "Residual": actual_values - y_pred,
            "Absolute_Error": np.abs(actual_values - y_pred),
            "Percentage_Error": np.abs((actual_values - y_pred) / actual_values) * 100
        })       
        return predictions_df
    
    def _calculate_metrics(self) -> dict[str, float]:
        """Calculate comprehensive regression metrics"""
        actual = self.predictions_df_["Actual"]
        predicted = self.predictions_df_["Predicted"]
        
        metrics = {
            # Core regression metrics
            'mae': mean_absolute_error(actual, predicted),
            'mse': mean_squared_error(actual, predicted),
            'rmse': np.sqrt(mean_squared_error(actual, predicted)),
            'r2_score': r2_score(actual, predicted),
            'mape': mean_absolute_percentage_error(actual, predicted) * 100,
            
            # Additional useful metrics for salary prediction
            'mean_actual': actual.mean(),
            'mean_predicted': predicted.mean(),
            'std_actual': actual.std(),
            'std_predicted': predicted.std(),
            
            # Percentage-based metrics
            'mae_percentage': (mean_absolute_error(actual, predicted) / actual.mean()) * 100,
            'rmse_percentage': (np.sqrt(mean_squared_error(actual, predicted)) / actual.mean()) * 100,
            
            # Residual analysis
            'residual_mean': self.predictions_df_["Residual"].mean(),
            'residual_std': self.predictions_df_["Residual"].std(),
            
            # Accuracy within ranges (useful for salary predictions)
            'accuracy_within_5_percent': (self.predictions_df_["Percentage_Error"] <= 5).mean() * 100,
            'accuracy_within_10_percent': (self.predictions_df_["Percentage_Error"] <= 10).mean() * 100,
            'accuracy_within_15_percent': (self.predictions_df_["Percentage_Error"] <= 15).mean() * 100,
        }
        
        return metrics
    
    def _generate_report(self) -> str:
        """Generate a comprehensive evaluation report"""
        report = f"""
        SALARY PREDICTION MODEL EVALUATION REPORT
        ==========================================
        
        Dataset Information:
        - Test samples: {len(self.X_test)}
        - Features: {len(self.X_test.columns)}
        
        Core Performance Metrics:
        - R² Score: {self.metrics_['r2_score']:.4f}
        - Mean Absolute Error (MAE): ${self.metrics_['mae']:,.2f}
        - Root Mean Square Error (RMSE): ${self.metrics_['rmse']:,.2f}
        - Mean Absolute Percentage Error (MAPE): {self.metrics_['mape']:.2f}%
        
        Relative Performance:
        - MAE as % of mean salary: {self.metrics_['mae_percentage']:.2f}%
        - RMSE as % of mean salary: {self.metrics_['rmse_percentage']:.2f}%
        
        Prediction Accuracy:
        - Within 5% of actual: {self.metrics_['accuracy_within_5_percent']:.1f}% of predictions
        - Within 10% of actual: {self.metrics_['accuracy_within_10_percent']:.1f}% of predictions
        - Within 15% of actual: {self.metrics_['accuracy_within_15_percent']:.1f}% of predictions
        
        Statistical Summary:
        - Mean actual salary: ${self.metrics_['mean_actual']:,.2f}
        - Mean predicted salary: ${self.metrics_['mean_predicted']:,.2f}
        - Residual mean: ${self.metrics_['residual_mean']:,.2f}
        - Residual std: ${self.metrics_['residual_std']:,.2f}
        
        Model Interpretation:
        """
        
        # Add interpretation based on R² score
        if self.metrics_['r2_score'] >= 0.9:
            report += "- Excellent model performance (R² ≥ 0.90)\n"
        elif self.metrics_['r2_score'] >= 0.8:
            report += "- Good model performance (R² ≥ 0.80)\n"
        elif self.metrics_['r2_score'] >= 0.7:
            report += "- Moderate model performance (R² ≥ 0.70)\n"
        else:
            report += "- Model performance needs improvement (R² < 0.70)\n"
        
        # Add MAPE interpretation
        if self.metrics_['mape'] <= 10:
            report += "- High prediction accuracy (MAPE ≤ 10%)\n"
        elif self.metrics_['mape'] <= 20:
            report += "- Moderate prediction accuracy (MAPE ≤ 20%)\n"
        else:
            report += "- Consider model improvements (MAPE > 20%)\n"
        
        return report
    
    def _create_visualizations(self):
        """Create evaluation visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted scatter plot
        axes[0, 0].scatter(self.predictions_df_["Actual"], self.predictions_df_["Predicted"], 
                          alpha=0.4, color='blue')
        min_val = min(self.predictions_df_["Actual"].min(), self.predictions_df_["Predicted"].min())
        max_val = max(self.predictions_df_["Actual"].max(), self.predictions_df_["Predicted"].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Salary')
        axes[0, 0].set_ylabel('Predicted Salary')
        axes[0, 0].set_title(f'Actual vs Predicted (R² = {self.metrics_["r2_score"]:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        axes[0, 1].scatter(self.predictions_df_["Predicted"], self.predictions_df_["Residual"], 
                          alpha=0.4, color='green')
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Salary')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals distribution
        axes[1, 0].hist(self.predictions_df_["Residual"], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residuals Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Error distribution
        axes[1, 1].hist(self.predictions_df_["Percentage_Error"], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 1].set_xlabel('Absolute Percentage Error (%)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Percentage Error Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # plt.show()
        return fig
    
    def _save_results(self):
        """Save evaluation results to files"""
        # Save report
        save_text(self.report_,
                  self.config.files.evaluation_report,
                  self.config.dirs.logs,
                  self.name_prefix)
        # Save metrics
        metrics_df = pd.DataFrame([self.metrics_])
        save_dataframe(metrics_df,
                       self.config.files.evaluation_metrics,
                       self.config.dirs.logs,
                       self.name_prefix)
            
        # Save predictions
        save_dataframe(self.predictions_df_,
                       self.config.files.predictions,
                       self.config.dirs.logs,
                       self.name_prefix)

        # Save evaluation plots
        if self.evaluation_figure:
            fig_path = os.path.join(self.config.dirs.logs,
                                    self.config.files.evaluation_plots)
            self.evaluation_figure.savefig(fig_path,
                                           dpi=300,
                                           bbox_inches='tight')
        # Save feature importance plots
        if self.feature_importance_figure:
            fig_path = os.path.join(self.config.dirs.logs,
                                    self.config.files.feature_importance_plot)
            self.feature_importance_figure.savefig(fig_path,
                                                   dpi=300,
                                                   bbox_inches='tight')

    def _get_feature_importance(self, top_n: int = 10):
        """Get feature importance if model supports it"""
        if hasattr(self.model, 'feature_importances_'):
            if self.feature_selection:
                columns_to_keep = load_object(self.config.files.selected_features,
                                              self.config.dirs.artifacts)
                self.X_test = select_features(self.X_test, columns_to_keep)
            print(f'-------------------self.X_test.shape: {self.X_test.shape}')
            importance_df = pd.DataFrame({
                'feature': self.X_test.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Plot top features
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=importance_df.head(top_n), x='importance', y='feature', ax=ax)
            ax.set_title(f'Top {top_n} Feature Importance')
            ax.set_xlabel('Importance')
            plt.tight_layout()
            # plt.show()
            return fig
            
        else:
            print("Model doesn't support feature importance")
            return None
    
    def print_summary(self):
        """Print a concise summary of results"""
        print("="*50)
        print("MODEL EVALUATION SUMMARY")
        print("="*50)
        print(f"R² Score: {self.metrics_['r2_score']:.4f}")
        print(f"RMSE: ${self.metrics_['rmse']:,.2f}")
        print(f"MAE: ${self.metrics_['mae']:,.2f}")
        print(f"MAPE: {self.metrics_['mape']:.2f}%")
        print(f"Predictions within 10%: {self.metrics_['accuracy_within_10_percent']:.1f}%")
        print("="*50)
