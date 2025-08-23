import pandas as pd

from omegaconf import OmegaConf


class Evaluator:
    def __init__(self,
                 config: OmegaConf,
                 X_test: pd.DataFrame,
                 y_test: pd.DataFrame):
        self.config = config
        self.X_test = X_test
        self.y_test = y_test
    
    def run(self):
        pass

    def _predict(self, model) -> pd.DataFrame:
        y_pred = model.predict(self.X_test)
        return pd.DataFrame({"Actual": self.y_test, "Predicted": y_pred})
