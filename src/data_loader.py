from omegaconf import OmegaConf
import pandas as pd

from .utils import load_dataframe


class DataLoader:
    """
    Utility class to load a dataset from a CSV file and separate
    features from the target.

    Parameters
    ----------
    file_path : str
        name of the CSV file containing the dataset.
    dir_path : str
        path to the directory containing the dataset.
    target : str
        Name of the target column.
    
    Public Methods
    --------------
    load() -> tuple[pd.DataFrame, pd.Series]
        Loads the dataset and splits it into features and target.

    """
    def __init__(self,
                 config: OmegaConf,
                 file_path: str = None,
                 ):
        self.file_path = file_path
        self.dir_path = config.dirs.data
        self.target = config.preprocessing.target[0]

    def load(self, df: pd.DataFrame = None):
        """
        Load the dataset from the CSV file and split into features and target.

        Returns
        -------
        X : pandas.DataFrame
            Feature matrix obtained by dropping the target column.
        y : pandas.Series
            Target variable corresponding to the target column.
        """
        if df is None:
            if self.file_path is None:
                raise ValueError("File path must be specified.")
            df = load_dataframe(file_path=self.file_path,
                                dir_path=self.dir_path)
        X = df.drop(columns=[self.target], axis=1)
        y = df[self.target]
        return X, y
