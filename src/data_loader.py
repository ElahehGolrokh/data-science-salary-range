import pandas as pd


class DataLoader:
    """
    Utility class to load a dataset from a CSV file and separate features from the target.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the dataset.
    target : str
        Name of the target column in the dataset.

    Attributes
    ----------
    file_path : str
        Path to the CSV file.
    target : str
        Name of the target column.
    """
    def __init__(self, file_path: str, target):
        self.file_path = file_path
        self.target = target

    def load(self):
        """
        Load the dataset from the CSV file and split into features and target.

        Returns
        -------
        X : pandas.DataFrame
            Feature matrix obtained by dropping the target column.
        y : pandas.Series
            Target variable corresponding to the target column.
        """
        df = pd.read_csv(self.file_path)
        X = df.drop(columns=[self.target], axis=1)
        y = df[self.target]
        return X, y
