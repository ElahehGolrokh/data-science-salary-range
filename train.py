import argparse
import pandas as pd
import yaml

from omegaconf import OmegaConf

from src.training import ModelingPipeline


# parser = argparse.ArgumentParser(
#     prog='train.py',
#     description='Modeling pipeline on the preprocessed data',
#     epilog=f'Thanks for using.'
# )

# parser.add_argument('--', help='set the training data file path')
# args = parser.parse_args()

config = OmegaConf.load('config.yaml')


def main():
    train_scaled = pd.read_csv('data/preprocessed_train_df.csv')
    X_train = train_scaled.drop('mean_salary', axis=1)
    y_train = train_scaled['mean_salary']

    test_scaled = pd.read_csv('data/preprocessed_test_df.csv')
    X_test = test_scaled.drop('mean_salary', axis=1)
    y_test = test_scaled['mean_salary']

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    pipeline = ModelingPipeline(config,
                                X_train,
                                y_train,
                                X_test,
                                y_test,
                                default_model=None,
                                )
    pipeline.run_pipeline()
    print(f'pipeline.best_feature_counts_ : {pipeline.best_feature_counts_}')

if __name__ == "__main__":
    main()