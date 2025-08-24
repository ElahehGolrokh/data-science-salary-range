import argparse

from omegaconf import OmegaConf

from src.data_loader import DataLoader
from src.evaluation import Evaluator
from src.utils import load_dataframe


parser = argparse.ArgumentParser(
    prog='train.py',
    description='Modeling pipeline on the preprocessed data',
    epilog=f'Thanks for using.'
)

parser.add_argument('-s', '--save',
                    action='store_true',
                    help='Whether to save evaluation results')
args = parser.parse_args()

config = OmegaConf.load('config.yaml')


def main(save_results: bool):
    loader = DataLoader(config,
                        file_path=config.files.preprocessed_test)
    X_test, y_test = loader.load()
    src_df = load_dataframe(config.files.train_data,
                            config.dirs.data)
    evaluator = Evaluator(config,
                          X_test,
                          y_test,
                          src_df,
                          save_results)
    evaluator.run()
    evaluator.print_summary()

    # Get feature importance (if supported)
    # feature_importance = evaluator._get_feature_importance()
    # print(f'Feature Importance: {feature_importance}')
    # Example usage:
    """
    # Initialize evaluator
    evaluator = Evaluator(config, X_test, y_test)

    # Run evaluation
    results = evaluator.run(model=your_trained_model)

    # Print summary
    evaluator.print_summary()

    # Get feature importance (if supported)
    feature_importance = evaluator.get_feature_importance()
    """


if __name__ == "__main__":
    main(args.save)
