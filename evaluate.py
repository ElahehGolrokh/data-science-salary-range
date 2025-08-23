import argparse

from omegaconf import OmegaConf

from src.data_loader import DataLoader
from src.evaluation import Evaluator


parser = argparse.ArgumentParser(
    prog='train.py',
    description='Modeling pipeline on the preprocessed data',
    epilog=f'Thanks for using.'
)

parser.add_argument('-fs', '--feature_selection',
                    action='store_true',
                    help='Enable feature selection in the pipeline')
parser.add_argument('-cm', '--compare_models',
                    action='store_true',
                    help='Enable model comparison in the pipeline')
parser.add_argument('-t', '--train',
                    action='store_true',
                    help='Enable model training in the pipeline')
args = parser.parse_args()

config = OmegaConf.load('config.yaml')


def main(feature_selection: bool,
         compare_models: bool,
         train_flag: bool):
    loader = DataLoader(config,
                        file_path=config.files.preprocessed_test)
    X_test, y_test = loader.load()

    evaluator = Evaluator(config,
                          X_test,
                          y_test)
    evaluator.run()


if __name__ == "__main__":
    main(args.feature_selection,
         args.compare_models,
         args.train)