import argparse

from omegaconf import OmegaConf

from src.data_loader import DataLoader
from src.evaluation import Evaluator
from src.utils import load_dataframe


parser = argparse.ArgumentParser(
    prog='evaluate.py',
    description='Evaluate the trained model on the test set',
    epilog=f'Thanks for using.'
)

parser.add_argument('-s', '--save',
                    action='store_true',
                    help='Whether to save evaluation results')
parser.add_argument('-np', '--name_prefix',
                    type=str,
                    help='Prefix for saved evaluation result files')
args = parser.parse_args()

config = OmegaConf.load('config.yaml')


def main(save_results: bool,
         name_prefix: str = None):
    loader = DataLoader(config,
                        file_path=config.files.preprocessed_test)
    X_test, y_test = loader.load()
    src_df = load_dataframe(config.files.train_data,
                            config.dirs.data)
    evaluator = Evaluator(config,
                          X_test,
                          y_test,
                          src_df,
                          save_results,
                          name_prefix)
    evaluator.run()
    evaluator.print_summary()


if __name__ == "__main__":
    main(args.save,
         args.name_prefix)
