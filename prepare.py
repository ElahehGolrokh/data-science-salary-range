import argparse

from omegaconf import OmegaConf
from src.preprocessing import Splitter

parser = argparse.ArgumentParser(
    prog='prepare.py',
    description='Scrape data science job postings',
    epilog=f'Thanks for using.'
)

parser.add_argument('--train_path', help='set the training data file path')
parser.add_argument('--test_path', help='set the testing data file path')
parser.add_argument('--file_path', help='set the file path of input data')
parser.add_argument('--train_size', help='set the proportion of the dataset to include in the train split')
parser.add_argument('--random_state', help='set the random seed for reproducibility')


args = parser.parse_args()

config = OmegaConf.load('config.yaml')
FILE_PATH = args.file_path if args.file_path else config.paths.feature_engineered
TRAIN_SIZE = args.train_size if args.train_size else config.preprocessing.train_size
RANDOM_STATE = args.random_state if args.random_state else config.preprocessing.random_state
TRAIN_PATH = args.train_path if args.train_path else config.paths.train_data
TEST_PATH = args.test_path if args.test_path else config.paths.test_data


def main(file_path: str,
         train_path: str,
         test_path: str,
         train_size: float,
         random_state: int,
         ):
    train_df_, test_df_ = Splitter(file_path,
                                   train_path,
                                   test_path,
                                   train_size,
                                   random_state).split()
    print(train_df_.shape, test_df_.shape)


if __name__ == '__main__':
    main(file_path=FILE_PATH,
         train_path=TRAIN_PATH,
         test_path=TEST_PATH,
         train_size=TRAIN_SIZE,
         random_state=RANDOM_STATE)
