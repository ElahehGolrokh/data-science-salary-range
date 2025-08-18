import argparse

from omegaconf import OmegaConf
from src.preprocessing import Splitter, Preprocessor

parser = argparse.ArgumentParser(
    prog='prepare.py',
    description='Scrape data science job postings',
    epilog=f'Thanks for using.'
)

# data directory path
parser.add_argument('--data_dir_path', help='set the directory path for saving data')
parser.add_argument('--artifacts_dir_path', help='set the directory path for saving artifacts')

# Splitter object's parameters
parser.add_argument('--train_path', help='set the training data file path')
parser.add_argument('--test_path', help='set the testing data file path')
parser.add_argument('--file_path', help='set the file path of input data')
parser.add_argument('--train_size', help='set the proportion of the dataset to include in the train split')
parser.add_argument('--random_state', help='set the random seed for reproducibility')

# Preprocessor object's parameters
parser.add_argument('--one_hot_encoder_path', help='set the file path for saving the one hot encoder')
parser.add_argument('--mlb_path', help='set the file path for saving the multi-label binarizer')
parser.add_argument('--scaler_path', help='set the file path for saving the scaler')
parser.add_argument('--preprocessed_train_path', help='set the file path for saving the preprocessed training data')
parser.add_argument('--preprocessed_test_path', help='set the file path for saving the preprocessed testing data')

args = parser.parse_args()

config = OmegaConf.load('config.yaml')

# directory paths
DATA_DIR_PATH = args.data_dir_path if args.data_dir_path else config.paths.data_dir
ARTIFACTS_DIR_PATH = args.artifacts_dir_path if args.artifacts_dir_path else config.paths.artifacts_dir

# splitter object's parameters
FILE_PATH = args.file_path if args.file_path else config.paths.feature_engineered
TRAIN_SIZE = args.train_size if args.train_size else config.preprocessing.train_size
RANDOM_STATE = args.random_state if args.random_state else config.preprocessing.random_state
TRAIN_PATH = args.train_path if args.train_path else config.paths.train_data
TEST_PATH = args.test_path if args.test_path else config.paths.test_data

# preprocessor object's parameters
ONE_HOT_ENCODER_PATH = config.paths.one_hot_encoder
MLB_PATH = config.paths.mlb
SCALER_PATH = config.paths.scaler
PREPROCESSED_TRAIN_PATH = args.preprocessed_train_path if args.preprocessed_train_path else config.paths.preprocessed_train
PREPROCESSED_TEST_PATH = args.preprocessed_test_path if args.preprocessed_test_path else config.paths.preprocessed_test


def main(data_dir_path: str,
         artifacts_dir_path: str,
         file_path: str,
         train_path: str,
         test_path: str,
         train_size: float,
         random_state: int,
         one_hot_encoder_path: str,
         mlb_path: str,
         scaler_path: str,
         preprocessed_train_path: str,
         preprocessed_test_path: str
         ):
    train_df_, test_df_ = Splitter(data_dir_path,
                                   file_path,
                                   train_path,
                                   test_path,
                                   train_size,
                                   random_state).split()
    print(train_df_.shape, test_df_.shape)

    preprocessor = Preprocessor(data_dir_path=data_dir_path,
                                artifacts_dir_path=artifacts_dir_path,
                                one_hot_encoder_path=one_hot_encoder_path,
                                mlb_path=mlb_path,
                                scaler_path=scaler_path)

    preprocessor.run(input_df=train_df_,
                     preprocessed_path=preprocessed_train_path)
    preprocessor.run(input_df=test_df_,
                     src_df=train_df_,
                     preprocessed_path=preprocessed_test_path)

if __name__ == '__main__':
    main(
         data_dir_path=DATA_DIR_PATH,
         artifacts_dir_path=ARTIFACTS_DIR_PATH,
         file_path=FILE_PATH,
         train_path=TRAIN_PATH,
         test_path=TEST_PATH,
         train_size=TRAIN_SIZE,
         random_state=RANDOM_STATE,
         one_hot_encoder_path=ONE_HOT_ENCODER_PATH,
         mlb_path=MLB_PATH,
         scaler_path=SCALER_PATH,
         preprocessed_train_path=PREPROCESSED_TRAIN_PATH,
         preprocessed_test_path=PREPROCESSED_TEST_PATH)
