import argparse

from omegaconf import OmegaConf
from src.preprocessing import Splitter, Preprocessor

parser = argparse.ArgumentParser(
    prog='prepare.py',
    description='Scrape data science job postings',
    epilog=f'Thanks for using.'
)

parser.add_argument('-s', '--save_flag',
                    action='store_true',
                    help='Whether to save the preprocessed data and artifacts')
parser.add_argument('--transform_target',
                    action='store_true',
                    help='Whether to transform the target variable')

args = parser.parse_args()

config = OmegaConf.load('config.yaml')


def main(save_flag: bool, transform_target: bool):
    train_df_, test_df_ = Splitter(config, save_flag).split()
    print(train_df_.shape, test_df_.shape)

    PREPROCESSED_TRAIN_PATH = config.paths.preprocessed_train
    PREPROCESSED_TEST_PATH = config.paths.preprocessed_test

    preprocessor = Preprocessor(config, save_flag, transform_target)

    preprocessor.run(input_df=train_df_,
                     preprocessed_path=PREPROCESSED_TRAIN_PATH)
    preprocessor.run(input_df=test_df_,
                     src_df=train_df_,
                     preprocessed_path=PREPROCESSED_TEST_PATH)

if __name__ == '__main__':
    main(args.save_flag, args.transform_target)
