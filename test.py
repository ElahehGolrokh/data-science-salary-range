import argparse
from omegaconf import OmegaConf

from src.inference import InferencePipeline
from src.preprocessing import Preprocessor
from src.utils import load_dataframe, load_object

parser = argparse.ArgumentParser(
    prog='train.py',
    description='Modeling pipeline on the preprocessed data',
    epilog=f'Thanks for using.'
)

parser.add_argument('-t', '--transform_target',
                    action='store_true',
                    help='Whether to transform the target variable')
args = parser.parse_args()

config = OmegaConf.load('config.yaml')

transform_target = args.transform_target if args.transform_target is not None \
            else config.preprocessing.transform_target

def main(transform_target: bool):
    """Run inference on a single example."""
    config = OmegaConf.load('config.yaml')
    model = load_object(config.files.final_model,
                        config.dirs.artifacts)
    input_df = load_dataframe(config.files.test_data,
                              config.dirs.data)\
               .iloc[[0]]\
               .drop(config.preprocessing.columns_to_drop, axis=1)
    src_df = load_dataframe(config.files.train_data,
                            config.dirs.data)
    preprocessor = Preprocessor(config,
                                save_flag=False)
    preprocessed_df = preprocessor.run(input_df=input_df,
                                       src_df=src_df,
                                       phase='inference',
                                       transform_target=transform_target)
    inference_pipeline = InferencePipeline(config,
                                           model,
                                           input_df=preprocessed_df,
                                           src_df=src_df)
    result = inference_pipeline.run()
    print('result: {}'.format(result))


if __name__ == '__main__':
    main(transform_target)
