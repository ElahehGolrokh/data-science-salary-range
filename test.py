from omegaconf import OmegaConf

from src.inference import InferencePipeline
from src.preprocessing import Preprocessor
from src.utils import load_dataframe, load_object


def main():
    """Run inference on a single example."""
    config = OmegaConf.load('config.yaml')
    model = load_object(config.training.model_path)
    input_df = load_dataframe(config.paths.test_data)\
               .iloc[[0]]\
               .drop(config.preprocessing.columns_to_drop, axis=1)
    src_df = load_dataframe(config.paths.train_data)
    preprocessor = Preprocessor(config,
                                save_flag=False,
                                transform_target=False)
    preprocessed_df = preprocessor.run(input_df=input_df,
                                       src_df=src_df,
                                       phase='inference')
    inference_pipeline = InferencePipeline(config,
                                           model,
                                           input_df=preprocessed_df,
                                           src_df=src_df)
    result = inference_pipeline.run()
    print('result: {}'.format(result))


if __name__ == '__main__':
    main()
