from omegaconf import OmegaConf

from src.inference import InferencePipeline
from src.utils import load_dataframe, load_object_from_file


def main():
    """Run inference on a single example."""
    config = OmegaConf.load('config.yaml')
    model = load_object_from_file(config.training.model_path)
    input_df = load_dataframe(config.paths.test_data)\
               .iloc[[0]]\
               .drop(config.preprocessing.columns_to_drop, axis=1)
    src_df = load_dataframe(config.paths.train_data)

    inference_pipeline = InferencePipeline(config,
                                           model,
                                           input_df,
                                           src_df)
    result = inference_pipeline.run()
    print('result: {}'.format(result))


if __name__ == '__main__':
    main()
