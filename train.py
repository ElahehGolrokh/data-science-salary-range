import argparse

from omegaconf import OmegaConf

from src.data_loader import DataLoader
from src.training import ModelSelector, ModelTrainer


parser = argparse.ArgumentParser(
    prog='train.py',
    description='Modeling pipeline on the preprocessed data',
    epilog=f'Thanks for using.'
)

parser.add_argument('-cm', '--compare_models',
                    action='store_true',
                    help='Enable model comparison in the pipeline')
parser.add_argument('-t', '--train',
                    action='store_true',
                    help='Enable model training in the pipeline')
parser.add_argument('-mn', '--model_name',
                    type=str,
                    help='Name of the model to train')
args = parser.parse_args()

config = OmegaConf.load('config.yaml')


def main(compare_models: bool,
         train: bool,
         model_name: str = None):
    loader = DataLoader(config,
                        file_path=config.files.preprocessed_train)
    X_train, y_train = loader.load()

    # Feature selection and model comparison
    # feature_selection = config.inference.feature_selection
    # if feature_selection or compare_models:
    model_selector = ModelSelector(config,
                                   X_train,
                                   y_train,
                                   compare_models=compare_models,
                                   feature_count_=20)
    best_model_name_, selected_features_ = model_selector.run()


    # Model training
    model_name = model_name if model_name else best_model_name_
    if train:
        model_trainer = ModelTrainer(config,
                                     X_train,
                                     y_train,
                                     model_name=model_name,
                                     selected_features_=selected_features_)
        model_trainer.run()


if __name__ == "__main__":
    main(args.compare_models,
         args.train,
         args.model_name)
