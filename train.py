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
args = parser.parse_args()

config = OmegaConf.load('config.yaml')


def main(compare_models: bool,
         train: bool):
    loader = DataLoader(config,
                        file_path=config.files.preprocessed_train)
    X_train, y_train = loader.load()
    best_model_name_, selected_features_ = None, None

    # Feature selection and model comparison
    feature_selection = config.inference.feature_selection
    if feature_selection or compare_models:
        model_selector = ModelSelector(config,
                                       X_train,
                                       y_train,
                                       feature_selection=feature_selection,
                                       compare_models=compare_models)
        best_model_name_, selected_features_ = model_selector.run()
        print(f'***************** model_selector.best_feature_counts_ : {model_selector.best_feature_counts_}')


    # Model training
    if train:
        model_trainer = ModelTrainer(config,
                                     X_train,
                                     y_train,
                                     best_model_name_=best_model_name_,
                                     selected_features_=selected_features_)
        model_trainer.run()


if __name__ == "__main__":
    main(args.compare_models,
         args.train)
