import argparse

from omegaconf import OmegaConf

from src.data_loader import DataLoader
from src.training import ModelSelector, ModelTrainer


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
                        file_path=config.files.preprocessed_train)
    X_train, y_train = loader.load()
    best_model_name_, selected_features_ = None, None

    # Feature selection and model comparison
    if feature_selection or compare_models:
        model_selector = ModelSelector(config,
                                       X_train,
                                       y_train,
                                       feature_selection_flag=feature_selection,
                                       compare_models_flag=compare_models)
        best_model_name_, selected_features_ = model_selector.run()
        print(f'***************** model_selector.best_feature_counts_ : {model_selector.best_feature_counts_}')


    # Model training
    if train_flag:
        model_trainer = ModelTrainer(config,
                                     X_train,
                                     y_train,
                                     best_model_name_=best_model_name_,
                                     selected_features_=selected_features_)
        model_trainer.run()


if __name__ == "__main__":
    main(args.feature_selection,
         args.compare_models,
         args.train)
