import argparse

from omegaconf import OmegaConf

from src.data_loader import DataLoader
from src.training import ModelSelector, ModelTrainer, RandomForestFineTuner


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
parser.add_argument('-fn', '--fine_tune',
                    action='store_true',
                    help='Enable fine-tuning for the model')
args = parser.parse_args()

config = OmegaConf.load('config.yaml')


def main(compare_models: bool,
         train: bool,
         model_name: str = None,
         fine_tune: bool = False):
    loader = DataLoader(config,
                        file_path=config.files.preprocessed_train)
    X_train, y_train = loader.load()

    # Feature selection and model comparison
    model_selector = ModelSelector(config,
                                   X_train,
                                   y_train,
                                   compare_models=compare_models,
                                   feature_count_=15)
    best_model_name_, selected_features_ = model_selector.run()

    # Model fine-tuning. Since the best model in our case is a RandomForest,
    # we use the specialized tuner. For a more general case, it should be
    # replaced to support other model types.
    if fine_tune:
        if model_name is not None and model_name != "RandomForest":
            raise ValueError("Fine-tuning is only supported for RandomForest.")
        else:
            fine_tuner = RandomForestFineTuner(config)
            fine_tuner.fit(X_train[selected_features_], y_train)
            model_name = fine_tuner.best_model

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
         args.model_name,
         args.fine_tune)
