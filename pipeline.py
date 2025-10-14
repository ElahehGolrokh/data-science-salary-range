from metaflow import FlowSpec, step, Parameter
from omegaconf import OmegaConf
from src.data_loader import DataLoader
from src.evaluation import Evaluator
from src.preprocessing import Preprocessor, Splitter
from src.training import ModelSelector, ModelTrainer, RandomForestFineTuner
from src.utils import load_dataframe


class Pipeline(FlowSpec):
    config_path: str = Parameter('config-path',
                                 default='config.yaml',
                                 help='path to the config file')
    save_flag: bool = Parameter('save-flag',
                                is_flag=True,
                                help='whether to save the preprocessed data & evaluation results')
    output_dir: str = Parameter('output-dir',
                                default='data',
                                help='where to write the prepared data')
    do_prepare: bool = Parameter('prepare',
                                 is_flag=True,
                                 help='whether to execute the preparation step.')
    do_train: bool = Parameter('train',
                               is_flag=True,
                               help='whether to execute the training step.')
    do_fine_tuning: bool = Parameter('fine-tune',
                                      is_flag=True,
                                      help='whether to execute the fine-tuning.')
    do_evaluation: bool = Parameter('evaluate',
                                    is_flag=True,
                                    help='whether to execute the evaluation step.')

    @step
    def start(self):
        print("Starting pipeline...")
        self.config = OmegaConf.load(self.config_path)
        self.model = None
        self.model_name = None
        self.next(self.prepare)

    @step
    def prepare(self):
        if self.do_prepare:
            print("Preparing data...")
            train_df_, test_df_ = Splitter(self.config, self.save_flag).split()

            if self.save_flag:
                PREPROCESSED_TRAIN_Name = self.config.files.preprocessed_train
                PREPROCESSED_TEST_Name = self.config.files.preprocessed_test
            else:
                PREPROCESSED_TRAIN_Name = None
                PREPROCESSED_TEST_Name = None

            preprocessor = Preprocessor(self.config, self.save_flag)

            self.preprocessed_train = preprocessor.run(input_df=train_df_,
                                                       phase='train',
                                                       preprocessed_path=PREPROCESSED_TRAIN_Name,
                                                       transform_target=True)
            self.preprocessed_test = preprocessor.run(input_df=test_df_,
                                                      src_df=train_df_,
                                                      phase='evaluation',
                                                      preprocessed_path=PREPROCESSED_TEST_Name,
                                                      transform_target=False)
        else:
            self.preprocessed_train = load_dataframe(file_path=self.config.files.preprocessed_train,
                                                     dir_path=self.output_dir)
            self.preprocessed_test = load_dataframe(file_path=self.config.files.preprocessed_test,
                                                    dir_path=self.output_dir)
        self.next(self.train)

    @step
    def train(self):
        if self.do_train:
            print("Training model...")
            loader = DataLoader(self.config)
            X_train, y_train = loader.load(self.preprocessed_train)

            # Feature selection
            model_selector = ModelSelector(self.config,
                                           X_train,
                                           y_train,
                                           compare_models=False,
                                           feature_count_=15)
            best_model_name_, selected_features_ = model_selector.run()

            # Model fine-tuning. Since the best model in our case is a RandomForest,
            # we use the specialized tuner.
            if self.do_fine_tuning:
                fine_tuner = RandomForestFineTuner(self.config)
                fine_tuner.fit(X_train[selected_features_], y_train)
                self.model_name = fine_tuner.best_model

            # Model training
            model_name = self.model_name if self.model_name else best_model_name_
            model_trainer = ModelTrainer(self.config,
                                         X_train,
                                         y_train,
                                         model_name=model_name,
                                         selected_features_=selected_features_)
            self.model = model_trainer.run()
        self.next(self.evaluate)

    @step
    def evaluate(self):
        if self.do_evaluation:
            print("Evaluating model...")
            loader = DataLoader(self.config)
            X_test, y_test = loader.load(self.preprocessed_test)
            src_df = load_dataframe(self.config.files.train_data,
                                    self.config.dirs.data)
            evaluator = Evaluator(self.config,
                                  X_test,
                                  y_test,
                                  src_df,
                                  self.save_flag)
            evaluator.run()
            evaluator.print_summary()
        self.next(self.end)

    @step
    def end(self):
        print('done.')


if __name__ == '__main__':
    Pipeline()
