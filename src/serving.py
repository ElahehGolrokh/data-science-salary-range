import gradio as gr
from huggingface_hub import hf_hub_download
import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from .utils import load_object_from_file, load_dataframe
from .inference import InferencePipeline
from .preprocessing import Preprocessor


class GradioApp:
    def __init__(self, config: OmegaConf):
        self.config = config
        self.repo_id = config.repo_id
        self.features = [feat for feat in config.preprocessing.all_features
                        if feat not in config.preprocessing.columns_to_drop
                        and feat not in config.preprocessing.target]
        self.src_df = load_dataframe(config.paths.train_data)

    def launch(self):
        self._load_artifacts()
        app = self._get_interface()
        print('self.features = {}'.format(self.features))
        app.launch(share=True)

    def _download_artifacts(self):
        artifacts = {
            "model": "final_model.pkl",
            "scaler": "scaler.pkl",
            "mlb": "mlb.pkl",
            "one_hot_encoder": "one_hot_encoder.pkl",
            "features": "final_selected_features.pkl",   # which features to use
            "best_model_name": "best_model_name.txt"
        }
        downloaded = {}
        for key, filename in artifacts.items():
            try:
                path = hf_hub_download(repo_id=self.repo_id, filename=filename)
                downloaded[key] = path
            except Exception as e:
                print(f"⚠️ Could not download {filename}: {e}")
        return downloaded

    def _load_artifacts(self):
        downloaded = self._download_artifacts()
        self.model_ = load_object_from_file(downloaded["model"])
        self.scaler_ = load_object_from_file(downloaded["scaler"])
        self.mlb_ = load_object_from_file(downloaded["mlb"])
        self.one_hot_encoder_ = load_object_from_file(downloaded["one_hot_encoder"])
        self.selected_features_ = load_object_from_file(downloaded["features"])

        with open(downloaded["best_model_name"], "r") as f:
            self.best_model_name = f.read().strip()

        print(f"✅ Loaded model: {self.best_model_name}")

    def _get_user_inputs(self, user_inputs):
        """
        user_inputs: dict of feature_name -> value
        """
        try:
            # Build input vector in correct feature order
            X = []
            for feat in self.features:
                X.append(user_inputs.get(feat, None))  # default None if missing
            return X
        except Exception as e:
            print(f"⚠️ Error during input vector construction: {e}")
            return None

    def _predict(self, user_inputs):
        X = self._get_user_inputs(user_inputs)
        if X is None:
            return "⚠️ Error during input vector construction"
        try:
            X = np.array(X).reshape(1, -1)
            X = pd.DataFrame(X, columns=self.features)
            print('X = {}'.format(X))
            preprocessor = Preprocessor(self.config,
                                        save_flag=False,
                                        transform_target=False,
                                        one_hot_encoder_=self.one_hot_encoder_,
                                        mlb_=self.mlb_,
                                        scaler_=self.scaler_)
            preprocessed_df = preprocessor.run(input_df=X,
                                               src_df=self.src_df,
                                               phase='inference')
            inference_pipeline = InferencePipeline(self.config,
                                                   self.model_,
                                                   input_df=preprocessed_df,
                                                   src_df=self.src_df,
                                                   columns_to_keep=self.selected_features_)
            result = inference_pipeline.run()
            print('-----------------------------------------result = {}'.format(result))
            return result

        except Exception as e:
            return f"⚠️ Error during prediction: {e}"

    def _get_interface(self):
        inputs = []
        for feat in self.features:
            if feat in self.config.preprocessing.numerical_features:
                inputs.append(gr.Number(label=feat))
            elif feat in self.config.preprocessing.categorical_features:
                # Get unique categories from the source DataFrame
                categories = self.src_df[feat].dropna().unique().tolist()
                inputs.append(gr.Dropdown(choices=categories, label=feat))
            else:
                inputs.append(gr.Textbox(label=feat,
                                         lines=2,
                                         placeholder="Enter text here, each one separated by a comma"))
        app = gr.Interface(
            fn=lambda *args: self._predict(dict(zip(self.features, args))),
            inputs=inputs,
            outputs="text",
            title=f"Inference App ({self.best_model_name})",
            description="Upload new inputs and see predictions from the trained model."
        )
        return app
