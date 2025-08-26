import gradio as gr
from huggingface_hub import hf_hub_download
import numpy as np
from omegaconf import OmegaConf
import pandas as pd

from .utils import load_object, load_dataframe
from .inference import InferencePipeline
from .preprocessing import Preprocessor


class GradioApp:
    """
    Gradio-based application for model inference.

    This class wraps preprocessing, model loading, and prediction logic
    into a Gradio interface for interactive inference.

    Parameters
    ----------
    config : OmegaConf
        Configuration object with paths, preprocessing, and model settings.

    Attributes
    ----------
    repo_id : str
        Hugging Face Hub repository ID for downloading artifacts.
    features : list of str
        Model features excluding dropped columns and target column(s).
    src_df : pandas.DataFrame
        Source dataset (training data) used for categories and preprocessing.
    model_ : estimator
        Trained model loaded from Hugging Face Hub.
    scaler_ : object
        Fitted scaler loaded from Hugging Face Hub.
    mlb_ : object
        MultiLabelBinarizer loaded from Hugging Face Hub.
    one_hot_encoder_ : object
        OneHotEncoder loaded from Hugging Face Hub.
    selected_features_ : list of str
        Final set of features used by the trained model.
    best_model_name : str
        Name of the best model loaded from Hugging Face Hub.

    Examples
    --------
    >>> from omegaconf import OmegaConf
    >>> from src.gradio_app import GradioApp
    >>> config = OmegaConf.load("config.yaml")
    >>> app = GradioApp(config)
    >>> app.launch()  # Starts a Gradio server for interactive inference
    """
    def __init__(self, config: OmegaConf):
        self.config = config
        self.repo_id = config.repo_id
        self.features = [feat for feat in config.preprocessing.all_features
                        if feat not in config.preprocessing.columns_to_drop
                        and feat not in config.preprocessing.target]
        self.src_df = load_dataframe(config.files.train_data,
                                     config.dirs.data)

    def launch(self) -> None:
        """
        Launch the Gradio application.

        Loads artifacts, builds the Gradio interface,
        and starts the app server.

        Notes
        -----
        This method blocks execution until the server is stopped.
        """
        self._load_artifacts()
        app = self._get_interface()
        app.launch(share=True)

    def _download_artifacts(self) -> dict:
        """
        Download required artifacts from Hugging Face Hub.

        Returns
        -------
        dict
            Dictionary mapping artifact keys (e.g. "model", "scaler") to
            local file paths where artifacts are cached.
        """
        artifacts = {
            "model": self.config.files.final_model,
            "scaler": self.config.files.scaler,
            "mlb": self.config.files.mlb,
            "one_hot_encoder": self.config.files.one_hot_encoder,
            "features": self.config.files.selected_features,   # which features to use
            "best_model_name": self.config.files.best_model_name
        }
        downloaded = {}
        for key, filename in artifacts.items():
            try:
                path = hf_hub_download(repo_id=self.repo_id, filename=filename)
                downloaded[key] = path
            except Exception as e:
                raise ValueError(f"⚠️ Could not download {filename}: {e}")
        return downloaded

    def _load_artifacts(self) -> None:
        """
        Load model and preprocessing artifacts into memory.

        Artifacts include the model, scaler, multi-label binarizer, encoders,
        selected features, and best model name.
        """
        downloaded = self._download_artifacts()
        self.model_ = load_object(downloaded["model"])
        self.scaler_ = load_object(downloaded["scaler"])
        self.mlb_ = load_object(downloaded["mlb"])
        self.one_hot_encoder_ = load_object(downloaded["one_hot_encoder"])
        self.selected_features_ = load_object(downloaded["features"])

        with open(downloaded["best_model_name"], "r") as f:
            self.best_model_name = f.read().strip()

        print(f"✅ Loaded model: {self.best_model_name}")

    def _get_user_inputs(self, user_inputs: dict) -> list:
        """
        Construct input vector in the correct feature order.

        Parameters
        ----------
        user_inputs : dict
            Mapping from feature names to user-provided values.

        Returns
        -------
        list
            Ordered list of feature values aligned with `self.features`.
        """
        try:
            X = []
            for feat in self.features:
                X.append(user_inputs.get(feat, None))  # default None if missing
            return X
        except Exception as e:
            print(f"⚠️ Error during input vector construction: {e}")
            return None

    def _prepare_inputs(self, user_inputs: dict) -> pd.DataFrame:
        """
        Prepare input features for model prediction.

        Parameters
        ----------
        user_inputs : dict
            Mapping from feature names to user-provided values.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the prepared input features.
        """
        X = self._get_user_inputs(user_inputs)
        if X is None:
            raise ValueError("⚠️ Error during input vector construction")
        try:
            X = np.array(X).reshape(1, -1)
            X = pd.DataFrame(X, columns=self.features)
            print('X = {}'.format(X))
            preprocessor = Preprocessor(self.config,
                                        save_flag=False,
                                        one_hot_encoder_=self.one_hot_encoder_,
                                        mlb_=self.mlb_,
                                        scaler_=self.scaler_)
            preprocessed_df = preprocessor.run(input_df=X,
                                               src_df=self.src_df,
                                               phase='inference',
                                               transform_target=False)
            return preprocessed_df
        except Exception as e:
            raise ValueError(f"⚠️ Error during preprocessing: {e}")

    def _predict(self, user_inputs: dict) -> str:
        """
        Run model inference.

        Parameters
        ----------
        user_inputs : dict
            Mapping from feature names to user-provided values.

        Returns
        -------
        str
            Prediction result or error message.
        """
        preprocessed_df = self._prepare_inputs(user_inputs)
        try:
            inference_pipeline = InferencePipeline(self.config,
                                                   self.model_,
                                                   input_df=preprocessed_df,
                                                   src_df=self.src_df,
                                                   columns_to_keep=self.selected_features_)
            result = inference_pipeline.run()
            print(f"Raw prediction result: {result}")
            try:
                # Round to nearest 1,000
                rounded = int(round(result[0], -3))  
                formatted = f"€ {int(rounded):,}"
                print(f"Formatted prediction result: {formatted}")
                return f"Estimated annual salary: {formatted}"
            except Exception:
                return f"Estimated annual salary: € {result}"

        except Exception as e:
            raise ValueError(f"⚠️ Error during prediction: {e}")

    def _get_interface(self) -> gr.Interface:
        """
        Build the Gradio interface for interactive inference.

        Returns
        -------
        gr.Interface
            Configured Gradio interface object with feature inputs and
            prediction output.
        """
        inputs = []
        for feat in self.features:
            if feat in self.config.preprocessing.numerical_features:
                inputs.append(gr.Number(label=feat))
            elif feat in self.config.preprocessing.categorical_features or \
                feat in self.config.preprocessing.ordinal_features:
                # Get unique categories from the source DataFrame
                categories = self.src_df[feat].dropna().unique().tolist()
                # remove Remote from locations
                if 'Remote' in categories:
                    categories.remove('Remote')
                inputs.append(gr.Dropdown(choices=categories, label=feat))
            else:  # skills
                inputs.append(gr.Textbox(label=feat,
                                         lines=2,
                                         placeholder="Enter text here, each one separated by a comma"))
        app = gr.Interface(
            fn=lambda *args: self._predict(dict(zip(self.features, args))),
            inputs=inputs,
            outputs=gr.Textbox(label="Predicted Salary (Annual, in €)"),
            title=f"Data Science Salary Predictor",
            description="This app uses machine learning to estimate salaries for data science roles. \
                         Try it out with your own profile!"
        )
        return app
