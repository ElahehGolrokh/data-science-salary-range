import joblib
import json
import gradio as gr
from huggingface_hub import hf_hub_download
import numpy as np
from omegaconf import OmegaConf

# ----------------------------
# Repo ID (replace with yours)
# ----------------------------
config = OmegaConf.load('private_settings.yaml')
REPO_ID = config.repo_id

numerical_features = config.numerical_features
# ----------------------------
# Download artifacts
# ----------------------------
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
        path = hf_hub_download(repo_id=REPO_ID, filename=filename)
        downloaded[key] = path
    except Exception as e:
        print(f"⚠️ Could not download {filename}: {e}")

# ----------------------------
# Load artifacts
# ----------------------------
model = joblib.load(downloaded["model"])
scaler = joblib.load(downloaded["scaler"])
mlb = joblib.load(downloaded["mlb"])
one_hot_encoder = joblib.load(downloaded["one_hot_encoder"])
selected_features = joblib.load(downloaded["features"])

with open(downloaded["best_model_name"], "r") as f:
    best_model_name = f.read().strip()

print(f"✅ Loaded model: {best_model_name}")

# ----------------------------
# Inference Function
# ----------------------------
def predict(user_inputs):
    """
    user_inputs: dict of feature_name -> value
    """
    try:
        # Build input vector in correct feature order
        X = []
        for feat in selected_features:
            X.append(user_inputs.get(feat, 0))  # default 0 if missing
            if feat in numerical_features:
                # scaler issue
                pass

        X = np.array(X).reshape(1, -1)
        # Predict
        print('predincting...')
        y_pred = model.predict(X)

        return f"Prediction: {y_pred[0]}"

    except Exception as e:
        return f"⚠️ Error during prediction: {e}"

# ----------------------------
# Gradio UI
# ----------------------------
# Build dynamic input fields (example: numeric sliders)
# You may adjust depending on your features
inputs = []
for feat in selected_features:
    inputs.append(gr.Number(label=feat))

demo = gr.Interface(
    fn=lambda *args: predict(dict(zip(selected_features, args))),
    inputs=inputs,
    outputs="text",
    title=f"Inference App ({best_model_name})",
    description="Upload new inputs and see predictions from the trained model."
)

if __name__ == "__main__":
    demo.launch(share=True)