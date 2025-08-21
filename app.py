import joblib
import gradio as gr
from huggingface_hub import hf_hub_download
import numpy as np
from omegaconf import OmegaConf

from src.serving import GradioApp


config = OmegaConf.load('private_settings.yaml')


def main():
    app = GradioApp(config)
    app.launch()

# ----------------------------
# Download artifacts
# ----------------------------

# ----------------------------
# Load artifacts
# ----------------------------

# ----------------------------
# Inference Function
# ----------------------------

# ----------------------------
# Gradio UI
# ----------------------------
# Build dynamic input fields (example: numeric sliders)
# You may adjust depending on your features


if __name__ == "__main__":
    main()