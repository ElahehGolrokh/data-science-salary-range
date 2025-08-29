from omegaconf import OmegaConf

from src.serving import GradioApp


config = OmegaConf.load('private_settings.yaml')

app = GradioApp(config)
demo = app.build()


if __name__ == "__main__":
    demo.launch(share=True)
