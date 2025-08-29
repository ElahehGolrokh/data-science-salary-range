from omegaconf import OmegaConf

from src.serving import GradioApp


config = OmegaConf.load('config.yaml')

app = GradioApp(config)
demo = app.build()


if __name__ == "__main__":
    demo.launch(share=True)
