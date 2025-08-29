from omegaconf import OmegaConf

from src.serving import GradioApp


config = OmegaConf.load('config.yaml')


def main():
    app = GradioApp(config)
    app.launch()


if __name__ == "__main__":
    main()