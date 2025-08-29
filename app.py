from omegaconf import OmegaConf
import os

from src.serving import GradioApp


base_conf = OmegaConf.load('config.yaml')
private_conf = OmegaConf.create({
    "repo_id": os.getenv("REPO_ID"),
    "hf_token": os.getenv("HF_TOKEN")
})
config = OmegaConf.merge(base_conf, private_conf)

app = GradioApp(config)
demo = app.build()


if __name__ == "__main__":
    demo.launch()
