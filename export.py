import argparse
from omegaconf import OmegaConf

from src.exporting import Exporter


parser = argparse.ArgumentParser(
    prog='export.py',
    description='Export model artifacts to Hugging Face Hub',
    epilog=f'Thanks for using.'
)

parser.add_argument("-ri", "--repo_id", type=str, help="Hugging Face repo ID where model is stored")
parser.add_argument("--api_token", type=str, help="Hugging Face API token")

args = parser.parse_args()

config = OmegaConf.load('private_settings.yaml')
REPO_ID = args.repo_id if args.repo_id else config.exporting.repo_id
API_TOKEN = args.api_token if args.api_token else config.exporting.api_token


def main(repo_id: str, api_token: str):
    exporter = Exporter(config=config, repo_id=repo_id, api_token=api_token)
    exporter.export()


if __name__ == "__main__":
    main(repo_id=REPO_ID, api_token=API_TOKEN)
