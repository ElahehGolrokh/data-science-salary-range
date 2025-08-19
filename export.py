import argparse
from huggingface_hub import login
from omegaconf import OmegaConf
from pathlib import Path

from src.exporting import export_artifacts


parser = argparse.ArgumentParser(
    prog='export.py',
    description='Export model artifacts to Hugging Face Hub',
    epilog=f'Thanks for using.'
)
parser.add_argument("-ri", "--repo_id", type=str, help="Hugging Face repo ID where model is stored")
parser.add_argument("--api_token", type=str, help="Hugging Face API token")

args = parser.parse_args()

config = OmegaConf.load('config.yaml')
REPO_ID = args.repo_id if args.repo_id else config.exporting.repo_id
API_TOKEN = args.api_token if args.api_token else config.exporting.api_token


def main(repo_id: str, api_token: str):
    # ARTIFACTS = [
    #     "artifacts/best_model_name.txt",
    #     "artifacts/final_model.pkl",
    #     "artifacts/final_selected_features.pkl",
    #     "artifacts/scaler.pkl",
    #     "artifacts/mlb.pkl",
    #     "artifacts/one_hot_encoder.pkl"
    # ]
    # Project root (where export.py lives)
    ROOT = Path(__file__).parent

    # Directory containing artifacts (here: root folder)
    # Change ROOT / "artifacts" if you move them into a subdir
    ARTIFACT_DIR = ROOT / "artifacts"

    # Collect artifact files automatically
    ARTIFACTS = [str(p.relative_to(ROOT)) for p in ARTIFACT_DIR.glob("*") if p.is_file()]

    print("📂 Found artifacts:")
    for f in ARTIFACTS:
        print(" -", f)
    login(api_token)
    export_artifacts(artifact_paths=ARTIFACTS, repo_id=repo_id)

    
if __name__ == "__main__":
    main(repo_id=REPO_ID, api_token=API_TOKEN)
