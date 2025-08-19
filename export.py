import argparse
from huggingface_hub import login
from omegaconf import OmegaConf
from src.exporting import export_artifacts


parser = argparse.ArgumentParser(
    prog='export.py',
    description='Export model artifacts to Hugging Face Hub',
    epilog=f'Thanks for using.'
)
parser.add_argument("-ri", "--repo_id", type=str, help="Hugging Face repo ID where model is stored")
parser.add_argument("--local_artifacts_path", type=str, help="Path to the local artifacts directory")
parser.add_argument("--api_token", type=str, help="Hugging Face API token")

args = parser.parse_args()

config = OmegaConf.load('config.yaml')
REPO_ID = args.repo_id if args.repo_id else config.exporting.repo_id
LOCAL_ARTIFACTS_PATH = args.local_artifacts_path if args.local_artifacts_path else config.paths.artifacts_dir
API_TOKEN = args.api_token if args.api_token else config.exporting.api_token


def main(repo_id: str, api_token: str, local_artifacts_path: str):
    ARTIFACTS = [
        "artifacts/best_model_name.txt",
        "artifacts/final_model.pkl",
        "artifacts/final_selected_features.pkl",
        "artifacts/scaler.pkl",
        "artifacts/mlb.pkl",
        "artifacts/one_hot_encoder.pkl"
    ]
    # LOCAL_MODEL_PATH = "artifacts/final_model.pkl"   # path to your trained model file
    login(api_token)
    urls = export_artifacts(artifact_paths=ARTIFACTS, repo_id=repo_id)
    print(urls)

    
if __name__ == "__main__":
    main(repo_id=REPO_ID, api_token=API_TOKEN, local_artifacts_path=LOCAL_ARTIFACTS_PATH)
