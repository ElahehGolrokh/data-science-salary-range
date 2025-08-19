import os
from huggingface_hub import HfApi, upload_file, login

from pathlib import Path

from .utils import get_root


class Exporter:
    """
    Handles the export of model artifacts to the Hugging Face Hub.

    This class manages authentication, artifact discovery, and uploading 
    of model-related files (e.g., model weights, scalers, encoders) 
    to a specified Hugging Face model repository.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing paths and other settings.
    repo_id : str
        Hugging Face repository identifier in the format ``"username/repo-name"``
        where artifacts are uploaded.
    api_token : str
        Hugging Face API token used to authenticate the upload process.
    artifacts_dir : str
        Local directory path containing the model artifacts to be exported.
    """
    def __init__(self,
                 config: dict,
                 repo_id: str,
                 api_token: str):
        self.repo_id = repo_id
        self.artifacts_dir = Path(config.paths.artifacts_dir)
        login(api_token)

    def _read_artifacts(self) -> list:
        """
        Read artifact files from the local directory.

        Returns:
            list[str]: List of artifact file paths.
        """
        # Project root (where export.py lives)
        ROOT = get_root()
        print(f'root = {ROOT}')
        self.artifacts_dir = ROOT / self.artifacts_dir

        # Collect artifact files automatically
        artifacts = [str(p.relative_to(ROOT)) for p in self.artifacts_dir.glob("*") if p.is_file()]

        print("📂 Found artifacts:")
        for f in artifacts:
            print(" -", f)
        return artifacts

    def export(self):
        """
        Upload multiple artifacts (model, scalers, encoders, etc.) to a Hugging Face model repo.

        Args:
        artifact_paths (list[str]): Local file paths to upload.
        repo_id (str): Hugging Face repo in format "username/repo-name".

        Returns:
            list[str]: URLs of uploaded files.
        """
        api = HfApi()
        # ✅ Ensure repo exists (will not error if already created)
        api.create_repo(repo_id=self.repo_id, repo_type="model", exist_ok=True)

        artifacts = self._read_artifacts()
        uploaded_urls = []
        for path in artifacts:
            filename = os.path.basename(path)
            print(f"📤 Uploading {filename} → {self.repo_id}/{filename} ...")

            url = upload_file(
                path_or_fileobj=path,
                path_in_repo=filename,
                repo_id=self.repo_id,
                repo_type="model"
            )
            uploaded_urls.append(url)

        print("✅ All artifacts uploaded successfully.")
        return uploaded_urls
