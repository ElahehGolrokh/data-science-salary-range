import os
from huggingface_hub import HfApi, upload_file


def export_artifacts(artifact_paths: str, repo_id: str):
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
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    uploaded_urls = []
    for path in artifact_paths:
        filename = os.path.basename(path)
        print(f"📤 Uploading {filename} → {repo_id}/{filename} ...")

        url = upload_file(
            path_or_fileobj=path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model"
        )
        uploaded_urls.append(url)

    print("✅ All artifacts uploaded successfully.")
    return uploaded_urls
