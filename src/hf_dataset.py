import os
from huggingface_hub import HfApi, snapshot_download

def push_dataset(config: dict):
    """
    Uploads the local dataset folder to a Hugging Face Hub dataset repository.
    Make sure you have logged in using `huggingface-cli login` first.
    """
    base_path = config["data"]["base_path"]
    repo_id = config["data"].get("hf_repo_id")
    
    if not repo_id:
        print("ERROR: hf_repo_id is not set in config.yaml under the 'data' section.")
        print("Please add 'hf_repo_id: your_username/your_dataset_name' to config.")
        return

    if not os.path.exists(base_path):
        print(f"ERROR: Dataset path '{base_path}' doesn't exist.")
        return
    
    api = HfApi()
    print(f"Uploading '{base_path}' to Hugging Face Dataset '{repo_id}'...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        # Ignore errors if repo exists, we will catch permissions errors later during push
        pass
        
    try:
        api.upload_large_folder(
            folder_path=base_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print("Upload complete! You can now safely delete the heavy files in the dataset folder on your computer.")
    except Exception as e:
        print(f"ERROR: Could not upload to {repo_id}.")
        print("Please ensure you are logged in by running `huggingface-cli login` in the terminal.")
        print(f"Details: {e}")

def pull_dataset(config: dict):
    """
    Downloads the dataset from Hugging Face Hub into the local dataset folder.
    """
    base_path = config["data"]["base_path"]
    repo_id = config["data"].get("hf_repo_id")
    
    if not repo_id:
        print("ERROR: hf_repo_id is not set in config.yaml under the 'data' section.")
        return

    print(f"Downloading dataset from '{repo_id}' to '{base_path}'...")
    os.makedirs(base_path, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=base_path,
            local_dir_use_symlinks=False
        )
        print("Download complete!")
    except Exception as e:
        print(f"ERROR: Could not download the dataset {repo_id}.")
        print("Is it private? If so, run `huggingface-cli login` first.")
        print(f"Details: {e}")
