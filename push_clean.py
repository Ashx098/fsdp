#!/usr/bin/env python3
import os
import time
from huggingface_hub import HfApi
from tqdm import tqdm

# -------- CONFIG --------
REPO_ID = "juspay/GLM-4.5-Air-HS"
BASE_REMOTE_FOLDER = "GLM_FSDP_LoRA"
LOCAL_DIR = "/workspace/MoE Training/fsdp/glm_output"
COMMIT_MESSAGE = "Upload cleaned FSDP LoRA checkpoints + logs"
# ------------------------

# Patterns to include (whitelist)
INCLUDE_EXT = [
    "adapter_model.safetensors",
    "adapter_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "metadata.json",
    "README.md",
]

# logs
ALWAYS_INCLUDE = [
    "train_log.jsonl",
    "eval_log.jsonl",
]

# NEVER upload FSDP shards
SKIP_DIRS = [
    "optimizer_0",
    "pytorch_model_fsdp_0",
]

api = HfApi()


def get_token():
    tok = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not tok:
        raise RuntimeError("Missing HF token. Export HUGGINGFACE_HUB_TOKEN")
    return tok


def should_upload(fname):
    # Check extension whitelist
    if fname in INCLUDE_EXT:
        return True
    # logs
    if fname in ALWAYS_INCLUDE:
        return True
    return False


def walk_clean_files(root):
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip unwanted folders
        if any(skip in dirpath for skip in SKIP_DIRS):
            continue

        for f in filenames:
            if should_upload(f):
                local_path = os.path.join(dirpath, f)
                rel = os.path.relpath(local_path, root)
                repo_path = f"{BASE_REMOTE_FOLDER}/{rel}"
                files.append((local_path, repo_path, os.path.getsize(local_path)))
    return files


def upload_file(local_path, repo_path, token):
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=REPO_ID,
        token=token,
        repo_type="model",
        commit_message=COMMIT_MESSAGE,
    )
    return True


def main():
    token = get_token()
    print(f"📂 Cleaning directory: {LOCAL_DIR}")
    print(f"📁 Uploading into HF folder: {BASE_REMOTE_FOLDER}/")

    # Collect clean files
    files = walk_clean_files(LOCAL_DIR)
    total_size = sum(sz for _, _, sz in files)

    print(f"\nTotal files selected: {len(files)}")
    print(f"Total upload size: {total_size/1024**3:.2f} GB\n")

    pbar = tqdm(total=total_size, unit="B", unit_scale=True, desc="Uploading")

    for local, remote, sz in files:
        upload_file(local, remote, token)
        pbar.update(sz)
        print(f"✔ Uploaded: {remote}")

    pbar.close()

    print("\n===============================")
    print("🎉 Clean LoRA Upload Completed")
    print("===============================")


if __name__ == "__main__":
    main()
