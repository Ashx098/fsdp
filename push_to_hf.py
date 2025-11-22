#!/usr/bin/env python3
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi
from tqdm import tqdm

# -------- CONFIG --------
REPO_ID = "juspay/GLM-4.5-Air-HS"
REPO_PATH = "GLM_FSDP_LoRA"
LOCAL_DIR = "/workspace/MoE Training/fsdp/glm_output"
COMMIT_MESSAGE = "Parallel upload: FSDP LoRA outputs"
MAX_WORKERS = 4

# If you want to skip huge optimizer/model shards, add patterns here:
SKIP_PATTERNS = [
    # "optimizer_0/",
    # "pytorch_model_fsdp_0/",
]
# ------------------------

api = HfApi()

def get_token():
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("Set HUGGINGFACE_HUB_TOKEN")
    return token

def should_skip(rel_path: str):
    return any(pat in rel_path for pat in SKIP_PATTERNS)

def list_all_files(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            local_path = os.path.join(dirpath, f)
            rel_path = os.path.relpath(local_path, root)

            if should_skip(rel_path):
                continue

            repo_dest = f"{REPO_PATH}/{rel_path}"
            size = os.path.getsize(local_path)
            files.append((local_path, repo_dest, size))
    return files

def upload_one(local_path, repo_path, token, retry=2):
    last_err = None
    for attempt in range(retry + 1):
        try:
            api.upload_file(
                path_or_fileobj=local_path,   # ✅ STRICTLY PASS STRING PATH
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model",
                token=token,
                commit_message=COMMIT_MESSAGE
            )
            return True, None
        except Exception as e:
            last_err = e
            time.sleep(2 * attempt + 1)
    return False, last_err

def main():
    token = get_token()

    print(f"📂 Uploading from: {LOCAL_DIR}")
    print(f"📁 To repo path: {REPO_PATH}/")
    print(f"⚡ Parallel workers: {MAX_WORKERS}\n")

    files = list_all_files(LOCAL_DIR)
    total_files = len(files)
    total_bytes = sum(sz for _, _, sz in files)

    print(f"Total files to upload: {total_files}")
    print(f"Total size: {total_bytes/1024**3:.2f} GB\n")

    start = time.time()
    uploaded_bytes = 0

    # Global progress bar in bytes with ETA
    pbar = tqdm(
        total=total_bytes,
        unit="B",
        unit_scale=True,
        desc="Uploading",
        dynamic_ncols=True
    )

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as exe:
        futures = {
            exe.submit(upload_one, local, dest, token): (local, dest, size)
            for local, dest, size in files
        }

        for fut in as_completed(futures):
            local, dest, size = futures[fut]
            ok, err = fut.result()

            if ok:
                uploaded_bytes += size
                pbar.update(size)
                pbar.set_postfix_str(f"{uploaded_bytes/total_bytes*100:.1f}%")
                print(f"✔ Uploaded: {local}  →  {dest}")
            else:
                print(f"❌ Failed: {local}")
                print("Error:", err)

    pbar.close()

    mins = (time.time() - start) / 60
    print("\n===============================")
    print("🎉 Parallel Upload Completed")
    print(f"⏱ Time: {mins:.2f} mins")
    print("===============================")

if __name__ == "__main__":
    main()
