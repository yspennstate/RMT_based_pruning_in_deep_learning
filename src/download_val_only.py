"""Download ONLY ImageNet validation parquet files from HF — not the full dataset."""
import os
# HF token must be set externally: `export HF_TOKEN=...` (HuggingFace)
if "HF_TOKEN" not in os.environ:
    raise RuntimeError("Set HF_TOKEN env var to download HuggingFace assets.")
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/hf_cache/hub"

from huggingface_hub import HfApi, hf_hub_download, login
_token = os.environ.get("HF_TOKEN")
if not _token:
    raise RuntimeError("Set HF_TOKEN env var to download HuggingFace assets.")
login(_token)

api = HfApi()
files = api.list_repo_files("ILSVRC/imagenet-1k", repo_type="dataset")
val_files = [
    f for f in files
    if f.endswith(".parquet") and ("val" in f.lower().split("/")[-1])
]
print(f"Total repo files: {len(files)}")
print(f"Validation parquet files: {len(val_files)}")
for f in val_files:
    print(" ", f)

print("\nDownloading val files only...", flush=True)
local_paths = []
for f in val_files:
    p = hf_hub_download(
        "ILSVRC/imagenet-1k",
        f,
        repo_type="dataset",
        cache_dir="/workspace/hf_cache/hub",
    )
    local_paths.append(p)
    print(f"  OK: {f} -> {p}", flush=True)

print(f"\nDownloaded {len(local_paths)} validation files.", flush=True)
for p in local_paths:
    sz = os.path.getsize(p) / 1e6
    print(f"  {sz:7.1f} MB  {p}")
