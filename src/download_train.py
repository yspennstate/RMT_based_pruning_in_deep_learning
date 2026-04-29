"""Download ImageNet train parquet files in parallel."""
import os
# HF token must be set externally: `export HF_TOKEN=...` (HuggingFace)
if "HF_TOKEN" not in os.environ:
    raise RuntimeError("Set HF_TOKEN env var to download HuggingFace assets.")
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/hf_cache/hub"

from huggingface_hub import HfApi, hf_hub_download, login
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

_token = os.environ.get("HF_TOKEN")
if not _token:
    raise RuntimeError("Set HF_TOKEN env var to download HuggingFace assets.")
login(_token)
api = HfApi()
files = api.list_repo_files("ILSVRC/imagenet-1k", repo_type="dataset")
train_files = sorted([f for f in files if f.endswith(".parquet") and "train" in f.lower().split("/")[-1]])
print(f"Train parquet files: {len(train_files)}", flush=True)

start = time.time()
done = 0

def dl(f):
    return hf_hub_download(
        "ILSVRC/imagenet-1k", f,
        repo_type="dataset",
        cache_dir="/workspace/hf_cache/hub",
    )

with ThreadPoolExecutor(max_workers=8) as ex:
    futures = {ex.submit(dl, f): f for f in train_files}
    for fut in as_completed(futures):
        f = futures[fut]
        try:
            fut.result()
            done += 1
            elapsed = time.time() - start
            rate = done / elapsed * 60
            eta = (len(train_files) - done) / max(rate/60, 0.01)
            print(f"[{done}/{len(train_files)}]  {f}  ({rate:.1f}/min, ETA {eta/60:.1f} min)", flush=True)
        except Exception as e:
            print(f"FAIL: {f} — {e}", flush=True)

print(f"\nDone: {done}/{len(train_files)} files in {(time.time()-start)/60:.1f} min", flush=True)
