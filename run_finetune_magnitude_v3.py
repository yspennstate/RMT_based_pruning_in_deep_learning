"""
Magnitude-only baseline — iterative pruning + fine-tuning pipeline for ViT-B/16.

Same cumulative sparsity schedule (5%, 10%, ..., 70%) and fine-tune hyperparameters
as run_finetune_pipeline.py, but replaces the regime-adapted sigma+-budget + Haar
SV preprocessing with plain global magnitude pruning. This gives the head-to-head
magnitude-vs-RMT comparison under identical fine-tuning conditions.

Results land in a separate directory so the two pipelines can run in parallel on
the same volume without colliding:
    /workspace/finetune_results_magnitude/
"""
import os
# HF token must be set externally: `export HF_TOKEN=...` (HuggingFace)
if "HF_TOKEN" not in os.environ:
    raise RuntimeError("Set HF_TOKEN env var to download HuggingFace assets.")
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/hf_cache/hub"
os.environ.setdefault("OMP_NUM_THREADS", "8")

import sys, json, time, math, glob, datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "src"))
from theory_pruning import iter_target_layers_modules  # noqa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("/workspace/finetune_results_magnitude_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)
RESULTS_FILE = OUT_DIR / "finetune_results.json"
LOG_FILE = OUT_DIR / "finetune_log.txt"


def log(msg):
    ts = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── Cycle configuration (same epochs/LR as RMT pipeline; method = magnitude only)
CYCLES = [
    # v1 phase (mirrors RMT pipeline cycles 1-8): short FT, warmup 500, LR floor 1%%
    (0.05, 1, 3.00e-5,  500, 0.01),
    (0.10, 1, 3.00e-5,  500, 0.01),
    (0.15, 1, 3.00e-5,  500, 0.01),
    (0.20, 1, 3.00e-5,  500, 0.01),
    (0.25, 1, 3.00e-5,  500, 0.01),
    (0.30, 2, 3.00e-5,  500, 0.01),
    (0.35, 2, 3.00e-5,  500, 0.01),
    (0.40, 2, 3.00e-5,  500, 0.01),
    # v2 phase (mirrors RMT pipeline v2): long FT, warmup 5000, LR floor 5%%
    (0.45,  4, 2.5e-5, 5000, 0.05),
    (0.50,  5, 2.2e-5, 5000, 0.05),
    (0.55,  6, 2.0e-5, 5000, 0.05),
    (0.60,  8, 1.8e-5, 5000, 0.05),
    (0.65, 10, 1.6e-5, 5000, 0.05),
    (0.70, 12, 1.4e-5, 5000, 0.05),
]

BATCH_SIZE_TRAIN = 128
BATCH_SIZE_VAL   = 256
NUM_WORKERS      = 8
LABEL_SMOOTHING  = 0.1
GRAD_CLIP        = 1.0
WEIGHT_DECAY     = 1e-6
WARMUP_STEPS     = 500
MIN_LR_FRACTION  = 0.01
SEED             = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class HFParquetDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, transform):
        self.dataset = hf_ds
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_loaders(preprocess_train, preprocess_val):
    from datasets import load_dataset
    train_files = sorted(glob.glob(
        "/workspace/hf_cache/hub/datasets--ILSVRC--imagenet-1k/snapshots/*/data/train-*.parquet"
    ))
    val_files = sorted(glob.glob(
        "/workspace/hf_cache/hub/datasets--ILSVRC--imagenet-1k/snapshots/*/data/validation-*.parquet"
    ))
    log(f"Train parquet files: {len(train_files)}, val parquet files: {len(val_files)}")
    train_hf = load_dataset("parquet", data_files=train_files, split="train")
    val_hf   = load_dataset("parquet", data_files=val_files,   split="train")

    # FULL 50K validation set (matches official top-1 reporting; baseline = 85.106%).
    train_ds = HFParquetDataset(train_hf, preprocess_train)
    val_ds   = HFParquetDataset(val_hf, preprocess_val)
    log(f"Train dataset: {len(train_ds)}, val subset: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE_TRAIN, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
        drop_last=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE_VAL, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True,
    )
    return train_loader, val_loader


# ── Pure global magnitude pruning ────────────────────────────────────────────

def apply_global_magnitude(model, target_sparsity):
    """Classic global magnitude pruning: pick the smallest |W| across all target
    layers and zero the bottom `target_sparsity` fraction."""
    all_scores = []
    layer_meta = []
    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        score = np.abs(W2)
        layer_meta.append((name, mod, W.shape, W2, score))
        all_scores.append(score.ravel())
    flat = np.concatenate(all_scores)
    n_total = flat.size
    n_zero = int(n_total * target_sparsity)
    if n_zero > 0 and n_zero < n_total:
        threshold = np.partition(flat, n_zero)[n_zero]
    elif n_zero >= n_total:
        threshold = flat.max() + 1
    else:
        threshold = -1.0
    total_zeroed = 0
    total_params = 0
    for name, mod, shape, W2, score in layer_meta:
        mask = (score >= threshold).astype(np.float32)
        W_new = W2 * mask
        total_zeroed += int((mask == 0).sum())
        total_params += mask.size
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(shape)).float())
    return total_zeroed / total_params if total_params > 0 else 0.0


def build_mask(model):
    masks = {}
    for name, mod in iter_target_layers_modules(model):
        with torch.no_grad():
            masks[name] = (mod.weight != 0).to(DEVICE)
    return masks


def apply_mask(model, masks):
    with torch.no_grad():
        for name, mod in iter_target_layers_modules(model):
            if name in masks:
                mod.weight.mul_(masks[name])


@torch.no_grad()
def evaluate_model(model, val_loader, label=""):
    model.eval()
    correct = 0
    total = 0
    t0 = time.time()
    for x, y in val_loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    top1 = 100.0 * correct / total
    log(f"  eval {label}: top1={top1:.2f}%  ({total} imgs, {time.time()-t0:.1f}s)")
    return top1


def finetune_one_cycle(model, masks, train_loader, val_loader, epochs, base_lr, cycle_idx, warmup_steps=None, min_lr_fraction=None):
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    total_steps = epochs * len(train_loader)
    warmup = min(warmup_steps if warmup_steps is not None else WARMUP_STEPS, total_steps // 4)
    min_lr = base_lr * (min_lr_fraction if min_lr_fraction is not None else MIN_LR_FRACTION)

    def lr_at(step):
        if step < warmup:
            return min_lr + (base_lr - min_lr) * step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

    step = 0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t0 = time.time()
        for i, (x, y) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            lr = lr_at(step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(x)
                loss = criterion(out, y)
            loss.backward()
            for name, mod in iter_target_layers_modules(model):
                if name in masks and mod.weight.grad is not None:
                    mod.weight.grad.mul_(masks[name])
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            apply_mask(model, masks)

            with torch.no_grad():
                pred = out.argmax(dim=1)
                epoch_correct += (pred == y).sum().item()
                epoch_total += y.size(0)
                epoch_loss += loss.item() * y.size(0)

            step += 1
            if (i + 1) % 100 == 0:
                log(f"  cycle {cycle_idx} ep {epoch+1}/{epochs} step {i+1}/{len(train_loader)} "
                    f"loss={epoch_loss/epoch_total:.4f}  acc={100.0*epoch_correct/epoch_total:.2f}%  lr={lr:.2e}")
        log(f"  cycle {cycle_idx} epoch {epoch+1}/{epochs} done in {(time.time()-t0)/60:.1f} min "
            f"train_loss={epoch_loss/epoch_total:.4f}  train_acc={100.0*epoch_correct/epoch_total:.2f}%")
    return model


def load_results():
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return []


def save_results(results):
    tmp = RESULTS_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    tmp.replace(RESULTS_FILE)


def ckpt_path(cycle_idx):
    return CKPT_DIR / f"cycle_{cycle_idx:02d}_s{int(CYCLES[cycle_idx][0]*100):02d}.pt"


def save_checkpoint(model, cycle_idx, masks):
    path = ckpt_path(cycle_idx)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "cycle_idx": cycle_idx,
        "model_state_dict": model.state_dict(),
        "masks": {k: v.cpu() for k, v in masks.items()},
    }, tmp)
    tmp.replace(path)
    log(f"  checkpoint saved: {path.name}")


def load_checkpoint(model, cycle_idx):
    path = ckpt_path(cycle_idx)
    if not path.exists():
        return None
    data = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(data["model_state_dict"])
    masks = {k: v.to(DEVICE) for k, v in data["masks"].items()}
    return masks


def main():
    log("=" * 70)
    log("Magnitude-only baseline: iterative pruning + fine-tuning")
    log(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    log(f"Results: {RESULTS_FILE}")
    log(f"Checkpoints: {CKPT_DIR}")
    log("=" * 70)

    log("loading ViT-B/16 (pretrained)...")
    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    preprocess_val = timm.data.create_transform(**data_config, is_training=False)
    preprocess_train = timm.data.create_transform(**data_config, is_training=True)
    model.to(DEVICE)

    log("building loaders...")
    train_loader, val_loader = build_loaders(preprocess_train, preprocess_val)

    results = load_results()
    if not results:
        baseline_top1 = evaluate_model(model, val_loader, label="baseline")
        results.append({"cycle_idx": -1, "target_s": 0.0, "method": "baseline",
                        "epochs": 0, "lr": 0.0,
                        "pre_ft_top1": baseline_top1, "post_ft_top1": baseline_top1,
                        "achieved_s": 0.0,
                        "ts": datetime.datetime.utcnow().isoformat()})
        save_results(results)

    start_cycle = 0
    global_masks = None
    for c in range(len(CYCLES) - 1, -1, -1):
        masks_loaded = load_checkpoint(model, c)
        if masks_loaded is not None:
            start_cycle = c + 1
            model.to(DEVICE)
            log(f"Resumed from checkpoint cycle {c} (next: cycle {start_cycle})")
            global_masks = masks_loaded
            break

    pipeline_start = time.time()
    for cycle_idx in range(start_cycle, len(CYCLES)):
        target_s, epochs, base_lr, warmup_steps_c, min_lr_fraction_c = CYCLES[cycle_idx]
        t0 = time.time()
        log("-" * 70)
        log(f"Cycle {cycle_idx+1}/{len(CYCLES)}: target_s={target_s:.2f}  "
            f"epochs={epochs}  lr={base_lr:.2e}  method=magnitude")

        model.cpu()
        achieved = apply_global_magnitude(model, target_s)
        model.to(DEVICE)
        log(f"  achieved sparsity: {achieved:.4f}")
        global_masks = build_mask(model)

        pre_ft_top1 = evaluate_model(model, val_loader, label=f"pre-FT s={target_s:.2f}")

        model = finetune_one_cycle(model, global_masks, train_loader, val_loader,
                                   epochs, base_lr, cycle_idx + 1,
                                   warmup_steps=warmup_steps_c, min_lr_fraction=min_lr_fraction_c)

        post_ft_top1 = evaluate_model(model, val_loader, label=f"post-FT s={target_s:.2f}")

        save_checkpoint(model, cycle_idx, global_masks)
        results.append({
            "cycle_idx": cycle_idx,
            "target_s": target_s,
            "achieved_s": achieved,
            "method": "magnitude",
            "epochs": epochs,
            "lr": base_lr,
            "pre_ft_top1": pre_ft_top1,
            "post_ft_top1": post_ft_top1,
            "cycle_elapsed_min": (time.time() - t0) / 60,
            "ts": datetime.datetime.utcnow().isoformat(),
        })
        save_results(results)
        log(f"  cycle {cycle_idx+1} summary: pre-FT={pre_ft_top1:.2f}%  post-FT={post_ft_top1:.2f}%  "
            f"gain={post_ft_top1-pre_ft_top1:+.2f}pp  elapsed={(time.time()-t0)/60:.1f}min")

    log("=" * 70)
    log(f"PIPELINE DONE. Total time: {(time.time()-pipeline_start)/60:.1f} min")
    log("=" * 70)


if __name__ == "__main__":
    main()
