"""
Iterative pruning + fine-tuning pipeline for ViT-B/16 on ImageNet.

For each cumulative sparsity target, this script:
  1. Applies the regime-adapted sigma+-budget pruning criterion (plus Haar SV
     preprocessing at cycle 1) to reach the new target sparsity. Weights
     previously zeroed by earlier cycles stay zeroed because magnitude
     pruning selects the smallest-magnitude weights first.
  2. Evaluates the pruned-but-not-yet-fine-tuned model on the 10K val subset.
  3. Fine-tunes N epochs on ImageNet train with mixed precision, SGD+cosine,
     gradient clipping, and a binary mask re-applied after each optimizer step.
  4. Evaluates post-fine-tune accuracy on the same val subset.
  5. Saves a checkpoint and appends a line to the results JSON.

Resume: if the checkpoint for cycle N exists, the script loads it and skips
to cycle N+1. No work is ever lost.
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
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

# Import the existing sweep utilities
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "src"))
from magnitude_rmt_sweep import (  # noqa
    compute_layer_signals,
    layer_weights_from_signal,
    apply_modulated_magnitude,
    sv_prune_haar,
    parse_method_name,
)
from theory_pruning import iter_target_layers_modules  # noqa

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path("/workspace/finetune_results_magnitude_v4")
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


# ── Cycle configuration ──────────────────────────────────────────────────────
# (target_sparsity, epochs, base_lr, method_str, apply_sv)
CYCLES = [
    # v1 phase - short FT
    (0.05, 1, 3.00e-5, "magnitude", False),
    (0.10, 1, 3.00e-5, "magnitude", False),
    (0.15, 1, 3.00e-5, "magnitude", False),
    (0.20, 1, 3.00e-5, "magnitude", False),
    (0.25, 1, 3.00e-5, "magnitude", False),
    (0.30, 2, 3.00e-5, "magnitude", False),
    (0.35, 2, 3.00e-5, "magnitude", False),
    (0.40, 2, 3.00e-5, "magnitude", False),
    # v2 phase - long FT
    (0.45,  4, 2.5e-5, "magnitude", False),
    (0.50,  5, 2.2e-5, "magnitude", False),
    (0.55,  6, 2.0e-5, "magnitude", False),
    (0.60,  8, 1.8e-5, "magnitude", False),
    (0.65, 10, 1.6e-5, "magnitude", False),
    (0.70, 12, 1.4e-5, "magnitude", False),
]

BATCH_SIZE_TRAIN = 128
BATCH_SIZE_VAL   = 256
NUM_WORKERS      = 8
LABEL_SMOOTHING  = 0.1
GRAD_CLIP        = 1.0
WEIGHT_DECAY     = 1e-6
WARMUP_STEPS     = 5000
MIN_LR_FRACTION  = 0.05  # final LR = base_lr * this
SEED             = 42

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ── Dataset: HF parquet-backed, train + val ─────────────────────────────────

class HFParquetDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace Dataset (from local parquet files) with a transform."""
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
    assert len(train_files) > 0, "Train parquet files not found. Run the download step first."
    assert len(val_files) > 0, "Val parquets not found"

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


# ── Pruning operations ──────────────────────────────────────────────────────

def apply_sv_preprocessing(model, signals):
    """Apply Haar SV sparsification (z=0.5, p=3) to all target layers in-place.
    Runs only at cycle 1 to denoise the baseline before iterative pruning begins.
    """
    alphas = [signals[n].get("alpha") for n in signals if signals[n].get("alpha") is not None]
    alpha_mean = float(np.mean(alphas)) if alphas else None

    for name, mod in iter_target_layers_modules(model):
        W = mod.weight.detach().cpu().numpy()
        W2 = W.reshape(W.shape[0], -1) if W.ndim == 4 else W
        splus = signals[name].get("splus")
        alpha = signals[name].get("alpha")
        if splus is None:
            continue
        W_new = sv_prune_haar(W2, splus, z_base=0.5, alpha=alpha, alpha_mean=alpha_mean, power=3)
        with torch.no_grad():
            mod.weight.copy_(torch.from_numpy(W_new.reshape(W.shape)).float())


def prune_to_target(model, target_sparsity, method_str, apply_sv, signals):
    """Pure global magnitude pruning. method_str/apply_sv/signals are ignored."""
    model.cpu()
    all_abs = []
    for name, mod in iter_target_layers_modules(model):
        all_abs.append(mod.weight.detach().abs().flatten())
    flat = torch.cat(all_abs)
    k = int(target_sparsity * flat.numel())
    if k <= 0:
        model.to(DEVICE)
        return 0.0
    threshold = torch.kthvalue(flat, k).values.item()
    zeroed = 0
    total = 0
    with torch.no_grad():
        for name, mod in iter_target_layers_modules(model):
            keep = mod.weight.abs() > threshold
            mod.weight.mul_(keep.to(mod.weight.dtype))
            zeroed += (~keep).sum().item()
            total += keep.numel()
    model.to(DEVICE)
    return zeroed / total


def build_mask(model):
    """Return {layer_name: bool tensor} marking non-zero weights on GPU."""
    masks = {}
    for name, mod in iter_target_layers_modules(model):
        with torch.no_grad():
            masks[name] = (mod.weight != 0).to(DEVICE)
    return masks


def apply_mask(model, masks):
    """Re-zero any weight that should be zero per the mask."""
    with torch.no_grad():
        for name, mod in iter_target_layers_modules(model):
            if name in masks:
                mod.weight.mul_(masks[name])


# ── Eval ──────────────────────────────────────────────────────────────────

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


# ── Fine-tune one cycle ─────────────────────────────────────────────────

def finetune_one_cycle(model, masks, train_loader, val_loader, epochs, base_lr, cycle_idx):
    """SGD momentum 0.9 + warmup + cosine decay, with BF16 autocast and mask re-application."""
    optimizer = optim.SGD(
        model.parameters(), lr=base_lr, momentum=0.9, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    total_steps = epochs * len(train_loader)
    warmup = min(WARMUP_STEPS, total_steps // 4)
    min_lr = base_lr * MIN_LR_FRACTION

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
            # Zero gradients at masked (pruned) positions so SGD+momentum does not move them
            for name, mod in iter_target_layers_modules(model):
                if name in masks and mod.weight.grad is not None:
                    mod.weight.grad.mul_(masks[name])
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            optimizer.step()
            # Belt-and-suspenders: re-zero any weight that drifted off the mask
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


# ── Result durability ──────────────────────────────────────────────────────

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


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    log("=" * 70)
    log("Iterative pruning + fine-tuning pipeline")
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

    signals = None  # magnitude does not need RMT signals; skip

    # Baseline eval (cycle 0)
    results = load_results()
    if not results:
        baseline_top1 = evaluate_model(model, val_loader, label="baseline")
        results.append({"cycle_idx": -1, "target_s": 0.0, "method": "baseline",
                        "epochs": 0, "lr": 0.0,
                        "pre_ft_top1": baseline_top1, "post_ft_top1": baseline_top1,
                        "achieved_s": 0.0,
                        "ts": datetime.datetime.utcnow().isoformat()})
        save_results(results)

    # Determine where to resume
    start_cycle = 0
    for c in range(len(CYCLES) - 1, -1, -1):
        masks_loaded = load_checkpoint(model, c)
        if masks_loaded is not None:
            start_cycle = c + 1
            model.to(DEVICE)
            log(f"Resumed from checkpoint cycle {c} (next: cycle {start_cycle})")
            global_masks = masks_loaded
            break
    else:
        global_masks = None

    pipeline_start = time.time()
    for cycle_idx in range(start_cycle, len(CYCLES)):
        target_s, epochs, base_lr, method, apply_sv = CYCLES[cycle_idx]
        t0 = time.time()
        log("-" * 70)
        log(f"Cycle {cycle_idx+1}/{len(CYCLES)}: target_s={target_s:.2f}  "
            f"epochs={epochs}  lr={base_lr:.2e}  method={method}  sv={apply_sv}")

        achieved = prune_to_target(model, target_s, method, apply_sv, signals)
        log(f"  achieved sparsity: {achieved:.4f}")
        global_masks = build_mask(model)

        pre_ft_top1 = evaluate_model(model, val_loader, label=f"pre-FT s={target_s:.2f}")

        model = finetune_one_cycle(model, global_masks, train_loader, val_loader,
                                   epochs, base_lr, cycle_idx + 1)

        post_ft_top1 = evaluate_model(model, val_loader, label=f"post-FT s={target_s:.2f}")

        save_checkpoint(model, cycle_idx, global_masks)
        results.append({
            "cycle_idx": cycle_idx,
            "target_s": target_s,
            "achieved_s": achieved,
            "method": method,
            "epochs": epochs,
            "lr": base_lr,
            "sv_preprocessing": apply_sv,
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
