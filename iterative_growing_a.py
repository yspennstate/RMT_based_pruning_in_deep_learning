"""
Strategy A variant: NO mask. Each iteration prunes K% smallest (cumulative target),
where K grows: 5, 10, 15, ..., then SV prune (Haar z+bulk z=0.14 cut=0.76).

Each iteration starts from baseline-rebuilt model state (no carry-over),
prunes K% smallest of current dense matrix per layer, then SV prunes.
Strategy B already done in iter5pct_log.txt — not re-run here.
"""

import sys, os, math, time, copy, gc, ctypes, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Force UTF-8 stdout so any non-ASCII chars in log messages don't crash on
# Windows cp1252. Bit us once already (UnicodeEncodeError on '→').
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import torch
import timm
import numpy as np

from pruning import count_nonzero_params, count_total_params, replace_layers
from SplittableLayers import SplittableConv, SplittableLinear
from validation import evaluate, get_val_dataset
from RMT import bema_inside


# ───── Adaptive throttle: when the user is actively at the keyboard/mouse,
# shrink eval kernels so Chrome/YouTube/games can slip in between them.
# When idle, run at full (still inside the 50% GPU + 8-core CPU caps).
class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]


def user_idle_seconds():
    """Seconds since last keyboard or mouse input on this Windows session."""
    try:
        lii = _LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(_LASTINPUTINFO)
        if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            return 9999
        tick = ctypes.windll.kernel32.GetTickCount()
        return max(0, (tick - lii.dwTime) / 1000.0)
    except Exception:
        return 9999


IDLE_THRESHOLD_SEC = 10            # >10 s of no input → assume user away
ACTIVE_BATCH_SIZE = 1              # tiny kernels → Chrome/games stay smooth
IDLE_BATCH_SIZE   = 8              # full speed (still inside GPU 50% cap)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Half-GPU rule: protect the live IBKR trading bots running on this box.
# After the 2026-04-06 hard crash, all GPU sims must cap at ~50% VRAM.
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    torch.backends.cudnn.benchmark = False  # deterministic memory pattern
ALPHA, BETA, GOF = 0.25, 0.8, 1
SV_Z = 0.1407
SV_CUT = 0.760
PRUNE_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]  # cumulative %
LOG_FILE = os.path.join(os.path.dirname(__file__), "iter_growing_a_log.txt")

# ───── RMT cache ─────
# We cache the once-only-needed RMT work for the baseline ViT-B model:
#   * `rmt_splus_metrics.json` — per-layer Marchenko–Pastur bulk-edge sigma_+
#                                values (from compute_splus / BEMA). These are
#                                the "RMT metrics" of each weight matrix and
#                                depend only on the baseline weights.
#   * `sv_pruned_baseline_state.pt` — the full ViT-B state_dict AFTER applying
#                                Haar z+bulk SV pruning (z=SV_Z, cut=SV_CUT) to
#                                every Splittable layer. Built ONCE; reused for
#                                every iteration AND every future run.
# Cache is keyed on (model='vit_base_patch16_224', SV_Z, SV_CUT). If you change
# SV_Z or SV_CUT the cache is invalidated and rebuilt automatically.
RMT_CACHE_DIR     = os.path.join(os.path.dirname(__file__), "rmt_cache")
SPLUS_CACHE_FILE  = os.path.join(RMT_CACHE_DIR, "rmt_splus_metrics.json")
SVPRUNED_STATE_FILE = os.path.join(RMT_CACHE_DIR, "sv_pruned_baseline_state.pt")
os.makedirs(RMT_CACHE_DIR, exist_ok=True)


def log(msg):
    print(msg)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")


def compute_splus(W_np):
    M, N = W_np.shape
    p = min(M, N)
    n = max(M, N)
    gram = (W_np @ W_np.T / N) if M <= N else (W_np.T @ W_np / N)
    eigenvals = np.sort(np.linalg.eigvalsh(gram))
    sigma_sq, lamda_plus, l2 = bema_inside(p, n, eigenvals, ALPHA, 0.8)
    return math.sqrt(N * lamda_plus)


def sv_prune_z_bulk(W_np, splus, z_thresh, bulk_cutoff):
    U, S, Vt = np.linalg.svd(W_np, full_matrices=False)
    M, N = W_np.shape
    thresh_U = z_thresh / math.sqrt(M)
    thresh_V = z_thresh / math.sqrt(N)
    for i in range(len(S)):
        if S[i] / splus >= bulk_cutoff:
            continue
        U[:, i] = np.where(np.abs(U[:, i]) < thresh_U, 0, U[:, i])
        Vt[i, :] = np.where(np.abs(Vt[i, :]) < thresh_V, 0, Vt[i, :])
    return U @ np.diag(S) @ Vt


def get_subs(layer):
    return [s for s in layer.modules() if isinstance(s, (torch.nn.Conv2d, torch.nn.Linear))]


def prune_smallest_per_layer(model, frac):
    """Zero the smallest `frac` of nonzero entries in each Splittable layer."""
    for name, layer in model.named_modules():
        if not isinstance(layer, (SplittableConv, SplittableLinear)):
            continue
        subs = get_subs(layer)
        if not subs:
            continue
        all_abs = []
        for sub in subs:
            w = sub.weight.data.abs()
            nz = w[w > 0]
            if len(nz) > 0:
                all_abs.append(nz)
        if not all_abs:
            continue
        all_abs = torch.cat(all_abs)
        n_prune = int(len(all_abs) * frac)
        if n_prune <= 0:
            continue
        sorted_vals, _ = torch.sort(all_abs)
        threshold = sorted_vals[n_prune - 1].item()
        with torch.no_grad():
            for sub in subs:
                mask = sub.weight.data.abs() > threshold
                sub.weight.data *= mask.float()


def sv_prune_all_layers(model):
    """Compute splus + Haar z+bulk SV prune for every Splittable layer.
       Used only when building the cache; subsequent calls load from disk."""
    splus_metrics = {}
    for name, layer in model.named_modules():
        if not isinstance(layer, (SplittableConv, SplittableLinear)):
            continue
        W = layer.get_matrix()
        splus = compute_splus(W)
        splus_metrics[name] = float(splus)
        W_pruned = sv_prune_z_bulk(W, splus, SV_Z, SV_CUT)
        layer.set_params("layer1", torch.from_numpy(W_pruned).float(),
                         bias=None, change_bias=False)
    return splus_metrics


def get_sv_pruned_baseline(base_state):
    """Return a state_dict for ViT-B with Haar SV pruning already applied to
    every Splittable layer. Built ONCE per (SV_Z, SV_CUT) pair, then cached on
    disk forever — eliminates ~all CPU SVD work after the first run."""
    cache_key = {"model": "vit_base_patch16_224", "SV_Z": SV_Z, "SV_CUT": SV_CUT}

    if os.path.exists(SVPRUNED_STATE_FILE) and os.path.exists(SPLUS_CACHE_FILE):
        try:
            with open(SPLUS_CACHE_FILE) as f:
                meta = json.load(f)
            if (meta.get("SV_Z") == SV_Z and meta.get("SV_CUT") == SV_CUT
                    and meta.get("model") == "vit_base_patch16_224"):
                log(f"[rmt-cache] HIT: loading SV-pruned baseline from {SVPRUNED_STATE_FILE}")
                log(f"[rmt-cache] {len(meta.get('splus', {}))} cached layer splus values")
                return torch.load(SVPRUNED_STATE_FILE, map_location="cpu", weights_only=False)
            log(f"[rmt-cache] MISS: SV_Z/SV_CUT changed, rebuilding cache")
        except Exception as e:
            log(f"[rmt-cache] failed to load cache ({e}), rebuilding")

    log(f"[rmt-cache] BUILDING: one-time SV prune of baseline ViT-B (z={SV_Z}, cut={SV_CUT})")
    t0 = time.time()
    m = build_model(base_state)
    splus_metrics = sv_prune_all_layers(m)
    sv_state = copy.deepcopy(m.cpu().state_dict())
    del m; gc.collect()

    torch.save(sv_state, SVPRUNED_STATE_FILE)
    with open(SPLUS_CACHE_FILE, "w") as f:
        json.dump({**cache_key, "splus": splus_metrics,
                   "n_layers": len(splus_metrics),
                   "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)
    log(f"[rmt-cache] BUILT in {time.time()-t0:.0f}s -> {SVPRUNED_STATE_FILE}")
    log(f"[rmt-cache] saved {len(splus_metrics)} layer splus values -> {SPLUS_CACHE_FILE}")
    return sv_state


def build_model(state):
    m = timm.create_model("vit_base_patch16_224", pretrained=False)
    replace_layers(m, ALPHA, BETA, GOF, depth=0)
    m.load_state_dict(state)
    m.eval()
    return m


def eval_model(model, preprocess, total_params):
    # Reclaim any leftover allocator slabs from the previous iteration BEFORE
    # we ship a fresh model to GPU — fragmentation across iterations was the
    # crash vector on 2026-04-06.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Adaptive: pick batch size + CUDA stream priority based on user activity.
    idle = user_idle_seconds()
    user_active = idle < IDLE_THRESHOLD_SEC
    bs = ACTIVE_BATCH_SIZE if user_active else IDLE_BATCH_SIZE
    mode_str = f"ACTIVE bs={bs}" if user_active else f"IDLE bs={bs}"
    log(f"  [throttle] user idle {idle:.0f}s -> {mode_str}")

    val_loader = get_val_dataset(preprocess=preprocess, batch_size=bs)

    model.to(DEVICE)
    nz = count_nonzero_params(model)
    pct = 100 * nz / total_params

    if user_active and torch.cuda.is_available():
        # Low-priority CUDA stream → kernels from this process queue BEHIND
        # any pending Chrome/game/desktop GPU work. Combined with bs=1 (very
        # short kernels) this lets foreground apps stay smooth.
        low_stream = torch.cuda.Stream(priority=-1)
        with torch.cuda.stream(low_stream):
            top1, _ = evaluate(val_loader, model, DEVICE)
        torch.cuda.current_stream().wait_stream(low_stream)
    else:
        top1, _ = evaluate(val_loader, model, DEVICE)

    top1 = top1.cpu().item() if isinstance(top1, torch.Tensor) else top1
    model.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return top1, pct


if __name__ == "__main__":
    with open(LOG_FILE, "w") as f:
        f.write(f"Strategy A growing-prune (NO mask) started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"SV params: z={SV_Z}, cutoff={SV_CUT}\n")
        f.write(f"Cumulative prune levels: {PRUNE_LEVELS}\n\n")
    log(f"Device: {DEVICE}")

    model = timm.create_model("vit_base_patch16_224", pretrained=True)
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)

    replace_layers(model, ALPHA, BETA, GOF, depth=0)
    total_params = count_total_params(model)
    base_state = copy.deepcopy(model.cpu().state_dict())
    del model; gc.collect(); torch.cuda.empty_cache()

    # One-time (cached) SV pruning of baseline. Future runs load from disk.
    sv_pruned_state = get_sv_pruned_baseline(base_state)

    log("\n=== Strategy A: SV first (cached), then cumulative K% mag prune (no mask) ===")
    log(f"{'iter':>4}  {'K%':>4}  {'top1':>7}  {'kept':>7}  {'time':>6}")
    for it, K in enumerate(PRUNE_LEVELS, 1):
        t0 = time.time()
        m = build_model(sv_pruned_state)        # already SV-pruned
        prune_smallest_per_layer(m, K / 100.0)  # then drop K% smallest
        top1, pct = eval_model(m, preprocess, total_params)
        log(f"{it:4d}  {K:3d}%  {top1:6.2f}%  {pct:6.2f}%  {time.time()-t0:5.0f}s")
        del m; gc.collect(); torch.cuda.empty_cache()

    log(f"\nFinished {time.strftime('%Y-%m-%d %H:%M:%S')}")
