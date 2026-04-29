"""
Comprehensive analysis of refined4 pruning sweep results.
Phase 1: 14 methods x 11 sparsities (uniform beta)
Phase 3: 25 (ba,bm) combos x 4 sparsities (layer-type beta)
"""
import json, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from collections import defaultdict

CACHE = "C:/Users/owner/Projects/rmt_pruning_vit/optuna_run/rmt_cache"
p1 = json.load(open(f"{CACHE}/splus_refined4_phase1_results.json"))
p3 = json.load(open(f"{CACHE}/splus_refined4_phase3_results.json"))

# Parse method params
def parse_method(m):
    """Return dict with b, sd, p, ba, bm parsed from method name."""
    if m == "magnitude":
        return {"b": 0, "sd": 0, "p": 0, "ba": 0, "bm": 0}
    d = {}
    for kv in re.findall(r'(b|sd|p|ba|bm)([\d.]+)', m):
        key, val = kv
        d[key] = float(val)
    return d

# ============================================================
# 1. PHASE 1: Full 14x11 grid
# ============================================================
print("=" * 130)
print("PHASE 1: Full 14-method x 11-sparsity grid (top1 accuracy, 250-batch)")
print("=" * 130)

methods1 = sorted(set(r["method"] for r in p1))
spars1 = sorted(set(r["target_sparsity"] for r in p1))

# Build lookup
p1_lookup = {}
for r in p1:
    p1_lookup[(r["method"], r["target_sparsity"])] = r["top1"]

# Short names for methods
def short_name(m):
    if m == "magnitude":
        return "magnitude"
    params = parse_method(m)
    parts = []
    if "b" in params: parts.append(f"b{params['b']:.2f}")
    if "sd" in params: parts.append(f"sd{params['sd']:.2f}")
    if "p" in params and params["p"] > 0: parts.append(f"p{params['p']:.1f}")
    if "ba" in params and params["ba"] > 0: parts.append(f"ba{params['ba']:.2f}")
    if "bm" in params and params["bm"] > 0: parts.append(f"bm{params['bm']:.2f}")
    return " ".join(parts)

# Print header
hdr = f"{'Method':<32}"
for s in spars1:
    hdr += f" {s*100:5.1f}%"
print(hdr)
print("-" * 130)

# Magnitude first as baseline
mag_row = "magnitude"
line = f"{'** magnitude (baseline) **':<32}"
for s in spars1:
    v = p1_lookup.get(("magnitude", s), None)
    line += f" {v:5.2f}" if v is not None else "    --"
print(line)
print("-" * 130)

# All splus methods
for m in methods1:
    if m == "magnitude":
        continue
    line = f"{short_name(m):<32}"
    for s in spars1:
        v = p1_lookup.get((m, s), None)
        line += f" {v:5.2f}" if v is not None else "    --"
    print(line)

# Delta vs magnitude
print()
print("=" * 130)
print("PHASE 1: Delta vs magnitude (positive = better than magnitude)")
print("=" * 130)
hdr = f"{'Method':<32}"
for s in spars1:
    hdr += f" {s*100:5.1f}%"
print(hdr)
print("-" * 130)

for m in methods1:
    if m == "magnitude":
        continue
    line = f"{short_name(m):<32}"
    for s in spars1:
        v = p1_lookup.get((m, s), None)
        mag = p1_lookup.get(("magnitude", s), None)
        if v is not None and mag is not None:
            d = v - mag
            line += f" {d:+5.2f}"
        else:
            line += "    --"
    print(line)

# Best per sparsity in Phase 1
print()
print("=" * 100)
print("PHASE 1: Best method at each sparsity")
print("=" * 100)
p1_best = {}
for s in spars1:
    best_m, best_v = None, -999
    for m in methods1:
        v = p1_lookup.get((m, s), None)
        if v is not None and v > best_v:
            best_v = v
            best_m = m
    mag = p1_lookup.get(("magnitude", s), None)
    delta = best_v - mag if mag else 0
    p1_best[s] = (best_m, best_v)
    print(f"  s={s*100:5.1f}%: {short_name(best_m):<35} top1={best_v:.2f}  (Δmag={delta:+.2f})")

# ============================================================
# 2. PHASE 3: 5x5 heatmap at each sparsity
# ============================================================
print()
print("=" * 100)
print("PHASE 3: Layer-type beta (ba_attn x bm_mlp) heatmaps")
print("  Base params: b=1.50, sd=0.85, p=1.0")
print("=" * 100)

spars3 = sorted(set(r["target_sparsity"] for r in p3))
p3_lookup = {}
for r in p3:
    params = parse_method(r["method"])
    ba = params.get("ba", 1.0)
    bm = params.get("bm", 1.0)
    p3_lookup[(r["target_sparsity"], ba, bm)] = r["top1"]

ba_vals = sorted(set(parse_method(r["method"]).get("ba", 1.0) for r in p3))
bm_vals = sorted(set(parse_method(r["method"]).get("bm", 1.0) for r in p3))

for s in spars3:
    mag = p1_lookup.get(("magnitude", s), None)
    uniform = p1_lookup.get(("splus_budget_b1.50_sd0.85_p1.0", s), None)
    print(f"\n  Sparsity = {s*100:.1f}%   (magnitude={mag}, uniform b1.50={uniform})")

    # Header
    hdr = f"  {'ba\\bm':<8}"
    for bm in bm_vals:
        hdr += f" {bm:6.2f}"
    print(hdr)
    print("  " + "-" * 45)

    best_v_s, best_ba_s, best_bm_s = -999, 0, 0
    for ba in ba_vals:
        line = f"  {ba:<8.2f}"
        for bm in bm_vals:
            v = p3_lookup.get((s, ba, bm), None)
            if v is not None:
                line += f" {v:6.2f}"
                if v > best_v_s:
                    best_v_s, best_ba_s, best_bm_s = v, ba, bm
            else:
                line += "     --"
        print(line)

    delta_mag = best_v_s - mag if mag else 0
    print(f"  Best: ba={best_ba_s:.2f} bm={best_bm_s:.2f} → {best_v_s:.2f} (Δmag={delta_mag:+.2f})")

# ============================================================
# 3. ABSOLUTE CHAMPION: Phase 1 + Phase 3 combined
# ============================================================
print()
print("=" * 100)
print("ABSOLUTE CHAMPION at each sparsity (Phase 1 + Phase 3 combined)")
print("=" * 100)

# Collect ALL results into one pool
all_results = defaultdict(list)  # sparsity -> [(top1, method_desc, params_dict)]

for r in p1:
    s = r["target_sparsity"]
    params = parse_method(r["method"])
    desc = r["method"]
    all_results[s].append((r["top1"], desc, params))

for r in p3:
    s = r["target_sparsity"]
    params = parse_method(r["method"])
    desc = r["method"]
    all_results[s].append((r["top1"], desc, params))

all_spars = sorted(all_results.keys())
champs = {}
for s in all_spars:
    entries = all_results[s]
    entries.sort(key=lambda x: -x[0])
    best = entries[0]
    mag = p1_lookup.get(("magnitude", s), None)
    delta = best[0] - mag if mag else 0
    champs[s] = best
    print(f"  s={s*100:5.1f}%: top1={best[0]:6.2f}  Δmag={delta:+5.2f}  method={short_name(best[1])}")

# ============================================================
# 4. CHAMPION PARAMETER TABLE
# ============================================================
print()
print("=" * 100)
print("CHAMPION PARAMETERS for definitive comparison")
print("=" * 100)
print(f"  {'Sparsity':<10} {'top1':>6} {'Δmag':>6}  {'β':>5} {'sd':>5} {'p':>4} {'βa':>5} {'βm':>5}  {'Source':<8}")
print("  " + "-" * 75)

for s in all_spars:
    top1, desc, params = champs[s]
    mag = p1_lookup.get(("magnitude", s), None)
    delta = top1 - mag if mag else 0
    b = params.get("b", 0)
    sd = params.get("sd", 0)
    p_val = params.get("p", 0)
    ba = params.get("ba", 0)
    bm = params.get("bm", 0)
    source = "P3" if "ba" in desc else "P1" if desc != "magnitude" else "mag"
    is_mag = desc == "magnitude"
    if is_mag:
        print(f"  {s*100:5.1f}%     {top1:6.2f} {delta:+5.2f}  magnitude (no splus method beats it)")
    else:
        print(f"  {s*100:5.1f}%     {top1:6.2f} {delta:+5.2f}  {b:5.2f} {sd:5.2f} {p_val:4.1f} {ba:5.2f} {bm:5.2f}  {source}")

# ============================================================
# 5. SIMPLE RECOMMENDATION for s=50-70%
# ============================================================
print()
print("=" * 100)
print("SIMPLE RECOMMENDATION: Best single config across s=50-70%")
print("=" * 100)

target_range = [s for s in all_spars if 0.50 <= s <= 0.70]
print(f"  Evaluating sparsities: {[f'{s*100:.0f}%' for s in target_range]}")

# For each unique method, sum its delta-vs-magnitude across target range
method_scores = defaultdict(lambda: {"sum_delta": 0, "count": 0, "details": []})
for s in target_range:
    mag = p1_lookup.get(("magnitude", s), None)
    if mag is None:
        continue
    for top1, desc, params in all_results[s]:
        if desc == "magnitude":
            continue
        d = top1 - mag
        method_scores[desc]["sum_delta"] += d
        method_scores[desc]["count"] += 1
        method_scores[desc]["details"].append((s, d))

# Rank by average delta
ranked = []
for desc, info in method_scores.items():
    if info["count"] == len(target_range):
        avg = info["sum_delta"] / info["count"]
        ranked.append((avg, desc, info["details"]))
    elif info["count"] >= 3:  # at least 3 sparsities covered
        avg = info["sum_delta"] / info["count"]
        ranked.append((avg, desc, info["details"]))

ranked.sort(key=lambda x: -x[0])

print(f"\n  Top 10 methods by average Δmag across s=50-70%:")
print(f"  {'Rank':<5} {'Avg Δmag':>8} {'Method':<55} {'Per-sparsity deltas'}")
print("  " + "-" * 110)
for i, (avg, desc, details) in enumerate(ranked[:10]):
    detail_str = ", ".join(f"{s*100:.0f}%:{d:+.2f}" for s, d in sorted(details))
    print(f"  {i+1:<5} {avg:+8.2f}  {short_name(desc):<55} {detail_str}")

# Also show magnitude baseline for context
print(f"\n  Magnitude baseline at target sparsities:")
for s in target_range:
    v = p1_lookup.get(("magnitude", s), None)
    print(f"    s={s*100:.0f}%: {v:.2f}")

# Top-2 and top-3 at each sparsity for deeper context
print()
print("=" * 100)
print("TOP-3 at each sparsity (for context)")
print("=" * 100)
for s in all_spars:
    entries = all_results[s]
    entries.sort(key=lambda x: -x[0])
    mag = p1_lookup.get(("magnitude", s), None)
    print(f"\n  s={s*100:.1f}%  (magnitude={mag:.2f}):")
    for i, (top1, desc, params) in enumerate(entries[:3]):
        delta = top1 - mag if mag else 0
        print(f"    #{i+1}: {top1:6.2f} (Δ={delta:+.2f})  {short_name(desc)}")
