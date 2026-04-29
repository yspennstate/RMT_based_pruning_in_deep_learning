"""Parse haar_optuna_log.txt into haar_optuna_results.json"""
import re, json, os

log_path = os.path.join(os.path.dirname(__file__), "haar_optuna_log.txt")
out_path = os.path.join(os.path.dirname(__file__), "haar_optuna_results.json")

trials = []
baseline_top1 = None

with open(log_path, 'r') as f:
    for line in f:
        # Baseline
        m = re.match(r'Baseline \(no SV\): ([\d.]+)%', line)
        if m:
            baseline_top1 = float(m.group(1))
            continue

        # Successful trial
        m = re.match(r'Trial\s+(\d+)\s+\[(\w+)\s*\]:\s+\S+\s+z=([\d.]+).*?(cut=([\d.]+))?\s*(pow=(\d+))?\s+([\d.]+)%\s+kept=([\d.]+)%', line)
        if m:
            num = int(m.group(1))
            method = m.group(2)
            z = float(m.group(3))
            cutoff = float(m.group(5)) if m.group(5) else None
            power = int(m.group(7)) if m.group(7) else None
            top1 = float(m.group(8))

            params = {"method": method}
            if method == "just_z":
                params["jz_z"] = z
            elif method == "z_bulk":
                params["zb_z"] = z
                params["zb_cutoff"] = cutoff if cutoff else 1.0
            elif method == "z_graduated":
                params["zg_z"] = z
                params["zg_cutoff"] = cutoff if cutoff else 1.0
                params["zg_power"] = power if power else 1

            trials.append({"number": num, "params": params, "top1": top1})
            continue

        # Failed trial
        m = re.match(r'Trial\s+(\d+)\s+\[(\w+)\s*\]:\s+.*FAILED', line)
        if m:
            trials.append({"number": int(m.group(1)), "params": {"method": m.group(2)}, "top1": None})

# Find best
valid = [t for t in trials if t['top1'] is not None and t['top1'] > 0]
best = max(valid, key=lambda t: t['top1']) if valid else None

results = {
    "baseline_top1": baseline_top1,
    "best_params": best['params'] if best else None,
    "best_top1": best['top1'] if best else None,
    "trials": trials
}

with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Parsed {len(trials)} trials, baseline={baseline_top1}%")
if best:
    print(f"Best: {best['top1']:.2f}% {best['params']}")
print(f"Saved to {out_path}")
