"""experiments/xd/{mode}__{seed}.json を 1 ファイルに統合.

出力: experiments/ablation_x_difficulty.json
      {config, results: {mode: {per_case: [...全seed...], per_seed_summary: {...}}}}
"""
import json, glob, re
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent
XD = ROOT / "experiments" / "xd"
OUT = ROOT / "experiments" / "ablation_x_difficulty.json"
METRIC = "Recall@K_correct"

merged_percase = defaultdict(list)
per_seed_metric = defaultdict(lambda: defaultdict(dict))  # mode -> metric -> {seed: val}
config = None
files = sorted(glob.glob(str(XD / "*.json")))
print(f"merging {len(files)} files")
for f in files:
    d = json.load(open(f, encoding="utf-8"))
    if config is None:
        config = d.get("config", {})
    for mode, r in d.get("results", {}).items():
        pc = r.get("per_case", [])
        merged_percase[mode].extend(pc)
        # per-seed aggregate (from summary if present)
        for k, v in r.items():
            if isinstance(v, dict) and "mean" in v:
                pass
        for a in r.get("per_seed", []):
            sd = a.get("seed")
            for mk in ["Recall@K_correct", "MAP", "FullRecall@3"]:
                if mk in a:
                    per_seed_metric[mode][mk][sd] = a[mk]

results = {}
for mode, pc in merged_percase.items():
    seeds = sorted({r["seed"] for r in pc})
    results[mode] = {
        "n_per_case": len(pc),
        "seeds": seeds,
        "per_case": pc,
    }
    print(f"  {mode:18s} per_case={len(pc):5d} seeds={seeds}")

json.dump({"config": config, "results": results}, open(OUT, "w", encoding="utf-8"),
          ensure_ascii=False)
print(f"Saved: {OUT}")
