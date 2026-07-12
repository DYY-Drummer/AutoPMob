#!/usr/bin/env python3
"""baseline vs reranker-10S の対応あり有意差検定（層化分割・10乱数）.

入力: experiments/strat_A.json / experiments/strat_B.json の results[mode]["per_seed"]
出力: experiments/significance_stats.json
      （Wilcoxon符号順位検定・対応あり t 検定・Cohen dz、指標は Recall@K_correct / MAP / Recall@20）
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"

METRICS = ["Recall@K_correct", "MAP", "Recall@20"]
SETTINGS = {"A": "strat_A.json", "B": "strat_B.json"}


def load_per_seed(path, mode, metric):
    doc = json.load(open(path))
    return [p[metric] for p in doc["results"][mode]["per_seed"]]


def paired_stats(base, rer):
    b = np.asarray(base, dtype=float)
    r = np.asarray(rer, dtype=float)
    d = r - b
    t = stats.ttest_rel(r, b)
    w = stats.wilcoxon(r, b)
    return {
        "n": int(len(d)),
        "mean_base": round(float(b.mean()), 4),
        "mean_rer": round(float(r.mean()), 4),
        "mean_delta": round(float(d.mean()), 4),
        "std_delta": round(float(d.std(ddof=1)), 4),
        "p_ttest": round(float(t.pvalue), 5),
        "p_wilcoxon": round(float(w.pvalue), 5),
        "cohen_dz": round(float(d.mean() / d.std(ddof=1)), 3),
    }


def main():
    out = {}
    for name, fname in SETTINGS.items():
        path = EXP / fname
        doc = json.load(open(path))
        out[name] = {
            "file": fname,
            "seeds": [p["seed"] for p in doc["results"]["baseline"]["per_seed"]],
        }
        for m in METRICS:
            out[name][m] = paired_stats(
                load_per_seed(path, "baseline", m),
                load_per_seed(path, "reranker-10S", m),
            )
    dst = EXP / "significance_stats.json"
    json.dump(out, open(dst, "w"), ensure_ascii=False, indent=1)
    print(f"wrote {dst}")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
