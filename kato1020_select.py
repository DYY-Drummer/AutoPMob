"""2026-10-20：提案手法の第 1 段の重みと訓練の不正解の選び方を、分割ごとに検証用のケースの Recall@20 で選ぶ.

入力: experiments/kato1020/raw/select/{A,B}_{w}_{neg}__{seed}.json（kato1020_run.py select の出力。per-case に split）
選び方: 検証用のケースの Recall@20 の平均が最も高い (重み, 不正解の選び方)。同じなら検証用の MAP、それでも同じなら
        候補の並び（重みは (0.7, 0.3)・(0.5, 0.5)・(0.3, 0.7)・(0.0, 1.0)、不正解は無作為・hard の順）の先のもの。
出力: experiments/kato1020/selected.json
  selected[A|B][seed] = {w, neg, N1, val, test}、grid[A|B][w|neg] = 検証用とテストの平均 ± 標準偏差
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

from kato1014_common import METRIC_FNS, load_records

ROOT = Path(__file__).parent
# 出力先と外すケースの一覧は環境変数で切り替える（例 2026-10-21：KATO_OUT=experiments/kato1021、KATO_EXCLUDE=experiments/kato1020/excluded_cases.json。設定しなければ前回と同じ）
OUT = Path(os.environ.get("KATO_OUT", ROOT / "experiments" / "kato1020"))
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
WEIGHTS = ["w70-30", "w50-50", "w30-70", "w00-100"]
NEGS = ["random", "hard"]
M = ("Recall@20", "MAP", "MAP@20", "R-Precision")
N1_FIX = {k: int(v) for k, v in (x.split("=") for x in os.environ.get("KATO_N1_FIX", "").split(",") if x)}


def choose(cands):
    """cands: [(w, neg, val_metrics)]（候補の並び順）→ 選んだ (w, neg, val_metrics)."""
    best = None
    for k, (w, neg, v) in enumerate(cands):
        key = (v["Recall@20"], v["MAP"], -k)
        if best is None or key > best[0]:
            best = (key, (w, neg, v))
    return best[1]


def split_means(recs, n, split):
    rs = [r for r in recs if r.get("split") == split]
    return {m: float(np.mean([METRIC_FNS[m](r, n) for r in rs])) for m in M}, len(rs)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix-weight", default=None, help="第 1 段の重みをこの値に固定し、不正解の選び方だけを選ぶ（案 b）")
    ap.add_argument("--pooled", action="store_true", help="10 分割の検証用の平均で 1 つの設定を選び、全分割に使う（案 c）")
    ap.add_argument("--out", default="selected.json")
    a = ap.parse_args()
    rule = json.load(open(OUT / "n1_rule.json", encoding="utf-8"))["settings"]
    doc = {"selection": "検証用のケースの Recall@20（同じなら MAP、それでも同じなら候補の並びの先）",
           "fixed": {"m": 5, "gamma": 0.1, "n_neg": 8, "epochs": 15, "hidden": 64, "lr": 1e-3, "weight_decay": 1e-4,
                     "batch_size": 16, "split": "stratified", "N1": "重みと分割ごとに n1_rule.json"},
           "selected": {}, "grid": {}}
    for S in ("A", "B"):
        doc["selected"][S], grid = {}, {}
        per_seed_cands, all_tests = {}, {}
        for seed in SEEDS:
            cands, tests = [], {}
            for w in WEIGHTS:
                for neg in NEGS:
                    f = OUT / "raw" / "select" / f"{S}_{w}_{neg}__{seed}.json"
                    recs, n = load_records(f)
                    v, nv = split_means(recs, n, "val")
                    t, nt = split_means(recs, n, "test")
                    teq = [dict(r, ranks=r["ranks_equiv"]) for r in recs if r.get("split") == "test"]
                    tq = {m: float(np.mean([METRIC_FNS[m](r, n) for r in teq])) for m in M}
                    cands.append((w, neg, v))
                    tests[(w, neg)] = (t, tq, nv, nt, n)
                    g = grid.setdefault(f"{w}|{neg}", {"val": {m: [] for m in M}, "test": {m: [] for m in M}})
                    for m in M:
                        g["val"][m].append(v[m]); g["test"][m].append(t[m])
            if a.fix_weight:
                cands = [x for x in cands if x[0] == a.fix_weight]
            per_seed_cands[seed] = cands
            w, neg, v = choose(cands)
            t, tq, nv, nt, n = tests[(w, neg)]
            all_tests[seed] = tests
            assert n == N1_FIX.get(w, rule[S][str(seed)][w]["N1"])
            doc["selected"][S][str(seed)] = {"w": w, "neg": neg, "N1": n, "val": v, "test": t, "test_equiv": tq,
                                             "n_val": nv, "n_test": nt}
        if a.pooled:
            # 10 分割の検証用の平均で 1 つの設定を選び、全分割に使う
            keys = [(w, neg) for w, neg, _ in per_seed_cands[SEEDS[0]]]
            pooled = [(w, neg, {m: float(np.mean([dict(((x[0], x[1]), x[2]) for x in per_seed_cands[sd])[(w, neg)][m]
                                                 for sd in SEEDS])) for m in ("Recall@20", "MAP")}) for w, neg in keys]
            w, neg, v = choose(pooled)
            for seed in SEEDS:
                t, tq, nv, nt, n = all_tests[seed][(w, neg)]
                vv = dict(((x[0], x[1]), x[2]) for x in per_seed_cands[seed])[(w, neg)]
                doc["selected"][S][str(seed)] = {"w": w, "neg": neg, "N1": n, "val": vv, "test": t, "test_equiv": tq,
                                                 "n_val": nv, "n_test": nt}
        doc["grid"][S] = {k: {part: {m: {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1))}
                                     for m, x in g[part].items()} for part in g} for k, g in grid.items()}
        cnt = Counter((x["w"], x["neg"]) for x in doc["selected"][S].values())
        n1s = Counter(x["N1"] for x in doc["selected"][S].values())
        print(f"[{S}] 選ばれた設定 {dict(cnt)}  N1 {dict(n1s)}")
        for k in sorted(doc["grid"][S]):
            g = doc["grid"][S][k]
            print(f"   {k:16s} val R@20 {g['val']['Recall@20']['mean']:.4f}  test R@20 {g['test']['Recall@20']['mean']:.4f}  "
                  f"test MAP {g['test']['MAP']['mean']:.4f}")
    doc["mode"] = {"fix_weight": a.fix_weight, "pooled": a.pooled}
    json.dump(doc, open(OUT / a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {OUT / a.out}")


if __name__ == "__main__":
    main()
