"""§5・§8 の PFI（2026-10-14 の指示 5）：新しい設定の一括の並べ直しで、特徴量のシャッフルによる低下を測る.

手順は analyze_pfi.py（報告書 §5 の図、§8 (3)）と同じ:
  - 訓練済みのモデル（reranker-10S、10 特徴量）を固定し、テストのケースごとに候補どうしで特徴量の列を
    シャッフルする（群は同じ行の並べ替えを群の全列に当てる）。乱数は RandomState(seed*1000 + i)、i = 0..19。
  - 対象: 10 特徴量 1 個ずつ、変数の 6 特徴量（x2, x4, x5, x6, x8, x9）、話題の 4 特徴量（x1, x3, x7, x10）、
    補完性と一貫性の組。
変えた点: モデルは set_aware_reranker.run_mode で訓練する（on_trained で受け取る。C の一括の並べ直しと
同じモデル）。第 1 段の重みは 0.3/0.7、候補数は --top-k。指標は Recall@20 と MAP（標準）。旧主指標 Recall@K も記録。

使い方: python kato1014_pfi.py --seed 42 --top-k 2000 --output experiments/kato1014/raw/pfi/pfi_A__42.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from analyze_pfi import FEATURES, PERM_TARGETS, sep_for_feature
from kato1014_common import ap_at, ap_std, recall_at, rk

METRICS = ("Recall@20", "MAP", "MAP@20", "Recall@K")


def permute_case(feats_list, cols, perm_seed):
    """analyze_pfi.permute_feats(scope="case") と同じ並べ替え（ケースの順に同じ乱数列から）."""
    prng = np.random.RandomState(perm_seed)
    out = []
    for f0 in feats_list:
        f = f0.copy()
        perm = prng.permutation(f.shape[0])
        f[:, cols] = f0[np.ix_(perm, cols)]
        out.append(f)
    return out


def case_metrics_from_scores(cands, scores, corr) -> dict:
    """一括の並べ直しと同じ並べ方（点数の降順、同点は候補の順）で、正解式の順位と指標を求める."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    pos = {cands[k]: r for r, k in enumerate(order, 1)}
    n = len(cands)
    miss = 10_000 if n < 10_000 else 10 ** 9  # set_aware_reranker.run_mode の miss_rank と同じ
    ranks = sorted(pos.get(c, miss) for c in corr)
    return {"ranks": ranks, "Recall@20": recall_at(ranks, 20), "MAP": ap_std(ranks, n),
            "MAP@20": ap_at(ranks, 20), "Recall@K": rk(ranks)}


def run_pfi(ctx, cl, cases, n_perm):
    import torch
    seed, model = ctx["seed"], ctx["model"]
    cache = []
    for ci in ctx["te"]:
        cands = ctx["get_cands"](ci)
        cache.append({"ci": ci, "cands": cands, "corr": set(cl[ci]),
                      "feats": np.asarray(ctx["get_feats"](ci, cands), dtype=np.float32)})

    def score_all(feats_list):
        with torch.no_grad():
            return [case_metrics_from_scores(rec["cands"],
                                             model(torch.tensor(f)).numpy().ravel(), rec["corr"])
                    for rec, f in zip(cache, feats_list)]

    base = score_all([r["feats"] for r in cache])
    col_of = {name: c for c, name, _ in FEATURES}
    seps = {name: [sep_for_feature(r["feats"], r["cands"], r["corr"], c) for r in cache]
            for name, c in col_of.items()}
    per_target = {}
    for lab, cols in PERM_TARGETS.items():
        sums = {m: np.zeros(len(cache)) for m in METRICS}
        agg = {m: [] for m in METRICS}
        for pi in range(n_perm):
            res = score_all(permute_case([r["feats"] for r in cache], cols, seed * 1000 + pi))
            for m in METRICS:
                v = np.array([x[m] for x in res])
                sums[m] += v
                agg[m].append(float(v.mean()))
        per_target[lab] = {
            "agg_perm_mean": {m: float(np.mean(agg[m])) for m in METRICS},
            "case_perm_mean": {m: (sums[m] / n_perm).tolist() for m in METRICS},
        }
    return {
        "seed": seed, "n_test": len(cache),
        "base": {m: float(np.mean([b[m] for b in base])) for m in METRICS},
        "cases": [{"case_id": cases[r["ci"]].get("case_id"), "n_correct": len(r["corr"]),
                   "variant": cases[r["ci"]].get("variant_type"), "n_cands": len(r["cands"]),
                   "base": {m: base[i][m] for m in METRICS}, "ranks": base[i]["ranks"],
                   "sep": {name: seps[name][i] for name in col_of}}
                  for i, r in enumerate(cache)],
        "targets": per_target,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--top-k", type=int, required=True)
    ap.add_argument("--n-perm", type=int, default=20)
    ap.add_argument("--stage1-w-text", type=float, default=0.3)
    ap.add_argument("--stage1-w-var", type=float, default=0.7)
    ap.add_argument("--hard-neg", action="store_true", help="訓練の不正解を点数の高い順に選ぶ（C と同じ訓練にする）")
    ap.add_argument("--exclude", default=None, help="外すケースの一覧の JSON（2026-10-20）")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    from analyze_stage1_n1 import load_setting_A
    from set_aware_reranker import run_mode
    D = load_setting_A(a.exclude)
    cases, ek, et, ev, ed, es, cl, cs = (D[k] for k in ("cases", "ek", "et", "ev", "ed", "es", "cl", "cs"))
    holder = {}
    t0 = time.time()

    def hook(ctx):
        holder["pfi"] = run_pfi(ctx, cl, cases, a.n_perm)

    r = run_mode("reranker-10S", [a.seed], cases, ek, et, ev, ed, es, cl, cs, a.top_k, 15, 1e-3,
                 save_per_case=True, split_mode="stratified", stage1_w_text=a.stage1_w_text,
                 stage1_w_var=a.stage1_w_var, fast=True, hard_neg=a.hard_neg, on_trained=hook)
    # PFI の基準値（シャッフル前）が run_mode の評価と同じであることを確かめる
    want = {rec["case_id"]: rec["ranks"] for rec in r["per_case"]}
    pfi = holder["pfi"]
    n_same = sum(1 for c in pfi["cases"] if want.get(c["case_id"]) == c["ranks"])
    assert n_same == len(pfi["cases"]), f"基準の順位が評価と一致しない: {n_same}/{len(pfi['cases'])}"
    out = {"config": {**vars(a), "setting": "A", "mode": "reranker-10S", "fast": True,
                      "variants": "original,multisource_,dae_", "split": "stratified",
                      "perm_seed": "seed*1000+i", "scope": "case", "elapsed_sec": round(time.time() - t0, 1)},
           "pfi": pfi}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.output, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"Saved: {a.output}  base R@20 {pfi['base']['Recall@20']:.4f}  "
          f"GROUP_var → {pfi['targets']['GROUP_var']['agg_perm_mean']['Recall@20']:.4f}  "
          f"GROUP_topic → {pfi['targets']['GROUP_topic']['agg_perm_mean']['Recall@20']:.4f}")


if __name__ == "__main__":
    main()
