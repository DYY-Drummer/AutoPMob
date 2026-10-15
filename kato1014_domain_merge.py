"""§8（2026-10-14 の指示 5）：分野ラベルをまとめた場合の x7 の効果を、新しい設定でやり直す.

run_domain_merge_ablation.py と同じ合併（L1 = 表記の正規化、K100・K40 = ラベルの TF-IDF の k-means）を使い、
reranker-2+Dom（x1, x2 に分野ラベルの一致 x7）を訓練し直して reranker-2 と比べる。
条件は run_kato1014.sh と同じ（設定 A・層化分割・第 1 段の重み 0.3/0.7・--top-k・--fast）。

x7 の数え方（旧版から直した点、2026-10-15 の最終レビュー）：旧版は式のラベルを群の名前（K100・K40 では "C12" など）に
置き換えてから、説明文に含まれるかを部分文字列で比べたので、K100・K40 では x7 がほとんど 1 にならず
（設定 A で 1,823 ケース中 5 ケース）、合併の効果を測れていなかった。ここでは、説明文が「式の群に入るどれかの
元のラベル」を元の規則（小文字にして、一方が他方に含まれる）で含めば x7 = 1 とする。合併しない（群 = ラベル 1 つ）
ときは元の x7 と同じ値になる（tests/test_kato1014_domain_merge.py）。

出力は set_aware_reranker.py --save-per-case と同じ形の JSON（Recall@20 と MAP を後で数える）。
使い方: python kato1014_domain_merge.py --merge K40 --seed 42 --top-k 2000 --hard-neg --output out.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import stage2_fast
from run_domain_merge_ablation import build_maps, load_lists
from set_aware_reranker import run_mode
from stage2_fast import FastBank
from two_stage_query_conditioned import case_text


def make_cluster_bank(cluster_of: dict):
    """x7 だけを群で数える FastBank を返す（x10 などほかの列は元のラベルのまま）."""

    class ClusterBank(FastBank):
        def __init__(self, X_ctx, X_eq, cases, eq_vars_list, eq_domains, svd_sim):
            super().__init__(X_ctx, X_eq, cases, eq_vars_list, eq_domains, svd_sim)
            labels = sorted({d for d in eq_domains if d})
            self._clabels = [(d.lower(), cluster_of.get(d, d)) for d in labels]
            self._cl_eq = np.array([cluster_of.get(d, d) if d else "" for d in eq_domains], dtype=object)

        def features(self, ci, cands, set_mask=()):
            F = super().features(ci, cands, set_mask)
            dl = case_text(self.cases[ci]).lower()
            hit = {c for low, c in self._clabels if (low and low in dl) or (dl and dl in low)}
            cl = self._cl_eq[np.asarray(cands, dtype=np.int64)]
            F[:, 6] = np.array([1.0 if (c and c in hit) else 0.0 for c in cl], dtype=np.float32)
            return F

    return ClusterBank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", required=True, choices=["orig", "L1", "K100", "K40"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--top-k", type=int, required=True)
    ap.add_argument("--stage1-w-text", type=float, default=0.3)
    ap.add_argument("--stage1-w-var", type=float, default=0.7)
    ap.add_argument("--mode", default="reranker-2+Dom")
    ap.add_argument("--hard-neg", action="store_true", help="訓練の不正解を点数の高い順に選ぶ（run_kato1014.sh の EXTRA と合わせる）")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    ctx = load_lists()
    cases, ek, et, ev, ed, es, cl, cs = ctx
    labels = sorted({d for d in ed if d})
    maps = build_maps(labels)
    n_groups = len({maps[a.merge][d] for d in labels if maps[a.merge][d]})
    # run_mode は呼ばれた時点で stage2_fast.FastBank を読むので、x7 を群で数える版に差し替える
    stage2_fast.FastBank = make_cluster_bank(maps[a.merge])
    r = run_mode(a.mode, [a.seed], cases, ek, et, ev, ed, es, cl, cs,
                 a.top_k, 15, 1e-3, save_per_case=True, split_mode="stratified",
                 stage1_w_text=a.stage1_w_text, stage1_w_var=a.stage1_w_var, fast=True,
                 hard_neg=a.hard_neg)
    cfg = {**vars(a), "top_k": a.top_k, "n_domain_groups": n_groups, "fast": True,
           "x7_rule": "説明文が式の群のどれかの元のラベルを含めば 1（cluster-aware）",
           "variants": "original,multisource_,dae_", "split": "stratified"}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": cfg, "results": {a.mode: r}}, open(a.output, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"Saved: {a.output}  ({a.mode}@{a.merge}, {n_groups} 群, seed={a.seed}, N1={a.top_k})")


if __name__ == "__main__":
    main()
