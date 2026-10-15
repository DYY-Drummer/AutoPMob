"""§9.1（2026-10-14 の指示 5）：誤りの位置（第 1 段の見逃しか、第 2 段の並べ方か）を新しい設定でやり直す.

旧版 analyze_case_outcomes.py（乱数の種 42、第 1 段の重み 0.7/0.3、N1 = 50 と 200、上位 K 件で判定）と同じ数え方を、
第 1 段の重み 0.3/0.7・候補数 --top-k・上位 20 件（Recall@20）で行う。上位 K 件での数え方も併記する。
  - 上位 n 件に入らなかった正解式を、候補に入っていなかった（第 1 段の見逃し）か、候補にはいたが
    上位 n 件から漏れた（第 2 段の見逃し）かに分ける（analyze_case_outcomes.failure_split）。
  - 正解式をすべて上位 n 件に並べたケースとそうでないケースの属性の差（Mann–Whitney の U 検定）。
  - 第 2 段で漏れた正解式と、上位 n 件に入った不正解式の組で、正解式の値の方が大きい割合（topic_rescue_fraction）。
モデルは set_aware_reranker.run_mode（--fast）で訓練し、on_trained で受け取る（C の一括の並べ直しと同じモデル）。
出力: experiments/kato1014/case_outcomes_<tag>.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from analyze_case_outcomes import FEAT_COL, _mean_sem, _mw, failure_split, topic_rescue_fraction
from kato1014_common import OUT

RESCUE = ("svd_sim", "text_sim", "domain", "specificity", "io_jaccard")


def outcomes(recs, top_n_of):
    """recs: ケースごとの {cands, feats, scores, corr, attrs}。top_n_of(rec) は判定に使う件数."""
    rows = []
    for r in recs:
        order = np.argsort(-r["scores"], kind="stable")
        ranked = [r["cands"][k] for k in order]
        n = top_n_of(r)
        fs = failure_split(ranked, r["cands"], r["corr"], n)
        idx_of = {j: k for k, j in enumerate(r["cands"])}
        wrong = set(j for j in ranked[:n] if j not in r["corr"])
        val = {f: {j: float(r["feats"][idx_of[j], FEAT_COL[f]]) for j in (fs["reranker_miss"] | wrong)}
               for f in RESCUE}
        rows.append({**r["attrs"], "recall": fs["recall"],
                     "coverage": len(r["corr"] & set(r["cands"])) / len(r["corr"]),
                     "n_stage1_miss": len(fs["stage1_miss"]), "n_reranker_miss": len(fs["reranker_miss"]),
                     "rescue": {f: topic_rescue_fraction(fs["reranker_miss"], wrong, val[f]) for f in RESCUE}})
    solved = [x for x in rows if x["recall"] == 1.0]
    uns = [x for x in rows if x["recall"] < 1.0]
    s1 = sum(x["n_stage1_miss"] for x in rows)
    s2 = sum(x["n_reranker_miss"] for x in rows)
    return {
        "n_cases": len(rows), "n_gt": int(sum(x["n_correct"] for x in rows)),
        "mean_recall": float(np.mean([x["recall"] for x in rows])),
        "missed_total": s1 + s2, "stage1_miss": s1, "stage2_miss": s2,
        "stage1_share": s1 / (s1 + s2) if (s1 + s2) else None,
        "solved": len(solved), "unsolved": len(uns),
        "solved_vs_unsolved": {a: _mw([x[a] for x in solved], [x[a] for x in uns])
                               for a in ("n_correct", "n_input", "n_sources", "coverage")},
        "rescue_fraction": {f: _mean_sem([x["rescue"][f] for x in rows if x["rescue"][f] == x["rescue"][f]])
                            for f in RESCUE},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage1-w-text", type=float, default=0.3)
    ap.add_argument("--stage1-w-var", type=float, default=0.7)
    ap.add_argument("--hard-neg", action="store_true", help="訓練の不正解を点数の高い順に選ぶ（C と同じ訓練にする）")
    a = ap.parse_args()
    import torch
    from analyze_stage1_n1 import load_setting_A
    from set_aware_reranker import run_mode
    from two_stage_query_conditioned import in_vars, out_vars

    D = load_setting_A()
    cases, ek, et, ev, ed, es, cl, cs = (D[k] for k in ("cases", "ek", "et", "ev", "ed", "es", "cl", "cs"))
    holder = {}

    def hook(ctx):
        recs = []
        with torch.no_grad():
            for ci in ctx["te"]:
                cands = ctx["get_cands"](ci)
                feats = np.asarray(ctx["get_feats"](ci, cands), dtype=np.float32)
                corr = set(cl[ci])
                recs.append({"cands": cands, "feats": feats, "corr": corr,
                             "scores": ctx["model"](torch.tensor(feats)).numpy().ravel(),
                             "attrs": {"case_id": cases[ci].get("case_id"), "n_correct": len(corr),
                                       "n_input": len(in_vars(cases[ci])), "n_output": len(out_vars(cases[ci])),
                                       "n_sources": len({ek[j].split("__")[0] for j in corr})}})
        holder["recs"] = recs

    run_mode("reranker-10S", [a.seed], cases, ek, et, ev, ed, es, cl, cs, a.top_k, 15, 1e-3,
             split_mode="stratified", stage1_w_text=a.stage1_w_text, stage1_w_var=a.stage1_w_var,
             fast=True, hard_neg=a.hard_neg, on_trained=hook)
    recs = holder["recs"]
    out = {"config": {**vars(a), "setting": "A", "mode": "reranker-10S", "n_test": len(recs)},
           "top20": outcomes(recs, lambda r: 20),
           "topK": outcomes(recs, lambda r: len(r["corr"]))}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"case_outcomes_{a.tag}.json"
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {p}")
    for key in ("top20", "topK"):
        o = out[key]
        print(f"[{key}] 正解式 {o['n_gt']} 本中、漏れ {o['missed_total']}（第 1 段 {o['stage1_miss']}・第 2 段 {o['stage2_miss']}）"
              f"、全部入ったケース {o['solved']}/{o['n_cases']}")
        for att, v in o["solved_vs_unsolved"].items():
            print(f"   {att:10s} 全部 {v['mean_solved']:.2f} / 漏れあり {v['mean_unsolved']:.2f}  p={v['p']:.2g}")
        print("   救済: " + "  ".join(f"{f} {v['mean']:.3f}" for f, v in o["rescue_fraction"].items()))


if __name__ == "__main__":
    main()
