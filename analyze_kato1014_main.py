"""C（2026-10-14 の指示 3・4）：主結果（報告書 §4 の表、付録 A の BM25・E5、付録 B の内訳）を新しい設定で.

新しい設定 = 第 1 段の重み (0.3, 0.7)、候補数 N1 は指示 3 の実験で決めた値（--rundir の実行に入っている）。
指標は Recall@20（主指標）と MAP（標準の定義）。旧主指標 Recall@K と被覆率は対応表のためだけに出す。

入力:
  --rundir の {static,infer,train}_{A,B}__{seed}.json（run_kato1014.sh の出力）
  experiments/kato1014/raw/baselines_{A,B}.json（external_baselines.py --save-per-case）
  experiments/kato1014/baselines_stats.json（Recall@20 で選び直した最良の重み）
  N1 = 50 の旧版（参照）: experiments/xs1/reranker-10S_w30-70__*.json、experiments/xs1_B/ 同
出力: experiments/kato1014/main_<tag>.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from kato1014_common import (
    EXP, N_EQ, compare, dump, load_records, summarize,
)

METRICS = ("Recall@20", "MAP", "MAP@20")
REF_METRICS = ("Recall@20", "MAP", "MAP@20", "Recall@K", "coverage")


def baseline_records(setting: str, label: str):
    d = json.load(open(EXP / "kato1014" / "raw" / f"baselines_{setting}.json", encoding="utf-8"))["results"]
    return d[label]["per_case"], N_EQ


def b_in(n: int) -> str:
    return "<=5" if n <= 5 else "6-10" if n <= 10 else "11-15" if n <= 15 else "16-20" if n <= 20 else ">=21"


def buckets(groups: dict, metrics=METRICS) -> dict:
    """正解式の数・入力変数の数ごとの、10 分割のテストのケースを合わせた平均（付録 B と同じ数え方）."""
    from kato1014_common import METRIC_FNS
    out = {}
    for name, key in (("n_correct", lambda c: str(c["n_correct"])), ("n_input", lambda c: b_in(c["n_input"]))):
        agg = defaultdict(dict)
        for g, (recs, n) in groups.items():
            by = defaultdict(list)
            for c in recs:
                by[key(c)].append(c)
            for k, cs in by.items():
                agg[k]["n_records"] = len(cs)
                for m in metrics:
                    agg[k][f"{g}|{m}"] = float(np.mean([METRIC_FNS[m](c, n) for c in cs]))
        out[name] = dict(agg)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rundir", required=True)
    ap.add_argument("--tag", required=True, help="出力名 main_<tag>.json（例: n2000）")
    a = ap.parse_args()

    bstats = json.load(open(EXP / "kato1014" / "baselines_stats.json", encoding="utf-8"))
    out = {"rundir": a.rundir, "metrics": list(METRICS), "table": {}, "tests": {}, "buckets": {}}
    for S in ("A", "B"):
        best = {sc: bstats["best"][S][sc]["w"] for sc in ("tfidf", "bm25", "e5")}
        G = {
            f"baseline_tfidf_{best['tfidf']}": baseline_records(S, f"tfidf__none__{best['tfidf']}"),
            "baseline_tfidf_w40-60": baseline_records(S, "tfidf__none__w40-60"),
            "baseline_tfidf_w70-30": baseline_records(S, "tfidf__none__w70-30"),
            f"baseline_bm25_{best['bm25']}": baseline_records(S, f"bm25__minmax__{best['bm25']}"),
            f"baseline_e5_{best['e5']}": baseline_records(S, f"e5__minmax__{best['e5']}"),
            "proposed_static_N1_50": load_records(
                (EXP / "xs1" / "reranker-10S_w30-70__*.json") if S == "A"
                else (EXP / "xs1_B" / "reranker-10S_w30-70__*.json")),
        }
        for kind in ("static", "infer", "train"):
            recs, n = load_records(f"{a.rundir}/{kind}_{S}__*.json")
            if recs:
                G[f"proposed_{kind}"] = (recs, n)
        out["table"][S] = {g: {"n_list": n, **summarize(r, n, REF_METRICS)} for g, (r, n) in G.items()}
        bb = f"baseline_tfidf_{best['tfidf']}"
        T = {}
        for g in ("proposed_static", "proposed_infer", "proposed_train", "proposed_static_N1_50"):
            if g not in G:
                continue
            T[f"{g} vs {bb}"] = compare(*G[g], *G[bb], METRICS)
            for other in (f"baseline_bm25_{best['bm25']}", f"baseline_e5_{best['e5']}"):
                T[f"{g} vs {other}"] = compare(*G[g], *G[other], METRICS)
        if "proposed_static" in G:
            T["proposed_static vs proposed_static_N1_50"] = compare(
                *G["proposed_static"], *G["proposed_static_N1_50"], REF_METRICS)
        for x, y in (("proposed_infer", "proposed_static"), ("proposed_train", "proposed_infer"),
                     ("proposed_train", "proposed_static")):
            if x in G and y in G:
                T[f"{x} vs {y}"] = compare(*G[x], *G[y], METRICS)
        out["tests"][S] = T
        bg = {k: G[k] for k in (bb, "baseline_tfidf_w70-30", "proposed_static", "proposed_train") if k in G}
        out["buckets"][S] = buckets(bg)
        print(f"== 設定 {S}（最良の baseline {bb}）")
        for g, v in out["table"][S].items():
            print(f"  {g:32s} N1={v['n_list']:6d}  R@20 {v['Recall@20']['mean']:.4f}±{v['Recall@20']['std']:.4f}"
                  f"  MAP {v['MAP']['mean']:.4f}±{v['MAP']['std']:.4f}  (R@K {v['Recall@K']['mean']:.4f})")
        for k, v in T.items():
            print(f"  {k:60s} " + "  ".join(
                f"{m} {v[m]['mean_delta']:+.4f} ({v[m]['wins']}/10, p={v[m]['p_wilcoxon']})" for m in METRICS))
    dump(out, f"main_{a.tag}.json")


if __name__ == "__main__":
    main()
