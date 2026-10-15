"""指示 4（2026-10-14）：古典的情報検索手法（baseline）の重みを Recall@20 で選び直し、MAP を標準の定義で出す.

入力: experiments/kato1014/raw/baselines_{A,B}.json
      （external_baselines.py --scorers tfidf:none,bm25:minmax,e5:minmax --save-per-case の出力。
       学習しないので再実行は数分。集計値は experiments/external_baselines_{A,B}.json と同じになる）
出力: experiments/kato1014/baselines_stats.json
  - weights[setting][scorer][w]: Recall@20・MAP（標準）・Recall@K（旧主指標、対応表用）の 10 分割の平均±標準偏差
  - best[setting][scorer]: Recall@20 の 10 分割の平均が最も高い重み
  - check: 既存の集計 JSON の Recall@20 と一致するか
"""
from __future__ import annotations

import json

from kato1014_common import EXP, N_EQ, dump, seed_means, seed_summary

SCORERS = {"tfidf": "tfidf__none", "bm25": "bm25__minmax", "e5": "e5__minmax"}
METRICS = ("Recall@20", "MAP", "MAP@20", "Recall@K")


def main():
    out = {"source": "experiments/kato1014/raw/baselines_{A,B}.json", "n_list": N_EQ,
           "weights": {}, "best": {}, "check": {}}
    for S in ("A", "B"):
        d = json.load(open(EXP / "kato1014" / "raw" / f"baselines_{S}.json", encoding="utf-8"))["results"]
        old = json.load(open(EXP / f"external_baselines_{S}.json", encoding="utf-8"))["results"]
        out["weights"][S], out["best"][S], out["check"][S] = {}, {}, {}
        for sc, prefix in SCORERS.items():
            rows = {}
            for lab, r in d.items():
                if not lab.startswith(prefix + "__"):
                    continue
                w = lab.split("__")[-1]
                recs = r["per_case"]
                rows[w] = {m: seed_summary(seed_means(recs, m, N_EQ)) for m in METRICS}
                rows[w]["w_text"], rows[w]["w_var"] = r["w_text"], r["w_var"]
                same = abs(rows[w]["Recall@20"]["mean"] - old[lab]["Recall@20"]["mean"]) < 6e-5
                out["check"][S][lab] = same
            best = max(rows, key=lambda w: rows[w]["Recall@20"]["mean"])
            out["weights"][S][sc] = rows
            out["best"][S][sc] = {"w": best, "w_text": rows[best]["w_text"], "w_var": rows[best]["w_var"],
                                  **{m: {k: rows[best][m][k] for k in ("mean", "std")} for m in METRICS}}
        print(f"[{S}] 既存の集計と Recall@20 が一致: {sum(out['check'][S].values())}/{len(out['check'][S])}")
        for sc in SCORERS:
            b = out["best"][S][sc]
            print(f"  {sc:6s} 最良 {b['w']}  R@20 {b['Recall@20']['mean']:.4f}±{b['Recall@20']['std']:.4f}  "
                  f"MAP {b['MAP']['mean']:.4f}±{b['MAP']['std']:.4f}")
    dump(out, "baselines_stats.json")


if __name__ == "__main__":
    main()
