"""報告書 §9.3（第 1 段の重みの見直し）の数値を、Recall@20 と MAP（標準）で数え直す（既存の実行のみ、再実行なし）.

§9.3 は N1 = 50（と逐次選択の N1 = 200）で第 1 段の重みを比べた節で、(0.3, 0.7) を選んだ根拠である。
2026-10-14 の指示 4 で主指標を Recall@20 にしたので、同じ実行の per-case から指標だけを数え直す。
入力: experiments/xd/reranker-10S__*.json（w70-30）、experiments/xs1/reranker-10S_{w50-50,w30-70,w00-100}__*.json、
      experiments/xg_A/train__*.json（w70-30、N1 = 200）、experiments/xs1_k200/train_w30-70__*.json、
      experiments/stage1_weight_coverage.json（被覆率）
出力: experiments/kato1014/weights_n50.json
"""
from __future__ import annotations

import json

from kato1014_common import EXP, compare, dump, load_records, summarize

M = ("Recall@20", "MAP", "MAP@20", "Recall@K", "coverage")


def main():
    runs = {"w70-30": load_records(EXP / "xd" / "reranker-10S__*.json"),
            "w50-50": load_records(EXP / "xs1" / "reranker-10S_w50-50__*.json"),
            "w30-70": load_records(EXP / "xs1" / "reranker-10S_w30-70__*.json"),
            "w00-100": load_records(EXP / "xs1" / "reranker-10S_w00-100__*.json")}
    out = {"N1": 50, "setting": "A", "static": {w: summarize(r, n, M) for w, (r, n) in runs.items()},
           "static_vs_w70-30": {w: compare(*runs[w], *runs["w70-30"], M) for w in ("w50-50", "w30-70", "w00-100")},
           "static_vs_w70-30_K>=8": {w: compare(*runs[w], *runs["w70-30"], M, sub=lambda r: r["n_correct"] >= 8)
                                     for w in ("w50-50", "w30-70", "w00-100")}}
    tg = {"w70-30": load_records(EXP / "xg_A" / "train__*.json"),
          "w30-70": load_records(EXP / "xs1_k200" / "train_w30-70__*.json")}
    out["train_greedy_N1_200"] = {w: summarize(r, n, M) for w, (r, n) in tg.items()}
    out["train_greedy_N1_200_w30-70_vs_w70-30"] = compare(*tg["w30-70"], *tg["w70-30"], M)
    out["train_greedy_N1_200_w30-70_vs_w70-30_K>=8"] = compare(*tg["w30-70"], *tg["w70-30"], M,
                                                                sub=lambda r: r["n_correct"] >= 8)
    cov = json.load(open(EXP / "stage1_weight_coverage.json", encoding="utf-8"))["coverage"]
    out["coverage_all_cases"] = {w: {k: cov[w][k]["mean"] for k in ("50", "200")} for w in cov}
    for w, s in out["static"].items():
        print(f"  {w:8s} R@20 {s['Recall@20']['mean']:.4f}±{s['Recall@20']['std']:.4f}  MAP {s['MAP']['mean']:.4f}  "
              f"(R@K {s['Recall@K']['mean']:.4f}, 被覆率 {s['coverage']['mean']:.4f})")
    for w, v in out["static_vs_w70-30"].items():
        print(f"  {w} - w70-30: " + "  ".join(f"{m} {v[m]['mean_delta']:+.4f} ({v[m]['wins']}/10, p={v[m]['p_wilcoxon']:.3g})"
                                         for m in ("Recall@20", "MAP")))
    v = out["train_greedy_N1_200_w30-70_vs_w70-30"]
    print("  逐次（訓練も）N1=200 w30-70 - w70-30: " + "  ".join(
        f"{m} {v[m]['mean_delta']:+.4f} ({v[m]['wins']}/10, p={v[m]['p_wilcoxon']:.3g})" for m in ("Recall@20", "MAP")))
    dump(out, "weights_n50.json")


if __name__ == "__main__":
    main()
