"""2026-10-20：91 件を外した後の、第 1 段の重みごと・分割ごとの N1（訓練と検証のケースで全正解式が入る値）.

決め方（2026-10-14 と同じ）：分割ごとに、訓練と検証のケースの「全正解式が候補に入る最小の N1」（同点は最も後ろの順位）の
最大値を求め、千の位に切り上げる。テストのケースは使わない。第 1 段の重みの候補 4 通りのそれぞれについて求める
（提案手法の第 1 段の重みは検証用のケースで選ぶので、どの重みが選ばれても N1 がその重みの規則で決まるようにする）。

出力: experiments/kato1020/n1_rule.json
  settings[A|B][seed][wlabel] = {"n1_trainval_max", "N1", "test_full_at_N1", "test_coverage_at_N1"}
  all_cases[A|B][wlabel] = 全ケースで全正解式が入る N1（参考、テストを含む）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_stage1_n1 import SEEDS, coverage_curve, jaccard_matrix, load_setting_A, split_for, worst_tie_ranks

ROOT = Path(__file__).parent
OUT = ROOT / "experiments" / "kato1020" / "n1_rule.json"
WEIGHTS = {"w70-30": (0.7, 0.3), "w50-50": (0.5, 0.5), "w30-70": (0.3, 0.7), "w00-100": (0.0, 1.0)}


def round_up_thousand(n: int, n_eq: int | None = None) -> int:
    v = int(-(-int(n) // 1000) * 1000)
    return min(v, n_eq) if n_eq else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default=str(ROOT / "experiments" / "kato1020" / "excluded_cases.json"))
    a = ap.parse_args()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from two_stage_query_conditioned import case_text, io_vars

    D = load_setting_A(a.exclude)
    cases, ek, et, ev, cl, cs = (D[k] for k in ("cases", "ek", "et", "ev", "cl", "cs"))
    n_eq = len(ek)
    tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = tfidf.fit_transform(et)
    TS = cosine_similarity(tfidf.transform([case_text(c) for c in cases]), X_eq)
    VS = jaccard_matrix([io_vars(c) for c in cases], ev)
    fam_B = [i for i, c in enumerate(cases) if str(c.get("variant_type", "")).startswith("dae_")]
    print(f"設定 A {len(cases)} 件（外したあと）、設定 B {len(fam_B)} 件")

    out = {"exclude": a.exclude, "rule": "分割ごとに訓練＋検証のケースで全正解式が入る N1 の最大値を千の位に切り上げ（DB の式数が上限）",
           "weights": {k: list(v) for k, v in WEIGHTS.items()}, "n_eq": n_eq,
           "n_cases": {"A": len(cases), "B": len(fam_B)}, "settings": {"A": {}, "B": {}}, "all_cases": {"A": {}, "B": {}}}
    for wl, (wt, wv) in WEIGHTS.items():
        S = wt * TS + wv * VS
        mins = {}
        ranks = {}
        for ci in range(len(cases)):
            if not cl[ci]:
                continue
            r = worst_tie_ranks(S[ci], cl[ci])
            ranks[ci] = r
            mins[ci] = max(r)
        for setting, idx in (("A", list(range(len(cases)))), ("B", fam_B)):
            sub_cases = [cases[i] for i in idx]
            sub_cl = [cl[i] for i in idx]
            sub_cs = [cs[i] for i in idx]
            out["all_cases"][setting][wl] = int(max(mins[i] for i in idx if i in mins))
            for seed in SEEDS:
                tr, trv, te = split_for(sub_cases, sub_cl, ek, sub_cs, seed)
                n_trv = max(mins[idx[i]] for i in trv)
                N1 = round_up_thousand(n_trv, n_eq)
                cur = coverage_curve([ranks[idx[i]] for i in te], [N1])
                out["settings"][setting].setdefault(str(seed), {})[wl] = {
                    "n1_trainval_max": int(n_trv), "N1": int(N1),
                    "test_full_at_N1": cur["full"][0], "test_coverage_at_N1": cur["coverage"][0],
                    "n_train": len(tr), "n_trainval": len(trv), "n_test": len(te)}
        print(f"  {wl}: 全ケース A {out['all_cases']['A'][wl]:,} / B {out['all_cases']['B'][wl]:,}  "
              f"N1（分割ごと, A）{sorted({v[wl]['N1'] for v in out['settings']['A'].values()})}  "
              f"B {sorted({v[wl]['N1'] for v in out['settings']['B'].values()})}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
