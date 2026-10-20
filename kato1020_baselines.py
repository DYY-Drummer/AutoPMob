"""2026-10-20：古典的情報検索手法（baseline）の重みを、分割ごとに検証用のケースの Recall@20 で選ぶ（91 件を外した後）.

対象: TF-IDF の文章類似度（正規化なし）と BM25・E5（どちらも全 DB で 0〜1 に正規化）のそれぞれで、重み 12 通り
（external_baselines.py と同じ (1.0, 0.0) 〜 (0.0, 1.0)）。全 DB の式を s = w_t × 文章の点数 + w_v × 変数の一致度 で並べる。
分割は set_aware_reranker.py と同じ層化分割（91 件を外したケースの並びで作る）。検証用のケース = 分割の val の出典のケース。
選び方: 検証用のケースの Recall@20 の平均が最も高い重み。同じなら検証用の MAP、それでも同じなら重みの並びの先のもの。
同じ式（別の出典にある、空白などを除いて同じ LaTeX）を正解とみなした順位 ranks_equiv も数える（set_aware_reranker.equiv_ranks と同じ規則）。

出力: experiments/kato1020/baselines.json（選ばれた重み、テストの値）、
      experiments/kato1020/raw/baselines_percase_{A,B}.json（選ばれた重みのテストのケースの per-case。分析用、非追跡）
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
OUT = ROOT / "experiments" / "kato1020"
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
WEIGHTS = [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7),
           (0.2, 0.8), (0.1, 0.9), (0.05, 0.95), (0.0, 1.0)]
SCORERS = [("tfidf", "none"), ("bm25", "minmax"), ("e5", "minmax")]


def wlabel(wt, wv):
    return f"w{int(round(wt * 100)):02d}-{int(round(wv * 100)):02d}"


def members_by_key(ids):
    m = defaultdict(list)
    for j, k in enumerate(ids):
        m[k].append(j)
    return m


def equiv_from_pos(pos, corr, ids, src, members):
    """全式の順位 pos（1 始まり）から、同じ式を正解とみなした順位（昇順）を求める.

    set_aware_reranker.equiv_ranks を全式の並びに当てたものと同じ（並びを全部たどらずに、正解式と同じ印の式だけを見る）。
    """
    gts = sorted(corr)
    cand = {g: {g} | {o for o in members[ids[g]] if o != g and src[o] != src[g]} for g in gts}
    items = sorted({(int(pos[j]), j) for g in gts for j in cand[g]})
    got = {}
    for p, j in items:
        for g in gts:
            if g not in got and j in cand[g]:
                got[g] = p
                break
    return sorted(got.get(g, 10 ** 9) for g in gts)


def split_ids(cases, cl, ek, cs, seed):
    from two_stage_query_conditioned import in_vars, out_vars
    from set_aware_reranker import stratified_src_split
    feats = [(len(cl[i]), len(in_vars(cases[i])), len(out_vars(cases[i])),
              len({ek[j].split("__")[0] for j in cl[i]})) for i in range(len(cases))]
    sp = stratified_src_split(cs, feats, seed)
    pick = lambda part: [i for i, s in enumerate(cs) if s in sp[part] and cl[i]]  # noqa: E731
    return pick("train"), pick("val"), pick("test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exclude", default=str(OUT / "excluded_cases.json"))
    a = ap.parse_args()
    from external_baselines import jaccard_matrix, load_setting, minmax_rows, text_scores
    from kato1014_common import METRIC_FNS, N_EQ
    from set_aware_reranker import get_src, latex_key
    from two_stage_query_conditioned import eq_key, in_vars, io_vars, load_equations

    eqs = [e for e in load_equations() if eq_key(e)]
    ids = [latex_key(e.get("equation")) for e in eqs]
    src = [get_src(e) for e in eqs]
    members = members_by_key(ids)
    ex = set(json.load(open(a.exclude, encoding="utf-8"))["excluded_case_ids"])
    M = ("Recall@20", "MAP", "MAP@20", "R-Precision")
    doc = {"exclude": a.exclude, "weights": [list(w) for w in WEIGHTS], "selection": "検証用のケースの Recall@20（同じなら MAP）",
           "settings": {}}
    for S in ("A", "B"):
        data = load_setting(S)
        keep = [i for i, c in enumerate(data["cases"]) if c.get("case_id") not in ex]
        cases = [data["cases"][i] for i in keep]
        cl = [data["correct_lists"][i] for i in keep]
        cs = [data["case_srcs"][i] for i in keep]
        ek = data["eq_keys"]
        assert ek == [eq_key(e) for e in eqs]
        J = jaccard_matrix([io_vars(c) for c in cases], data["eq_vars"])
        splits = {s: split_ids(cases, cl, ek, cs, s) for s in SEEDS}
        res = {}
        percase_sel = []
        percase_all = {}
        for scorer, nm in SCORERS:
            T = text_scores(scorer, data["eq_texts"], data["cases"], None)[keep]
            if nm == "minmax":
                T = minmax_rows(T)
            recs = {}
            for wt, wv in WEIGHTS:
                wl = wlabel(wt, wv)
                per = {}
                for ci in range(len(cases)):
                    if not cl[ci]:
                        continue
                    order = np.argsort(-(wt * T[ci] + wv * J[ci]))
                    pos = np.empty(len(order), dtype=np.int64); pos[order] = np.arange(1, len(order) + 1)
                    corr = set(cl[ci])
                    per[ci] = {"case_id": cases[ci].get("case_id"), "variant": cases[ci].get("variant_type"),
                               "n_correct": len(corr), "n_input": len(in_vars(cases[ci])),
                               "ranks": sorted(int(pos[j]) for j in corr),
                               "ranks_equiv": equiv_from_pos(pos, corr, ids, src, members)}
                recs[wl] = per
            # 分割ごとに検証用のケースで選ぶ
            sel, table = {}, {wl: {"val": {m: [] for m in M}, "test": {m: [] for m in M},
                                   "test_equiv": {m: [] for m in M}} for wl in recs}
            for seed in SEEDS:
                tr, va, te = splits[seed]
                best = None
                for k, (wt, wv) in enumerate(WEIGHTS):
                    wl = wlabel(wt, wv)
                    v = {m: float(np.mean([METRIC_FNS[m](recs[wl][ci], N_EQ) for ci in va])) for m in M}
                    t = {m: float(np.mean([METRIC_FNS[m](recs[wl][ci], N_EQ) for ci in te])) for m in M}
                    te_eq = [dict(recs[wl][ci], ranks=recs[wl][ci]["ranks_equiv"]) for ci in te]
                    tq = {m: float(np.mean([METRIC_FNS[m](r, N_EQ) for r in te_eq])) for m in M}
                    for m in M:
                        table[wl]["val"][m].append(v[m]); table[wl]["test"][m].append(t[m]); table[wl]["test_equiv"][m].append(tq[m])
                    key = (v["Recall@20"], v["MAP"], -k)
                    if best is None or key > best[0]:
                        best = (key, wl, v, t, tq)
                _, wl, v, t, tq = best
                sel[str(seed)] = {"w": wl, "val": v, "test": t, "test_equiv": tq,
                                  "n_val": len(va), "n_test": len(te)}
                if scorer == "tfidf":
                    for ci in te:
                        percase_sel.append({"seed": seed, "mode": f"baseline_{wl}", **recs[wl][ci]})
                    # §5 (i) の「提案手法と同じ重みの baseline」のため、TF-IDF は全 12 通りの per-case も残す
                    for wt2, wv2 in WEIGHTS:
                        wl2 = wlabel(wt2, wv2)
                        for part, ids_ in (("val", va), ("test", te)):
                            for ci in ids_:
                                percase_all.setdefault(wl2, []).append({"seed": seed, "split": part, **recs[wl2][ci]})
            summ = lambda key: {m: {"mean": float(np.mean([sel[s][key][m] for s in sel])),  # noqa: E731
                                     "std": float(np.std([sel[s][key][m] for s in sel], ddof=1))} for m in M}
            res[scorer] = {"selected": sel, "summary_test": summ("test"), "summary_test_equiv": summ("test_equiv"),
                           "by_weight": {wl: {part: {m: {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1))}
                                                     for m, x in table[wl][part].items()}
                                              for part in ("val", "test", "test_equiv")} for wl in table}}
            ws = [sel[s]["w"] for s in sel]
            print(f"[{S}] {scorer}: 選ばれた重み {sorted(set(ws), key=ws.count, reverse=True)}（{ {w: ws.count(w) for w in set(ws)} }）  "
                  f"テスト R@20 {res[scorer]['summary_test']['Recall@20']['mean']:.4f}  MAP {res[scorer]['summary_test']['MAP']['mean']:.4f}  "
                  f"（同じ式も正解: R@20 {res[scorer]['summary_test_equiv']['Recall@20']['mean']:.4f}  MAP {res[scorer]['summary_test_equiv']['MAP']['mean']:.4f}）")
        doc["settings"][S] = {"n_cases": len(cases), "scorers": res}
        (OUT / "raw").mkdir(parents=True, exist_ok=True)
        json.dump({"config": {"setting": S, "top_k": N_EQ, "mode": "baseline_selected"},
                   "results": {"baseline": {"n_features": 0, "per_case": percase_sel}}},
                  open(OUT / "raw" / f"baselines_percase_{S}.json", "w", encoding="utf-8"), ensure_ascii=False)
        json.dump({"config": {"setting": S, "top_k": N_EQ, "scorer": "tfidf:none"},
                   "results": {wl: {"n_features": 0, "per_case": v} for wl, v in percase_all.items()}},
                  open(OUT / "raw" / f"baselines_tfidf_all_{S}.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(doc, open(OUT / "baselines.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {OUT / 'baselines.json'}")


if __name__ == "__main__":
    main()
