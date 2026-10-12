"""8/31 報告の改訂（加藤先生の注釈 78 件への対応、2026-10-12）で使う数値をまとめて JSON に書き出す.

出力: experiments/kato_0831_followup_stats.json
  1. stage1_B_w30-70   : 設定 B で第 1 段の重み 0.3/0.7 の提案手法（experiments/xs1_B/）の
                          Recall@K・MAP・Recall@20（seed 平均±sd）と、重み 0.7/0.3（strat_B.json）および
                          最良重みの baseline（external_baselines_B.json の tfidf__none__w40-60）との対応のある検定。
  2. vs_best_baseline   : 提案手法（0.7/0.3 と 0.3/0.7）と最良重みの baseline（0.4/0.6）の検定（設定 A・B、3 指標）。
  3. bucket_A           : 設定 A の正解式数・入力変数数ごとの Recall@K（提案手法 0.7/0.3、baseline 0.4/0.6
                          （experiments/xk50/baseline40_A__*.json）、baseline 0.7/0.3）。10 分割の per-case を合併した平均。
  5. map_standard       : 表 3 の各行の MAP を標準の定義（K で割り、並べた中にない正解式は 0）で数え直した値と、実装の値。
  4. gt_checks          : 評価 1,823 件の正解集合が Kato and Kano (2025) の要件 1〜3 を満たすか、式と未知変数の
                          完全マッチングがあるか、正解式に「同じ文字列（空白などを除く）の式が別出典にある」ものの数。
"""
from __future__ import annotations

import ast
import collections
import glob
import json
import re
import statistics as st
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.stats import wilcoxon

ROOT = Path(__file__).parent
EXP = ROOT / "experiments"
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
METRICS = ["Recall@K_correct", "MAP", "Recall@20"]


def _val(x):
    return x["mean"] if isinstance(x, dict) else x


def per_seed_runs(pattern, mode="reranker-10S"):
    out = {}
    for f in glob.glob(str(pattern)):
        seed = int(re.search(r"__(\d+)\.json$", f).group(1))
        r = json.load(open(f))["results"][mode]
        out[seed] = {m: _val(r[m]) for m in METRICS}
    return out


def per_seed_strat(setting, mode):
    r = json.load(open(EXP / f"strat_{setting}.json"))["results"][mode]
    return {p["seed"]: {m: p[m] for m in METRICS} for p in r["per_seed"]}


def per_seed_ext(setting, label):
    r = json.load(open(EXP / f"external_baselines_{setting}.json"))["results"][label]
    return {p["seed"]: {m: p[m] for m in METRICS} for p in r["per_seed"]}


def per_seed_stage1_A_w3070():
    d = json.load(open(EXP / "stage1_weight_stats.json"))["overall"]["reranker-10S"]["w30-70"]["per_seed"]
    rk = {int(s): v for s, v in d.items()}
    full = per_seed_runs(EXP / "xs1" / "reranker-10S_w30-70__*.json")
    return {s: full[s] for s in rk}


def summary(ps):
    return {m: {"mean": round(st.mean(ps[s][m] for s in SEEDS), 4),
                "std": round(st.stdev(ps[s][m] for s in SEEDS), 4)} for m in METRICS}


def test(a, b):
    res = {}
    for m in METRICS:
        x = [a[s][m] for s in SEEDS]; y = [b[s][m] for s in SEEDS]
        d = [i - j for i, j in zip(x, y)]
        res[m] = {"mean_delta": round(st.mean(d), 4), "wins": sum(v > 0 for v in d),
                  "p_wilcoxon": round(float(wilcoxon(x, y).pvalue), 5)}
    return res


def bucket_A():
    def pc(pattern):
        recs = []
        for f in sorted(glob.glob(str(pattern))):
            r = json.load(open(f))["results"]
            recs += r[list(r)[0]]["per_case"]
        return recs
    d = json.load(open(EXP / "strat_A.json"))["results"]
    groups = {"proposed_w70-30": d["reranker-10S"]["per_case"], "baseline_w70-30": d["baseline"]["per_case"],
              "baseline_w40-60": pc(EXP / "xk50" / "baseline40_A__*.json")}

    def b_in(n):
        return "<=5" if n <= 5 else "6-10" if n <= 10 else "11-15" if n <= 15 else "16-20" if n <= 20 else ">=21"
    out = {}
    for name, fn in (("n_correct", lambda c: str(c["n_correct"])), ("n_input", lambda c: b_in(c["n_input"]))):
        agg = collections.defaultdict(dict)
        for g, recs in groups.items():
            by = collections.defaultdict(list)
            for c in recs:
                by[fn(c)].append(c["Recall@K_correct"])
            for k, v in by.items():
                agg[k][g] = round(float(np.mean(v)), 4)
                agg[k]["n_records"] = len(v)
        out[name] = dict(agg)
    return out


def gt_checks():
    eqs = json.load(open(ROOT / "unified_equations.json"))
    V = {f"{e['source_id']}__{e['eq_id']}": set(e["variables"].keys()) for e in eqs}

    def nz(s):
        return re.sub(r"\\left|\\right|\\,|\\;|\\!|\\quad|\s+", "", s or "")
    lx = {f"{e['source_id']}__{e['eq_id']}": (nz(e["equation"]), e["source_id"]) for e in eqs}
    by_lx = collections.defaultdict(list)
    for k, (s, _) in lx.items():
        by_lx[s].append(k)
    cases = json.load(open(ROOT / "training_cases.json"))
    P = lambda s: ast.literal_eval(s) if isinstance(s, str) else s  # noqa: E731
    stats = collections.defaultdict(collections.Counter)
    gt_all = set()
    for c in cases:
        vt = c["variant_type"]
        fam = ("single" if vt == "original" else "cross" if vt.startswith("multisource")
               else "synth" if vt.startswith("dae_") else None)
        if fam is None:
            continue
        I = set(P(c["input_variables"])); O = set(P(c["output_variables"])); R = P(c["correct_model_ids"])
        gt_all.update(R)
        allv = set().union(*[V[r] for r in R])
        unk = sorted(allv - I)
        occ = collections.Counter(v for r in R for v in V[r])
        s = stats[fam]
        s["n"] += 1
        s["req1_io_included"] += (I | O) <= allv
        s["req2_unknowns_eq_equations"] += len(unk) == len(R)
        s["req3_intermediates_in_2plus"] += all(occ[v] >= 2 for v in allv - (I | O))
        idx = {u: i for i, u in enumerate(unk)}
        rows, cols = [], []
        for i, r in enumerate(R):
            for v in V[r] - I:
                rows.append(i); cols.append(idx[v])
        m = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(R), len(unk)))
        s["perfect_matching"] += int((maximum_bipartite_matching(m, perm_type="column") >= 0).sum() == len(R) == len(unk))
    dup = [g for g in gt_all if any(lx[o][1] != lx[g][1] for o in by_lx[lx[g][0]] if o != g)]
    return {"by_family": {k: dict(v) for k, v in stats.items()},
            "gt_equations": len(gt_all), "identical_latex_in_other_source": len(dup)}


MISS_RANK = 10_000  # evaluate_multi_eq.case_metrics と同じ（候補外の正解式の順位）


def map_standard():
    """MAP を標準の定義（AP = 正解式ごとの順位までの適合率の和 / K、並べた中にない正解式は 0）で数え直す.

    evaluate_multi_eq.case_metrics の AP は「並べた中にある正解式だけ」で平均するため、第 1 段の候補に入らなかった
    正解式を除く提案手法の値が高めに出る。baseline は全式を並べるので、どちらの定義でも同じ値になる。
    """
    def ap(ranks, standard):
        rs = sorted(ranks)
        v = [k / r for k, r in enumerate(rs, 1) if r < MISS_RANK]
        if standard:
            return sum(v) / len(rs) if rs else 0.0
        return sum(v) / len(v) if v else 0.0

    def files(pattern):
        recs = []
        for f in sorted(glob.glob(str(pattern))):
            r = json.load(open(f))["results"]
            recs += r[list(r)[0]]["per_case"]
        return recs
    sA = json.load(open(EXP / "strat_A.json"))["results"]
    sB = json.load(open(EXP / "strat_B_percase.json"))["results"]
    groups = {
        "A": {"baseline_w70-30": sA["baseline"]["per_case"], "baseline_w40-60": files(EXP / "xk50" / "baseline40_A__*.json"),
              "proposed_w70-30": sA["reranker-10S"]["per_case"], "proposed_w30-70": files(EXP / "xs1" / "reranker-10S_w30-70__*.json")},
        "B": {"baseline_w70-30": sB["baseline"]["per_case"], "baseline_w40-60": files(EXP / "xk50" / "baseline40_B__*.json"),
              "proposed_w70-30": sB["reranker-10S"]["per_case"], "proposed_w30-70": files(EXP / "xs1_B" / "reranker-10S_w30-70__*.json")},
    }
    out = {}
    for setting, g in groups.items():
        res, seed_std = {}, {}
        for name, recs in g.items():
            by_std, by_imp = collections.defaultdict(list), collections.defaultdict(list)
            for c in recs:
                by_std[c["seed"]].append(ap(c["ranks"], True))
                by_imp[c["seed"]].append(ap(c["ranks"], False))
            std_vals = [float(np.mean(by_std[s])) for s in SEEDS]
            imp_vals = [float(np.mean(by_imp[s])) for s in SEEDS]
            seed_std[name] = std_vals
            res[name] = {"MAP_standard": {"mean": round(st.mean(std_vals), 4), "std": round(st.stdev(std_vals), 4)},
                         "MAP_implemented": {"mean": round(st.mean(imp_vals), 4), "std": round(st.stdev(imp_vals), 4)}}
        for w in ("proposed_w70-30", "proposed_w30-70"):
            x, y = seed_std[w], seed_std["baseline_w40-60"]
            d = [i - j for i, j in zip(x, y)]
            res[w]["MAP_standard_vs_best_baseline"] = {"mean_delta": round(st.mean(d), 4), "wins": sum(v > 0 for v in d),
                                                       "p_wilcoxon": round(float(wilcoxon(x, y).pvalue), 5)}
        out[setting] = res
    return out


def main():
    out = {}
    pB = per_seed_runs(EXP / "xs1_B" / "reranker-10S_w30-70__*.json")
    out["stage1_B_w30-70"] = {
        "summary": summary(pB),
        "vs_w70-30": test(pB, per_seed_strat("B", "reranker-10S")),
        "vs_best_baseline_w40-60": test(pB, per_seed_ext("B", "tfidf__none__w40-60")),
    }
    pA = per_seed_stage1_A_w3070()
    out["vs_best_baseline"] = {}
    for S, p30 in (("A", pA), ("B", pB)):
        best = per_seed_ext(S, "tfidf__none__w40-60")
        out["vs_best_baseline"][S] = {
            "proposed_w70-30": test(per_seed_strat(S, "reranker-10S"), best),
            "proposed_w30-70": test(p30, best),
        }
    out["bucket_A"] = bucket_A()
    out["gt_checks"] = gt_checks()
    out["map_standard"] = map_standard()
    path = EXP / "kato_0831_followup_stats.json"
    json.dump(out, open(path, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(out["stage1_B_w30-70"]["summary"]), json.dumps(out["gt_checks"]))
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
