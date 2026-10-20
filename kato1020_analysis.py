"""2026-10-20 の再実験の集計（91 件を外し、定数を検証用のケースで選んだ設定）.

検定はしない（本人の決定）。2 つの手法の差は、10 通りの分割のうち何通りで上回ったか（下回ったか）と、差の平均 ± 標準偏差で示す。
指標は Recall@20（主）、R-Precision（旧 Recall@K）、MAP（標準）、MAP@20。同じ式を正解とみなした値は ranks_equiv から。

サブコマンド:
  main   : §4 の表（baseline・BM25・E5 は検証用のケースで選んだ重み、提案手法は選んだ設定）、内訳（正解式の数・入力変数の数）、同じ式
  greedy : §6（一括の並べ直し・逐次選択（推論のみ）・逐次選択（訓練も逐次）、全ケースと正解式 8 本以上）
  mech   : §5・§8（アブレーション、集合の特徴量の上乗せの内訳、話題の特徴量、分野ラベルの合併）
  pfi    : §5・§8 の PFI（低下が正の分割の数、平均 ± 標準偏差、x10 の打ち消し合い）と図
  sweep  : 付録の N1 の比較（Recall@20、MAP、時間）
出力: experiments/kato1020/{main,greedy,mech,pfi,sweep}.json
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from kato1014_common import METRIC_FNS, N_EQ, load_records, seed_means

ROOT = Path(__file__).parent
OUT = ROOT / "experiments" / "kato1020"
RAW = OUT / "raw"
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
M = ("Recall@20", "R-Precision", "MAP", "MAP@20")
NIN_BINS = [("1-3", lambda v: v <= 3), ("4-6", lambda v: 4 <= v <= 6), ("7-10", lambda v: 7 <= v <= 10),
            ("11-15", lambda v: 11 <= v <= 15), ("16+", lambda v: v >= 16)]
NC_BINS = [(str(k), (lambda v, k=k: v == k)) for k in range(1, 11)]
SELECTED = "selected.json"   # --selected で案 a/b/c を切り替える
TAG = ""                     # --tag：final・sweep の入力先と出力名の接尾辞


# ---------------------------------------------------------------------------
# 小道具
# ---------------------------------------------------------------------------

def counts(a: dict, b: dict) -> dict:
    """a − b を分割で対にして、上回った・下回った・同じだった分割の数と、差の平均 ± 標準偏差（検定はしない）."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    return {"n": len(seeds), "mean_a": float(np.mean([a[s] for s in seeds])), "mean_b": float(np.mean([b[s] for s in seeds])),
            "mean_delta": float(d.mean()), "std_delta": float(d.std(ddof=1)) if len(d) > 1 else None,
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}


def summ(sm: dict) -> dict:
    v = [sm[s] for s in sorted(sm)]
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else None, "n": len(v),
            "per_seed": {str(s): sm[s] for s in sorted(sm)}}


def test_only(recs):
    return [r for r in recs if r.get("split", "test") == "test"]


def equiv(recs):
    return [dict(r, ranks=r["ranks_equiv"]) for r in recs]


def SM(recs, n, metrics=M, sub=None):
    return {m: seed_means(recs, m, n, sub) for m in metrics}


def selected():
    return json.load(open(OUT / SELECTED, encoding="utf-8"))["selected"]


def FIN():
    return RAW / f"final{TAG}"


def proposed_static(S):
    """各分割で選ばれた (重み, 不正解) の select 実行のテストのケース."""
    sel = selected()[S]
    recs, n_by = [], {}
    for seed in SEEDS:
        c = sel[str(seed)]
        r, n = load_records(RAW / "select" / f"{S}_{c['w']}_{c['neg']}__{seed}.json")
        recs += test_only(r)
        n_by[seed] = n
    return recs, n_by


def seed_means_n(recs, metric, n_by, sub=None):
    """分割ごとに N1 が違ってよい seed_means."""
    by = defaultdict(list)
    for r in recs:
        if sub is not None and not sub(r):
            continue
        by[r["seed"]].append(METRIC_FNS[metric](r, n_by[r["seed"]]))
    return {s: float(np.mean(v)) for s, v in by.items()}


def load_final(pattern):
    recs, n_by = [], {}
    for f in sorted(glob.glob(str(pattern))):
        r, n = load_records(f)
        r = test_only(r)
        recs += r
        for x in r:
            n_by[x["seed"]] = n
    return recs, n_by


def b_in(n):
    return "<=5" if n <= 5 else "6-10" if n <= 10 else "11-15" if n <= 15 else "16-20" if n <= 20 else ">=21"


def dump(obj, name):
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(OUT / name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {OUT / name}")


# ---------------------------------------------------------------------------
# main（§4）
# ---------------------------------------------------------------------------

def cmd_main():
    bl = json.load(open(OUT / "baselines.json", encoding="utf-8"))["settings"]
    out = {"metrics": list(M), "table": {}, "table_equiv": {}, "compare": {}, "buckets": {}, "selected_weights": {}}
    for S in ("A", "B"):
        G, nb = {}, {}
        G["proposed_static"], nb["proposed_static"] = proposed_static(S)
        for k in ("infer", "train"):
            G[f"proposed_{k}"], nb[f"proposed_{k}"] = load_final(FIN() / f"{k}_{S}__*.json")
        b = json.load(open(RAW / f"baselines_percase_{S}.json", encoding="utf-8"))["results"]["baseline"]["per_case"]
        G["baseline_tfidf"], nb["baseline_tfidf"] = b, {s: N_EQ for s in SEEDS}
        T, TQ = {}, {}
        sm_all, smq_all = {}, {}
        for g in G:
            if not G[g]:
                continue
            sm_all[g] = {m: seed_means_n(G[g], m, nb[g]) for m in M}
            smq_all[g] = {m: seed_means_n(equiv(G[g]), m, nb[g]) for m in M}
            T[g] = {m: summ(sm_all[g][m]) for m in M}
            TQ[g] = {m: summ(smq_all[g][m]) for m in M}
        for sc in ("bm25", "e5"):
            sel = bl[S]["scorers"][sc]["selected"]
            sm_all[f"baseline_{sc}"] = {m: {int(s): sel[s]["test"][m] for s in sel} for m in M}
            smq_all[f"baseline_{sc}"] = {m: {int(s): sel[s]["test_equiv"][m] for s in sel} for m in M}
            T[f"baseline_{sc}"] = {m: summ(sm_all[f"baseline_{sc}"][m]) for m in M}
            TQ[f"baseline_{sc}"] = {m: summ(smq_all[f"baseline_{sc}"][m]) for m in M}
        out["selected_weights"][S] = {sc: {s: v["w"] for s, v in bl[S]["scorers"][sc]["selected"].items()}
                                      for sc in ("tfidf", "bm25", "e5")}
        out["table"][S], out["table_equiv"][S] = T, TQ
        C = {}
        for p in ("proposed_static", "proposed_infer", "proposed_train"):
            if p not in sm_all:
                continue
            for bname in ("baseline_tfidf", "baseline_bm25", "baseline_e5"):
                C[f"{p} vs {bname}"] = {m: counts(sm_all[p][m], sm_all[bname][m]) for m in M}
                C[f"{p} vs {bname} (equiv)"] = {m: counts(smq_all[p][m], smq_all[bname][m]) for m in M}
        out["compare"][S] = C
        # 内訳（分割のテストのケースを合わせた平均。付録 B と同じ数え方）
        bk = {}
        for name, key in (("n_correct", lambda c: str(c["n_correct"])), ("n_input", lambda c: b_in(c["n_input"]))):
            agg = defaultdict(dict)
            for g in ("baseline_tfidf", "proposed_static", "proposed_train"):
                if g not in G or not G[g]:
                    continue
                by = defaultdict(list)
                for c in G[g]:
                    by[key(c)].append(c)
                for k, cs in by.items():
                    agg[k]["n_records"] = len(cs)
                    for m in M:
                        agg[k][f"{g}|{m}"] = float(np.mean([METRIC_FNS[m](c, nb[g][c["seed"]]) for c in cs]))
            bk[name] = dict(agg)
        out["buckets"][S] = bk
        print(f"== 設定 {S}")
        for g, v in T.items():
            print(f"  {g:18s} R@20 {v['Recall@20']['mean']:.4f}±{v['Recall@20']['std']:.4f}  R-Prec {v['R-Precision']['mean']:.4f}  "
                  f"MAP {v['MAP']['mean']:.4f}±{v['MAP']['std']:.4f}  MAP@20 {v['MAP@20']['mean']:.4f}  | 同じ式も正解: R@20 "
                  f"{TQ[g]['Recall@20']['mean']:.4f} MAP {TQ[g]['MAP']['mean']:.4f}")
        for k, v in C.items():
            if "(equiv)" in k:
                continue
            print(f"  {k:42s} " + "  ".join(f"{m} {v[m]['mean_delta']:+.4f}（{v[m]['wins']}/{v[m]['n']}）" for m in M))
    dump(out, f"main{TAG}.json")


# ---------------------------------------------------------------------------
# greedy（§6）
# ---------------------------------------------------------------------------

def cmd_greedy():
    out = {"summary": {}, "compare": {}}
    for S in ("A", "B"):
        G, nb = {}, {}
        G["static"], nb["static"] = proposed_static(S)
        for k in ("infer", "train"):
            G[k], nb[k] = load_final(FIN() / f"{k}_{S}__*.json")
        out["summary"][S], out["compare"][S] = {}, {}
        for sub_name, sub in (("all", None), ("K>=8", lambda r: r["n_correct"] >= 8)):
            sm = {k: {m: seed_means_n(G[k], m, nb[k], sub) for m in M} for k in G}
            out["summary"][S][sub_name] = {k: {m: summ(sm[k][m]) for m in M} for k in sm}
            out["compare"][S][sub_name] = {f"{x} - {y}": {m: counts(sm[x][m], sm[y][m]) for m in M}
                                           for x, y in (("infer", "static"), ("train", "infer"), ("train", "static"))}
            s = out["summary"][S][sub_name]
            print(f"[{S} {sub_name}] " + "  ".join(f"{k} R@20 {s[k]['Recall@20']['mean']:.4f} MAP {s[k]['MAP']['mean']:.4f}" for k in s))
            for k, v in out["compare"][S][sub_name].items():
                print(f"    {k:14s} " + "  ".join(f"{m} {v[m]['mean_delta']:+.4f}（{v[m]['wins']}/{v[m]['n']}）" for m in ("Recall@20", "MAP")))
    dump(out, f"greedy{TAG}.json")


# ---------------------------------------------------------------------------
# mech（§5・§8、設定 A）
# ---------------------------------------------------------------------------

def lift(mode_recs, n_m, base_recs, n_b, key, bins, metric):
    rows = []
    for lbl, pred in bins:
        sub = (lambda r, pred=pred: pred(r[key]))
        a = seed_means_n(mode_recs, metric, n_m, sub)
        b = seed_means_n(base_recs, metric, n_b, sub)
        c = counts(a, b) if set(a) & set(b) else None
        rows.append({"bin": lbl, "n_records": sum(1 for r in mode_recs if pred(r[key])), **({} if c is None else c)})
    return rows


def cmd_mech():
    sel = selected()["A"]
    G, nb = {}, {}
    for name in ("reranker-2", "reranker-7", "reranker-2_Spec", "reranker-2_Svd", "reranker-2_Dom",
                 "reranker-7_Comp", "reranker-7_Coh", "reranker-7_Dom"):
        G[name], nb[name] = load_final(FIN() / f"{name}_A__*.json")
    G["reranker-10S"], nb["reranker-10S"] = proposed_static("A")
    for mg in ("orig", "L1", "K100", "K40"):
        G[f"Dom@{mg}"], nb[f"Dom@{mg}"] = load_final(FIN() / "domain_merge" / f"{mg}__*.json")
    # 提案手法と同じ第 1 段の重み（分割ごとに選ばれた重み）の baseline
    allw = json.load(open(RAW / "baselines_tfidf_all_A.json", encoding="utf-8"))["results"]
    G["baseline_same_w"] = [r for wl, v in allw.items() for r in v["per_case"]
                            if r["split"] == "test" and sel[str(r["seed"])]["w"] == wl]
    nb["baseline_same_w"] = {s: N_EQ for s in SEEDS}
    S_ = {g: {m: seed_means_n(G[g], m, nb[g]) for m in M} for g in G if G[g]}
    cmp_ = lambda x, y: ({m: counts(S_[x][m], S_[y][m]) for m in M} if x in S_ and y in S_ else None)  # noqa: E731
    out = {"summary": {g: {m: summ(S_[g][m]) for m in M} for g in S_},
           "steps": {"(i) baseline_same_w -> reranker-2": cmp_("reranker-2", "baseline_same_w"),
                     "(ii) reranker-2 -> reranker-7": cmp_("reranker-7", "reranker-2"),
                     "(ii') reranker-2 -> reranker-2+Spec": cmp_("reranker-2_Spec", "reranker-2"),
                     "(iii) reranker-7 -> reranker-10S": cmp_("reranker-10S", "reranker-7"),
                     "total baseline_same_w -> reranker-10S": cmp_("reranker-10S", "baseline_same_w")},
           "single_set_additions": {x: cmp_(x, "reranker-7") for x in ("reranker-7_Comp", "reranker-7_Coh", "reranker-7_Dom")},
           "topic_additions": {"x7": cmp_("reranker-2_Dom", "reranker-2"), "x3": cmp_("reranker-2_Svd", "reranker-2"),
                               "x10": cmp_("reranker-7_Dom", "reranker-7")},
           "domain_merge": {mg: cmp_(f"Dom@{mg}", "reranker-2") for mg in ("L1", "K100", "K40")},
           "lift_by_n_correct": {}, "lift_by_n_input": {}}
    if G.get("Dom@orig") and G.get("reranker-2_Dom"):
        key = lambda r: (r["seed"], r["case_id"])  # noqa: E731
        A_ = {key(r): r["ranks"] for r in G["Dom@orig"]}
        B_ = {key(r): r["ranks"] for r in G["reranker-2_Dom"]}
        out["domain_merge"]["orig_equals_reranker-2_Dom"] = {"n": len(A_), "identical_ranks": sum(1 for k in A_ if B_.get(k) == A_[k])}
    for mode in ("reranker-7_Comp", "reranker-7_Coh", "reranker-7_Dom", "reranker-10S"):
        if mode in G and G[mode] and G["reranker-7"]:
            out["lift_by_n_correct"][mode] = {m: lift(G[mode], nb[mode], G["reranker-7"], nb["reranker-7"], "n_correct", NC_BINS, m) for m in M}
            out["lift_by_n_input"][mode] = {m: lift(G[mode], nb[mode], G["reranker-7"], nb["reranker-7"], "n_input", NIN_BINS, m) for m in M}
    for g, v in out["summary"].items():
        print(f"  {g:18s} R@20 {v['Recall@20']['mean']:.4f}±{v['Recall@20']['std']:.4f}  MAP {v['MAP']['mean']:.4f}")
    for sect in ("steps", "single_set_additions", "topic_additions", "domain_merge"):
        for k, v in out[sect].items():
            if v and "Recall@20" in v:
                print(f"  [{sect}] {k:40s} " + "  ".join(f"{m} {v[m]['mean_delta']:+.4f}（{v[m]['wins']}/{v[m]['n']}）" for m in ("Recall@20", "MAP")))
    dump(out, f"mech{TAG}.json")


# ---------------------------------------------------------------------------
# pfi
# ---------------------------------------------------------------------------

def cmd_pfi():
    import analyze_kato1014_pfi as P
    files = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(str(FIN() / "pfi" / "pfi_A__*.json")))]
    seeds = [d["pfi"]["seed"] for d in files]
    mets = ("Recall@20", "MAP", "MAP@20", "Recall@K")
    out = {"seeds": seeds, "n_perm": files[0]["config"]["n_perm"], "base": {}, "targets": {}, "cancellation": {}}
    for m in mets:
        b = [d["pfi"]["base"][m] for d in files]
        out["base"][m] = {"mean": float(np.mean(b)), "std": float(np.std(b, ddof=1))}
    for lab in files[0]["pfi"]["targets"]:
        out["targets"][lab] = {}
        for m in mets:
            drops = np.array([d["pfi"]["base"][m] - d["pfi"]["targets"][lab]["agg_perm_mean"][m] for d in files])
            out["targets"][lab][m] = {"drop_mean": float(drops.mean()), "drop_std": float(drops.std(ddof=1)),
                                      "n_positive": int((drops > 0).sum()), "n_negative": int((drops < 0).sum()),
                                      "n_splits": len(drops),
                                      "shuffled_mean": float(np.mean([d["pfi"]["targets"][lab]["agg_perm_mean"][m] for d in files]))}
    for f in ("gDom", "domain", "svd_sim", "text_sim"):
        out["cancellation"][f] = {}
        for m in ("Recall@20", "MAP"):
            c = P.cancellation(files, f, m)
            c.pop("mannwhitney_p", None)   # 検定はしない（2026-10-20 本人の決定）
            out["cancellation"][f][m] = c
    dump(out, f"pfi{TAG}.json")
    P.figure(out, f"final{TAG}")
    m = "Recall@20"
    print(f"base {m} {out['base'][m]['mean']:.4f}±{out['base'][m]['std']:.4f}")
    for lab, v in sorted(out["targets"].items(), key=lambda kv: -kv[1][m]["drop_mean"]):
        print(f"  {lab:14s} 低下 {v[m]['drop_mean']:+.4f}±{v[m]['drop_std']:.4f}（正 {v[m]['n_positive']}/{v[m]['n_splits']}）→ {v[m]['shuffled_mean']:.4f}"
              f" | MAP 低下 {v['MAP']['drop_mean']:+.4f}（正 {v['MAP']['n_positive']}/{v['MAP']['n_splits']}）")


# ---------------------------------------------------------------------------
# sweep（付録の N1）
# ---------------------------------------------------------------------------

def cmd_sweep():
    sel = selected()
    out = {"N1": {}, "note": "時間は 1 回の実行（訓練＋評価）の壁時計の時間。6 並列で実行したので、1 本だけで実行するより長い。"}
    for S in ("A", "B"):
        out["N1"][S] = {}
        rows = {}
        for f in sorted(glob.glob(str(RAW / f"sweep{TAG}" / f"static_{S}_N*__*.json"))):
            n1 = int(Path(f).name.split("_N")[1].split("__")[0])
            d = json.load(open(f, encoding="utf-8"))
            r = list(d["results"].values())[0]
            rows.setdefault(n1, {"recs": [], "time_train": [], "time_eval": []})
            rows[n1]["recs"] += test_only(r["per_case"])
            for ps in r["per_seed"]:
                rows[n1]["time_train"].append(ps.get("time_train_sec")); rows[n1]["time_eval"].append(ps.get("time_eval_sec"))
        # 選んだ N1（select の実行）も並べる
        sel_rows = {"recs": [], "time_train": [], "time_eval": []}
        for seed in SEEDS:
            c = sel[S][str(seed)]
            d = json.load(open(RAW / "select" / f"{S}_{c['w']}_{c['neg']}__{seed}.json", encoding="utf-8"))
            r = list(d["results"].values())[0]
            sel_rows["recs"] += test_only(r["per_case"])
            for ps in r["per_seed"]:
                sel_rows["time_train"].append(ps.get("time_train_sec")); sel_rows["time_eval"].append(ps.get("time_eval_sec"))
        n1s_sel = sorted({sel[S][str(s)]["N1"] for s in SEEDS})
        for n1, v in sorted(rows.items()) + [(f"selected({'/'.join(map(str, n1s_sel))})", sel_rows)]:
            n_by = {s: (n1 if isinstance(n1, int) else sel[S][str(s)]["N1"]) for s in SEEDS}
            sm = {m: seed_means_n(v["recs"], m, n_by) for m in M + ("coverage",)}
            tt = [x for x in v["time_train"] if x is not None]; te = [x for x in v["time_eval"] if x is not None]
            tot = [a_ + b_ for a_, b_ in zip(v["time_train"], v["time_eval"]) if a_ is not None and b_ is not None]
            out["N1"][S][str(n1)] = {**{m: summ(sm[m]) for m in sm},
                                     "time_train_sec": {"mean": float(np.mean(tt)), "std": float(np.std(tt, ddof=1))} if tt else None,
                                     "time_eval_sec": {"mean": float(np.mean(te)), "std": float(np.std(te, ddof=1))} if te else None,
                                     "time_total_sec": {"mean": float(np.mean(tot)), "std": float(np.std(tot, ddof=1))} if tot else None}
            x = out["N1"][S][str(n1)]
            print(f"[{S}] N1={str(n1):18s} R@20 {x['Recall@20']['mean']:.4f}  MAP {x['MAP']['mean']:.4f}  被覆率 {x['coverage']['mean']:.4f}  "
                  f"時間 訓練 {x['time_train_sec']['mean'] if x['time_train_sec'] else float('nan'):.0f}s 評価 {x['time_eval_sec']['mean'] if x['time_eval_sec'] else float('nan'):.0f}s")
    # 逐次選択（N1 = 2,000 だけ。§6 の実行）を並べる
    out["sequential_N1_2000"] = {}
    for S in ("A", "B"):
        out["sequential_N1_2000"][S] = {}
        for k in ("infer", "train"):
            recs, n_by = load_final(FIN() / f"{k}_{S}__*.json")
            if not recs:
                continue
            tt, te_ = [], []
            for f in sorted(glob.glob(str(FIN() / f"{k}_{S}__*.json"))):
                for ps in list(json.load(open(f, encoding="utf-8"))["results"].values())[0]["per_seed"]:
                    tt.append(ps.get("time_train_sec")); te_.append(ps.get("time_eval_sec"))
            tot = [a_ + b_ for a_, b_ in zip(tt, te_) if a_ is not None and b_ is not None]
            out["sequential_N1_2000"][S][k] = {**{m: summ(seed_means_n(recs, m, n_by)) for m in M + ("coverage",)},
                                               "time_total_sec": {"mean": float(np.mean(tot)), "std": float(np.std(tot, ddof=1))} if tot else None,
                                               "note": "検証用のケースも評価した実行なので、評価の時間は一括の行より長い"}
    # 確認：N1 = 2,000 の sweep と select（同じ設定）のテストの値が同じか
    for S in ("A", "B"):
        a_, b_ = out["N1"][S].get("2000"), next((v for k, v in out["N1"][S].items() if k.startswith("selected")), None)
        if a_ and b_:
            out.setdefault("check_sweep2000_equals_selected", {})[S] = all(
                abs(a_[m]["mean"] - b_[m]["mean"]) < 1e-12 for m in M)
    dump(out, f"sweep{TAG}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["main", "greedy", "mech", "pfi", "sweep"])
    ap.add_argument("--selected", default="selected.json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    SELECTED, TAG = a.selected, a.tag
    {"main": cmd_main, "greedy": cmd_greedy, "mech": cmd_mech, "pfi": cmd_pfi, "sweep": cmd_sweep}[a.cmd]()
