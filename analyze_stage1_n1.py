"""加藤先生の指示 3（2026-10-14）：全正解式が第 1 段の候補に入るには N1 がいくつ要るか.

第 1 段の点数 s1 = w_t * 文章類似度 + w_v * 変数の一致度（既定は見直し後の 0.3 / 0.7）で
DB の全 11,146 式を並べ、各ケースの各正解式の順位を求める。同点の式が並ぶときは、同点の中で
最も後ろの順位をとる（点数 0 の正解式は DB の最後になる）。ケースごとの
「全正解式が候補に入る最小の N1」＝正解式の順位の最大値。

出力（experiments/kato1014/n1_coverage.json）:
  - 設定 A（1,823 件）と B（合成 1,000 件）の曲線: N1 ごとの「全正解式が入るケースの割合」と被覆率
  - 全ケースを 100%（と 99% など）にするのに要る N1、種類別（単一出典・出典横断・合成）
  - 下位に残る正解式の一覧（ケース、点数の内訳、変数の重なり）
  - 分割ごと（乱数の種 10 通り、set_aware_reranker と同じ層化分割）に、訓練と検証のケースだけで
    決めた N1 と、その N1 でのテストの被覆率（テストのケースは N1 の決定に使わない）

s1 は DB だけで当てはめた TF-IDF とケースの入出力変数で決まり、乱数の種にも設定にもよらない。
stage1_score_matrix は two_stage_query_conditioned.stage1 と同じ dtype・同じ式で全ケースを一度に計算する
（実データでの一致は run() の中で確かめる）。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "experiments" / "kato1014"
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
SHARES = [0.90, 0.95, 0.99, 0.995, 0.999, 1.0]
REPORT_N1 = [10, 20, 25, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000,
             3000, 5000, 8000, 11146]


# ---------------------------------------------------------------------------
# 数え方（テスト対象）
# ---------------------------------------------------------------------------

def worst_tie_ranks(scores: np.ndarray, idx) -> list[int]:
    """各 idx の順位（1 始まり）。同点の中で最も後ろの順位 = 点数がそれ以上の式の数."""
    srt = np.sort(scores)
    n = len(scores)
    return [int(n - np.searchsorted(srt, scores[j], side="left")) for j in idx]


def best_tie_ranks(scores: np.ndarray, idx) -> list[int]:
    """同点の中で最も前の順位 = 1 + 点数がより高い式の数（参考）."""
    srt = np.sort(scores)
    n = len(scores)
    return [int(n - np.searchsorted(srt, scores[j], side="right") + 1) for j in idx]


def min_n1_full(ranks) -> int:
    """全正解式が上位 N1 件に入る最小の N1."""
    return int(max(ranks))


def jaccard_matrix(query_sets, eq_sets) -> np.ndarray:
    """J[q, e] = |q ∩ e| / |q ∪ e|（和集合が空なら 0）を float32 で返す.

    two_stage_query_conditioned.jaccard を全組に当てて float32 にしたものと同じ値になる
    （整数どうしの割り算は float64 で正しく丸まり、その後 float32 に丸める）。
    """
    vocab = {}
    for s in list(eq_sets) + list(query_sets):
        for v in s:
            vocab.setdefault(v, len(vocab))

    def inc(sets):
        rows, cols = [], []
        for i, s in enumerate(sets):
            for v in s:
                rows.append(i); cols.append(vocab[v])
        data = np.ones(len(rows), dtype=np.int64)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(sets), max(1, len(vocab))))

    Q, E = inc(query_sets), inc(eq_sets)
    inter = (Q @ E.T).toarray().astype(np.int64)
    qn = np.array([len(s) for s in query_sets], dtype=np.int64)[:, None]
    en = np.array([len(s) for s in eq_sets], dtype=np.int64)[None, :]
    union = qn + en - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return J.astype(np.float32)


def stage1_score_matrix(X_ctx, X_eq, query_sets, eq_sets, w_text=0.3, w_var=0.7) -> np.ndarray:
    """全ケース×全式の s1。stage1() の `w_text*ts + w_var*vs` と同じ dtype の計算
    （ts は float64 のコサイン類似度、vs は float32 の Jaccard、和は float64）."""
    from sklearn.metrics.pairwise import cosine_similarity
    ts = cosine_similarity(X_ctx, X_eq)
    vs = jaccard_matrix(query_sets, eq_sets)
    return w_text * ts + w_var * vs


def coverage_curve(ranks_per_case, grid) -> dict:
    """N1 ごとの「全正解式が入るケースの割合」(full) と被覆率のケース平均 (coverage)."""
    mins = np.array([max(r) for r in ranks_per_case])
    full, cov = [], []
    for n1 in grid:
        full.append(float(np.mean(mins <= n1)))
        cov.append(float(np.mean([np.mean(np.asarray(r) <= n1) for r in ranks_per_case])))
    return {"n1": list(grid), "full": full, "coverage": cov}


def n1_for_fraction(min_n1s, share: float) -> int:
    """全正解式が入るケースの割合が share 以上になる最小の N1."""
    a = np.sort(np.asarray(min_n1s))
    need = int(np.ceil(share * len(a) - 1e-9))
    return int(a[max(need, 1) - 1])


# ---------------------------------------------------------------------------
# 実データ
# ---------------------------------------------------------------------------

def family(v: str) -> str:
    if v == "original":
        return "single_source"
    if v.startswith("multisource_"):
        return "cross_source"
    if v.startswith("dae_"):
        return "synthesized"
    return "other"


def load_setting_A():
    """set_aware_reranker.main と同じ手順で DB とケース（設定 A）を作る."""
    from two_stage_query_conditioned import (
        load_equations, load_cases, norm, eq_key, eq_text, eq_vars, case_src,
    )
    from set_aware_reranker import get_src
    eqs = load_equations()
    cases = [c for c in load_cases() if family(str(c.get("variant_type", ""))) != "other"]
    ek, et, ev, ed, es, eq_raw = [], [], [], [], [], []
    for e in eqs:
        k = eq_key(e)
        if not k:
            continue
        ek.append(k); et.append(eq_text(e)); ev.append(eq_vars(e))
        ed.append(norm(e.get("domain") or "")); es.append(get_src(e)); eq_raw.append(e)
    ki = {k: i for i, k in enumerate(ek)}
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki] for c in cases]
    cs = [case_src(c) for c in cases]
    return dict(cases=cases, ek=ek, et=et, ev=ev, ed=ed, es=es, eq_raw=eq_raw, cl=cl, cs=cs)


def split_for(cases, cl, ek, cs, seed):
    """set_aware_reranker.run_mode（split_mode='stratified'）と同じ層化分割."""
    from two_stage_query_conditioned import in_vars, out_vars
    from set_aware_reranker import stratified_src_split
    feats = [(len(cl[i]), len(in_vars(cases[i])), len(out_vars(cases[i])),
              len({ek[j].split("__")[0] for j in cl[i]})) for i in range(len(cases))]
    sp = stratified_src_split(cs, feats, seed)
    te = [i for i, s in enumerate(cs) if s in sp["test"] and cl[i]]
    trv = [i for i, s in enumerate(cs) if (s in sp["train"] or s in sp["val"]) and cl[i]]
    tr = [i for i, s in enumerate(cs) if s in sp["train"] and cl[i]]
    return tr, trv, te


def _grid(n_eq):
    g = sorted(set([int(round(x)) for x in np.logspace(0, np.log10(n_eq), 160)] + REPORT_N1))
    return [x for x in g if 1 <= x <= n_eq]


def run(w_text=0.3, w_var=0.7, out_json=None, check_cases=60):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from two_stage_query_conditioned import case_text, io_vars, stage1, in_vars, out_vars

    D = load_setting_A()
    cases, ek, et, ev, ed, cl, cs = (D[k] for k in ("cases", "ek", "et", "ev", "ed", "cl", "cs"))
    n_eq = len(ek)
    tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = tfidf.fit_transform(et)
    X_ctx = tfidf.transform([case_text(c) for c in cases])
    ios = [io_vars(c) for c in cases]
    from sklearn.metrics.pairwise import cosine_similarity
    TS = cosine_similarity(X_ctx, X_eq)
    VS = jaccard_matrix(ios, ev)
    S = w_text * TS + w_var * VS
    print(f"設定 A {len(cases)} 件 × DB {n_eq} 式、重み ({w_text}, {w_var})")

    # stage1() との一致（実データ、check_cases 件）：点数の配列と上位 200 件の順
    rng = np.random.default_rng(0)
    chk = sorted(rng.choice(len(cases), size=min(check_cases, len(cases)), replace=False).tolist())
    n_same_order = 0
    for ci in chk:
        ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
        vs = np.array([__import__("two_stage_query_conditioned").jaccard(ios[ci], ev[j])
                       for j in range(n_eq)], dtype=np.float32)
        ref = w_text * ts + w_var * vs
        assert np.array_equal(ref, S[ci]), f"s1 mismatch case {ci}"
        want = stage1(ci, X_ctx, X_eq, ios[ci], ev, 200, w_text=w_text, w_var=w_var)
        n_same_order += int(np.argsort(-S[ci])[:200].tolist() == want)
    print(f"stage1() と点数が完全一致: {len(chk)}/{len(chk)} 件、上位 200 件の順が一致: {n_same_order}/{len(chk)} 件")

    per_case = []
    for ci, c in enumerate(cases):
        corr = cl[ci]
        if not corr:
            continue
        sc = S[ci]
        wr = worst_tie_ranks(sc, corr)
        br = best_tie_ranks(sc, corr)
        order = np.argsort(-sc)
        pos = np.empty(n_eq, dtype=np.int64); pos[order] = np.arange(1, n_eq + 1)
        gts = []
        for j, r_w, r_b in zip(corr, wr, br):
            gts.append({
                "eq": ek[j], "rank_worst": r_w, "rank_best": r_b, "rank_argsort": int(pos[j]),
                "s1": float(sc[j]), "text_sim": float(TS[ci, j]), "var_jaccard": float(VS[ci, j]),
                "n_eq_vars": len(ev[j]), "n_shared": len(ev[j] & ios[ci]), "domain": ed[j],
            })
        per_case.append({
            "ci": ci, "case_id": c.get("case_id"), "variant": c.get("variant_type"),
            "family": family(str(c.get("variant_type", ""))), "K": len(corr),
            "n_input": len(in_vars(c)), "n_output": len(out_vars(c)), "n_io": len(ios[ci]),
            "min_n1": min_n1_full(wr), "min_n1_argsort": int(max(g["rank_argsort"] for g in gts)),
            "gt": gts,
        })

    grid = _grid(n_eq)
    out = {"config": {"w_text": w_text, "w_var": w_var, "n_eq": n_eq,
                      "n_cases_A": len(per_case), "tie_rule": "worst (rank = #eq with s1 >= s1(gt))",
                      "stage1_check": {"n_checked": len(chk), "score_equal": len(chk),
                                       "top200_order_equal": n_same_order}},
           "settings": {}, "families": {}, "per_seed": {}, "tail": []}

    def summarize(sub):
        ranks = [[g["rank_worst"] for g in p["gt"]] for p in sub]
        mins = [p["min_n1"] for p in sub]
        cur = coverage_curve(ranks, grid)
        at = coverage_curve(ranks, REPORT_N1)
        ranks_a = [[g["rank_argsort"] for g in p["gt"]] for p in sub]
        at_a = coverage_curve(ranks_a, REPORT_N1)
        zero = sum(1 for p in sub for g in p["gt"] if g["s1"] == 0.0)
        return {
            "n_cases": len(sub), "n_gt": int(sum(len(p["gt"]) for p in sub)),
            "n_gt_zero_score": int(zero),
            "n1_for_share": {str(s): n1_for_fraction(mins, s) for s in SHARES},
            "max_min_n1": int(max(mins)),
            "at": {str(n): {"full": at["full"][i], "coverage": at["coverage"][i],
                            "full_argsort": at_a["full"][i], "coverage_argsort": at_a["coverage"][i]}
                   for i, n in enumerate(REPORT_N1)},
            "curve": cur,
        }

    subsets = {"A": per_case, "B": [p for p in per_case if p["family"] == "synthesized"]}
    for name, sub in subsets.items():
        out["settings"][name] = summarize(sub)
    for fam in ("single_source", "cross_source", "synthesized"):
        out["families"][fam] = summarize([p for p in per_case if p["family"] == fam])

    # 分割ごと：訓練＋検証で N1 を決め、テストの被覆率を見る（テストは決定に使わない）
    for name, sub in subsets.items():
        idx_of = {p["ci"]: p for p in sub}
        cases_s = [cases[p["ci"]] for p in sub]
        cl_s = [cl[p["ci"]] for p in sub]
        cs_s = [cs[p["ci"]] for p in sub]
        rows = []
        for seed in SEEDS:
            tr, trv, te = split_for(cases_s, cl_s, ek, cs_s, seed)
            P = lambda ids: [idx_of[sub[i]["ci"]] for i in ids]
            p_trv, p_te = P(trv), P(te)
            m_trv = [p["min_n1"] for p in p_trv]
            row = {"seed": seed, "n_train": len(tr), "n_trainval": len(trv), "n_test": len(te),
                   "n1_trainval": {str(s): n1_for_fraction(m_trv, s) for s in SHARES},
                   "test_at_trainval_n1": {}}
            for s in SHARES:
                n1 = row["n1_trainval"][str(s)]
                c_te = coverage_curve([[g["rank_worst"] for g in p["gt"]] for p in p_te], [n1])
                row["test_at_trainval_n1"][str(s)] = {"n1": n1, "full": c_te["full"][0],
                                                       "coverage": c_te["coverage"][0]}
            c50 = coverage_curve([[g["rank_argsort"] for g in p["gt"]] for p in p_te], [50])
            row["test_coverage_n1_50_argsort"] = c50["coverage"][0]
            rows.append(row)
        out["per_seed"][name] = rows

    # 下位に残る正解式：全正解式に要る N1 が大きいケース上位 40 件
    tail = sorted(per_case, key=lambda p: -p["min_n1"])[:40]
    for p in tail:
        c = cases[p["ci"]]
        worst = max(p["gt"], key=lambda g: g["rank_worst"])
        j = ek.index(worst["eq"])
        raw = D["eq_raw"][j]
        out["tail"].append({
            "case_id": p["case_id"], "variant": p["variant"], "K": p["K"], "n_io": p["n_io"],
            "min_n1": p["min_n1"], "context": (c.get("context") or "")[:300],
            "inputs": c.get("input_variables"), "outputs": c.get("output_variables"),
            "worst_gt": {**worst, "equation": (raw.get("equation") or "")[:200],
                         "vars": sorted(ev[j]),
                         "context_text": (raw.get("context_text") or "")[:200]},
        })
    out["per_case"] = [{k: v for k, v in p.items() if k != "ci"} for p in per_case]

    out_json = Path(out_json) if out_json else OUT_DIR / f"n1_coverage_w{int(w_text*100):02d}-{int(w_var*100):02d}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Saved: {out_json}")
    for name in ("A", "B"):
        s = out["settings"][name]
        print(f"[{name}] n={s['n_cases']} GT={s['n_gt']} GT点数0={s['n_gt_zero_score']} "
              f"N1(100%)={s['max_min_n1']} N1 by share={s['n1_for_share']}")
        print("   " + "  ".join(f"N1={n}: full {s['at'][str(n)]['full']:.3f} cov {s['at'][str(n)]['coverage']:.4f}"
                                for n in (50, 200, 500, 1000, 2000)))
    return out


def make_figure(in_json=None, out_png="docs/figures/fig_n1_full_coverage_ja.png"):
    """N1 ごとの「全正解式が入るケースの割合」と被覆率（設定 A・B）。判断材料と報告書用."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Hiragino Sans"
    plt.rcParams["axes.unicode_minus"] = False
    d = json.load(open(in_json or OUT_DIR / "n1_coverage_w30-70.json", encoding="utf-8"))
    col = {"A": "#2a78d6", "B": "#eb6834"}
    lab = {"A": "設定 A（1,823 件）", "B": "設定 B（合成 1,000 件）"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, key, title in ((axes[0], "full", "(a) 全正解式が候補に入るケースの割合"),
                           (axes[1], "coverage", "(b) 被覆率（候補に入る正解式の割合）")):
        for S in ("A", "B"):
            cur = d["settings"][S]["curve"]
            ax.plot(cur["n1"], cur[key], color=col[S], lw=2, label=lab[S])
        for x, ls in ((50, ":"), (200, ":"), (1880, "--")):
            ax.axvline(x, color="#888888", ls=ls, lw=1)
        ax.text(1880, 0.705, " 1,880\n（全ケース 100%）", fontsize=8.5, color="#52514e", va="bottom")
        ax.text(50, 0.705, " 50", fontsize=8.5, color="#52514e", va="bottom")
        ax.text(200, 0.705, " 200", fontsize=8.5, color="#52514e", va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(10, d["config"]["n_eq"])
        ax.set_ylim(0.70, 1.005)
        ax.set_xlabel("第 1 段の候補の件数 $N_1$（対数軸）")
        ax.set_ylabel("割合")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="upper left" if key == "full" else "center right")
    fig.tight_layout()
    Path(ROOT / out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / out_png, dpi=150, bbox_inches="tight")
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-text", type=float, default=0.3)
    ap.add_argument("--w-var", type=float, default=0.7)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--figure-only", action="store_true", help="保存済み JSON から図だけ作る")
    a = ap.parse_args()
    if not a.figure_only:
        run(a.w_text, a.w_var, a.output)
    make_figure(a.output)
