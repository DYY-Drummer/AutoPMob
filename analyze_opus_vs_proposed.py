"""Claude Opus 5.5 が提案手法を上回ったケースの特徴（LLM との比較の本実行の分析。API は呼ばない）.

比べ方：ケースごとに、Opus は 3 回の平均、提案手法はそのケースがテストに入る分割の平均。
Recall@20・MAP@20 がそれより高いケースを「上回った」とする。主は設定 A、設定 B で同じ傾向かを確かめる。

使うもの
- results_claude-opus-5-5__eqd__ex-none__high__v4.json（evaluate3 の出力。上位 20 件の式もある）
- proposed_top20/{A,B}__{種}.json（dump_proposed_top20.py。提案手法の上位 20 件。既存の出力と順位が一致）
- 第 1 段の順位（TF-IDF の文章類似度 0.3 ＋ 変数の Jaccard 0.7。分割によらないのでここで計算し直す）

出力: experiments/llm_db_baseline/opus_vs_proposed.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_llm_db_baseline as rl  # noqa: E402
from two_stage_query_conditioned import (  # noqa: E402
    load_equations, eq_key, eq_text, eq_vars, case_text, io_vars, jaccard, norm,
)
from set_aware_reranker import get_src, latex_key  # noqa: E402

OUT = rl.OUT_DIR / "opus_vs_proposed.json"
TOP = ROOT / "experiments" / "llm_db_baseline" / "proposed_top20"
W_TEXT, W_VAR = 0.3, 0.7          # 選ばれた第 1 段の重み（selected.json：全分割 w30-70）
EPS = 1e-9
# 同じ教科書が別々の出典として DB に 2 回以上入っているもの（式の LaTeX の重なりで確かめた）
DUP_BOOKS = {"SEBORG_Process_Dynamic_and_Control_Chap_2.pdf", "SEBORG_3rd_Edition_Process_Dynamics_and.pdf",
             "2018_Book_ProcessControl.pdf", "2018_Book_ProcessControl_Chapter1.pdf",
             "2018_Book_ProcessControl_Chapter1_6.pdf", "9781852337766-c1.pdf"}


def mean(x) -> float:
    x = list(x)
    return float(np.mean(x)) if x else float("nan")


def cmp(o: float, p: float) -> str:
    return "win" if o > p + EPS else ("loss" if o < p - EPS else "tie")


def load_top(setting: str) -> dict:
    out = defaultdict(dict)
    for f in sorted(TOP.glob(f"{setting}__*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        assert d["rank_mismatch"] == 0
        for c, v in d["cases"].items():
            out[c][d["seed"]] = v
    return out


def main() -> None:
    cases, splits = rl.load_case_index(), rl.load_splits()
    ek, et, ev, ed, es, ex = [], [], [], [], [], []
    for e in load_equations():
        k = eq_key(e)
        if not k:
            continue
        ek.append(k); et.append(eq_text(e)); ev.append(eq_vars(e))
        ed.append(norm(e.get("domain") or "")); es.append(get_src(e)); ex.append(latex_key(e.get("equation")))
    ki = {k: i for i, k in enumerate(ek)}
    lat_src = defaultdict(set)
    for j, x in enumerate(ex):
        lat_src[x].add(es[j])

    opus = json.loads((rl.OUT_DIR / "results_claude-opus-5-5__eqd__ex-none__high__v4.json").read_text(encoding="utf-8"))
    oby = defaultdict(dict)
    for r in opus["per_case"]:
        oby[r["case_id"]][int(r.get("run") or 1)] = r

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = tfidf.fit_transform(et)

    result = {"definition": "case-level: Opus mean over 3 runs vs proposed mean over the splits in which the case is a test case",
              "settings": {}}
    for setting in ("A", "B"):
        top = load_top(setting)
        ids = [c for c in cases if c in top]
        X_ctx = tfidf.transform([case_text(cases[c]) for c in ids])
        rows = []
        for qi, c in enumerate(ids):
            io = io_vars(cases[c])
            corr = list(dict.fromkeys(ki[norm(x)] for x in cases[c]["correct_model_ids"] if norm(x) in ki))
            ts = cosine_similarity(X_ctx[qi], X_eq).ravel()
            vs = np.array([jaccard(io, ev[j]) for j in range(len(ek))], dtype=np.float32)
            order = np.argsort(-(W_TEXT * ts + W_VAR * vs))
            pos = np.empty(len(ek), dtype=np.int64); pos[order] = np.arange(1, len(ek) + 1)
            prop_lists = [v["top"] for v in top[c].values()]
            pm = [rl.metrics_from_ranks(v["ranks"]) for v in top[c].values()]
            o = [oby[c][k] for k in (1, 2, 3)]
            corr_keys = {ek[j] for j in corr}
            clat = {ex[j] for j in corr}
            eqs = []
            for j in corr:
                eqs.append({"key": ek[j], "jac": float(vs[j]), "ts": float(ts[j]), "stage1_rank": int(pos[j]),
                            "n_vars": len(ev[j]), "twin": len(lat_src[ex[j]] - {es[j]}) > 0,
                            "prop_found": mean(ek[j] in l for l in prop_lists),
                            "opus_found": mean(ek[j] in (r["ranked_keys"] or []) for r in o)})

            def wrong(lists):
                st = defaultdict(list)
                for l in lists:
                    nw = nall = ntw = 0
                    for k in l:
                        if k is None or k in corr_keys:
                            continue
                        if ex[ki[k]] in clat:
                            ntw += 1
                            continue
                        nw += 1
                        v = ev[ki[k]]
                        nall += bool(v) and v <= io
                    st["wrong"].append(nw); st["all_vars_in_io"].append(nall); st["twin"].append(ntw)
                return {k: mean(v) for k, v in st.items()}

            def rank1(lists):
                cnt = Counter()
                for l in lists:
                    if not l:
                        continue
                    k = l[0]
                    if k is None:
                        cnt["invalid"] += 1
                    elif k in corr_keys:
                        cnt["correct"] += 1
                    elif ex[ki[k]] in clat:
                        cnt["twin"] += 1
                    elif ev[ki[k]] and ev[ki[k]] <= io:
                        cnt["wrong_all_vars_in_io"] += 1
                    else:
                        cnt["wrong_other"] += 1
                return cnt

            main_src = Counter(es[j] for j in corr).most_common(1)[0][0]
            rows.append({
                "case_id": c, "kind": rl.case_kind(cases[c]["variant_type"]), "K": len(corr), "n_io": len(io),
                "dup_book": main_src in DUP_BOOKS, "eqs": eqs,
                "opus_R": mean(r["recall@20"] for r in o), "opus_AP": mean(r["AP@20"] for r in o),
                "opus_R_min": min(r["recall@20"] for r in o), "opus_AP_min": min(r["AP@20"] for r in o),
                "prop_R": mean(x["recall@20"] for x in pm), "prop_AP": mean(x["AP@20"] for x in pm),
                "prop_R_max": max(x["recall@20"] for x in pm), "prop_AP_max": max(x["AP@20"] for x in pm),
                "wrong_prop": wrong(prop_lists), "wrong_opus": wrong([r["ranked_keys"] or [] for r in o]),
                "rank1_prop": rank1(prop_lists), "rank1_opus": rank1([r["ranked_keys"] or [] for r in o]),
            })
        S = {"n_cases": len(rows)}
        for met in ("R", "AP"):
            g = {r["case_id"]: cmp(r[f"opus_{met}"], r[f"prop_{met}"]) for r in rows}
            S[f"{met}_groups"] = dict(Counter(g.values()))
            S[f"{met}_groups_by_kind"] = {kd: dict(Counter(g[r["case_id"]] for r in rows if r["kind"] == kd))
                                          for kd in ("single", "cross", "synth")}
            S[f"{met}_robust_wins"] = sum(r[f"opus_{met}_min"] > r[f"prop_{met}_max"] + EPS for r in rows)
            for r in rows:
                r[f"g_{met}"] = g[r["case_id"]]
        win = [r for r in rows if r["g_R"] == "win"]
        S["R_win_K"] = dict(sorted(Counter(r["K"] for r in win).items()))
        S["R_win_from_dup_books"] = sum(r["dup_book"] for r in win)
        S["R_win_means"] = {k: mean(r[k] for r in win) for k in ("opus_R", "prop_R", "opus_AP", "prop_AP")}
        big = [r for r in rows if r["kind"] == "synth" and r["K"] >= 7]
        S["synth_K7plus"] = {name: {"n": len(s), "R_win_rate": mean(r["g_R"] == "win" for r in s),
                                    "R_loss_rate": mean(r["g_R"] == "loss" for r in s),
                                    "prop_R": mean(r["prop_R"] for r in s), "opus_R": mean(r["opus_R"] for r in s),
                                    "twins_per_list_prop": mean(r["wrong_prop"]["twin"] for r in s),
                                    "twins_per_list_opus": mean(r["wrong_opus"]["twin"] for r in s)}
                             for name, s in (("dup_books", [r for r in big if r["dup_book"]]),
                                             ("other_sources", [r for r in big if not r["dup_book"]]))}
        # 上回ったケースの正解式：Opus だけが上位 20 件に入れた（3 回中 2 回以上・提案手法は分割の半分未満）式
        E = [e for r in win for e in r["eqs"]]
        groups = {"both": [e for e in E if e["prop_found"] >= .5 and e["opus_found"] >= .5],
                  "only_opus": [e for e in E if e["prop_found"] < .5 and e["opus_found"] >= .5],
                  "only_proposed": [e for e in E if e["prop_found"] >= .5 and e["opus_found"] < .5],
                  "neither": [e for e in E if e["prop_found"] < .5 and e["opus_found"] < .5]}
        S["R_win_correct_eqs"] = {k: {"n": len(v), "jaccard": mean(e["jac"] for e in v),
                                      "text_sim": mean(e["ts"] for e in v),
                                      "stage1_rank_median": float(np.median([e["stage1_rank"] for e in v])) if v else None,
                                      "n_vars": mean(e["n_vars"] for e in v), "has_twin": mean(e["twin"] for e in v)}
                                  for k, v in groups.items()}
        S["wrong_items_per_list_synth_K7plus"] = {
            g: {who: {k: mean(r[f"wrong_{who}"][k] for r in big if r["g_R"] == g) for k in ("wrong", "all_vars_in_io", "twin")}
                for who in ("prop", "opus")} for g in ("win", "tie", "loss")}
        cross = [r for r in rows if r["kind"] == "cross"]
        if cross:
            mj = np.array([min(e["jac"] for e in r["eqs"]) for r in cross])
            qs = np.quantile(mj, [0, 1 / 3, 2 / 3, 1])
            terc = []
            for i in range(3):
                s = [r for r, x in zip(cross, mj) if qs[i] <= x < qs[i + 1] + (1e-9 if i == 2 else 0)]
                terc.append({"min_jaccard": [float(qs[i]), float(qs[i + 1])], "n": len(s),
                             "AP_win_rate": mean(r["g_AP"] == "win" for r in s),
                             "opus_AP": mean(r["opus_AP"] for r in s), "prop_AP": mean(r["prop_AP"] for r in s)})
            S["cross_AP_by_min_jaccard_tercile"] = terc
        for kd in ("cross", "synth"):
            aw = [r for r in rows if r["kind"] == kd and r["g_AP"] == "win"]
            if aw:
                cp, co = Counter(), Counter()
                for r in aw:
                    cp += r["rank1_prop"]; co += r["rank1_opus"]
                S[f"AP_win_rank1_{kd}"] = {"prop": {k: v / sum(cp.values()) for k, v in cp.most_common()},
                                           "opus": {k: v / sum(co.values()) for k, v in co.most_common()}}
        result["settings"][setting] = S
        print(f"setting {setting}: {json.dumps({k: S[k] for k in ('n_cases', 'R_groups', 'AP_groups', 'R_robust_wins', 'R_win_from_dup_books')}, ensure_ascii=False)}")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
