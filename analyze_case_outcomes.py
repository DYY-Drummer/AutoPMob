"""正解・不正解ケースの分析（最良モデル reranker-10S, seed=42）.

analyze_pfi.train_and_cache と同一手順で 1 seed だけ訓練し、テストケースを
解けたケース / 落ちたケースに分けて次を調べる：

  1. 失敗の分解      : 見逃した正解式が (i) stage1 で候補50件に入らなかったのか
                       (ii) 候補にはいたが再ランクで上位K_c から漏れたのか。
  2. 解けた/落ちた比較: ケース属性（正解式数・入力変数数・ソース数）と
                       特徴の自ケース分離度 sep の差（Mann-Whitney）。
  3. 話題特徴の救済可能性: 再ランク見逃しペア（見逃し正解 m, 誤上位 w）のうち
                       svd_sim(m) > svd_sim(w) となる割合。0.5 なら話題特徴を
                       強めても順位を戻せない。
  4. 選択効果        : 候補内での text_sim と変数系特徴の相関（stage1 の
                       混合選抜が生む Berkson 型の負相関の検証）。
  5. 具体例          : stage1 見逃し・再ランク見逃し・解けたケースの実例
                       （文脈・正解式・上位候補と特徴値）。

出力: experiments/case_outcomes_seed42.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent

FEAT_COL = {"text_sim": 0, "io_jaccard": 1, "svd_sim": 2, "input_cov": 3,
            "output_cov": 4, "specificity": 5, "domain": 6,
            "gComp": 7, "gCoh": 8, "gDom": 9}


# ---------------------------------------------------------------------------
# 純関数（tests/test_case_outcomes.py で検証）
# ---------------------------------------------------------------------------

def rank_candidates(cands: list, scores: np.ndarray) -> list:
    order = np.argsort(-scores)
    return [cands[k] for k in order]


def failure_split(ranked: list, cands: list, corr: set, k_c: int) -> dict:
    """上位 k_c と正解集合から、命中 / stage1見逃し / 再ランク見逃しを分ける."""
    top = set(ranked[:k_c])
    hit = corr & top
    missed = corr - top
    stage1_miss = {m for m in missed if m not in set(cands)}
    reranker_miss = missed - stage1_miss
    return {"hit": hit, "stage1_miss": stage1_miss, "reranker_miss": reranker_miss,
            "recall": len(hit) / len(corr) if corr else float("nan")}


def topic_rescue_fraction(missed: set, wrong_above: set, value_of: dict) -> float:
    """(m, w) 全ペア中、value(m) > value(w) となる割合（同値は0.5）."""
    pairs = [(m, w) for m in missed for w in wrong_above
             if m in value_of and w in value_of]
    if not pairs:
        return float("nan")
    wins = sum(1.0 if value_of[m] > value_of[w] else
               (0.5 if value_of[m] == value_of[w] else 0.0) for m, w in pairs)
    return wins / len(pairs)


def within_case_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _mean_sem(arr) -> dict:
    a = np.asarray([x for x in arr if x == x], dtype=float)
    if a.size == 0:
        return {"mean": float("nan"), "sem": float("nan"), "n": 0}
    sem = float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0
    return {"mean": float(a.mean()), "sem": sem, "n": int(a.size)}


def _mw(a, b) -> dict:
    from scipy import stats
    a = np.asarray([x for x in a if x == x]); b = np.asarray([x for x in b if x == x])
    out = {"mean_solved": float(a.mean()) if a.size else float("nan"),
           "mean_unsolved": float(b.mean()) if b.size else float("nan"),
           "n_solved": int(a.size), "n_unsolved": int(b.size), "p": float("nan")}
    if a.size and b.size:
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        out["p"] = float(p)
    return out


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------

def run(seed: int = 42, out_json: str = "experiments/case_outcomes_seed42.json"):
    import torch
    from two_stage_query_conditioned import (
        load_equations, load_cases, norm, eq_key, eq_text, eq_vars,
        case_text, in_vars, out_vars, case_src,
    )
    from set_aware_reranker import get_src
    from analyze_pfi import train_and_cache, keep_setting_A

    eqs = load_equations()
    cases_all = load_cases()
    cases = [c for c in cases_all if keep_setting_A(str(c.get("variant_type", "")))]
    ek, et, ev, ed, es = [], [], [], [], []
    eq_rec = []
    for e in eqs:
        k = eq_key(e)
        if not k:
            continue
        ek.append(k); et.append(eq_text(e)); ev.append(eq_vars(e))
        ed.append(norm(e.get("domain") or "")); es.append(get_src(e))
        eq_rec.append(e)
    ki = {k: i for i, k in enumerate(ek)}
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki]
          for c in cases]
    cs = [case_src(c) for c in cases]
    case_by_id = {c.get("case_id", f"idx_{i}"): c for i, c in enumerate(cases)}
    print(f"設定A ケース {len(cases)} 件 / 式 {len(ek)} 件 → seed={seed} で訓練")

    model, cache = train_and_cache(seed, cases, ek, et, ev, ed, es, cl, cs)
    print(f"訓練完了。テスト {len(cache)} ケースを採点")

    # 各テストケースの順位・失敗分解・特徴統計
    recs = []
    with torch.no_grad():
        for rec in cache:
            feats = rec["feats"]
            scores = model(torch.tensor(feats, dtype=torch.float32)).numpy().ravel()
            ranked = rank_candidates(rec["cands"], scores)
            k_c = len(rec["corr"])
            fs = failure_split(ranked, rec["cands"], rec["corr"], k_c)
            idx_of = {j: k for k, j in enumerate(rec["cands"])}
            val = {name: {j: float(feats[idx_of[j], col]) for j in rec["cands"]}
                   for name, col in FEAT_COL.items()}
            wrong_above = [j for j in ranked[:k_c] if j not in rec["corr"]]
            rescue = {name: topic_rescue_fraction(fs["reranker_miss"],
                                                  set(wrong_above), val[name])
                      for name in ("svd_sim", "text_sim", "domain", "specificity",
                                   "io_jaccard")}
            pos_mask = np.array([j in rec["corr"] for j in rec["cands"]], dtype=bool)
            sep = {}
            for name, col in FEAT_COL.items():
                p, n = feats[pos_mask, col], feats[~pos_mask, col]
                sep[name] = float(p.mean() - n.mean()) if p.size and n.size else float("nan")
            recs.append({
                "case_id": rec["case_id"], "variant": rec["variant"],
                "recall": fs["recall"], "n_correct": k_c,
                "n_input": rec["n_input"], "n_output": rec["n_output"],
                "n_sources": rec["n_sources"],
                "coverage": len(rec["corr"] & set(rec["cands"])) / k_c,
                "n_stage1_miss": len(fs["stage1_miss"]),
                "n_reranker_miss": len(fs["reranker_miss"]),
                "stage1_miss": sorted(fs["stage1_miss"]),
                "reranker_miss": sorted(fs["reranker_miss"]),
                "wrong_above": wrong_above, "ranked_top": ranked[:max(k_c, 5)],
                "rescue": rescue, "sep": sep,
                "corr_ts_vs": within_case_corr(feats[:, 0], feats[:, 1]),
                "corr_ts_spec": within_case_corr(feats[:, 0], feats[:, 5]),
            })

    solved = [r for r in recs if r["recall"] == 1.0]
    unsolved = [r for r in recs if r["recall"] < 1.0]
    base_R = float(np.mean([r["recall"] for r in recs]))

    # 1. 失敗の分解（見逃し式の本数ベース）
    tot_s1 = sum(r["n_stage1_miss"] for r in unsolved)
    tot_rr = sum(r["n_reranker_miss"] for r in unsolved)
    # 2. 解けた/落ちた比較
    comp_attrs = {a: _mw([r[a] for r in solved], [r[a] for r in unsolved])
                  for a in ("n_correct", "n_input", "n_sources", "coverage")}
    comp_sep = {f: _mw([r["sep"][f] for r in solved], [r["sep"][f] for r in unsolved])
                for f in ("specificity", "io_jaccard", "svd_sim", "text_sim")}
    # 3. 話題特徴の救済可能性（再ランク見逃しがあるケースのみ）
    rescue_agg = {name: _mean_sem([r["rescue"][name] for r in recs
                                   if r["rescue"][name] == r["rescue"][name]])
                  for name in ("svd_sim", "text_sim", "domain", "specificity",
                               "io_jaccard")}
    # 4. 選択効果（候補内相関）
    sel = {"corr_ts_vs": _mean_sem([r["corr_ts_vs"] for r in recs]),
           "corr_ts_spec": _mean_sem([r["corr_ts_spec"] for r in recs])}

    # 5. 具体例の抽出（決定的：recall 昇順 → n_correct 昇順 → case_id）
    def _eq_view(j):
        e = eq_rec[j]
        return {"idx": j, "eq_id": ek[j], "equation": (e.get("equation") or "")[:120],
                "domain": (e.get("domain") or "")[:60],
                "vars": sorted(ev[j])[:12]}

    cache_by_id = {rec["case_id"]: rec for rec in cache}

    def _example_full(r, kind):
        c = case_by_id.get(r["case_id"], {})
        rec = cache_by_id[r["case_id"]]
        idx_of = {j: k for k, j in enumerate(rec["cands"])}
        feats = rec["feats"]

        def _fv(j):
            if j not in idx_of:
                return None
            k = idx_of[j]
            return {"specificity": round(float(feats[k, 5]), 3),
                    "io_jaccard": round(float(feats[k, 1]), 3),
                    "svd_sim": round(float(feats[k, 2]), 3),
                    "text_sim": round(float(feats[k, 0]), 3)}

        def _view(j, tag):
            v = _eq_view(j); v["feat"] = _fv(j); v["tag"] = tag
            return v

        top = [_view(j, "正解" if j in rec["corr"] else "誤り")
               for j in r["ranked_top"][:min(r["n_correct"] + 2, 7)]]
        missed = ([_view(j, "stage1見逃し") for j in r["stage1_miss"]]
                  + [_view(j, "再ランク見逃し") for j in r["reranker_miss"]])
        return {"kind": kind, "case_id": r["case_id"], "variant": r["variant"],
                "recall": r["recall"], "coverage": r["coverage"],
                "n_correct": r["n_correct"],
                "context": (c.get("context") or "")[:400],
                "input_variables": c.get("input_variables"),
                "output_variables": c.get("output_variables"),
                "top_ranked": top, "missed": missed[:6], "rescue": r["rescue"]}

    examples = []
    rr_cases = sorted((r for r in unsolved if r["n_reranker_miss"] > 0
                       and r["n_stage1_miss"] == 0),
                      key=lambda r: (r["recall"], r["n_correct"], r["case_id"]))
    s1_cases = sorted((r for r in unsolved if r["n_stage1_miss"] > 0),
                      key=lambda r: (r["recall"], r["n_correct"], r["case_id"]))
    ok_cases = sorted(solved, key=lambda r: (-r["n_correct"], r["case_id"]))
    if rr_cases:
        examples.append(_example_full(rr_cases[0], "reranker_miss"))
        if len(rr_cases) > 1:
            examples.append(_example_full(rr_cases[1], "reranker_miss"))
    if s1_cases:
        examples.append(_example_full(s1_cases[0], "stage1_miss"))
    if ok_cases:
        examples.append(_example_full(ok_cases[0], "solved"))

    out = {
        "config": {"seed": seed, "setting": "A", "model": "reranker-10S",
                   "n_test": len(recs)},
        "baseline_R": base_R,
        "outcome_counts": {"solved": len(solved), "partial_or_failed": len(unsolved),
                           "solved_share": len(solved) / len(recs)},
        "failure_decomposition": {
            "missed_equations_total": tot_s1 + tot_rr,
            "stage1_miss": tot_s1, "reranker_miss": tot_rr,
            "stage1_share": tot_s1 / (tot_s1 + tot_rr) if (tot_s1 + tot_rr) else float("nan"),
        },
        "solved_vs_unsolved_attrs": comp_attrs,
        "solved_vs_unsolved_sep": comp_sep,
        "topic_rescue_fraction": rescue_agg,
        "selection_effect_corr": sel,
        "examples": examples,
    }
    json.dump(out, open(ROOT / out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2, default=str)
    print(f"Saved: {out_json}")

    # コンソール要約
    print(f"\nbaseline R@K_correct(seed42) = {base_R:.4f}  "
          f"解けた {len(solved)}/{len(recs)} ({len(solved)/len(recs):.1%})")
    fd = out["failure_decomposition"]
    print(f"見逃し正解式 {fd['missed_equations_total']} 本の内訳: "
          f"stage1見逃し {fd['stage1_miss']} ({fd['stage1_share']:.1%}) / "
          f"再ランク見逃し {fd['reranker_miss']}")
    print("解けた/落ちた属性差 (Mann-Whitney p):")
    for a, d in comp_attrs.items():
        print(f"  {a:<12s} solved={d['mean_solved']:.3f} unsolved={d['mean_unsolved']:.3f} p={d['p']:.2e}")
    for f, d in comp_sep.items():
        print(f"  sep({f:<11s}) solved={d['mean_solved']:+.4f} unsolved={d['mean_unsolved']:+.4f} p={d['p']:.2e}")
    print("再ランク見逃しの救済可能性 P(missed > wrong_above):")
    for name, d in rescue_agg.items():
        print(f"  {name:<12s} {d['mean']:.3f}±{d['sem']:.3f} (n={d['n']}ケース)")
    print(f"候補内相関: corr(text_sim, io_jaccard)={sel['corr_ts_vs']['mean']:+.3f}, "
          f"corr(text_sim, specificity)={sel['corr_ts_spec']['mean']:+.3f}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default="experiments/case_outcomes_seed42.json")
    a = ap.parse_args()
    run(seed=a.seed, out_json=a.out_json)
