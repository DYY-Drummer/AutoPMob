#!/usr/bin/env python3
"""変数を記号でなく説明で照合する実験の学習と評価.

仕様: docs/superpowers/specs/2026-10-01-variable-description-matching-design.md §3.5
条件: S0（元の説明文・記号）, S1（記号なしの説明文・記号）, D1（言い換え）, D0（正解式の説明をそのまま）。
S1 の模型は、テストケースの入出力の一部を別の記号に替えた S1-r{50,100} でも評価する。
学習ループは set_aware_reranker.run_mode（reranker-10S, 静的）と同じ順序で乱数を消費するので、
S0 は既報（experiments/xs1/reranker-10S_w30-70__42.json）を再現する。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from two_stage_query_conditioned import Reranker, norm, in_vars, out_vars  # noqa: E402
from set_aware_reranker import stratified_src_split  # noqa: E402
from evaluate_multi_eq import compute_all_ranks, case_metrics, aggregate_metrics  # noqa: E402
from var_matching import (  # noqa: E402
    SYMBOL_TAU, K_REF, occurrence_incidence, stage1_match_counts, jaccard_from_counts,
    youden_tau, domain_flags, case_features,
)
from var_conditions import (  # noqa: E402
    CONDITIONS, load_data, load_var_descriptions, build_views, make_matcher,
    calibration_pairs, pair_sims,
)

SPEC = "docs/superpowers/specs/2026-10-01-variable-description-matching-design.md"
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
HP = dict(top_k=50, epochs=15, lr=1e-3, hidden_dim=64, margin=0.1, batch_size=16,
          n_neg_samples=8, weight_decay=1e-4)
SUMMARY_KEYS = ["MRR_first", "MRR_worst", "MRR_avg", "MAP",
                "Precision@C", "Recall@K_correct",
                "FullRecall@3", "FullRecall@10",
                "Recall@3", "Recall@5", "Recall@10", "Recall@20",
                "multi_only__MRR_first", "multi_only__MRR_worst",
                "multi_only__MAP", "multi_only__FullRecall@3", "multi_only__FullRecall@10",
                "multi_only__Recall@3", "multi_only__Recall@10",
                "multi_only__Recall@K_correct"]
COV_WEIGHTS = [(0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]


def wlabel(w) -> str:
    return f"w{int(round(w[0] * 100)):02d}-{int(round(w[1] * 100)):02d}"


def tau_for(matcher, views, data, idx, tau_delta=0.0) -> float:
    """記号版は SYMBOL_TAU。説明版は idx のケースの組で Youden の J が最大の τ（＋感度用の差分）.

    τ が有限でなければ ValueError（inf のまま使うと、説明版の条件がすべて「一致なし」になり、黙って結果が壊れる）."""
    if matcher.kind == "symbol":
        return SYMBOL_TAU
    pr = calibration_pairs(idx, views, data)
    tau = youden_tau(pair_sims(pr, matcher), pr["label"]) + tau_delta
    if not np.isfinite(tau):  # assert でなく明示的な検査（python -O でも効く）
        raise ValueError(f"τ が有限でない（{tau!r}）: 較正の組が退化していて（正例か負例がない、"
                         f"または類似度で区別できない）、roc_curve の先頭の閾値 inf が選ばれた可能性がある")
    return tau


def stage1(ci, view, matcher, tau, X_ctx, X_eq, data, inc, w_text, w_var, top_k):
    """第 1 段: (候補, 文章類似度 ts, 近似 |M|)."""
    from sklearn.metrics.pairwise import cosine_similarity
    ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
    full = matcher.q_occ(ci)
    m_approx = stage1_match_counts(full >= tau, inc)
    vs = jaccard_from_counts(m_approx, len(view.syms), data["n_vars"])
    cands = np.argsort(-(w_text * ts + w_var * vs))[:top_k].tolist()
    return cands, ts, full, m_approx


def prepare(idx, views, matcher, tau, X_ctx, X_eq, svd_sim, data, inc, w_text, w_var,
            top_k=HP["top_k"]):
    """各ケースの (候補, 特徴量) と、第 1 段の近似と厳密な |M| が食い違う割合."""
    eq_ptr, n_vars, doms = data["eq_ptr"], data["n_vars"], data["eq_domains"]
    prep, mismatch, total = {}, 0, 0
    for ci in idx:
        v = views[ci]
        cands, ts, full, m_approx = stage1(ci, v, matcher, tau, X_ctx, X_eq, data, inc,
                                           w_text, w_var, top_k)
        occ = np.concatenate([np.arange(eq_ptr[j], eq_ptr[j + 1]) for j in cands]).astype(np.int64)
        cptr = np.concatenate([[0], np.cumsum([n_vars[j] for j in cands])]).astype(np.int64)
        ref_end = int(cptr[min(K_REF, len(cands))])
        feats, m_exact = case_features(
            ts[cands], svd_sim[ci][cands], domain_flags(cands, doms, v.ctx), v.q_in, v.q_out,
            full[:, occ], matcher.occ_occ(occ, occ[:ref_end]), cptr, [doms[j] for j in cands], tau)
        mismatch += int((m_approx[cands] != m_exact).sum())
        total += len(cands)
        prep[ci] = (cands, feats)
    return prep, (mismatch / total if total else 0.0)


def train(seed, tr, prep, correct_lists, n_feat=10):
    """set_aware_reranker.run_mode の静的 reranker と同じ学習（同じ順序で乱数を消費する）."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = Reranker(n_feat, HP["hidden_dim"])
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=HP["weight_decay"])
    rng = random.Random(seed)
    tr = list(tr)
    for _ in range(HP["epochs"]):
        rng.shuffle(tr)
        for s in range(0, len(tr), HP["batch_size"]):
            losses = []
            for ci in tr[s:s + HP["batch_size"]]:
                corr = set(correct_lists[ci])
                if not corr:
                    continue
                cands, feats = prep[ci]
                pos = [j for j in cands if j in corr]
                neg = [j for j in cands if j not in corr]
                if not pos or not neg:
                    continue
                c2k = {j: k for k, j in enumerate(cands)}
                scores = model(torch.tensor(feats, dtype=torch.float32))
                chosen = rng.sample(neg, min(HP["n_neg_samples"], len(neg)))
                for p in pos:
                    for ng in chosen:
                        losses.append(F.relu(HP["margin"] - scores[c2k[p]] + scores[c2k[ng]]))
            if losses:
                loss = torch.stack(losses).mean()
                opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def evaluate(model, te, prep, data) -> list:
    cases, cl, ek = data["cases"], data["correct_lists"], data["eq_keys"]
    out = []
    with torch.no_grad():
        for ci in te:
            corr = set(cl[ci])
            if not corr:
                continue
            cands, feats = prep[ci]
            scores = model(torch.tensor(feats, dtype=torch.float32)).numpy().ravel()
            order = sorted(range(len(cands)), key=lambda k: -scores[k])
            cm = case_metrics(compute_all_ranks([cands[k] for k in order], corr, miss_rank=10_000))
            c = cases[ci]
            cm["variant"] = norm(c.get("variant_type") or "?")
            cm["case_id"] = c.get("case_id", f"idx_{ci}")
            cm["n_input"] = len(in_vars(c))
            cm["n_output"] = len(out_vars(c))
            cm["n_sources"] = len({ek[j].split("__")[0] for j in corr})
            out.append(cm)
    return out


def per_case_record(seed, label, cm) -> dict:
    rec = {"seed": seed, "mode": label, **cm}
    rec.pop("FullRecall", None); rec.pop("Recall", None)
    return rec


def summarize(label, aggs, per_case=None) -> dict:
    out = {"mode": label, "n_features": 10}
    for k in SUMMARY_KEYS:
        vals = [a.get(k) for a in aggs if a.get(k) is not None]
        if vals:
            out[k] = {"mean": round(float(np.mean(vals)), 4),
                      "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0, 4)}
    out["per_seed"] = aggs
    if per_case is not None:
        out["per_case"] = per_case
    return out


def run_condition(cond, data, vd, seeds, renames, tau_delta, w_text, w_var, save_per_case,
                  tfidf, X_eq):
    from sklearn.decomposition import TruncatedSVD
    cs, cl = data["case_srcs"], data["correct_lists"]
    inc = occurrence_incidence(data["eq_ptr"])
    views = build_views(cond, data, vd)
    matcher = make_matcher(cond, data, views)
    X_ctx = tfidf.transform([v.ctx for v in views])
    main_label = cond if tau_delta == 0 else f"{cond}-tau{tau_delta:+.2f}"
    rviews = ({f"S1-r{int(round(p * 100))}": build_views("S1", data, vd, p) for p in renames}
              if cond == "S1" else {})
    labels = [main_label] + list(rviews)
    aggs = {lab: [] for lab in labels}
    pcs = {lab: [] for lab in labels} if save_per_case else None
    taus, mism = {}, {}
    for seed in seeds:
        t0 = time.time()
        split = stratified_src_split(cs, data["feats"], seed)
        tr = [i for i, s in enumerate(cs) if s in split["train"] and cl[i]]
        te = [i for i, s in enumerate(cs) if s in split["test"] and cl[i]]
        svd = TruncatedSVD(n_components=256, random_state=seed)
        E = svd.fit_transform(X_eq)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

        def svd_sim_of(vs_):
            Q = svd.transform(tfidf.transform([v.io_text for v in vs_]))
            Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
            return Q @ E.T

        tau = tau_for(matcher, views, data, tr, tau_delta)
        taus[str(seed)] = tau
        prep, mism[str(seed)] = prepare(tr + te, views, matcher, tau, X_ctx, X_eq,
                                        svd_sim_of(views), data, inc, w_text, w_var)
        model = train(seed, tr, prep, cl)
        evals = {main_label: evaluate(model, te, prep, data)}
        for lab, rv in rviews.items():
            rprep, _ = prepare(te, rv, make_matcher("S1", data, rv), SYMBOL_TAU, X_ctx, X_eq,
                               svd_sim_of(rv), data, inc, w_text, w_var)
            evals[lab] = evaluate(model, te, rprep, data)
        for lab, res in evals.items():
            agg = aggregate_metrics(res)
            agg["seed"] = seed
            aggs[lab].append(agg)
            if pcs is not None:
                pcs[lab] += [per_case_record(seed, lab, cm) for cm in res]
            print(f"  [seed={seed}] {lab:12s} R@K={agg['Recall@K_correct']:.4f} MAP={agg['MAP']:.4f} "
                  f"R@20={agg['Recall@20']:.4f} n={agg['n_cases']} tau={tau:.4f} "
                  f"mismatch={mism[str(seed)]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    results = {lab: summarize(lab, aggs[lab], pcs[lab] if pcs is not None else None) for lab in labels}
    return results, {main_label: taus}, {main_label: mism}


def save_json(obj, path):
    """各条件の後に中間結果を安全に保存する（.tmp 経由で原子的に上書き）."""
    path = str(path)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp_path, path)


def coverage_sweep(conds, data, vd, tfidf, X_eq, k=50, seed=42) -> dict:
    """第 1 段の重みを変えたときの被覆率（上位 k 件に入る正解式の割合）。説明版の τ は seed の学習用ケースで決める."""
    from sklearn.metrics.pairwise import cosine_similarity
    cs, cl = data["case_srcs"], data["correct_lists"]
    inc = occurrence_incidence(data["eq_ptr"])
    split = stratified_src_split(cs, data["feats"], seed)
    tr = [i for i, s in enumerate(cs) if s in split["train"] and cl[i]]
    out = {}
    for cond in conds:
        views = build_views(cond, data, vd)
        matcher = make_matcher(cond, data, views)
        X_ctx = tfidf.transform([v.ctx for v in views])
        tau = tau_for(matcher, views, data, tr)
        acc = {wlabel(w): [] for w in COV_WEIGHTS}
        for ci in [i for i in range(len(cl)) if cl[i]]:
            ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
            vs = jaccard_from_counts(stage1_match_counts(matcher.q_occ(ci) >= tau, inc),
                                     len(views[ci].syms), data["n_vars"])
            corr = set(cl[ci])
            for w in COV_WEIGHTS:
                top = set(np.argsort(-(w[0] * ts + w[1] * vs))[:k].tolist())
                acc[wlabel(w)].append(len(corr & top) / len(corr))
        out[cond] = {"tau": tau, **{lab: float(np.mean(v)) for lab, v in acc.items()}}
        print(f"  coverage {cond}: " + "  ".join(f"{lab}={out[cond][lab]:.4f}" for lab in acc), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", choices=["A", "B"], required=True)
    ap.add_argument("--conds", default="S0,S1,D1,D0")
    ap.add_argument("--seed-list", default=None, help="カンマ区切り（既定: 正典の 10 seed）")
    ap.add_argument("--renames", default="0.5,1.0", help="S1 の模型で評価する付け替えの割合")
    ap.add_argument("--tau-delta", type=float, default=0.0, help="説明版の τ に足す値（感度の確認用）")
    ap.add_argument("--w-text", type=float, default=0.3)
    ap.add_argument("--w-var", type=float, default=0.7)
    ap.add_argument("--coverage", action="store_true", help="学習せず、第 1 段の被覆率の掃引だけ出す")
    ap.add_argument("--save-per-case", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    from sklearn.feature_extraction.text import TfidfVectorizer

    conds = [c.strip() for c in args.conds.split(",") if c.strip()]
    assert all(c in CONDITIONS for c in conds), conds
    seeds = [int(s) for s in args.seed_list.split(",")] if args.seed_list else SEEDS
    renames = [float(p) for p in args.renames.split(",") if p.strip()]
    data = load_data(args.setting)
    vd = load_var_descriptions() if any(c != "S0" for c in conds) else None
    tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = tfidf.fit_transform(data["eq_texts"])
    config = {**vars(args), **HP, "spec": SPEC, "n_cases": len(data["cases"])}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.coverage:
        out = {"config": config, "coverage": coverage_sweep(conds, data, vd, tfidf, X_eq)}
    else:
        results, taus, mism = {}, {}, {}
        for cond in conds:
            print(f"\n=== setting {args.setting} / {cond} ===", flush=True)
            r, t, m = run_condition(cond, data, vd, seeds, renames, args.tau_delta, args.w_text,
                                    args.w_var, args.save_per_case, tfidf, X_eq)
            results.update(r); taus.update(t); mism.update(m)
            # 各条件の後に中間結果を保存（失敗時に既処理部分を保護）
            out = {"config": config, "tau": taus, "approx_mismatch": mism, "results": results}
            save_json(out, args.output)
        out = {"config": config, "tau": taus, "approx_mismatch": mism, "results": results}
    save_json(out, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
