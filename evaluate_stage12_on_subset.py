"""LLM ベースラインと同じ 30 件のテストケースで Stage 1 / Stage 2 を評価する.

LLM 直接生成と公平に比較するため、`experiments/llm_baseline_test_cases.json`
に記載された 30 件の case_id について Stage 1 (baseline_mix) と Stage 2
(reranker-10) の数値を出す。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent))
from two_stage_query_conditioned import (
    BipartiteGraph,
    Reranker,
    case_src,
    case_text,
    compute_features,
    eq_key,
    eq_text,
    eq_vars,
    get_src,
    in_vars,
    io_vars,
    jaccard,
    load_cases,
    load_equations,
    norm,
    out_vars,
    src_split,
    stage1,
)
from evaluate_multi_eq import aggregate_metrics, case_metrics, compute_all_ranks


ROOT = Path(__file__).parent
SUBSET_JSON = ROOT / "experiments" / "llm_baseline_test_cases.json"
OUT_JSON = ROOT / "experiments" / "stage12_on_subset.json"


def evaluate_subset(mode: str, seeds: list[int], target_case_ids: set[str]):
    eqs = load_equations()
    cases = load_cases()
    ek, et, ev, ed, es = [], [], [], [], []
    for e in eqs:
        k = eq_key(e)
        if not k:
            continue
        ek.append(k)
        et.append(eq_text(e))
        ev.append(eq_vars(e))
        ed.append(norm(e.get("domain") or ""))
        es.append(get_src(e))
    ki = {k: i for i, k in enumerate(ek)}
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki] for c in cases]
    cs = [case_src(c) for c in cases]

    top_k = 50
    epochs = 15
    lr = 1e-3
    n_feat_map = {"baseline": 0, "reranker-7": 7, "reranker-10": 10}
    n_feat = n_feat_map[mode]
    graph = BipartiteGraph(ev) if mode == "reranker-10" else None
    n_eq = len(ek)

    per_seed_results = []
    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        split = src_split(es, seed)
        tr = [i for i, s in enumerate(cs) if s in split["train"] and cl[i]]
        # ターゲット case_id でフィルタ
        te = [
            i for i, c in enumerate(cases)
            if c.get("case_id") in target_case_ids and cl[i] and cs[i] in split["test"]
        ]
        # seed によって test split に入らないケースもあるので fallback: 強制的に全 30 件評価
        if len(te) < len(target_case_ids):
            te = [
                i for i, c in enumerate(cases)
                if c.get("case_id") in target_case_ids and cl[i]
            ]

        tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
        X_eq = tfidf.fit_transform(et)
        X_ctx = tfidf.transform([case_text(c) for c in cases])
        ios = [io_vars(c) for c in cases]

        svd = TruncatedSVD(n_components=256, random_state=seed)
        E = svd.fit_transform(X_eq); E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        Q = svd.transform(tfidf.transform([case_text(c, io=True) for c in cases]))
        Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        svd_sim = Q @ E.T

        case_results = []
        if mode == "baseline":
            for ci in te:
                corr = set(cl[ci])
                if not corr:
                    continue
                ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
                vs = np.array([jaccard(ios[ci], ev[j]) for j in range(n_eq)], dtype=np.float32)
                order = np.argsort(-(0.7 * ts + 0.3 * vs))
                ranks = compute_all_ranks(order.tolist(), corr)
                cm = case_metrics(ranks)
                cm["variant"] = norm(cases[ci].get("variant_type") or "?")
                case_results.append(cm)
        else:
            model = Reranker(n_feat, 64)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            rng = random.Random(seed)
            for ep in range(epochs):
                rng.shuffle(tr)
                for s in range(0, len(tr), 16):
                    batch = tr[s:s + 16]
                    losses = []
                    for ci in batch:
                        corr = set(cl[ci])
                        if not corr:
                            continue
                        cands = stage1(ci, X_ctx, X_eq, ios[ci], ev, top_k)
                        pos = [j for j in cands if j in corr]
                        neg = [j for j in cands if j not in corr]
                        if not pos or not neg:
                            continue
                        ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
                        feats = compute_features(
                            cands, ts, ios[ci], in_vars(cases[ci]), out_vars(cases[ci]),
                            svd_sim[ci], ev, ed, case_text(cases[ci]),
                            graph, ios[ci],
                        )
                        scores = model(torch.tensor(feats, dtype=torch.float32))
                        c2k = {j: k for k, j in enumerate(cands)}
                        ns = min(8, len(neg))
                        for p in pos:
                            for ng in rng.sample(neg, ns):
                                losses.append(F.relu(0.1 - scores[c2k[p]] + scores[c2k[ng]]))
                    if losses:
                        loss = torch.stack(losses).mean()
                        opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                for ci in te:
                    corr = set(cl[ci])
                    if not corr:
                        continue
                    cands = stage1(ci, X_ctx, X_eq, ios[ci], ev, top_k)
                    ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
                    feats = compute_features(
                        cands, ts, ios[ci], in_vars(cases[ci]), out_vars(cases[ci]),
                        svd_sim[ci], ev, ed, case_text(cases[ci]),
                        graph, ios[ci],
                    )
                    scores = model(torch.tensor(feats, dtype=torch.float32)).numpy()
                    order = sorted(range(len(cands)), key=lambda k: -scores[k])
                    ranked_topK = [cands[k] for k in order]
                    ranks = compute_all_ranks(ranked_topK, corr, miss_rank=10_000)
                    cm = case_metrics(ranks)
                    cm["variant"] = norm(cases[ci].get("variant_type") or "?")
                    case_results.append(cm)

        agg = aggregate_metrics(case_results)
        agg["seed"] = seed
        per_seed_results.append(agg)
        print(f"  [seed={seed}] {mode:12s}  "
              f"n={agg['n_cases']:3d}  "
              f"MRR_f={agg['MRR_first']:.4f}  "
              f"R@3={agg['Recall@3']:.4f}  "
              f"MAP={agg['MAP']:.4f}")

    # シード間平均
    keys = ["MRR_first", "Recall@3", "Recall@5", "Recall@10", "MAP"]
    summary = {"mode": mode, "n_seeds": len(seeds)}
    for k in keys:
        vals = [a[k] for a in per_seed_results if k in a]
        if vals:
            summary[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4),
            }
    summary["per_seed"] = per_seed_results
    return summary


def main():
    # サブセットケース ID
    with open(SUBSET_JSON, encoding="utf-8") as f:
        subset = json.load(f)
    target_ids = {c["case_id"] for c in subset}
    print(f"=== サブセット評価: {len(target_ids)} ケース ===")

    seeds = [42, 123, 456]  # 3 シードで素早く（baseline は決定的なので 1 シードでも OK）
    results = {}
    for mode in ["baseline", "reranker-10"]:
        print(f"\n=== Mode: {mode} ===")
        results[mode] = evaluate_subset(mode, seeds, target_ids)

    # 比較表
    print("\n" + "=" * 70)
    print(f"{'Method':<15s} | {'MRR_first':>15s} | {'Recall@3':>15s} | {'MAP':>15s}")
    print("-" * 70)
    for m, r in results.items():
        print(f"{m:<15s} | "
              f"{r['MRR_first']['mean']:.4f}±{r['MRR_first']['std']:.3f}   | "
              f"{r['Recall@3']['mean']:.4f}±{r['Recall@3']['std']:.3f}   | "
              f"{r['MAP']['mean']:.4f}±{r['MAP']['std']:.3f}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"target_case_ids": list(target_ids), "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
