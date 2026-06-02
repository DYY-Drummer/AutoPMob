"""複数式の組合せ retrieval 評価.

現行の MRR（「最初に見つかった正解の順位」）だけでなく、複数式の組合せ能力を
測る指標を実装し、baseline / reranker-7 / reranker-10 を再評価する。

新しい指標:
  1. MRR_first     : 最初の正解の逆順位（現行指標、参考）
  2. MRR_worst     : 最下位の正解の逆順位（全正解を上位に揃える能力）
  3. MRR_avg       : 各正解の逆順位の平均（ケース内平均）
  4. FullRecall@K  : 全正解が上位 K 件に含まれるケースの割合
  5. Precision@|C| : 上位 |C| 件のうち実際に正解だった割合（|C| = |correct|）
  6. MAP           : Mean Average Precision（情報検索の標準指標）

使い方:
  python evaluate_multi_eq.py --seeds 5
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from two_stage_query_conditioned import (  # type: ignore
    load_equations, load_cases, norm, eq_key, eq_text, eq_vars,
    case_text, io_vars, in_vars, out_vars, case_src, jaccard, src_split,
    BipartiteGraph, Reranker, stage1, compute_features, bci,
)


def get_src(e):
    return norm(e.get("source_id") or "unknown") or "unknown"


ROOT = Path(__file__).parent
OUT_DIR = ROOT / "experiments"
OUT_JSON = OUT_DIR / "multi_eq_evaluation.json"


def compute_all_ranks(ranked_list: list[int], correct: set[int],
                       miss_rank: int = 10_000) -> list[int]:
    """正解集合の各要素が ranked_list 内で何位かを返す.

    ranked_list に含まれない正解は miss_rank（例：10000）を割り当てる。
    """
    ranks = []
    pos_of = {j: r for r, j in enumerate(ranked_list, 1)}
    for c in correct:
        ranks.append(pos_of.get(c, miss_rank))
    return ranks


def case_metrics(ranks: list[int], K_list=(3, 5, 10, 20),
                  miss_rank: int = 10_000) -> dict:
    """1 ケース分のメトリクスを計算.

    Args:
        ranks: 各正解式の順位（昇順ソート前の生データ、miss = miss_rank）
        K_list: Recall@K の評価点
    """
    if not ranks:
        return {}
    sorted_ranks = sorted(ranks)  # 上位から
    n = len(ranks)
    # 1. 最初の正解（現行 MRR 相当）
    mrr_first = 1.0 / sorted_ranks[0] if sorted_ranks[0] < miss_rank else 0.0
    # 2. 最下位の正解（全正解を揃える能力）
    mrr_worst = 1.0 / sorted_ranks[-1] if sorted_ranks[-1] < miss_rank else 0.0
    # 3. 各正解の逆順位平均
    mrr_avg = np.mean([1.0 / r if r < miss_rank else 0.0 for r in sorted_ranks])
    # 4. Full Recall@K: 全正解が上位 K にあるか
    full_recall = {}
    recall = {}  # 標準 Recall@K: 上位 K に含まれる正解の割合
    for K in K_list:
        # 全正解の順位が K 以下なら成功
        full_recall[K] = 1.0 if sorted_ranks[-1] <= K else 0.0
        # 標準 Recall@K: |R_q ∩ Top_K(q)| / |R_q|
        n_in_topK = sum(1 for r in sorted_ranks if r <= K)
        recall[K] = n_in_topK / n if n else 0.0
    # 5. Precision@|C|: 上位 |C| 件のうち正解率
    # 上位 |C| 件の中に何件の正解が含まれているか
    n_correct_in_topN = sum(1 for r in sorted_ranks if r <= n)
    precision_at_C = n_correct_in_topN / n
    # 5b. ★ 新規：Recall@K_correct = Recall@n_correct
    # K = ケースの正解数 n_correct とする（R-Precision）
    # 4 正解ケースで Recall@3 ≤ 0.75 という天井問題を解消
    # 数学的には Precision@n と Recall@n が一致する（R-Precision）
    recall_at_n_correct = n_correct_in_topN / n  # = precision_at_C, definition equivalent
    # 6. Average Precision
    ap_values = []
    for k, r in enumerate(sorted_ranks, 1):
        if r < miss_rank:
            # rank r にあるこの正解は、上位 r 件中で k 件目の正解
            precision_at_r = k / r
            ap_values.append(precision_at_r)
    avg_precision = np.mean(ap_values) if ap_values else 0.0
    return {
        "n_correct": n,
        "MRR_first": float(mrr_first),
        "MRR_worst": float(mrr_worst),
        "MRR_avg": float(mrr_avg),
        "FullRecall": full_recall,
        "Recall": recall,  # 標準 Recall@K
        "Precision@C": float(precision_at_C),
        "Recall@K_correct": float(recall_at_n_correct),  # ★ K=n_correct で正解の天井問題解消
        "AP": float(avg_precision),
    }


def aggregate_metrics(case_results: list[dict]) -> dict:
    """ケース単位のメトリクスを集計."""
    if not case_results:
        return {}
    agg = {
        "n_cases": len(case_results),
        "MRR_first": float(np.mean([c["MRR_first"] for c in case_results])),
        "MRR_worst": float(np.mean([c["MRR_worst"] for c in case_results])),
        "MRR_avg": float(np.mean([c["MRR_avg"] for c in case_results])),
        "Precision@C": float(np.mean([c["Precision@C"] for c in case_results])),
        "Recall@K_correct": float(np.mean([c["Recall@K_correct"] for c in case_results])),  # ★
        "MAP": float(np.mean([c["AP"] for c in case_results])),
    }
    # Full Recall@K と 標準 Recall@K
    K_list = list(case_results[0]["FullRecall"].keys())
    for K in K_list:
        agg[f"FullRecall@{K}"] = float(np.mean([c["FullRecall"][K] for c in case_results]))
        agg[f"Recall@{K}"] = float(np.mean([c["Recall"][K] for c in case_results]))
    # Split by multi vs single correct
    single = [c for c in case_results if c["n_correct"] == 1]
    multi = [c for c in case_results if c["n_correct"] >= 2]
    if single:
        agg["single_only__MRR_first"] = float(np.mean([c["MRR_first"] for c in single]))
    if multi:
        agg["multi_only__MRR_first"] = float(np.mean([c["MRR_first"] for c in multi]))
        agg["multi_only__MRR_worst"] = float(np.mean([c["MRR_worst"] for c in multi]))
        agg["multi_only__MAP"]       = float(np.mean([c["AP"] for c in multi]))
        agg["multi_only__FullRecall@3"] = float(np.mean([c["FullRecall"][3] for c in multi]))
        agg["multi_only__FullRecall@10"] = float(np.mean([c["FullRecall"][10] for c in multi]))
        agg["multi_only__Recall@3"] = float(np.mean([c["Recall"][3] for c in multi]))
        agg["multi_only__Recall@10"] = float(np.mean([c["Recall"][10] for c in multi]))
        agg["multi_only__Recall@K_correct"] = float(np.mean([c["Recall@K_correct"] for c in multi]))  # ★
    agg["n_single"] = len(single)
    agg["n_multi"] = len(multi)
    return agg


def run_mode(mode, seeds, cases, eq_keys_l, eq_texts_l, eq_vars_l,
             eq_domains_l, eq_sources_l, correct_lists, case_srcs,
             top_k, epochs, lr):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity

    n_feat = {"baseline": 0, "reranker-7": 7, "reranker-10": 10}[mode]
    n_eq = len(eq_keys_l)
    graph = BipartiteGraph(eq_vars_l) if mode == "reranker-10" else None

    per_seed_aggs = []

    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        split = src_split(eq_sources_l, seed)
        tr = [i for i, s in enumerate(case_srcs) if s in split["train"] and correct_lists[i]]
        te = [i for i, s in enumerate(case_srcs) if s in split["test"] and correct_lists[i]]

        tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
        X_eq = tfidf.fit_transform(eq_texts_l)
        X_ctx = tfidf.transform([case_text(c) for c in cases])
        ios = [io_vars(c) for c in cases]

        svd = TruncatedSVD(n_components=256, random_state=seed)
        E = svd.fit_transform(X_eq); E = E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-12)
        Q = svd.transform(tfidf.transform([case_text(c, io=True) for c in cases]))
        Q = Q/(np.linalg.norm(Q,axis=1,keepdims=True)+1e-12)
        svd_sim = Q @ E.T

        case_results = []

        if mode == "baseline":
            # Stage 1 スコアで全式を直接ランキング（reranker なし）
            for ci in te:
                corr = set(correct_lists[ci])
                if not corr: continue
                ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
                vs = np.array([jaccard(ios[ci], eq_vars_l[j]) for j in range(n_eq)], dtype=np.float32)
                order = np.argsort(-(0.7*ts + 0.3*vs))
                ranks = compute_all_ranks(order.tolist(), corr)
                cm = case_metrics(ranks)
                cm["variant"] = norm(cases[ci].get("variant_type") or "?")
                case_results.append(cm)
        else:
            # Train reranker
            model = Reranker(n_feat, 64)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
            rng = random.Random(seed)

            for ep in range(epochs):
                rng.shuffle(tr)
                for s in range(0, len(tr), 16):
                    batch = tr[s:s+16]
                    losses = []
                    for ci in batch:
                        corr = set(correct_lists[ci])
                        if not corr: continue
                        cands = stage1(ci, X_ctx, X_eq, ios[ci], eq_vars_l, top_k)
                        pos = [j for j in cands if j in corr]
                        neg = [j for j in cands if j not in corr]
                        if not pos or not neg: continue

                        ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
                        feats = compute_features(
                            cands, ts, ios[ci], in_vars(cases[ci]), out_vars(cases[ci]),
                            svd_sim[ci], eq_vars_l, eq_domains_l, case_text(cases[ci]),
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

            # Evaluate
            model.eval()
            with torch.no_grad():
                for ci in te:
                    corr = set(correct_lists[ci])
                    if not corr: continue
                    cands = stage1(ci, X_ctx, X_eq, ios[ci], eq_vars_l, top_k)
                    ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
                    feats = compute_features(
                        cands, ts, ios[ci], in_vars(cases[ci]), out_vars(cases[ci]),
                        svd_sim[ci], eq_vars_l, eq_domains_l, case_text(cases[ci]),
                        graph, ios[ci],
                    )
                    scores = model(torch.tensor(feats, dtype=torch.float32)).numpy()
                    order = sorted(range(len(cands)), key=lambda k: -scores[k])
                    ranked_topK = [cands[k] for k in order]  # top_k candidates
                    # 候補外の正解は Stage 1 でも見つけられないので miss とする
                    ranks = compute_all_ranks(ranked_topK, corr, miss_rank=10_000)
                    cm = case_metrics(ranks)
                    cm["variant"] = norm(cases[ci].get("variant_type") or "?")
                    case_results.append(cm)

        agg = aggregate_metrics(case_results)
        agg["seed"] = seed
        per_seed_aggs.append(agg)
        print(f"  [seed={seed}] {mode:12s}  "
              f"MRR_f={agg['MRR_first']:.4f}  "
              f"MRR_w={agg.get('MRR_worst', 0):.4f}  "
              f"MAP={agg['MAP']:.4f}  "
              f"FullR@3={agg['FullRecall@3']:.4f}  "
              f"n={agg['n_cases']}")

    # Aggregate across seeds
    keys = ["MRR_first", "MRR_worst", "MRR_avg", "MAP",
            "FullRecall@3", "FullRecall@5", "FullRecall@10", "FullRecall@20",
            "Precision@C",
            "multi_only__MRR_first", "multi_only__MRR_worst",
            "multi_only__MAP", "multi_only__FullRecall@3", "multi_only__FullRecall@10"]
    summary = {"mode": mode}
    for k in keys:
        vals = [a.get(k) for a in per_seed_aggs if a.get(k) is not None]
        if vals:
            summary[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0, 4),
            }
    summary["per_seed"] = per_seed_aggs
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    seeds = [42, 123, 456, 789, 1024][:args.seeds]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eqs = load_equations(); cases = load_cases()
    ek, et, ev, ed, es = [], [], [], [], []
    for e in eqs:
        k = eq_key(e)
        if not k: continue
        ek.append(k); et.append(eq_text(e)); ev.append(eq_vars(e))
        ed.append(norm(e.get("domain") or "")); es.append(get_src(e))
    ki = {k: i for i, k in enumerate(ek)}
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki] for c in cases]
    cs = [case_src(c) for c in cases]

    results = {}
    for mode in ["baseline", "reranker-7", "reranker-10"]:
        print(f"\n{'='*60}\nMode: {mode}\n{'='*60}")
        results[mode] = run_mode(
            mode, seeds, cases, ek, et, ev, ed, es, cl, cs,
            args.top_k, args.epochs, args.lr,
        )

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"{'Method':<15s} | {'MRR_first':>10s} | {'MRR_worst':>10s} | {'MRR_avg':>10s} | "
          f"{'MAP':>10s} | {'FullR@3':>10s} | {'FullR@10':>10s}")
    print("-" * 90)
    for m, r in results.items():
        print(f"{m:<15s} | {r['MRR_first']['mean']:.4f}±{r['MRR_first']['std']:.3f} "
              f"| {r['MRR_worst']['mean']:.4f}±{r['MRR_worst']['std']:.3f} "
              f"| {r['MRR_avg']['mean']:.4f}±{r['MRR_avg']['std']:.3f} "
              f"| {r['MAP']['mean']:.4f}±{r['MAP']['std']:.3f} "
              f"| {r['FullRecall@3']['mean']:.4f}±{r['FullRecall@3']['std']:.3f} "
              f"| {r['FullRecall@10']['mean']:.4f}±{r['FullRecall@10']['std']:.3f}")

    # Multi-only
    print(f"\n=== 複数式ケースのみ ===")
    print(f"{'Method':<15s} | {'MRR_first':>10s} | {'MRR_worst':>10s} | {'MAP':>10s} | "
          f"{'FullR@3':>10s} | {'FullR@10':>10s}")
    print("-" * 85)
    for m, r in results.items():
        if "multi_only__MRR_first" in r:
            print(f"{m:<15s} "
                  f"| {r['multi_only__MRR_first']['mean']:.4f}±{r['multi_only__MRR_first']['std']:.3f} "
                  f"| {r['multi_only__MRR_worst']['mean']:.4f}±{r['multi_only__MRR_worst']['std']:.3f} "
                  f"| {r['multi_only__MAP']['mean']:.4f}±{r['multi_only__MAP']['std']:.3f} "
                  f"| {r['multi_only__FullRecall@3']['mean']:.4f}±{r['multi_only__FullRecall@3']['std']:.3f} "
                  f"| {r['multi_only__FullRecall@10']['mean']:.4f}±{r['multi_only__FullRecall@10']['std']:.3f}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
