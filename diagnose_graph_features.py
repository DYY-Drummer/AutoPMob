"""グラフ特徴量 (#7-9) の診断分析.

なぜ reranker-10 が reranker-7 より劣るようになったか、特徴量レベルで分析する。

実施する分析:
  (A) 各特徴量の正例/負例分布（mean, std, 正負差）
  (B) 単独 AUC（特徴量をそのままスコアにしたときの識別力）
  (C) 特徴量間相関（グラフ特徴量が基本特徴量と冗長でないか）
  (D) バリアント別の振る舞い
  (E) 特徴量の変動係数と範囲

出力: experiments/graph_feature_diagnosis.json / .md
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# two_stage_query_conditioned.py から必要な部品を流用
import sys
sys.path.insert(0, str(Path(__file__).parent))

from two_stage_query_conditioned import (  # type: ignore
    load_equations, load_cases, norm, eq_key, eq_text, eq_vars,
    case_text, io_vars, in_vars, out_vars, case_src, jaccard, src_split,
    BipartiteGraph, stage1, compute_features,
)

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "experiments"
OUT_JSON = OUT_DIR / "graph_feature_diagnosis.json"
OUT_MD = OUT_DIR / "graph_feature_diagnosis.md"


def get_src(e):
    return norm(e.get("source_id") or "unknown") or "unknown"


def eq_domain(e):
    return norm(e.get("domain") or "")


FEATURE_NAMES = [
    "f1_tfidf_cos",       # TF-IDF cosine
    "f2_io_jaccard",      # IO ∩ eq_vars / IO ∪ eq_vars
    "f3_svd_cos",         # SVD cosine
    "f4_input_cov",       # |input ∩ ev| / |input|
    "f5_output_cov",      # |output ∩ ev| / |output|
    "f6_reverse_cov",     # |io ∩ ev| / |ev|
    "f7_domain_match",    # domain name substring match
    "f8_2hop_reach",      # GRAPH: 2-hop reachability
    "f9_neighbor_jaccard",# GRAPH: neighbor Jaccard
    "f10_bridge_ratio",   # GRAPH: indirect bridge ratio
]
# NOTE: the 原コードでは特徴量の順序が異なる。実装に合わせる。
# 実装では feat[0..6] が base (7 features), feat[7..9] が graph.
# f7 は domain_match、その後 graph 特徴量。名前を合わせる。
FEATURE_NAMES_IMPL = [
    "f0_tfidf_cos",
    "f1_io_jaccard",
    "f2_svd_cos",
    "f3_input_cov",
    "f4_output_cov",
    "f5_reverse_cov",
    "f6_domain_match",
    "f7_2hop_reach",       # graph
    "f8_neighbor_jaccard", # graph
    "f9_bridge_ratio",     # graph
]
GRAPH_FEATURE_IDX = [7, 8, 9]


def auc_single_feature(feat_values: np.ndarray, labels: np.ndarray) -> float:
    """単一特徴量をスコアとしたときの AUC を Mann-Whitney U で計算."""
    pos = feat_values[labels == 1]
    neg = feat_values[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Mann-Whitney U の性質: AUC = U / (n_pos * n_neg)
    # 効率のため np.searchsorted 等を使用
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    # 同順位の平均ランク処理
    _, inv, counts = np.unique(combined, return_inverse=True, return_counts=True)
    rank_sum_by_val = np.zeros(len(counts))
    for i, r in zip(inv, ranks):
        rank_sum_by_val[i] += r
    avg_rank_by_val = rank_sum_by_val / counts
    ranks = avg_rank_by_val[inv]
    # pos は先頭 len(pos) 個
    rank_pos = ranks[: len(pos)].sum()
    n_pos = len(pos)
    n_neg = len(neg)
    u = rank_pos - n_pos * (n_pos + 1) / 2
    auc = u / (n_pos * n_neg)
    return float(auc)


def collect_features(seed: int = 42, top_k: int = 50):
    """Stage 1 で Top-K 候補を取り、各候補ペアの全10特徴量を収集."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD

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
        ed.append(eq_domain(e))
        es.append(get_src(e))

    ki = {k: i for i, k in enumerate(ek)}
    correct_lists = [
        [ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki]
        for c in cases
    ]
    cs = [case_src(c) for c in cases]

    # Build TF-IDF
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2))
    all_texts = [case_text(c) for c in cases] + et
    tfidf.fit(all_texts)
    X_ctx = tfidf.transform([case_text(c) for c in cases])
    X_eq = tfidf.transform(et)

    # SVD
    svd = TruncatedSVD(n_components=256, random_state=seed)
    svd.fit(np.vstack([X_ctx.toarray(), X_eq.toarray()]))
    Xc_svd = svd.transform(X_ctx)
    Xe_svd = svd.transform(X_eq)
    # L2 normalize
    Xc_svd = Xc_svd / (np.linalg.norm(Xc_svd, axis=1, keepdims=True) + 1e-9)
    Xe_svd = Xe_svd / (np.linalg.norm(Xe_svd, axis=1, keepdims=True) + 1e-9)

    # Split by source
    srcs = sorted(set(cs) - {None})
    split = src_split(srcs, seed=seed)
    test_s = split["test"]
    test_cases_idx = [i for i, s in enumerate(cs) if s in test_s]
    print(f"[Debug] train sources: {len(split['train'])}, "
          f"val: {len(split['val'])}, test: {len(test_s)}, "
          f"test cases: {len(test_cases_idx)}")

    graph = BipartiteGraph(ev)

    feats_all = []   # (n_pairs, 10)
    labels_all = []
    variants = []
    case_ids = []

    for ci in test_cases_idx:
        c = cases[ci]
        correct = set(correct_lists[ci])
        if not correct:
            continue
        io_set = io_vars(c)
        input_v = in_vars(c)
        output_v = out_vars(c)
        query_vars = io_set
        ctx = case_text(c, io=True)

        # Stage 1 候補
        cands = stage1(ci, X_ctx, X_eq, io_set, ev, k=top_k)
        # テキスト類似度
        ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
        vs = cosine_similarity(Xc_svd[ci : ci + 1], Xe_svd).ravel()

        feats = compute_features(
            cands, ts, io_set, input_v, output_v, vs,
            ev, ed, ctx,
            graph=graph, query_vars=query_vars,
        )
        for k_idx, j in enumerate(cands):
            feats_all.append(feats[k_idx])
            labels_all.append(1 if j in correct else 0)
            variants.append(c.get("variant_type", "?"))
            case_ids.append(c.get("case_id", ""))

    return (
        np.asarray(feats_all, dtype=np.float32),
        np.asarray(labels_all, dtype=np.int32),
        variants,
        case_ids,
    )


def analyze():
    print("=== Collecting features from Stage 1 Top-50 candidates on test set ===")
    feats, labels, variants, case_ids = collect_features(seed=42, top_k=50)
    print(f"Total pairs: {len(feats)}, positives: {int(labels.sum())}, "
          f"negatives: {int((labels == 0).sum())}")
    print()

    results = {
        "n_pairs": int(len(feats)),
        "n_positives": int(labels.sum()),
        "n_negatives": int((labels == 0).sum()),
        "per_feature": {},
        "correlations": {},
        "per_variant": {},
    }

    # (A) 正例/負例の分布と単独 AUC
    print("=== (A) 特徴量の正例/負例分布と単独AUC ===")
    print(f"{'feature':<22s} {'pos_mean':>10s} {'neg_mean':>10s} {'diff':>10s} "
          f"{'pos_std':>10s} {'neg_std':>10s} {'AUC':>8s}")
    for i, name in enumerate(FEATURE_NAMES_IMPL):
        fv = feats[:, i]
        pm = float(fv[labels == 1].mean()) if labels.sum() > 0 else 0.0
        nm = float(fv[labels == 0].mean()) if (labels == 0).sum() > 0 else 0.0
        ps = float(fv[labels == 1].std()) if labels.sum() > 0 else 0.0
        ns = float(fv[labels == 0].std()) if (labels == 0).sum() > 0 else 0.0
        auc = auc_single_feature(fv, labels)
        marker = " [GRAPH]" if i in GRAPH_FEATURE_IDX else ""
        print(f"{name:<22s} {pm:>10.4f} {nm:>10.4f} {pm-nm:>+10.4f} "
              f"{ps:>10.4f} {ns:>10.4f} {auc:>8.4f}{marker}")
        results["per_feature"][name] = {
            "pos_mean": round(pm, 4),
            "neg_mean": round(nm, 4),
            "pos_std": round(ps, 4),
            "neg_std": round(ns, 4),
            "diff_mean": round(pm - nm, 4),
            "auc": round(auc, 4),
            "is_graph": i in GRAPH_FEATURE_IDX,
        }
    print()

    # (B) 相関行列（グラフ特徴量 vs 基本特徴量）
    print("=== (B) グラフ特徴量と基本特徴量の相関 (Pearson) ===")
    corr = np.corrcoef(feats.T)
    print(f"{'':<22s}", end="")
    for j in GRAPH_FEATURE_IDX:
        print(f"{FEATURE_NAMES_IMPL[j]:>22s}", end="")
    print()
    for i in range(10):
        if i in GRAPH_FEATURE_IDX:
            continue
        print(f"{FEATURE_NAMES_IMPL[i]:<22s}", end="")
        row = {}
        for j in GRAPH_FEATURE_IDX:
            val = float(corr[i, j])
            print(f"{val:>22.4f}", end="")
            row[FEATURE_NAMES_IMPL[j]] = round(val, 4)
        print()
        results["correlations"][FEATURE_NAMES_IMPL[i]] = row
    # Graph vs graph
    print(f"{'':<22s}", end="")
    for j in GRAPH_FEATURE_IDX:
        print(f"{FEATURE_NAMES_IMPL[j]:>22s}", end="")
    print()
    for i in GRAPH_FEATURE_IDX:
        print(f"{FEATURE_NAMES_IMPL[i]:<22s}", end="")
        row = {}
        for j in GRAPH_FEATURE_IDX:
            val = float(corr[i, j])
            print(f"{val:>22.4f}", end="")
            row[FEATURE_NAMES_IMPL[j]] = round(val, 4)
        print()
        results["correlations"][FEATURE_NAMES_IMPL[i]] = row
    print()

    # (C) バリアント別分析
    print("=== (C) バリアント別のグラフ特徴量AUC ===")
    vt_set = sorted(set(variants))
    print(f"{'variant':<25s}", end="")
    for j in GRAPH_FEATURE_IDX:
        print(f"{FEATURE_NAMES_IMPL[j][:15]:>17s}", end="")
    print(f"{'pos_count':>12s}{'neg_count':>12s}")
    for vt in vt_set:
        mask = np.array([v == vt for v in variants])
        vf = feats[mask]
        vl = labels[mask]
        n_pos = int(vl.sum())
        n_neg = int((vl == 0).sum())
        print(f"{vt:<25s}", end="")
        vt_aucs = {}
        for j in GRAPH_FEATURE_IDX:
            auc = auc_single_feature(vf[:, j], vl) if n_pos > 0 and n_neg > 0 else 0.5
            print(f"{auc:>17.4f}", end="")
            vt_aucs[FEATURE_NAMES_IMPL[j]] = round(auc, 4)
        print(f"{n_pos:>12d}{n_neg:>12d}")
        results["per_variant"][vt] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "graph_aucs": vt_aucs,
        }
    print()

    # (D) グラフ特徴量の値の分散・ゼロ率
    print("=== (D) グラフ特徴量の値の分散・ゼロ率・ユニーク値数 ===")
    print(f"{'feature':<22s} {'min':>8s} {'max':>8s} {'mean':>8s} "
          f"{'std':>8s} {'zero_ratio':>12s} {'n_unique':>10s}")
    for i in GRAPH_FEATURE_IDX:
        fv = feats[:, i]
        name = FEATURE_NAMES_IMPL[i]
        zero = float((fv == 0).sum()) / len(fv)
        nu = int(len(np.unique(np.round(fv, 4))))
        print(f"{name:<22s} {fv.min():>8.4f} {fv.max():>8.4f} {fv.mean():>8.4f} "
              f"{fv.std():>8.4f} {zero:>12.4f} {nu:>10d}")
        results["per_feature"][name]["zero_ratio"] = round(zero, 4)
        results["per_feature"][name]["n_unique"] = nu
    print()

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    analyze()
