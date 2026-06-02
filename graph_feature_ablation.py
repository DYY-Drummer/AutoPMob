"""グラフ特徴量のアブレーション実験.

既存の reranker-7 / reranker-10 に加え、f9 の除去や部分組合せで評価。

Modes:
  baseline     : Stage 1 スコアをそのまま（0 特徴量）
  reranker-7   : 基本7特徴量のみ
  reranker-7+f7 : 基本7 + f7 (2-hop reach)のみ
  reranker-7+f8 : 基本7 + f8 (neighbor Jaccard)のみ
  reranker-9   : 基本7 + f7 + f8 (f9除去) ← Step 1 の検証対象
  reranker-10  : 基本7 + f7 + f8 + f9 (現行)

出力: experiments/graph_ablation_results.json
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 既存スクリプトから流用
import sys
sys.path.insert(0, str(Path(__file__).parent))
from two_stage_query_conditioned import (  # type: ignore
    load_equations, load_cases, norm, eq_key, eq_text, eq_vars,
    case_text, io_vars, in_vars, out_vars, case_src, jaccard, src_split,
    BipartiteGraph, Reranker, r2m, bci, stage1,
)


def get_src(e):
    return norm(e.get("source_id") or "unknown") or "unknown"


ROOT = Path(__file__).parent
OUT_DIR = ROOT / "experiments"
OUT_JSON = OUT_DIR / "graph_ablation_results.json"


# --- Feature configuration ---
# Each mode: list of graph feature indices to include
#   Indices: 7=2hop_reach, 8=neighbor_jaccard, 9=bridge_ratio,
#            D=idf_weighted_jaccard (new candidate D)
MODE_CONFIGS = {
    "baseline":       {"use_mlp": False, "graph_feats": []},
    "reranker-7":     {"use_mlp": True,  "graph_feats": []},
    "reranker-7+f7":  {"use_mlp": True,  "graph_feats": [7]},
    "reranker-7+f8":  {"use_mlp": True,  "graph_feats": [8]},
    "reranker-9":     {"use_mlp": True,  "graph_feats": [7, 8]},
    "reranker-10":    {"use_mlp": True,  "graph_feats": [7, 8, 9]},
    # --- Step 2: 新候補 D (IDF重み付きJaccard) ---
    "reranker-7+D":   {"use_mlp": True,  "graph_feats": ["D"]},
    "reranker-7_replaceF1withD": {"use_mlp": True,  "graph_feats": [],
                                   "replace_f1_with_D": True},
    # --- Step 3: 新候補 B (最短路距離) と S (Specificity) ---
    "reranker-7+B":   {"use_mlp": True,  "graph_feats": ["B"]},
    "reranker-7+S":   {"use_mlp": True,  "graph_feats": ["S"]},
    "reranker-7+BS":  {"use_mlp": True,  "graph_feats": ["B", "S"]},
    # --- Step 4: 新候補 C (変数説明文の意味類似度) ---
    "reranker-7+C":   {"use_mlp": True,  "graph_feats": ["C"]},
    "reranker-7+BSC": {"use_mlp": True,  "graph_feats": ["B", "S", "C"]},
    "reranker-7+SC":  {"use_mlp": True,  "graph_feats": ["S", "C"]},
    "reranker-7+BC":  {"use_mlp": True,  "graph_feats": ["B", "C"]},
}


def build_variable_idf(eq_vars_list: list[set], var_to_eqs: dict) -> dict:
    """各変数の IDF を計算: idf(v) = log(N_eq / |eq containing v|).

    レアな変数ほど高い IDF を持ち、情報価値が高い。時間 t や温度 T のような
    普遍変数は IDF が低く、η_ec のような固有変数は IDF が高い。
    """
    import math
    N = len(eq_vars_list)
    idf = {}
    for v, eq_set in var_to_eqs.items():
        df = len(eq_set)
        if df > 0:
            idf[v] = math.log(N / df)
        else:
            idf[v] = 0.0
    return idf


def idf_weighted_jaccard(a: set, b: set, idf: dict) -> float:
    """IDF で重み付けた Jaccard 係数.

    gD = Σ_{v ∈ a ∩ b} idf(v) / Σ_{v ∈ a ∪ b} idf(v)

    共通変数がレアであるほど高スコア。0〜1 の範囲。
    """
    if not a and not b:
        return 0.0
    inter = a & b
    union = a | b
    num = sum(idf.get(v, 0.0) for v in inter)
    den = sum(idf.get(v, 0.0) for v in union)
    return num / den if den > 0 else 0.0


def specificity_score(query_vars: set, cand_vars: set, idf: dict,
                       max_idf: float = 1.0) -> float:
    """候補式の特異性スコア (Specificity), [0, 1] に正規化.

    gS = Σ_{v ∈ cand_vars ∩ query_vars} idf(v) / (|cand_vars| * max_idf)

    候補式の全変数数×最大IDF で正規化することで、他の特徴量（0-1範囲）と
    スケールを揃える。これにより MLP 学習時のスケール支配問題を回避。
    """
    if not cand_vars:
        return 0.0
    inter = query_vars & cand_vars
    if not inter:
        return 0.0
    numerator = sum(idf.get(v, 0.0) for v in inter)
    return numerator / (len(cand_vars) * max_idf) if max_idf > 0 else 0.0


def build_variable_descriptions(equations: list) -> dict:
    """各変数シンボルに対する代表的な自然言語記述を収集.

    同じシンボルが複数の式で異なる記述を持つ場合、最頻値または最初の記述を採用。
    これを使って「変数の意味」を自然言語レベルで扱う。
    """
    from collections import Counter
    var_descs = {}  # symbol -> Counter of descriptions
    for e in equations:
        vs = e.get("variables") or {}
        for sym, desc in vs.items():
            if not isinstance(desc, str) or not desc.strip():
                continue
            sym_n = norm(sym)
            if sym_n not in var_descs:
                var_descs[sym_n] = Counter()
            var_descs[sym_n][desc.strip()] += 1
    # Pick most common description for each symbol
    return {sym: c.most_common(1)[0][0] for sym, c in var_descs.items()}


def build_variable_description_text(eq_variables: dict, var_desc_lookup: dict) -> str:
    """1 つの式について、全変数の説明文を連結したテキストを作る.

    変数の説明文が不足している場合は lookup から補完。
    """
    parts = []
    for sym, desc in (eq_variables or {}).items():
        if isinstance(desc, str) and desc.strip():
            parts.append(desc.strip())
        else:
            sym_n = norm(sym)
            if sym_n in var_desc_lookup:
                parts.append(var_desc_lookup[sym_n])
    return " ".join(parts)


def semantic_similarity(query_desc_vec, cand_desc_vec) -> float:
    """事前計算された query / candidate の description vector 間の cosine 類似度."""
    import numpy as np
    norm_q = np.linalg.norm(query_desc_vec)
    norm_c = np.linalg.norm(cand_desc_vec)
    if norm_q < 1e-9 or norm_c < 1e-9:
        return 0.0
    return float(np.dot(query_desc_vec, cand_desc_vec) / (norm_q * norm_c))


def shortest_path_score(cand_idx: int, query_vars: set, graph: BipartiteGraph,
                         max_hops: int = 4) -> float:
    """クエリ変数から候補式への最短路距離の平均.

    BFS で各クエリ変数 v から候補 eq_idx までの最短距離を計算し、平均を取る。
    未到達の場合は max_hops を使用。
    スコアは 1 / (1 + dist) で正規化（近いほど高い、0〜1）.

    f7 (2-hop binary) の連続化版: より細かい距離感を捉える。
    """
    if not query_vars:
        return 0.0
    cand_vars = graph.eq_vars[cand_idx]
    dists = []
    for v in query_vars:
        if v in cand_vars:
            dists.append(1)  # 直接含む = 距離 1 (var→eq)
            continue
        # BFS from v to cand_idx
        eqs_with_v = graph.var_to_eqs.get(v, set())
        if cand_idx in eqs_with_v:
            dists.append(1)
            continue
        # 2-hop: eq containing v → shared var → cand
        found = False
        visited_vars = set(cand_vars)  # 既知変数
        for eq_i in eqs_with_v:
            if eq_i == cand_idx:
                found = True
                break
            # check shared vars with cand
            if graph.eq_vars[eq_i] & cand_vars:
                found = True
                break
        if found:
            dists.append(2)
            continue
        # further hops: give up, use max_hops
        dists.append(max_hops)
    avg_dist = sum(dists) / len(dists)
    return 1.0 / (1.0 + avg_dist)


def compute_features_ablation(
    cand_indices, text_sim, io_set, input_v, output_v, svd_sim,
    eq_vars_list, eq_domains, case_ctx_str,
    graph: Optional[BipartiteGraph],
    query_vars: Optional[set],
    graph_feat_idx: list,
    var_idf: Optional[dict] = None,
    replace_f1_with_D: bool = False,
    max_idf: float = 1.0,
    semantic_sim: Optional[np.ndarray] = None,
) -> np.ndarray:
    """7基本特徴量 + 選択されたグラフ特徴量を計算.

    replace_f1_with_D=True の場合、f1_io_jaccard を gD で置換（特徴量数は7のまま）.
    """
    n = len(cand_indices)
    n_feat = 7 + len(graph_feat_idx)
    feats = np.zeros((n, n_feat), dtype=np.float32)
    dl = case_ctx_str.lower()
    use_graph = graph is not None and query_vars is not None

    # f7/f8/f9 のキャッシュ（候補ごとに 1 回のみ計算）
    need_legacy = any(x in (7, 8, 9) for x in graph_feat_idx)

    for k, j in enumerate(cand_indices):
        ev = eq_vars_list[j]
        feats[k, 0] = text_sim[j]
        if replace_f1_with_D and var_idf is not None:
            feats[k, 1] = idf_weighted_jaccard(io_set, ev, var_idf)
        else:
            feats[k, 1] = jaccard(io_set, ev)
        feats[k, 2] = svd_sim[j]
        feats[k, 3] = len(input_v & ev) / len(input_v) if input_v else 0.0
        feats[k, 4] = len(output_v & ev) / len(output_v) if output_v else 0.0
        feats[k, 5] = len(io_set & ev) / len(ev) if ev else 0.0
        d = eq_domains[j].lower()
        feats[k, 6] = 1.0 if (d and d in dl) or (dl and dl in d) else 0.0

        if graph_feat_idx:
            f7 = f8 = f9 = 0.0
            if need_legacy and use_graph:
                f7, f8, f9 = graph.query_conditioned_features(j, query_vars)
            col = 7
            for idx in graph_feat_idx:
                if idx == 7:
                    feats[k, col] = f7
                elif idx == 8:
                    feats[k, col] = f8
                elif idx == 9:
                    feats[k, col] = f9
                elif idx == "D":
                    if var_idf is not None and io_set is not None:
                        feats[k, col] = idf_weighted_jaccard(io_set, ev, var_idf)
                elif idx == "B":
                    # 最短路距離
                    if graph is not None and query_vars is not None:
                        feats[k, col] = shortest_path_score(j, query_vars, graph)
                elif idx == "S":
                    # Specificity (正規化版)
                    if var_idf is not None and query_vars is not None:
                        feats[k, col] = specificity_score(query_vars, ev, var_idf, max_idf)
                elif idx == "C":
                    # 変数説明文の意味類似度（事前計算済み semantic_sim[j] を使用）
                    if semantic_sim is not None:
                        feats[k, col] = float(semantic_sim[j])
                col += 1
    return feats


def run_mode(mode, seeds, cases, eq_keys_l, eq_texts_l, eq_vars_l, eq_domains_l, eq_sources_l,
             correct_lists, case_srcs, top_k, epochs, lr):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity

    cfg = MODE_CONFIGS[mode]
    use_mlp = cfg["use_mlp"]
    graph_feat_idx = cfg["graph_feats"]
    replace_f1_with_D = cfg.get("replace_f1_with_D", False)
    n_feat = 7 + len(graph_feat_idx)
    n_eq = len(eq_keys_l)

    # BipartiteGraph は f7/f8/f9 のどれか、または D を使う場合に必要
    need_graph = bool(graph_feat_idx) or replace_f1_with_D
    graph = BipartiteGraph(eq_vars_l) if need_graph else None

    # IDF が必要な特徴量 (D, S) または置換モード
    need_idf = ("D" in graph_feat_idx) or ("S" in graph_feat_idx) or replace_f1_with_D
    var_idf = None
    max_idf = 1.0
    if need_idf:
        var_idf = build_variable_idf(eq_vars_l, graph.var_to_eqs)
        max_idf = max(var_idf.values()) if var_idf else 1.0

    # 候補 C (意味類似度) が必要な場合: 変数説明文のTF-IDFで類似度を事前計算
    need_semantic = "C" in graph_feat_idx
    semantic_sim_matrix = None
    if need_semantic:
        from sklearn.feature_extraction.text import TfidfVectorizer as _Tfidf
        from sklearn.metrics.pairwise import cosine_similarity as _cossim
        # 1) 変数シンボル → 代表記述
        eqs_full = load_equations()
        var_desc_lookup = build_variable_descriptions(eqs_full)
        # 2) 各式の変数説明文テキストを作成（保存順で eq_keys_l と対応させる必要）
        eqs_by_key = {}
        for e in eqs_full:
            k = eq_key(e)
            if k:
                eqs_by_key[k] = e
        eq_desc_texts = []
        for key in eq_keys_l:
            e = eqs_by_key.get(key, {})
            eq_desc_texts.append(build_variable_description_text(
                e.get("variables") or {}, var_desc_lookup,
            ))
        # 3) 各ケースのクエリ変数説明文を作成
        case_desc_texts = []
        for c in cases:
            q_vars = io_vars(c)
            descs = [var_desc_lookup.get(norm(v), "") for v in q_vars]
            case_desc_texts.append(" ".join(d for d in descs if d))
        # 4) TF-IDF 学習 & 類似度計算
        tfidf_desc = _Tfidf(lowercase=True, max_features=20000, ngram_range=(1, 2), min_df=1)
        eq_desc_mat = tfidf_desc.fit_transform(eq_desc_texts)
        case_desc_mat = tfidf_desc.transform(case_desc_texts)
        # shape (n_cases, n_eq)
        semantic_sim_matrix = _cossim(case_desc_mat, eq_desc_mat)

    test_mrrs, var_mrrs = [], {}
    details = []

    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        split = src_split(eq_sources_l, seed)
        tr = [i for i, s in enumerate(case_srcs) if s in split["train"] and correct_lists[i]]
        te = [i for i, s in enumerate(case_srcs) if s in split["test"] and correct_lists[i]]

        tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1,2), min_df=1)
        X_eq = tfidf.fit_transform(eq_texts_l)
        X_ctx = tfidf.transform([case_text(c) for c in cases])
        ios = [io_vars(c) for c in cases]

        svd = TruncatedSVD(n_components=256, random_state=seed)
        E = svd.fit_transform(X_eq); E = E/(np.linalg.norm(E,axis=1,keepdims=True)+1e-12)
        Q = svd.transform(tfidf.transform([case_text(c, io=True) for c in cases]))
        Q = Q/(np.linalg.norm(Q,axis=1,keepdims=True)+1e-12)
        svd_sim = Q @ E.T

        if not use_mlp:
            # Baseline
            ranks, vr = [], {}
            for ci in te:
                corr = set(correct_lists[ci])
                if not corr: continue
                ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
                vs = np.array([jaccard(ios[ci], eq_vars_l[j]) for j in range(n_eq)], dtype=np.float32)
                order = np.argsort(-(0.7*ts + 0.3*vs))
                for r, j in enumerate(order, 1):
                    if int(j) in corr:
                        ranks.append(r)
                        vt = norm(cases[ci].get("variant_type") or "?")
                        vr.setdefault(vt, []).append(r)
                        break
            m = r2m(ranks); test_mrrs.append(m["MRR"])
            for vt, vranks in vr.items():
                var_mrrs.setdefault(vt, []).append(r2m(vranks)["MRR"])
            print(f"  [seed={seed}] {mode:20s}  MRR={m['MRR']:.4f}  R@1={m['Recall@1']:.4f}  n={m['n']:.0f}")
            continue

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
                    sem_sim_ci = semantic_sim_matrix[ci] if semantic_sim_matrix is not None else None
                    feats = compute_features_ablation(
                        cands, ts, ios[ci], in_vars(cases[ci]), out_vars(cases[ci]),
                        svd_sim[ci], eq_vars_l, eq_domains_l, case_text(cases[ci]),
                        graph, ios[ci], graph_feat_idx, var_idf, replace_f1_with_D,
                        max_idf, sem_sim_ci,
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
        ranks, vr = [], {}
        with torch.no_grad():
            for ci in te:
                corr = set(correct_lists[ci])
                if not corr: continue
                cands = stage1(ci, X_ctx, X_eq, ios[ci], eq_vars_l, top_k)
                ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
                feats = compute_features_ablation(
                    cands, ts, ios[ci], in_vars(cases[ci]), out_vars(cases[ci]),
                    svd_sim[ci], eq_vars_l, eq_domains_l, case_text(cases[ci]),
                    graph, ios[ci], graph_feat_idx,
                )
                scores = model(torch.tensor(feats, dtype=torch.float32)).numpy()
                order = sorted(range(len(cands)), key=lambda k: -scores[k])
                ranked = [cands[k] for k in order]
                for r, j in enumerate(ranked, 1):
                    if int(j) in corr:
                        ranks.append(r)
                        vt = norm(cases[ci].get("variant_type") or "?")
                        vr.setdefault(vt, []).append(r)
                        break
        m = r2m(ranks); test_mrrs.append(m["MRR"])
        for vt, vranks in vr.items():
            var_mrrs.setdefault(vt, []).append(r2m(vranks)["MRR"])
        print(f"  [seed={seed}] {mode:20s}  MRR={m['MRR']:.4f}  R@1={m['Recall@1']:.4f}  n={m['n']:.0f}")

    mn = float(np.mean(test_mrrs)); sd = float(np.std(test_mrrs, ddof=1)) if len(test_mrrs) > 1 else 0.0
    ci = bci(test_mrrs) if len(test_mrrs) > 1 else (mn, mn)
    va = {
        vt: {"MRR_mean": round(float(np.mean(ms)), 4),
             "MRR_std": round(float(np.std(ms, ddof=1)) if len(ms) > 1 else 0, 4)}
        for vt, ms in var_mrrs.items()
    }
    return {
        "mode": mode,
        "n_features": n_feat,
        "graph_feats_used": graph_feat_idx,
        "MRR_mean": round(mn, 4),
        "MRR_std": round(sd, 4),
        "MRR_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "by_variant": va,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modes", type=str, default=",".join(MODE_CONFIGS.keys()))
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

    modes = [m.strip() for m in args.modes.split(",") if m.strip() in MODE_CONFIGS]
    results = {}
    for mode in modes:
        print(f"\n{'='*60}\nMode: {mode}\n{'='*60}")
        results[mode] = run_mode(
            mode, seeds, cases, ek, et, ev, ed, es, cl, cs,
            args.top_k, args.epochs, args.lr,
        )

    print(f"\n{'='*70}")
    print(f"{'Method':<20s} | {'#feat':>5s} | {'MRR':>18s} | {'95% CI':>18s} | {'origMRR':>8s}")
    print("-" * 80)
    for m, r in results.items():
        ci = r["MRR_ci95"]
        orig = r["by_variant"].get("original", {}).get("MRR_mean", None)
        orig_s = f"{orig:.4f}" if orig is not None else "—"
        print(f"{m:<20s} | {r['n_features']:>5d} | {r['MRR_mean']:.4f} ± {r['MRR_std']:.4f} | "
              f"[{ci[0]:.4f}, {ci[1]:.4f}] | {orig_s:>8s}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
