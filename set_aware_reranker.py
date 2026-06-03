"""Set-aware 特徴量を用いた複数式向け reranker 実験.

目的:
  複数式ケース（全体の 44%）で GNN/グラフ情報の優位性を示すため、
  「候補同士の組合せ」を捉える Set-aware 特徴量を設計・評価する。

新特徴量 (3 つ、Stage 1 top-K_ref を参照集合として計算):
  gComp : |cand ∩ query - Union(top-K)| / |query|  ← 補完性
  gCoh  : |cand ∩ Union(top-K)| / |cand|           ← 一貫性
  gDom  : |{s ∈ top-K : domain(s) == domain(cand)}| / K  ← ドメイン一致

モード:
  baseline      : Stage 1 スコアそのまま
  reranker-7    : 基本 7 特徴量
  reranker-10   : 基本 7 + 既存グラフ 3
  reranker-13   : 基本 7 + 既存グラフ 3 + Set-aware 3
  reranker-10S  : 基本 7 + Set-aware 3（既存グラフ特徴量を置換）

出力: experiments/set_aware_results.json
"""
from __future__ import annotations

import argparse
import json
import random
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
from evaluate_multi_eq import (
    compute_all_ranks, case_metrics, aggregate_metrics,
)


def get_src(e):
    return norm(e.get("source_id") or "unknown") or "unknown"


ROOT = Path(__file__).parent
OUT_DIR = ROOT / "experiments"
OUT_JSON = OUT_DIR / "set_aware_results.json"

K_REF = 5  # Set-aware 特徴量の参照集合サイズ


def compute_set_aware_features(
    cand_indices: list[int],
    query_vars: set[str],
    eq_vars_list: list[set[str]],
    eq_domains: list[str],
    k_ref: int = K_REF,
    which: tuple = ("Comp", "Coh", "Dom"),
    output_vars: Optional[set] = None,
    ref_indices: Optional[list] = None,
) -> np.ndarray:
    """Set-aware 特徴量を計算.

    参照集合 = Stage 1 の上位 k_ref 件（候補自身は除外）

    特徴量（列）: 0=gComp 補完性, 1=gCoh 一貫性, 2=gDom ドメイン一致,
    3=gOutNov 出力補完性（候補が参照集合にない出力変数を固有に覆う割合）。
    """
    n = len(cand_indices)
    feats = np.zeros((n, 4), dtype=np.float32)  # Comp, Coh, Dom, OutNov
    output_vars = output_vars or set()

    # 参照集合：ref_indices があればそれ（greedy の既選択集合）、無ければ Stage 1 上位 k_ref
    ref_set = list(ref_indices) if ref_indices is not None else cand_indices[:k_ref]
    ref_vars_union = set()
    for r in ref_set:
        ref_vars_union |= eq_vars_list[r]
    ref_domains = [eq_domains[r] for r in ref_set]

    for k, j in enumerate(cand_indices):
        cand_vars = eq_vars_list[j]
        # 参照集合から自分自身を除く
        if j in ref_set:
            local_union = set()
            for r in ref_set:
                if r != j:
                    local_union |= eq_vars_list[r]
            local_domains = [eq_domains[r] for r in ref_set if r != j]
        else:
            local_union = ref_vars_union
            local_domains = ref_domains

        # gComp: 補完性
        # query のうち cand 固有（top-K にない）部分がどれだけあるか
        cand_contrib = (cand_vars & query_vars) - local_union
        feats[k, 0] = len(cand_contrib) / len(query_vars) if query_vars else 0.0

        # gCoh: 一貫性
        # cand の変数のうち、top-K と共有する割合
        feats[k, 1] = len(cand_vars & local_union) / len(cand_vars) if cand_vars else 0.0

        # gDom: ドメイン一致
        if local_domains:
            same = sum(1 for d in local_domains if d == eq_domains[j] and d)
            feats[k, 2] = same / len(local_domains)
        else:
            feats[k, 2] = 0.0

        # gOutNov: 出力補完性（参照集合に無い出力変数を候補がどれだけ固有に覆うか）
        if output_vars:
            out_contrib = (cand_vars & output_vars) - local_union
            feats[k, 3] = len(out_contrib) / len(output_vars)

    return feats


def compute_features_with_set(
    cand_indices, text_sim, io_set, input_v, output_v, svd_sim,
    eq_vars_list, eq_domains, case_ctx_str,
    graph: Optional[BipartiteGraph],
    query_vars: Optional[set],
    use_existing_graph: bool,
    set_aware_mask: tuple = (),  # ('Comp', 'Coh', 'Dom') 等のサブセット
) -> np.ndarray:
    """基本 7 + 任意で既存グラフ 3 + 選択された Set-aware 特徴量を結合."""
    n_set_feats = len(set_aware_mask)
    n = len(cand_indices)
    n_feat = 7 + (3 if use_existing_graph else 0) + n_set_feats
    feats = np.zeros((n, n_feat), dtype=np.float32)
    dl = case_ctx_str.lower()

    for k, j in enumerate(cand_indices):
        ev = eq_vars_list[j]
        feats[k, 0] = text_sim[j]
        feats[k, 1] = jaccard(io_set, ev)
        feats[k, 2] = svd_sim[j]
        feats[k, 3] = len(input_v & ev) / len(input_v) if input_v else 0.0
        feats[k, 4] = len(output_v & ev) / len(output_v) if output_v else 0.0
        feats[k, 5] = len(io_set & ev) / len(ev) if ev else 0.0
        d = eq_domains[j].lower()
        feats[k, 6] = 1.0 if (d and d in dl) or (dl and dl in d) else 0.0

    col = 7
    if use_existing_graph:
        if graph is not None and query_vars is not None:
            for k, j in enumerate(cand_indices):
                f7, f8, f9 = graph.query_conditioned_features(j, query_vars)
                feats[k, col] = f7
                feats[k, col + 1] = f8
                feats[k, col + 2] = f9
        col += 3

    if set_aware_mask:
        set_feats_all = compute_set_aware_features(
            cand_indices, query_vars or set(), eq_vars_list, eq_domains,
            output_vars=output_v,
        )
        # Select requested columns
        idx_map = {"Comp": 0, "Coh": 1, "Dom": 2, "OutNov": 3}
        for i, name in enumerate(set_aware_mask):
            if name in idx_map:
                feats[:, col + i] = set_feats_all[:, idx_map[name]]

    return feats


def greedy_order(cands, full_feats, init_scores, model, set_mask,
                 eq_vars_l, eq_domains_l, output_v, query_vars,
                 n_base, greedy_cap=20):
    """推論時 greedy 集合構築：1 件選ぶごとに set-aware 特徴量を「既選択集合」を
    参照に再計算して次を選ぶ（学習済みモデルをそのまま使用，新規学習不要）。

    full_feats : 通常スコアリング用の全特徴量（base + set-aware）。
    init_scores: full_feats による初期スコア（最初の 1 件と残りの順序付けに使用）。
    n_base     : set-aware より前の特徴量数（基本 7 [+既存グラフ 3]）。
    """
    idx_map = {"Comp": 0, "Coh": 1, "Dom": 2, "OutNov": 3}
    npos = len(cands)
    sel, remaining = [], list(range(npos))
    first = max(remaining, key=lambda k: init_scores[k])
    sel.append(first); remaining.remove(first)
    cap = min(greedy_cap, npos)
    while remaining and len(sel) < cap:
        ref_idx = [cands[k] for k in sel]            # 既選択集合（DB index）
        rem_db = [cands[k] for k in remaining]
        setf = compute_set_aware_features(
            rem_db, query_vars or set(), eq_vars_l, eq_domains_l,
            output_vars=output_v, ref_indices=ref_idx,
        )
        feat = np.array(full_feats[remaining], dtype=np.float32, copy=True)
        for i, name in enumerate(set_mask):
            feat[:, n_base + i] = setf[:, idx_map[name]]
        sc = model(torch.tensor(feat)).numpy().ravel()
        best = remaining[int(np.argmax(sc))]
        sel.append(best); remaining.remove(best)
    remaining.sort(key=lambda k: -init_scores[k])
    return [cands[k] for k in sel + remaining]


MODE_CONFIGS = {
    "baseline":         {"use_mlp": False, "use_existing_graph": False, "set_aware_mask": ()},
    "reranker-7":       {"use_mlp": True,  "use_existing_graph": False, "set_aware_mask": ()},
    "reranker-10":      {"use_mlp": True,  "use_existing_graph": True,  "set_aware_mask": ()},
    "reranker-10S":     {"use_mlp": True,  "use_existing_graph": False, "set_aware_mask": ("Comp", "Coh", "Dom")},
    "reranker-10S2":    {"use_mlp": True,  "use_existing_graph": False, "set_aware_mask": ("Comp", "Coh", "OutNov")},
    "reranker-11S":     {"use_mlp": True,  "use_existing_graph": False, "set_aware_mask": ("Comp", "Coh", "Dom", "OutNov")},
    "reranker-13":      {"use_mlp": True,  "use_existing_graph": True,  "set_aware_mask": ("Comp", "Coh", "Dom")},
    # --- アブレーション：単独特徴量 ---
    "reranker-7+Comp":  {"use_mlp": True,  "use_existing_graph": False, "set_aware_mask": ("Comp",)},
    "reranker-7+Coh":   {"use_mlp": True,  "use_existing_graph": False, "set_aware_mask": ("Coh",)},
    "reranker-7+Dom":   {"use_mlp": True,  "use_existing_graph": False, "set_aware_mask": ("Dom",)},
    # --- アブレーション：ペア ---
    "reranker-9S_CK":   {"use_mlp": True,  "use_existing_graph": False, "set_aware_mask": ("Comp", "Coh")},
    "reranker-9S_CD":   {"use_mlp": True,  "use_existing_graph": False, "set_aware_mask": ("Comp", "Dom")},
    "reranker-9S_KD":   {"use_mlp": True,  "use_existing_graph": False, "set_aware_mask": ("Coh", "Dom")},
}


def run_mode(mode, seeds, cases, eq_keys_l, eq_texts_l, eq_vars_l,
             eq_domains_l, eq_sources_l, correct_lists, case_srcs,
             top_k, epochs, lr, save_per_case=False,
             hidden_dim=64, margin=0.1, batch_size=16, n_neg_samples=8,
             weight_decay=1e-4, loss_type="pairwise", hard_neg=False, greedy=False):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity

    cfg = MODE_CONFIGS[mode]
    use_mlp = cfg["use_mlp"]
    use_existing = cfg["use_existing_graph"]
    set_mask = cfg.get("set_aware_mask", ())
    n_feat = 7 + (3 if use_existing else 0) + len(set_mask)
    n_eq = len(eq_keys_l)
    graph = BipartiteGraph(eq_vars_l) if use_existing else None

    per_seed_aggs = []
    per_case_records = []  # only populated when save_per_case=True

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

        if not use_mlp:
            for ci in te:
                corr = set(correct_lists[ci])
                if not corr: continue
                ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
                vs = np.array([jaccard(ios[ci], eq_vars_l[j]) for j in range(n_eq)], dtype=np.float32)
                order = np.argsort(-(0.7*ts + 0.3*vs))
                ranks = compute_all_ranks(order.tolist(), corr)
                cm = case_metrics(ranks)
                cm["variant"] = norm(cases[ci].get("variant_type") or "?")
                cm["case_id"] = cases[ci].get("case_id", f"idx_{ci}")
                cm["n_input"] = len(in_vars(cases[ci]))
                cm["n_output"] = len(out_vars(cases[ci]))
                cm["n_sources"] = len({(eq_keys_l[j].split("__")[0] if j < len(eq_keys_l) else "?") for j in corr})
                case_results.append(cm)
        else:
            model = Reranker(n_feat, hidden_dim)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            rng = random.Random(seed)

            for ep in range(epochs):
                rng.shuffle(tr)
                for s in range(0, len(tr), batch_size):
                    batch = tr[s:s+batch_size]
                    losses = []
                    for ci in batch:
                        corr = set(correct_lists[ci])
                        if not corr: continue
                        cands = stage1(ci, X_ctx, X_eq, ios[ci], eq_vars_l, top_k)
                        pos = [j for j in cands if j in corr]
                        neg = [j for j in cands if j not in corr]
                        if not pos or not neg: continue

                        ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
                        feats = compute_features_with_set(
                            cands, ts, ios[ci], in_vars(cases[ci]), out_vars(cases[ci]),
                            svd_sim[ci], eq_vars_l, eq_domains_l, case_text(cases[ci]),
                            graph, ios[ci], use_existing, set_mask,
                        )
                        scores = model(torch.tensor(feats, dtype=torch.float32))
                        c2k = {j: k for k, j in enumerate(cands)}
                        pos_k = [c2k[p] for p in pos]
                        if loss_type == "listwise":
                            # multi-positive softmax CE: 各正解をリスト全体より上へ押す
                            lse = torch.logsumexp(scores, dim=0)
                            losses.append(lse - scores[torch.tensor(pos_k)].mean())
                        else:
                            ns = min(n_neg_samples, len(neg))
                            if hard_neg:
                                # 現在スコアが高い（最も紛らわしい）negative を優先
                                neg_sorted = sorted(neg, key=lambda j: -float(scores[c2k[j]].detach()))
                                chosen = neg_sorted[:ns]
                            else:
                                chosen = rng.sample(neg, ns)
                            for pk in pos_k:
                                for ng in chosen:
                                    losses.append(F.relu(margin - scores[pk] + scores[c2k[ng]]))
                    if losses:
                        loss = torch.stack(losses).mean()
                        opt.zero_grad(); loss.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                for ci in te:
                    corr = set(correct_lists[ci])
                    if not corr: continue
                    cands = stage1(ci, X_ctx, X_eq, ios[ci], eq_vars_l, top_k)
                    ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
                    feats = compute_features_with_set(
                        cands, ts, ios[ci], in_vars(cases[ci]), out_vars(cases[ci]),
                        svd_sim[ci], eq_vars_l, eq_domains_l, case_text(cases[ci]),
                        graph, ios[ci], use_existing, set_mask,
                    )
                    scores = model(torch.tensor(feats, dtype=torch.float32)).numpy().ravel()
                    if greedy and set_mask:
                        ranked_topK = greedy_order(
                            cands, feats, scores, model, set_mask,
                            eq_vars_l, eq_domains_l, out_vars(cases[ci]), ios[ci],
                            n_feat - len(set_mask),
                        )
                    else:
                        order = sorted(range(len(cands)), key=lambda k: -scores[k])
                        ranked_topK = [cands[k] for k in order]
                    ranks = compute_all_ranks(ranked_topK, corr, miss_rank=10_000)
                    cm = case_metrics(ranks)
                    cm["variant"] = norm(cases[ci].get("variant_type") or "?")
                    cm["case_id"] = cases[ci].get("case_id", f"idx_{ci}")
                    cm["n_input"] = len(in_vars(cases[ci]))
                    cm["n_output"] = len(out_vars(cases[ci]))
                    cm["n_sources"] = len({(eq_keys_l[j].split("__")[0] if j < len(eq_keys_l) else "?") for j in corr})
                    case_results.append(cm)

        agg = aggregate_metrics(case_results)
        agg["seed"] = seed
        per_seed_aggs.append(agg)
        if save_per_case:
            for cm in case_results:
                rec = {"seed": seed, "mode": mode, **cm}
                # remove non-serializable / bulky inner dicts; keep scalar metrics + Recall@K_correct
                rec.pop("FullRecall", None); rec.pop("Recall", None)
                per_case_records.append(rec)
        print(f"  [seed={seed}] {mode:14s}  "
              f"MRR_f={agg['MRR_first']:.4f}  "
              f"MRR_w={agg.get('MRR_worst', 0):.4f}  "
              f"MAP={agg['MAP']:.4f}  "
              f"R@K_c={agg.get('Recall@K_correct', 0):.4f}  "
              f"FullR@3={agg['FullRecall@3']:.4f}  "
              f"multiFullR@3={agg.get('multi_only__FullRecall@3', 0):.4f}  "
              f"n={agg['n_cases']}")

    keys = ["MRR_first", "MRR_worst", "MRR_avg", "MAP",
            "Precision@C", "Recall@K_correct",
            "FullRecall@3", "FullRecall@10",
            "Recall@3", "Recall@5", "Recall@10", "Recall@20",
            "multi_only__MRR_first", "multi_only__MRR_worst",
            "multi_only__MAP", "multi_only__FullRecall@3", "multi_only__FullRecall@10",
            "multi_only__Recall@3", "multi_only__Recall@10",
            "multi_only__Recall@K_correct"]
    summary = {"mode": mode, "n_features": n_feat}
    for k in keys:
        vals = [a.get(k) for a in per_seed_aggs if a.get(k) is not None]
        if vals:
            summary[k] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0, 4),
            }
    summary["per_seed"] = per_seed_aggs
    if save_per_case:
        summary["per_case"] = per_case_records
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--modes", type=str,
                        default="reranker-7,reranker-10,reranker-10S,reranker-13")
    parser.add_argument("--variants", type=str, default=None,
                        help="使用する variant_type のカンマ区切りリスト "
                             "（例: original,multisource_v3,dae_X3）。"
                             "プレフィックス指定可（例: dae_）。"
                             "未指定なら全 variant を使用")
    parser.add_argument("--output", type=str, default=None,
                        help="結果 JSON の出力パス（指定なければ set_aware_results.json）")
    parser.add_argument("--save-per-case", action="store_true",
                        help="各ケースごとの評価結果も JSON に保存（特性別分析用）")
    # Phase 5: hyperparameter sweep 用の追加フラグ
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="Reranker MLP の隠れ層次元（default: 64）")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="Pairwise margin loss の margin（default: 0.1）")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="訓練バッチサイズ（default: 16）")
    parser.add_argument("--n-neg-samples", type=int, default=8,
                        help="positive 1 件あたりの negative サンプル数（default: 8）")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="AdamW の weight_decay（default: 1e-4）")
    parser.add_argument("--loss", type=str, default="pairwise",
                        choices=["pairwise", "listwise"],
                        help="損失関数（pairwise margin / listwise softmax CE）")
    parser.add_argument("--hard-neg", action="store_true",
                        help="hard negative マイニング（現在スコア上位の negative を採用）")
    parser.add_argument("--greedy", action="store_true",
                        help="推論時 greedy 集合構築（既選択集合を参照に set-aware を再計算）")
    args = parser.parse_args()

    all_seeds = [42, 123, 456, 789, 1024, 2024, 3141, 5926, 7777, 9999]
    seeds = all_seeds[:args.seeds]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    eqs = load_equations(); cases = load_cases()

    # variant フィルタ
    if args.variants:
        wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
        def matches(case_variant):
            for w in wanted:
                if w.endswith("_"):
                    if case_variant.startswith(w):
                        return True
                else:
                    if case_variant == w:
                        return True
            return False
        n_before = len(cases)
        cases = [c for c in cases if matches(c.get("variant_type", ""))]
        print(f"Variant filter: {wanted}")
        print(f"  ケース絞込み: {n_before} → {len(cases)} 件")
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
            save_per_case=args.save_per_case,
            hidden_dim=args.hidden_dim,
            margin=args.margin,
            batch_size=args.batch_size,
            n_neg_samples=args.n_neg_samples,
            weight_decay=args.weight_decay,
            loss_type=args.loss,
            hard_neg=args.hard_neg,
            greedy=args.greedy,
        )

    print(f"\n{'='*100}")
    print(f"=== 全体結果 ===")
    print(f"{'Method':<14s} | {'#f':>3s} | {'MRR_first':>10s} | {'FullR@3':>10s} | "
          f"{'MAP':>10s} | {'multi_FullR@3':>14s} | {'multi_MRR_w':>12s}")
    print("-" * 100)
    def _fmt(d, key, default="     -      "):
        v = d.get(key)
        if isinstance(v, dict) and "mean" in v:
            return f"{v['mean']:.4f}±{v['std']:.3f}"
        return default
    for m, r in results.items():
        print(f"{m:<14s} | {r['n_features']:>3d} | "
              f"{_fmt(r, 'MRR_first')} | "
              f"{_fmt(r, 'FullRecall@3')} | "
              f"{_fmt(r, 'MAP')} | "
              f"{_fmt(r, 'multi_only__FullRecall@3')} | "
              f"{_fmt(r, 'multi_only__MRR_worst')}")

    out_path = Path(args.output) if args.output else OUT_JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
