"""
Two-stage retrieval WITH GNN features.

Extends two_stage_retrieval.py by adding graph-based features to the reranker:
  Feature 7: GCN-refined equation embedding similarity (SVD space, residual GCN)
  Feature 8: GCN-refined equation embedding similarity (CodeBERT space)

The GNN is trained jointly with the reranker: during each training epoch,
the GNN updates equation embeddings, and these are used as reranker features.

Comparison modes:
  --mode baseline    : baseline_mix only (reference)
  --mode no-gnn      : 7-feature reranker (no graph)
  --mode gnn         : 9-feature reranker (with GNN features)
  --mode all         : run all three for comparison (default)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
UNIFIED_JSON = ROOT / "unified_equations.json"
TRAINING_JSON = ROOT / "training_cases.json"
GRAPH_PT = ROOT / "equation_graph.pt"
OUT_DIR = ROOT / "experiments"


# ---------------------------------------------------------------------------
# Shared helpers (same as two_stage_retrieval.py)
# ---------------------------------------------------------------------------

def load_equations() -> list[dict[str, Any]]:
    with open(UNIFIED_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "equations" in raw:
        return raw["equations"]
    return raw.get("papers", raw) if isinstance(raw, dict) else []


def load_training_cases() -> list[dict[str, Any]]:
    with open(TRAINING_JSON, encoding="utf-8") as f:
        return json.load(f)


def norm(s: str) -> str:
    return (s or "").strip()


def get_source_id(e: dict[str, Any]) -> str:
    return norm(e.get("source_id") or "unknown") or "unknown"


def equation_key(e: dict[str, Any]) -> str | None:
    eq_id = norm(e.get("eq_id") or "")
    if not eq_id:
        return None
    return eq_id if "__" in eq_id else f"{get_source_id(e)}__{eq_id}"


def case_source_id(c: dict[str, Any]) -> str:
    mids = [norm(m) for m in (c.get("correct_model_ids") or [])]
    prefixes = [m.split("__", 1)[0] for m in mids if "__" in m]
    return Counter(prefixes).most_common(1)[0][0] if prefixes else "unknown"


def build_equation_text(e: dict[str, Any]) -> str:
    parts = [norm(e.get("context_text") or ""), norm(e.get("equation") or ""), norm(e.get("domain") or "")]
    return "\n".join([p for p in parts if p])


def build_case_text(c: dict[str, Any], include_io: bool = False) -> str:
    ctx = norm(c.get("context") or "")
    if not include_io:
        return ctx
    ins = " ".join(norm(v) for v in (c.get("input_variables") or []) if norm(v))
    outs = " ".join(norm(v) for v in (c.get("output_variables") or []) if norm(v))
    return " ".join(filter(None, [ctx, f"INPUT {ins}" if ins else "", f"OUTPUT {outs}" if outs else ""]))


def get_io_vars(c: dict[str, Any]) -> set[str]:
    return {norm(v) for v in (c.get("input_variables") or []) + (c.get("output_variables") or []) if norm(v)}


def get_input_vars(c: dict[str, Any]) -> set[str]:
    return {norm(v) for v in (c.get("input_variables") or []) if norm(v)}


def get_output_vars(c: dict[str, Any]) -> set[str]:
    return {norm(v) for v in (c.get("output_variables") or []) if norm(v)}


def get_eq_vars(e: dict[str, Any]) -> set[str]:
    vd = e.get("variables") or {}
    return {norm(k) for k in vd.keys() if norm(k)} if isinstance(vd, dict) else set()


def jaccard(a: set[str], b: set[str]) -> float:
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def make_source_split(source_ids: list[str], seed: int = 42) -> dict[str, set[str]]:
    srcs = sorted(set(source_ids))
    rng = random.Random(seed)
    rng.shuffle(srcs)
    n_test = max(1, int(round(len(srcs) * 0.2)))
    n_val = max(1, int(round(len(srcs) * 0.2)))
    return {"test": set(srcs[:n_test]), "val": set(srcs[n_test:n_test + n_val]), "train": set(srcs[n_test + n_val:])}


def ranks_to_metrics(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        return {"n": 0.0, "MRR": 0.0, "Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0, "Recall@10": 0.0}
    n = len(ranks)
    mrr = sum(1.0 / r for r in ranks) / n
    r_at = lambda k: sum(1 for r in ranks if r <= k) / n
    return {"n": float(n), "MRR": float(mrr), "Recall@1": r_at(1), "Recall@3": r_at(3), "Recall@5": r_at(5), "Recall@10": r_at(10)}


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) < 2:
        m = values[0] if values else 0.0
        return (m, m)
    rng = np.random.RandomState(0)
    arr = np.array(values)
    boot = np.array([arr[rng.randint(0, len(arr), len(arr))].mean() for _ in range(n_boot)])
    return (float(np.percentile(boot, 100 * alpha / 2)), float(np.percentile(boot, 100 * (1 - alpha / 2))))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Reranker(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ResidualGCN(nn.Module):
    """Residual GCN refiner for equation embeddings."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(dim, dim)
        self.conv2 = GCNConv(dim, dim)
        self.dropout = dropout
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        a = torch.clamp(self.alpha, -1.0, 1.0)
        return F.normalize(x + a * h, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(
    cand_indices: list[int],
    text_sim: np.ndarray,
    io_set: set[str],
    input_vars: set[str],
    output_vars: set[str],
    svd_sim: np.ndarray,
    eq_vars: list[set[str]],
    eq_domains: list[str],
    case_ctx: str,
    gnn_svd_sim: np.ndarray | None = None,
    gnn_codebert_sim: np.ndarray | None = None,
) -> np.ndarray:
    """Compute feature vector for each (case, candidate) pair.

    Base features (0-6):
      0: TF-IDF cosine
      1: Jaccard (I/O)
      2: SVD cosine
      3: Input var overlap ratio
      4: Output var overlap ratio
      5: Eq var coverage
      6: Domain match

    GNN features (7-8, optional):
      7: GCN-refined SVD cosine (residual GCN on SVD embeddings)
      8: GCN-refined CodeBERT cosine (GCN on bipartite graph CodeBERT embeddings)
    """
    n = len(cand_indices)
    use_gnn = gnn_svd_sim is not None
    n_feat = 9 if use_gnn else 7
    feats = np.zeros((n, n_feat), dtype=np.float32)
    domain_lower = case_ctx.lower()

    for k, j in enumerate(cand_indices):
        ev = eq_vars[j]
        feats[k, 0] = text_sim[j]
        feats[k, 1] = jaccard(io_set, ev)
        feats[k, 2] = svd_sim[j]
        feats[k, 3] = len(input_vars & ev) / len(input_vars) if input_vars else 0.0
        feats[k, 4] = len(output_vars & ev) / len(output_vars) if output_vars else 0.0
        feats[k, 5] = len(io_set & ev) / len(ev) if ev else 0.0
        d = eq_domains[j].lower()
        feats[k, 6] = 1.0 if (d and d in domain_lower) or (domain_lower and domain_lower in d) else 0.0

        if use_gnn:
            feats[k, 7] = gnn_svd_sim[j] if gnn_svd_sim is not None else 0.0
            feats[k, 8] = gnn_codebert_sim[j] if gnn_codebert_sim is not None else 0.0

    return feats


# ---------------------------------------------------------------------------
# Stage 1: baseline retrieval
# ---------------------------------------------------------------------------

def stage1_retrieve(case_idx, X_case_ctx, X_eq_tfidf, io_set, eq_vars, top_k=50):
    from sklearn.metrics.pairwise import cosine_similarity
    text_sim = cosine_similarity(X_case_ctx[case_idx], X_eq_tfidf).ravel()
    var_sim = np.array([jaccard(io_set, eq_vars[j]) for j in range(len(eq_vars))], dtype=np.float32)
    scores = 0.7 * text_sim + 0.3 * var_sim
    return np.argsort(-scores)[:top_k].tolist()


# ---------------------------------------------------------------------------
# Training & Evaluation
# ---------------------------------------------------------------------------

def run_experiment(
    mode: str,  # "baseline", "no-gnn", "gnn"
    seeds: list[int],
    cases: list[dict[str, Any]],
    eq_keys: list[str],
    eq_texts: list[str],
    eq_vars_list: list[set[str]],
    eq_domains: list[str],
    eq_sources: list[str],
    correct_lists: list[list[int]],
    case_source_ids: list[str],
    graph_data: Any | None,
    top_k: int,
    epochs: int,
    lr: float,
) -> dict[str, Any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity

    n_eq = len(eq_keys)
    n_feat = {"baseline": 0, "no-gnn": 7, "gnn": 9}[mode]

    test_mrrs: list[float] = []
    variant_mrrs: dict[str, list[float]] = {}
    all_details: list[dict[str, Any]] = []

    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        split = make_source_split(eq_sources, seed=seed)
        train_idx = [i for i, s in enumerate(case_source_ids) if s in split["train"] and correct_lists[i]]
        test_idx = [i for i, s in enumerate(case_source_ids) if s in split["test"] and correct_lists[i]]

        # TF-IDF
        tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
        X_eq_tfidf = tfidf.fit_transform(eq_texts)
        case_ctx = [build_case_text(c, include_io=False) for c in cases]
        X_case_ctx = tfidf.transform(case_ctx)
        io_sets = [get_io_vars(c) for c in cases]

        # SVD
        svd = TruncatedSVD(n_components=256, random_state=seed)
        E_svd = svd.fit_transform(X_eq_tfidf)
        E_svd = E_svd / (np.linalg.norm(E_svd, axis=1, keepdims=True) + 1e-12)
        Q_svd = svd.transform(tfidf.transform([build_case_text(c, include_io=True) for c in cases]))
        Q_svd = Q_svd / (np.linalg.norm(Q_svd, axis=1, keepdims=True) + 1e-12)
        svd_sim = Q_svd @ E_svd.T

        # GNN embeddings (precomputed per seed)
        gnn_svd_sim = None
        gnn_cb_sim = None
        gnn_svd_model = None
        gnn_cb_model = None

        if mode == "gnn" and graph_data is not None:
            num_equations = int(getattr(graph_data, "num_equations", graph_data.x.shape[0]))
            edge_index_eq = getattr(graph_data, "edge_index_eq_eq", graph_data.edge_index)
            x_cb = F.normalize(graph_data.x[:num_equations], p=2, dim=-1)

            # Train SVD-space GCN refiner
            gnn_svd_model = ResidualGCN(dim=256, dropout=0.1)
            E_svd_t = torch.tensor(E_svd, dtype=torch.float32)
            opt_gnn_svd = torch.optim.AdamW(gnn_svd_model.parameters(), lr=5e-4, weight_decay=1e-4)
            rng_gnn = random.Random(seed)

            for _ in range(5):
                gnn_svd_model.train()
                E_refined = gnn_svd_model(E_svd_t, edge_index_eq)
                # InfoNCE-style loss
                gnn_losses = []
                sample_train = rng_gnn.sample(train_idx, min(64, len(train_idx)))
                for ci in sample_train:
                    pos_list = correct_lists[ci]
                    if not pos_list:
                        continue
                    pos = rng_gnn.choice(pos_list)
                    q = torch.tensor(Q_svd[ci], dtype=torch.float32)
                    pos_score = (E_refined[pos] @ q) / 0.07
                    negs = []
                    while len(negs) < 32:
                        j = rng_gnn.randrange(n_eq)
                        if j not in set(pos_list):
                            negs.append(j)
                    neg_scores = torch.stack([E_refined[j] @ q for j in negs]) / 0.07
                    logits = torch.cat([pos_score.unsqueeze(0), neg_scores])
                    gnn_losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([0])))
                if gnn_losses:
                    loss = torch.stack(gnn_losses).mean()
                    opt_gnn_svd.zero_grad()
                    loss.backward()
                    opt_gnn_svd.step()

            gnn_svd_model.eval()
            with torch.no_grad():
                E_gnn_svd = gnn_svd_model(E_svd_t, edge_index_eq).numpy()
            gnn_svd_sim = Q_svd @ E_gnn_svd.T

            # CodeBERT GCN (on bipartite graph, but extract eq-only embeddings)
            gnn_cb_model = ResidualGCN(dim=768, dropout=0.1)
            opt_gnn_cb = torch.optim.AdamW(gnn_cb_model.parameters(), lr=5e-4, weight_decay=1e-4)
            # Use bipartite graph edge_index for CodeBERT GCN
            full_x = graph_data.x  # includes variable nodes
            full_edge = graph_data.edge_index

            for _ in range(3):
                gnn_cb_model.train()
                z_full = gnn_cb_model(full_x, full_edge)
                z_eq = z_full[:num_equations]
                cb_losses = []
                sample_train = rng_gnn.sample(train_idx, min(64, len(train_idx)))
                for ci in sample_train:
                    pos_list = correct_lists[ci]
                    if not pos_list:
                        continue
                    pos = rng_gnn.choice(pos_list)
                    q = torch.tensor(Q_svd[ci], dtype=torch.float32)
                    # Project query to 768-dim via simple linear (q_svd → inner product with z_eq)
                    # Use SVD query as proxy: score = z_eq[j] @ mean(z_eq[pos_list])
                    q_proxy = z_eq[pos_list].mean(dim=0)
                    pos_score = (z_eq[pos] @ q_proxy) / 0.07
                    negs = []
                    while len(negs) < 32:
                        j = rng_gnn.randrange(n_eq)
                        if j not in set(pos_list):
                            negs.append(j)
                    neg_scores = torch.stack([z_eq[j] @ q_proxy for j in negs]) / 0.07
                    logits = torch.cat([pos_score.unsqueeze(0), neg_scores])
                    cb_losses.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([0])))
                if cb_losses:
                    loss = torch.stack(cb_losses).mean()
                    opt_gnn_cb.zero_grad()
                    loss.backward()
                    opt_gnn_cb.step()

            gnn_cb_model.eval()
            with torch.no_grad():
                z_full = gnn_cb_model(full_x, full_edge)
                z_eq_np = F.normalize(z_full[:num_equations], p=2, dim=-1).numpy()
            # CodeBERT GNN feature: for each equation, measure how much GNN
            # refinement shifts the embedding from its original position.
            # High shift = graph structure is informative for this equation.
            # This is a per-equation scalar, broadcast to all cases.
            cb_original = F.normalize(graph_data.x[:num_equations], p=2, dim=-1).numpy()
            gnn_shift = (z_eq_np * cb_original).sum(axis=1)  # cosine(refined, original) per eq
            gnn_cb_sim = np.tile(gnn_shift, (len(cases), 1))  # [n_cases, n_eq]

        if mode == "baseline":
            # Just evaluate baseline_mix
            ranks: list[int] = []
            vr: dict[str, list[int]] = {}
            for ci in test_idx:
                correct = set(correct_lists[ci])
                if not correct:
                    continue
                ts = cosine_similarity(X_case_ctx[ci], X_eq_tfidf).ravel()
                vs = np.array([jaccard(io_sets[ci], eq_vars_list[j]) for j in range(n_eq)], dtype=np.float32)
                scores = 0.7 * ts + 0.3 * vs
                order = np.argsort(-scores)
                for r, j in enumerate(order, start=1):
                    if int(j) in correct:
                        ranks.append(r)
                        vt = (cases[ci].get("variant_type") or "unknown").strip()
                        vr.setdefault(vt, []).append(r)
                        break
            metrics = ranks_to_metrics(ranks)
            test_mrrs.append(metrics["MRR"])
            for vt, vranks in vr.items():
                variant_mrrs.setdefault(vt, []).append(ranks_to_metrics(vranks)["MRR"])
            all_details.append({"seed": seed, "metrics": metrics, "by_variant": {vt: ranks_to_metrics(r) for vt, r in vr.items()}})
            print(f"  [seed={seed}] {mode:8s}  MRR={metrics['MRR']:.4f}  R@1={metrics['Recall@1']:.4f}  n={metrics['n']:.0f}")
            continue

        # Train reranker
        model = Reranker(n_features=n_feat, hidden=64)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        rng = random.Random(seed)
        margin = 0.1

        for epoch in range(epochs):
            rng.shuffle(train_idx)
            for start in range(0, len(train_idx), 16):
                batch = train_idx[start:start + 16]
                losses = []
                for ci in batch:
                    correct = set(correct_lists[ci])
                    if not correct:
                        continue
                    cands = stage1_retrieve(ci, X_case_ctx, X_eq_tfidf, io_sets[ci], eq_vars_list, top_k)
                    pos_in = [j for j in cands if j in correct]
                    neg_in = [j for j in cands if j not in correct]
                    if not pos_in or not neg_in:
                        continue

                    ts = cosine_similarity(X_case_ctx[ci], X_eq_tfidf).ravel()
                    feats = compute_features(
                        cands, ts, io_sets[ci], get_input_vars(cases[ci]), get_output_vars(cases[ci]),
                        svd_sim[ci], eq_vars_list, eq_domains, build_case_text(cases[ci]),
                        gnn_svd_sim[ci] if gnn_svd_sim is not None else None,
                        gnn_cb_sim[ci] if gnn_cb_sim is not None else None,
                    )
                    feats_t = torch.tensor(feats, dtype=torch.float32)
                    scores = model(feats_t)

                    # Map cand positions to scores
                    cand_to_k = {j: k for k, j in enumerate(cands)}
                    n_sample = min(8, len(neg_in))
                    neg_sample = rng.sample(neg_in, n_sample)
                    for p in pos_in:
                        for ng in neg_sample:
                            losses.append(F.relu(margin - scores[cand_to_k[p]] + scores[cand_to_k[ng]]))

                if losses:
                    loss = torch.stack(losses).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

        # Evaluate
        model.eval()
        ranks = []
        vr: dict[str, list[int]] = {}
        with torch.no_grad():
            for ci in test_idx:
                correct = set(correct_lists[ci])
                if not correct:
                    continue
                cands = stage1_retrieve(ci, X_case_ctx, X_eq_tfidf, io_sets[ci], eq_vars_list, top_k)
                if not any(j in correct for j in cands):
                    ranks.append(top_k + 1)
                    vt = (cases[ci].get("variant_type") or "unknown").strip()
                    vr.setdefault(vt, []).append(top_k + 1)
                    continue

                ts = cosine_similarity(X_case_ctx[ci], X_eq_tfidf).ravel()
                feats = compute_features(
                    cands, ts, io_sets[ci], get_input_vars(cases[ci]), get_output_vars(cases[ci]),
                    svd_sim[ci], eq_vars_list, eq_domains, build_case_text(cases[ci]),
                    gnn_svd_sim[ci] if gnn_svd_sim is not None else None,
                    gnn_cb_sim[ci] if gnn_cb_sim is not None else None,
                )
                scores = model(torch.tensor(feats, dtype=torch.float32)).numpy()
                reranked = sorted(range(len(cands)), key=lambda k: -scores[k])
                for r, k in enumerate(reranked, start=1):
                    if cands[k] in correct:
                        ranks.append(r)
                        vt = (cases[ci].get("variant_type") or "unknown").strip()
                        vr.setdefault(vt, []).append(r)
                        break

        metrics = ranks_to_metrics(ranks)
        test_mrrs.append(metrics["MRR"])
        for vt, vranks in vr.items():
            variant_mrrs.setdefault(vt, []).append(ranks_to_metrics(vranks)["MRR"])
        all_details.append({"seed": seed, "metrics": metrics, "by_variant": {vt: ranks_to_metrics(r) for vt, r in vr.items()}})
        print(f"  [seed={seed}] {mode:8s}  MRR={metrics['MRR']:.4f}  R@1={metrics['Recall@1']:.4f}  n={metrics['n']:.0f}")

    mean_mrr = float(np.mean(test_mrrs))
    std_mrr = float(np.std(test_mrrs, ddof=1)) if len(test_mrrs) > 1 else 0.0
    ci = bootstrap_ci(test_mrrs)

    # Variant aggregate
    variant_agg = {}
    for vt, mrrs in variant_mrrs.items():
        variant_agg[vt] = {
            "MRR_mean": round(float(np.mean(mrrs)), 4),
            "MRR_std": round(float(np.std(mrrs, ddof=1)) if len(mrrs) > 1 else 0.0, 4),
        }

    return {
        "mode": mode,
        "MRR_mean": round(mean_mrr, 4),
        "MRR_std": round(std_mrr, 4),
        "MRR_ci95": [round(ci[0], 4), round(ci[1], 4)],
        "n_features": n_feat,
        "by_variant": variant_agg,
        "per_seed": all_details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-stage retrieval with GNN features")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mode", choices=["baseline", "no-gnn", "gnn", "all"], default="all")
    args = parser.parse_args()

    seeds = [42, 123, 456, 789, 1024][:args.seeds]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    equations = load_equations()
    cases = load_training_cases()

    eq_keys, eq_texts, eq_vars_list, eq_domains, eq_sources = [], [], [], [], []
    for e in equations:
        k = equation_key(e)
        if not k:
            continue
        eq_keys.append(k)
        eq_texts.append(build_equation_text(e))
        eq_vars_list.append(get_eq_vars(e))
        eq_domains.append(norm(e.get("domain") or ""))
        eq_sources.append(get_source_id(e))

    key_to_idx = {k: i for i, k in enumerate(eq_keys)}
    correct_lists = []
    case_source_ids = []
    for c in cases:
        mids = [norm(m) for m in (c.get("correct_model_ids") or [])]
        correct_lists.append([key_to_idx[m] for m in mids if m in key_to_idx])
        case_source_ids.append(case_source_id(c))

    # Load graph
    graph_data = None
    if GRAPH_PT.exists():
        graph_data = torch.load(GRAPH_PT, weights_only=False)

    modes = [args.mode] if args.mode != "all" else ["baseline", "no-gnn", "gnn"]

    results = {}
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Mode: {mode}")
        print(f"{'='*60}")
        results[mode] = run_experiment(
            mode, seeds, cases, eq_keys, eq_texts, eq_vars_list, eq_domains, eq_sources,
            correct_lists, case_source_ids, graph_data, args.top_k, args.epochs, args.lr,
        )

    # Print comparison
    print(f"\n{'='*60}")
    print("COMPARISON (5-seed aggregate)")
    print(f"{'='*60}")
    print(f"{'Method':15s} | {'MRR':>15s} | {'95% CI':>20s} | {'orig MRR':>12s}")
    print("-" * 70)
    for mode, r in results.items():
        ci = r["MRR_ci95"]
        orig = r["by_variant"].get("original", {}).get("MRR_mean", "—")
        orig_s = f"{orig:.4f}" if isinstance(orig, float) else orig
        print(f"{mode:15s} | {r['MRR_mean']:.4f} ± {r['MRR_std']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {orig_s}")

    # Variant detail
    all_vts = set()
    for r in results.values():
        all_vts |= set(r["by_variant"].keys())
    if all_vts:
        print(f"\nBy variant type:")
        for vt in sorted(all_vts):
            line = f"  {vt:25s}"
            for mode, r in results.items():
                va = r["by_variant"].get(vt, {})
                mrr = va.get("MRR_mean", 0)
                line += f"  {mode}={mrr:.4f}"
            print(line)

    # Save
    out_json = OUT_DIR / "two_stage_gnn_comparison.json"
    payload = {"config": vars(args), "results": results}
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
