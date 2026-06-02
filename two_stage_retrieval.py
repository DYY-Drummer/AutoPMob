"""
Two-stage retrieval for equation selection.

Stage 1 (Recall-oriented):
  baseline_mix (0.7*TFIDF + 0.3*Jaccard) retrieves Top-K candidates.
  High recall, fast, no training needed.

Stage 2 (Precision-oriented, Reranker):
  Neural cross-encoder reranks the Top-K candidates using richer features:
  - TF-IDF cosine (context)
  - Jaccard (I/O vars)
  - SVD cosine (context+IO vs equation)
  - Variable overlap features (input match, output match, total match)
  - Domain similarity signal
  Trained with pairwise loss: score(positive) > score(negative) + margin.

Rationale (from Phase 0 failure analysis):
  Baseline fails on cross-domain / context-divergent cases (6.1%).
  The reranker learns to combine signals that baseline uses rigidly (fixed weights)
  and adds features that baseline lacks (SVD semantic similarity, fine-grained
  variable matching). On easy cases it preserves baseline quality; on hard cases
  it can correct baseline's ranking.

Multi-seed evaluation with 95% bootstrap CI included.
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
OUT_DIR = ROOT / "experiments"


# ---------------------------------------------------------------------------
# Shared helpers
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
    if not prefixes:
        return "unknown"
    return Counter(prefixes).most_common(1)[0][0] or "unknown"


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
    ins = {norm(v) for v in (c.get("input_variables") or []) if norm(v)}
    outs = {norm(v) for v in (c.get("output_variables") or []) if norm(v)}
    return ins | outs


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
    return {
        "test": set(srcs[:n_test]),
        "val": set(srcs[n_test:n_test + n_val]),
        "train": set(srcs[n_test + n_val:]),
    }


def ranks_to_metrics(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        return {"n": 0.0, "MRR": 0.0, "Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0, "Recall@10": 0.0}
    n = len(ranks)
    mrr = sum(1.0 / r for r in ranks) / n
    def r_at(k: int) -> float:
        return sum(1 for r in ranks if r <= k) / n
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
# Stage 1: Baseline retrieval (Top-K candidates)
# ---------------------------------------------------------------------------

def stage1_retrieve(
    case_idx: int,
    X_case_ctx,  # sparse matrix row
    X_eq_tfidf,  # sparse matrix
    io_set: set[str],
    eq_vars: list[set[str]],
    top_k: int = 50,
    w_text: float = 0.7,
    w_var: float = 0.3,
) -> list[int]:
    """Return top-K equation indices by baseline_mix score."""
    from sklearn.metrics.pairwise import cosine_similarity
    text_sim = cosine_similarity(X_case_ctx[case_idx], X_eq_tfidf).ravel()
    var_sim = np.array([jaccard(io_set, eq_vars[j]) for j in range(len(eq_vars))], dtype=np.float32)
    scores = w_text * text_sim + w_var * var_sim
    top_indices = np.argsort(-scores)[:top_k]
    return top_indices.tolist()


# ---------------------------------------------------------------------------
# Stage 2: Reranker model
# ---------------------------------------------------------------------------

class Reranker(nn.Module):
    """Simple feedforward reranker over handcrafted features."""

    def __init__(self, n_features: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def compute_reranker_features(
    case_idx: int,
    cand_indices: list[int],
    text_sim_all: np.ndarray,
    var_sim_all: np.ndarray,
    svd_sim_all: np.ndarray,
    input_vars: set[str],
    output_vars: set[str],
    eq_vars: list[set[str]],
    eq_domains: list[str],
    case_domain_hint: str,
) -> np.ndarray:
    """Compute feature vectors for (case, candidate) pairs.

    Features (7-dim):
      0: TF-IDF cosine (context → equation)
      1: Jaccard (all I/O vars → equation vars)
      2: SVD cosine (context+IO → equation)
      3: Input var overlap ratio = |input ∩ eq_vars| / |input|
      4: Output var overlap ratio = |output ∩ eq_vars| / |output|
      5: Eq var coverage = |io ∩ eq_vars| / |eq_vars|
      6: Domain match (1.0 if case context mentions equation domain, else 0.0)
    """
    n = len(cand_indices)
    feats = np.zeros((n, 7), dtype=np.float32)
    io_vars = input_vars | output_vars
    domain_hint_lower = case_domain_hint.lower()

    for k, j in enumerate(cand_indices):
        ev = eq_vars[j]
        feats[k, 0] = text_sim_all[j]
        feats[k, 1] = var_sim_all[j]
        feats[k, 2] = svd_sim_all[j]
        feats[k, 3] = len(input_vars & ev) / len(input_vars) if input_vars else 0.0
        feats[k, 4] = len(output_vars & ev) / len(output_vars) if output_vars else 0.0
        feats[k, 5] = len(io_vars & ev) / len(ev) if ev else 0.0
        d = eq_domains[j].lower()
        feats[k, 6] = 1.0 if (d and d in domain_hint_lower) or (domain_hint_lower and domain_hint_lower in d) else 0.0

    return feats


# ---------------------------------------------------------------------------
# Training the reranker
# ---------------------------------------------------------------------------

def train_reranker(
    model: Reranker,
    train_case_indices: list[int],
    correct_lists: list[list[int]],
    cases: list[dict[str, Any]],
    text_sim_matrix,
    var_sim_matrix: np.ndarray,
    svd_sim_matrix: np.ndarray,
    eq_vars: list[set[str]],
    eq_domains: list[str],
    X_case_ctx,
    X_eq_tfidf,
    io_sets: list[set[str]],
    n_eq: int,
    top_k: int = 50,
    epochs: int = 10,
    lr: float = 1e-3,
    margin: float = 0.1,
    seed: int = 42,
) -> None:
    """Train reranker with pairwise margin loss on Stage-1 candidates."""
    from sklearn.metrics.pairwise import cosine_similarity

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rng = random.Random(seed)

    for epoch in range(epochs):
        rng.shuffle(train_case_indices)
        total_loss = 0.0
        n_updates = 0

        # Mini-batch: accumulate loss over several cases, then backward
        batch_size = 16
        for start in range(0, len(train_case_indices), batch_size):
            batch = train_case_indices[start:start + batch_size]
            losses = []

            for ci in batch:
                correct = set(correct_lists[ci])
                if not correct:
                    continue

                # Stage 1: get candidates
                cands = stage1_retrieve(ci, X_case_ctx, X_eq_tfidf, io_sets[ci], eq_vars, top_k=top_k)

                # Find positives and negatives within candidates
                pos_in_cands = [j for j in cands if j in correct]
                neg_in_cands = [j for j in cands if j not in correct]

                if not pos_in_cands or not neg_in_cands:
                    continue

                # Compute features for all candidates
                text_sim = cosine_similarity(X_case_ctx[ci], X_eq_tfidf).ravel()
                var_sim_row = np.array([jaccard(io_sets[ci], eq_vars[j]) for j in range(n_eq)], dtype=np.float32)

                all_cand = pos_in_cands + neg_in_cands
                feats = compute_reranker_features(
                    ci, all_cand, text_sim, var_sim_row, svd_sim_matrix[ci],
                    get_input_vars(cases[ci]), get_output_vars(cases[ci]),
                    eq_vars, eq_domains, build_case_text(cases[ci]),
                )

                feats_t = torch.tensor(feats, dtype=torch.float32)
                scores = model(feats_t)

                n_pos = len(pos_in_cands)
                pos_scores = scores[:n_pos]
                neg_scores = scores[n_pos:]

                # Sample pairs
                n_sample = min(8, len(neg_in_cands))
                neg_idx = rng.sample(range(len(neg_in_cands)), n_sample)

                for pi in range(n_pos):
                    for ni in neg_idx:
                        losses.append(F.relu(margin - pos_scores[pi] + neg_scores[ni]))

            if losses:
                loss = torch.stack(losses).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
                n_updates += 1

    return


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_two_stage(
    model: Reranker,
    case_indices: list[int],
    correct_lists: list[list[int]],
    cases: list[dict[str, Any]],
    X_case_ctx,
    X_eq_tfidf,
    io_sets: list[set[str]],
    eq_vars: list[set[str]],
    eq_domains: list[str],
    svd_sim_matrix: np.ndarray,
    n_eq: int,
    top_k: int = 50,
    eq_keys: list[str] | None = None,
) -> tuple[list[int], dict[str, list[int]]]:
    """Evaluate 2-stage retrieval. Returns overall ranks and per-variant ranks."""
    from sklearn.metrics.pairwise import cosine_similarity

    model.eval()
    ranks: list[int] = []
    variant_ranks: dict[str, list[int]] = {}

    with torch.no_grad():
        for ci in case_indices:
            correct = set(correct_lists[ci])
            if not correct:
                continue

            # Stage 1
            cands = stage1_retrieve(ci, X_case_ctx, X_eq_tfidf, io_sets[ci], eq_vars, top_k=top_k)

            # Check if any correct answer is in candidates
            correct_in_cands = [j for j in cands if j in correct]

            if not correct_in_cands:
                # Stage 1 missed all correct answers — rank > top_k
                ranks.append(top_k + 1)
                vt = (cases[ci].get("variant_type") or "unknown").strip()
                variant_ranks.setdefault(vt, []).append(top_k + 1)
                continue

            # Stage 2: rerank candidates
            text_sim = cosine_similarity(X_case_ctx[ci], X_eq_tfidf).ravel()
            var_sim_row = np.array([jaccard(io_sets[ci], eq_vars[j]) for j in range(n_eq)], dtype=np.float32)

            feats = compute_reranker_features(
                ci, cands, text_sim, var_sim_row, svd_sim_matrix[ci],
                get_input_vars(cases[ci]), get_output_vars(cases[ci]),
                eq_vars, eq_domains, build_case_text(cases[ci]),
            )

            feats_t = torch.tensor(feats, dtype=torch.float32)
            scores = model(feats_t).numpy()

            # Rerank by neural score
            reranked = sorted(range(len(cands)), key=lambda k: -scores[k])

            best_rank = None
            for r, k in enumerate(reranked, start=1):
                if cands[k] in correct:
                    best_rank = r
                    break

            if best_rank is not None:
                ranks.append(best_rank)
                vt = (cases[ci].get("variant_type") or "unknown").strip()
                variant_ranks.setdefault(vt, []).append(best_rank)

    return ranks, variant_ranks


def evaluate_baseline_only(
    case_indices: list[int],
    correct_lists: list[list[int]],
    cases: list[dict[str, Any]],
    X_case_ctx,
    X_eq_tfidf,
    io_sets: list[set[str]],
    eq_vars: list[set[str]],
    n_eq: int,
) -> tuple[list[int], dict[str, list[int]]]:
    """Evaluate baseline_mix (for comparison)."""
    from sklearn.metrics.pairwise import cosine_similarity

    ranks: list[int] = []
    variant_ranks: dict[str, list[int]] = {}

    for ci in case_indices:
        correct = set(correct_lists[ci])
        if not correct:
            continue

        text_sim = cosine_similarity(X_case_ctx[ci], X_eq_tfidf).ravel()
        var_sim = np.array([jaccard(io_sets[ci], eq_vars[j]) for j in range(n_eq)], dtype=np.float32)
        scores = 0.7 * text_sim + 0.3 * var_sim
        order = np.argsort(-scores)

        best_rank = None
        for r, j in enumerate(order, start=1):
            if int(j) in correct:
                best_rank = r
                break

        if best_rank is not None:
            ranks.append(best_rank)
            vt = (cases[ci].get("variant_type") or "unknown").strip()
            variant_ranks.setdefault(vt, []).append(best_rank)

    return ranks, variant_ranks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage retrieval experiment")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=50, help="Stage 1 candidate pool size")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    seeds = [42, 123, 456, 789, 1024][:args.seeds]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    equations = load_equations()
    cases = load_training_cases()

    # Build equation index
    eq_keys: list[str] = []
    eq_texts: list[str] = []
    eq_vars_list: list[set[str]] = []
    eq_domains: list[str] = []
    eq_sources: list[str] = []

    for e in equations:
        k = equation_key(e)
        if not k:
            continue
        eq_keys.append(k)
        eq_texts.append(build_equation_text(e))
        eq_vars_list.append(get_eq_vars(e))
        eq_domains.append(norm(e.get("domain") or ""))
        eq_sources.append(get_source_id(e))

    n_eq = len(eq_keys)
    key_to_idx = {k: i for i, k in enumerate(eq_keys)}

    correct_lists: list[list[int]] = []
    case_source_ids: list[str] = []
    for c in cases:
        mids = [norm(m) for m in (c.get("correct_model_ids") or [])]
        correct_lists.append([key_to_idx[m] for m in mids if m in key_to_idx])
        case_source_ids.append(case_source_id(c))

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    # Results collectors
    baseline_test_mrrs: list[float] = []
    twostage_test_mrrs: list[float] = []
    baseline_variant_mrrs: dict[str, list[float]] = {}
    twostage_variant_mrrs: dict[str, list[float]] = {}
    all_details: list[dict[str, Any]] = []

    for seed in seeds:
        print(f"\n{'='*50}")
        print(f"Seed {seed}")
        print(f"{'='*50}")

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        split = make_source_split(eq_sources, seed=seed)
        train_idx = [i for i, s in enumerate(case_source_ids) if s in split["train"] and correct_lists[i]]
        val_idx = [i for i, s in enumerate(case_source_ids) if s in split["val"] and correct_lists[i]]
        test_idx = [i for i, s in enumerate(case_source_ids) if s in split["test"] and correct_lists[i]]

        print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

        # TF-IDF
        tfidf = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
        X_eq_tfidf = tfidf.fit_transform(eq_texts)

        case_ctx = [build_case_text(c, include_io=False) for c in cases]
        X_case_ctx = tfidf.transform(case_ctx)

        io_sets = [get_io_vars(c) for c in cases]

        # SVD similarities
        svd_dim = 256
        svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
        E_svd = svd.fit_transform(X_eq_tfidf)
        E_svd = E_svd / (np.linalg.norm(E_svd, axis=1, keepdims=True) + 1e-12)

        case_io_texts = [build_case_text(c, include_io=True) for c in cases]
        Q_svd = svd.transform(tfidf.transform(case_io_texts))
        Q_svd = Q_svd / (np.linalg.norm(Q_svd, axis=1, keepdims=True) + 1e-12)
        svd_sim_matrix = Q_svd @ E_svd.T  # [n_cases, n_eq]

        # --- Baseline evaluation ---
        bl_ranks, bl_vr = evaluate_baseline_only(
            test_idx, correct_lists, cases, X_case_ctx, X_eq_tfidf, io_sets, eq_vars_list, n_eq,
        )
        bl_metrics = ranks_to_metrics(bl_ranks)
        baseline_test_mrrs.append(bl_metrics["MRR"])
        for vt, vranks in bl_vr.items():
            vm = ranks_to_metrics(vranks)
            baseline_variant_mrrs.setdefault(vt, []).append(vm["MRR"])

        print(f"  Baseline:   test MRR={bl_metrics['MRR']:.4f}  R@1={bl_metrics['Recall@1']:.4f}  (n={bl_metrics['n']:.0f})")

        # --- Train reranker ---
        model = Reranker(n_features=7, hidden=64)
        train_reranker(
            model, train_idx, correct_lists, cases,
            None, None, svd_sim_matrix, eq_vars_list, eq_domains,
            X_case_ctx, X_eq_tfidf, io_sets, n_eq,
            top_k=args.top_k, epochs=args.epochs, lr=args.lr, seed=seed,
        )

        # --- Two-stage evaluation ---
        ts_ranks, ts_vr = evaluate_two_stage(
            model, test_idx, correct_lists, cases,
            X_case_ctx, X_eq_tfidf, io_sets, eq_vars_list, eq_domains,
            svd_sim_matrix, n_eq, top_k=args.top_k,
        )
        ts_metrics = ranks_to_metrics(ts_ranks)
        twostage_test_mrrs.append(ts_metrics["MRR"])
        for vt, vranks in ts_vr.items():
            vm = ranks_to_metrics(vranks)
            twostage_variant_mrrs.setdefault(vt, []).append(vm["MRR"])

        print(f"  Two-stage:  test MRR={ts_metrics['MRR']:.4f}  R@1={ts_metrics['Recall@1']:.4f}  (n={ts_metrics['n']:.0f})")

        # Per-variant detail
        for vt in sorted(set(list(bl_vr.keys()) + list(ts_vr.keys()))):
            bl_vm = ranks_to_metrics(bl_vr.get(vt, []))
            ts_vm = ranks_to_metrics(ts_vr.get(vt, []))
            diff = ts_vm["MRR"] - bl_vm["MRR"] if bl_vm["n"] else 0
            marker = "+" if diff > 0.01 else ("-" if diff < -0.01 else "=")
            print(f"    {vt:25s}  BL={bl_vm['MRR']:.3f}  2S={ts_vm['MRR']:.3f}  ({marker}{abs(diff):.3f})  n={bl_vm['n']:.0f}")

        all_details.append({
            "seed": seed,
            "n_test": len(test_idx),
            "baseline": bl_metrics,
            "two_stage": ts_metrics,
            "baseline_by_variant": {vt: ranks_to_metrics(r) for vt, r in bl_vr.items()},
            "two_stage_by_variant": {vt: ranks_to_metrics(r) for vt, r in ts_vr.items()},
        })

    # --- Aggregate ---
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS (mean ± std)")
    print(f"{'='*60}")

    bl_mean, bl_std = float(np.mean(baseline_test_mrrs)), float(np.std(baseline_test_mrrs, ddof=1))
    ts_mean, ts_std = float(np.mean(twostage_test_mrrs)), float(np.std(twostage_test_mrrs, ddof=1))
    bl_ci = bootstrap_ci(baseline_test_mrrs)
    ts_ci = bootstrap_ci(twostage_test_mrrs)

    print(f"\nBaseline:   MRR = {bl_mean:.4f} ± {bl_std:.4f}  95%CI [{bl_ci[0]:.4f}, {bl_ci[1]:.4f}]")
    print(f"Two-stage:  MRR = {ts_mean:.4f} ± {ts_std:.4f}  95%CI [{ts_ci[0]:.4f}, {ts_ci[1]:.4f}]")

    diff = ts_mean - bl_mean
    print(f"\nDifference: {diff:+.4f}")

    print("\nBy variant type:")
    for vt in sorted(set(list(baseline_variant_mrrs.keys()) + list(twostage_variant_mrrs.keys()))):
        bl_v = baseline_variant_mrrs.get(vt, [])
        ts_v = twostage_variant_mrrs.get(vt, [])
        if bl_v and ts_v:
            print(f"  {vt:25s}  BL={np.mean(bl_v):.4f}±{np.std(bl_v, ddof=1):.4f}  "
                  f"2S={np.mean(ts_v):.4f}±{np.std(ts_v, ddof=1):.4f}  "
                  f"diff={np.mean(ts_v)-np.mean(bl_v):+.4f}")

    # Save
    out_json = OUT_DIR / "two_stage_retrieval.json"
    payload = {
        "config": {"seeds": seeds, "top_k": args.top_k, "epochs": args.epochs, "lr": args.lr, "n_features": 7},
        "aggregate": {
            "baseline": {"MRR_mean": round(bl_mean, 4), "MRR_std": round(bl_std, 4), "MRR_ci95": [round(bl_ci[0], 4), round(bl_ci[1], 4)]},
            "two_stage": {"MRR_mean": round(ts_mean, 4), "MRR_std": round(ts_std, 4), "MRR_ci95": [round(ts_ci[0], 4), round(ts_ci[1], 4)]},
            "difference": round(diff, 4),
        },
        "per_seed": all_details,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
