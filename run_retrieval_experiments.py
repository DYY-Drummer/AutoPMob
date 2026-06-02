"""
Run retrieval experiments and hyperparameter search.

Goal:
  Try multiple methods and training hyperparameters, select best by validation MRR,
  then report test metrics.

Methods included:
  - baseline_mix: 0.7*TFIDF(context) + 0.3*Jaccard(IO vars, eq vars)
  - svd_cos: TFIDF(case_text incl IO) -> SVD(d), cosine with TFIDF(eq_text)->SVD(d)
  - q_mlp_to_x: train a query MLP to map SVD(case) -> equation_graph.x space; eq embeddings fixed to x (L2-norm)
  - gnn_refine_svd_residual: train residual GCN refiner on SVD eq embeddings; queries use SVD cosine space

Multi-seed mode (--seeds N):
  Run evaluation over N different random seeds (different train/val/test splits).
  Reports mean ± std and 95% bootstrap CI for each method.

Outputs:
  - experiments/retrieval_experiments.json
  - experiments/retrieval_experiments.md
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).parent
UNIFIED_JSON = ROOT / "unified_equations.json"
TRAINING_JSON = ROOT / "training_cases.json"
GRAPH_PT = ROOT / "equation_graph.pt"
OUT_DIR = ROOT / "experiments"
OUT_JSON = OUT_DIR / "retrieval_experiments.json"
OUT_MD = OUT_DIR / "retrieval_experiments.md"


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
    if "__" in eq_id:
        return eq_id
    return f"{get_source_id(e)}__{eq_id}"


def case_source_id(c: dict[str, Any]) -> str:
    mids = [norm(m) for m in (c.get("correct_model_ids") or [])]
    prefixes = [m.split("__", 1)[0] for m in mids if "__" in m]
    if not prefixes:
        return "unknown"
    return Counter(prefixes).most_common(1)[0][0] or "unknown"


def build_equation_text(e: dict[str, Any]) -> str:
    parts = [norm(e.get("context_text") or ""), norm(e.get("equation") or ""), norm(e.get("domain") or "")]
    return "\n".join([p for p in parts if p])


def build_case_text(c: dict[str, Any], include_io: bool = True) -> str:
    if not include_io:
        return norm(c.get("context") or "")
    ins = " ".join([norm(v) for v in (c.get("input_variables") or []) if norm(v)])
    outs = " ".join([norm(v) for v in (c.get("output_variables") or []) if norm(v)])
    return " ".join([norm(c.get("context") or ""), f"INPUT {ins}" if ins else "", f"OUTPUT {outs}" if outs else ""]).strip()


def get_case_io_vars(c: dict[str, Any]) -> set[str]:
    ins = [norm(v) for v in (c.get("input_variables") or [])]
    outs = [norm(v) for v in (c.get("output_variables") or [])]
    return {v for v in (ins + outs) if v}


def get_eq_vars(e: dict[str, Any]) -> set[str]:
    vars_dict = e.get("variables") or {}
    if isinstance(vars_dict, dict):
        return {norm(k) for k in vars_dict.keys() if norm(k)}
    return set()


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def make_source_split(source_ids: list[str], seed: int = 42, test_ratio: float = 0.2, val_ratio: float = 0.2) -> dict[str, set[str]]:
    srcs = sorted(set(source_ids))
    rng = random.Random(seed)
    rng.shuffle(srcs)
    n_test = max(1, int(round(len(srcs) * test_ratio))) if srcs else 0
    n_val = max(1, int(round(len(srcs) * val_ratio))) if srcs else 0
    test = set(srcs[:n_test])
    val = set(srcs[n_test : n_test + n_val])
    train = set(srcs[n_test + n_val :])
    return {"train": train, "val": val, "test": test}


def ranks_to_metrics(ranks: list[int]) -> dict[str, float]:
    if not ranks:
        return {"n": 0.0, "MRR": 0.0, "Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0, "Recall@10": 0.0}
    n = len(ranks)
    mrr = sum(1.0 / r for r in ranks) / n
    def r_at(k: int) -> float:
        return sum(1 for r in ranks if r <= k) / n
    return {"n": float(n), "MRR": float(mrr), "Recall@1": r_at(1), "Recall@3": r_at(3), "Recall@5": r_at(5), "Recall@10": r_at(10)}


def eval_by_scoring(
    case_indices: list[int],
    correct_lists: list[list[int]],
    score_fn: Callable[[int], np.ndarray],
) -> list[int]:
    ranks: list[int] = []
    for i in case_indices:
        correct = set(correct_lists[i])
        if not correct:
            continue
        scores = score_fn(i)
        order = np.argsort(-scores)
        best = None
        for r, j in enumerate(order, start=1):
            if int(j) in correct:
                best = r
                break
        if best is not None:
            ranks.append(int(best))
    return ranks


def eval_by_scoring_with_variant(
    case_indices: list[int],
    correct_lists: list[list[int]],
    cases: list[dict[str, Any]],
    score_fn: Callable[[int], np.ndarray],
) -> tuple[list[int], dict[str, list[int]]]:
    """Like eval_by_scoring but also returns per-variant_type ranks."""
    ranks: list[int] = []
    variant_ranks: dict[str, list[int]] = {}
    for i in case_indices:
        correct = set(correct_lists[i])
        if not correct:
            continue
        scores = score_fn(i)
        order = np.argsort(-scores)
        best = None
        for r, j in enumerate(order, start=1):
            if int(j) in correct:
                best = r
                break
        if best is not None:
            ranks.append(int(best))
            vt = (cases[i].get("variant_type") or "unknown").strip()
            variant_ranks.setdefault(vt, []).append(int(best))
    return ranks, variant_ranks


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Return (lower, upper) bootstrap percentile CI."""
    if len(values) < 2:
        m = values[0] if values else 0.0
        return (m, m)
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    boot_means = np.array([arr[rng.randint(0, len(arr), len(arr))].mean() for _ in range(n_boot)])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def aggregate_multi_seed_results(
    all_seed_results: dict[str, list[dict[str, float]]],
) -> dict[str, dict[str, Any]]:
    """Aggregate metrics across seeds: mean, std, 95% CI for each method."""
    agg = {}
    for method, seed_metrics_list in all_seed_results.items():
        metrics_keys = [k for k in seed_metrics_list[0].keys() if k != "n"]
        method_agg: dict[str, Any] = {"n_seeds": len(seed_metrics_list)}
        for mk in metrics_keys:
            vals = [sm[mk] for sm in seed_metrics_list]
            method_agg[f"{mk}_mean"] = round(float(np.mean(vals)), 4)
            method_agg[f"{mk}_std"] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)
            lo, hi = bootstrap_ci(vals)
            method_agg[f"{mk}_ci95"] = [round(lo, 4), round(hi, 4)]
        agg[method] = method_agg
    return agg


class QueryMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class ResidualGCNRefiner(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(dim, dim)
        self.conv2 = GCNConv(dim, dim)
        self.dropout = dropout
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        a = torch.clamp(self.alpha, -1.0, 1.0)
        y = x + a * h
        return F.normalize(y, p=2, dim=-1)


@dataclass
class ExperimentResult:
    method: str
    params: dict[str, Any]
    val: dict[str, float]
    test: dict[str, float]
    seconds: float


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    equations = load_equations()
    cases = load_training_cases()

    # equation lists aligned to graph order assumption (same as previous scripts)
    eq_keys: list[str] = []
    eq_texts: list[str] = []
    eq_vars: list[set[str]] = []
    eq_sources: list[str] = []
    for e in equations:
        k = equation_key(e)
        if not k:
            continue
        eq_keys.append(k)
        eq_texts.append(build_equation_text(e))
        eq_vars.append(get_eq_vars(e))
        eq_sources.append(get_source_id(e))

    data = torch.load(GRAPH_PT, weights_only=False)
    num_equations = int(getattr(data, "num_equations", data.x.shape[0]))
    # 式ノードのみの特徴（q_mlp_to_x で使用）。二部グラフの場合は先頭 num_equations 行。
    x_graph = data.x[:num_equations]
    # SVD refiner は式同士の隣接のみ使用（インデックス 0..num_equations-1）
    edge_index_eq = getattr(data, "edge_index_eq_eq", data.edge_index)
    edge_index = data.edge_index  # 二部グラフ全体（q_mlp では使わず、必要なら別途）
    if num_equations != len(eq_keys):
        raise ValueError(f"Graph num_equations ({num_equations}) != equations ({len(eq_keys)}).")

    key_to_idx = {k: i for i, k in enumerate(eq_keys)}
    correct_lists: list[list[int]] = []
    case_sources = []
    for c in cases:
        mids = [norm(m) for m in (c.get("correct_model_ids") or [])]
        correct = [key_to_idx[m] for m in mids if m in key_to_idx]
        correct_lists.append(correct)
        case_sources.append(case_source_id(c))

    split = make_source_split(eq_sources, seed=seed, test_ratio=0.2, val_ratio=0.2)
    train_cases = [i for i, s in enumerate(case_sources) if s in split["train"] and correct_lists[i]]
    val_cases = [i for i, s in enumerate(case_sources) if s in split["val"] and correct_lists[i]]
    test_cases = [i for i, s in enumerate(case_sources) if s in split["test"] and correct_lists[i]]

    # TF-IDF baseline matrices
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD

    tfidf_eq = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq_tfidf = tfidf_eq.fit_transform(eq_texts)

    # Precompute case TFIDF for context-only baseline (baseline_mix)
    case_contexts = [build_case_text(c, include_io=False) for c in cases]
    # We use the same vectorizer trained on eq texts for cosine
    X_case_context = tfidf_eq.transform(case_contexts)

    # Precompute SVD space (include IO in query)
    svd_dim = 256
    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    E_svd = svd.fit_transform(X_eq_tfidf)
    E_svd = E_svd / (np.linalg.norm(E_svd, axis=1, keepdims=True) + 1e-12)

    case_texts_io = [build_case_text(c, include_io=True) for c in cases]
    X_case_io = tfidf_eq.transform(case_texts_io)
    Q_svd = svd.transform(X_case_io)
    Q_svd = Q_svd / (np.linalg.norm(Q_svd, axis=1, keepdims=True) + 1e-12)

    results: list[ExperimentResult] = []

    # -------------------
    # Method 1: baseline_mix
    # -------------------
    t0 = time.time()
    w_text, w_var = 0.7, 0.3
    io_sets = [get_case_io_vars(c) for c in cases]

    def score_baseline_mix(i: int) -> np.ndarray:
        text_sim = cosine_similarity(X_case_context[i], X_eq_tfidf).ravel()
        var_sim = np.array([jaccard(io_sets[i], eq_vars[j]) for j in range(len(eq_vars))], dtype=np.float32)
        return w_text * text_sim + w_var * var_sim

    val_ranks = eval_by_scoring(val_cases, correct_lists, score_baseline_mix)
    test_ranks = eval_by_scoring(test_cases, correct_lists, score_baseline_mix)
    results.append(
        ExperimentResult(
            method="baseline_mix",
            params={"w_text": w_text, "w_var": w_var},
            val=ranks_to_metrics(val_ranks),
            test=ranks_to_metrics(test_ranks),
            seconds=time.time() - t0,
        )
    )

    # -------------------
    # Method 2: svd_cos
    # -------------------
    t0 = time.time()
    def score_svd_cos(i: int) -> np.ndarray:
        return (E_svd @ Q_svd[i]).astype(np.float32)
    val_ranks = eval_by_scoring(val_cases, correct_lists, score_svd_cos)
    test_ranks = eval_by_scoring(test_cases, correct_lists, score_svd_cos)
    results.append(
        ExperimentResult(
            method="svd_cos",
            params={"svd_dim": svd_dim},
            val=ranks_to_metrics(val_ranks),
            test=ranks_to_metrics(test_ranks),
            seconds=time.time() - t0,
        )
    )

    # -------------------
    # Method 3: q_mlp_to_x (grid search)
    # -------------------
    x0 = F.normalize(x_graph, p=2, dim=-1).to(device)  # [num_equations, 768]
    Q_t = torch.tensor(Q_svd, dtype=torch.float32).to(device)
    edge_index_dev = edge_index.to(device)  # q_mlp_to_x では GCN を使わない

    def train_q_mlp(lr: float, wd: float, epochs: int = 5, neg_per_case: int = 128, batch_size: int = 64, temperature: float = 0.07) -> QueryMLP:
        model = QueryMLP(in_dim=svd_dim, out_dim=x0.shape[1], dropout=0.1).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        rng = random.Random(seed)
        train_idx = train_cases[:]
        for _ in range(epochs):
            rng.shuffle(train_idx)
            for start in range(0, len(train_idx), batch_size):
                batch = train_idx[start : start + batch_size]
                if not batch:
                    continue
                losses = []
                for i in batch:
                    pos_list = correct_lists[i]
                    pos = rng.choice(pos_list)
                    # uniform negatives
                    negs = []
                    excl = set(pos_list)
                    while len(negs) < neg_per_case:
                        j = rng.randrange(0, x0.shape[0])
                        if j in excl:
                            continue
                        negs.append(j)
                    cand = [pos] + negs
                    q = model(Q_t[i].unsqueeze(0)).squeeze(0)  # [768]
                    logits = (x0[cand] @ q) / float(temperature)
                    target = torch.tensor([0], dtype=torch.long, device=device)
                    losses.append(F.cross_entropy(logits.unsqueeze(0), target))
                loss = torch.stack(losses).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return model

    def eval_q_mlp(model: QueryMLP, case_idx: list[int]) -> list[int]:
        model.eval()
        with torch.no_grad():
            ranks = []
            for i in case_idx:
                correct = set(correct_lists[i])
                if not correct:
                    continue
                q = model(Q_t[i].unsqueeze(0)).squeeze(0)  # [768]
                scores = (x0 @ q).detach().cpu().numpy()
                order = np.argsort(-scores)
                best = None
                for r, j in enumerate(order, start=1):
                    if int(j) in correct:
                        best = r
                        break
                if best is not None:
                    ranks.append(int(best))
            return ranks

    grid = [
        {"lr": 1e-3, "wd": 0.0},
        {"lr": 5e-4, "wd": 0.0},
        {"lr": 2e-4, "wd": 0.0},
        {"lr": 1e-3, "wd": 1e-4},
        {"lr": 5e-4, "wd": 1e-4},
        {"lr": 2e-4, "wd": 1e-4},
        {"lr": 1e-3, "wd": 1e-3},
        {"lr": 5e-4, "wd": 1e-3},
    ]

    best_val = -1.0
    best_model = None
    best_params = None
    for hp in grid:
        t0 = time.time()
        model = train_q_mlp(lr=hp["lr"], wd=hp["wd"], epochs=5, neg_per_case=128, batch_size=64)
        val_ranks = eval_q_mlp(model, val_cases)
        test_ranks = eval_q_mlp(model, test_cases)
        val_m = ranks_to_metrics(val_ranks)
        test_m = ranks_to_metrics(test_ranks)
        results.append(ExperimentResult(method="q_mlp_to_x", params=hp | {"epochs": 5, "neg_per_case": 128}, val=val_m, test=test_m, seconds=time.time() - t0))
        if val_m["MRR"] > best_val:
            best_val = val_m["MRR"]
            best_model = model
            best_params = hp

    # -------------------
    # Method 4: gnn_refine_svd_residual (small grid) — 式同士の隣接のみ使用
    # -------------------
    E0_t = torch.tensor(E_svd, dtype=torch.float32).to(device)
    Qd_t = torch.tensor(Q_svd, dtype=torch.float32).to(device)
    edge_index_eq_dev = edge_index_eq.to(device)

    def train_gnn_refiner(lr: float, wd: float, epochs: int = 5, neg_per_case: int = 128, batch_size: int = 64, temperature: float = 0.07) -> ResidualGCNRefiner:
        gnn = ResidualGCNRefiner(dim=svd_dim, dropout=0.1).to(device)
        opt = torch.optim.AdamW(gnn.parameters(), lr=lr, weight_decay=wd)
        rng = random.Random(seed)
        train_idx = train_cases[:]
        for _ in range(epochs):
            rng.shuffle(train_idx)
            for start in range(0, len(train_idx), batch_size):
                batch = train_idx[start : start + batch_size]
                if not batch:
                    continue
                gnn.train()
                E = gnn(E0_t, edge_index_eq_dev)
                losses = []
                for i in batch:
                    pos_list = correct_lists[i]
                    pos = rng.choice(pos_list)
                    negs = []
                    excl = set(pos_list)
                    while len(negs) < neg_per_case:
                        j = rng.randrange(0, E.shape[0])
                        if j in excl:
                            continue
                        negs.append(j)
                    cand = [pos] + negs
                    q = Qd_t[i]
                    logits = (E[cand] @ q) / float(temperature)
                    target = torch.tensor([0], dtype=torch.long, device=device)
                    losses.append(F.cross_entropy(logits.unsqueeze(0), target))
                loss = torch.stack(losses).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return gnn

    def eval_gnn_refiner(gnn: ResidualGCNRefiner, case_idx: list[int]) -> list[int]:
        gnn.eval()
        with torch.no_grad():
            E = gnn(E0_t, edge_index_eq_dev)
            ranks = []
            for i in case_idx:
                correct = set(correct_lists[i])
                if not correct:
                    continue
                q = Qd_t[i]
                scores = (E @ q).detach().cpu().numpy()
                order = np.argsort(-scores)
                best = None
                for r, j in enumerate(order, start=1):
                    if int(j) in correct:
                        best = r
                        break
                if best is not None:
                    ranks.append(int(best))
            return ranks

    grid2 = [
        {"lr": 5e-4, "wd": 1e-4},
        {"lr": 2e-4, "wd": 1e-4},
        {"lr": 5e-4, "wd": 1e-3},
    ]
    for hp in grid2:
        t0 = time.time()
        gnn = train_gnn_refiner(lr=hp["lr"], wd=hp["wd"], epochs=5, neg_per_case=128, batch_size=64)
        val_ranks = eval_gnn_refiner(gnn, val_cases)
        test_ranks = eval_gnn_refiner(gnn, test_cases)
        results.append(
            ExperimentResult(
                method="gnn_refine_svd_residual",
                params=hp | {"epochs": 5, "neg_per_case": 128},
                val=ranks_to_metrics(val_ranks),
                test=ranks_to_metrics(test_ranks),
                seconds=time.time() - t0,
            )
        )

    # Select best by val MRR
    best = max(results, key=lambda r: r.val.get("MRR", 0.0))

    payload = {
        "device": device,
        "seed": seed,
        "split": {k: sorted(list(v)) for k, v in split.items()},
        "counts": {"train_cases": len(train_cases), "val_cases": len(val_cases), "test_cases": len(test_cases)},
        "results": [asdict(r) for r in results],
        "best_by_val_mrr": asdict(best),
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown summary
    lines = []
    lines.append("# Retrieval experiment summary\n")
    lines.append(f"- Device: `{device}`\n")
    lines.append(f"- Cases: train={len(train_cases)}, val={len(val_cases)}, test={len(test_cases)}\n")
    lines.append("\n## Results (sorted by val MRR)\n")
    lines.append("| method | params | val MRR | val R@1 | val R@5 | test MRR | test R@1 | test R@5 | sec |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in sorted(results, key=lambda x: x.val.get("MRR", 0.0), reverse=True):
        p = json.dumps(r.params, ensure_ascii=False)
        lines.append(
            f"| `{r.method}` | `{p}` | {r.val.get('MRR',0):.4f} | {r.val.get('Recall@1',0):.4f} | {r.val.get('Recall@5',0):.4f} | "
            f"{r.test.get('MRR',0):.4f} | {r.test.get('Recall@1',0):.4f} | {r.test.get('Recall@5',0):.4f} | {r.seconds:.1f} |\n"
        )
    lines.append("\n## Best (by val MRR)\n")
    lines.append(f"- Method: `{best.method}`\n")
    lines.append(f"- Params: `{json.dumps(best.params, ensure_ascii=False)}`\n")
    lines.append(f"- Val: `{best.val}`\n")
    lines.append(f"- Test: `{best.test}`\n")
    OUT_MD.write_text("".join(lines), encoding="utf-8")

    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Best by val MRR: {best.method} {best.params} val={best.val} test={best.test}")


def main_multi_seed() -> None:
    """Run baseline methods over multiple seeds and report mean±std with CI."""
    parser = argparse.ArgumentParser(description="Retrieval experiments (multi-seed)")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds (default: 5)")
    parser.add_argument("--seed-list", type=str, default=None, help="Comma-separated seed list (e.g. 42,123,456)")
    parser.add_argument("--methods", type=str, default="baseline_mix,svd_cos",
                        help="Comma-separated methods to evaluate (default: baseline_mix,svd_cos)")
    parser.add_argument("--include-neural", action="store_true",
                        help="Also run q_mlp_to_x and gnn_refine_svd_residual (slow)")
    args = parser.parse_args()

    if args.seed_list:
        seeds = [int(s) for s in args.seed_list.split(",")]
    else:
        seeds = [42, 123, 456, 789, 1024][:args.seeds]

    methods = [m.strip() for m in args.methods.split(",")]
    if args.include_neural:
        methods = list(set(methods) | {"q_mlp_to_x", "gnn_refine_svd_residual"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    equations = load_equations()
    cases = load_training_cases()

    eq_keys: list[str] = []
    eq_texts: list[str] = []
    eq_vars: list[set[str]] = []
    eq_sources: list[str] = []
    for e in equations:
        k = equation_key(e)
        if not k:
            continue
        eq_keys.append(k)
        eq_texts.append(build_equation_text(e))
        eq_vars.append(get_eq_vars(e))
        eq_sources.append(get_source_id(e))

    key_to_idx = {k: i for i, k in enumerate(eq_keys)}
    correct_lists: list[list[int]] = []
    case_source_ids = []
    for c in cases:
        mids = [norm(m) for m in (c.get("correct_model_ids") or [])]
        correct = [key_to_idx[m] for m in mids if m in key_to_idx]
        correct_lists.append(correct)
        case_source_ids.append(case_source_id(c))

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD

    # Collect results: method -> list of per-seed metrics dicts
    test_by_method: dict[str, list[dict[str, float]]] = {}
    val_by_method: dict[str, list[dict[str, float]]] = {}
    # Per-variant results: method -> variant_type -> list of per-seed metrics
    test_by_method_variant: dict[str, dict[str, list[dict[str, float]]]] = {}

    all_seed_details: list[dict[str, Any]] = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        split = make_source_split(eq_sources, seed=seed)
        train_cases_idx = [i for i, s in enumerate(case_source_ids) if s in split["train"] and correct_lists[i]]
        val_cases_idx = [i for i, s in enumerate(case_source_ids) if s in split["val"] and correct_lists[i]]
        test_cases_idx = [i for i, s in enumerate(case_source_ids) if s in split["test"] and correct_lists[i]]

        seed_detail: dict[str, Any] = {
            "seed": seed,
            "n_train": len(train_cases_idx),
            "n_val": len(val_cases_idx),
            "n_test": len(test_cases_idx),
            "split_sources": {k: sorted(v) for k, v in split.items()},
            "results": {},
        }

        # TF-IDF (refit per seed is not necessary since corpus is the same, but split changes)
        tfidf_eq = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
        X_eq_tfidf = tfidf_eq.fit_transform(eq_texts)

        if "baseline_mix" in methods:
            w_text, w_var = 0.7, 0.3
            case_contexts = [build_case_text(c, include_io=False) for c in cases]
            X_case_ctx = tfidf_eq.transform(case_contexts)
            io_sets = [get_case_io_vars(c) for c in cases]

            def score_bm(i: int) -> np.ndarray:
                text_sim = cosine_similarity(X_case_ctx[i], X_eq_tfidf).ravel()
                var_sim = np.array([jaccard(io_sets[i], eq_vars[j]) for j in range(len(eq_vars))], dtype=np.float32)
                return w_text * text_sim + w_var * var_sim

            val_ranks, val_vr = eval_by_scoring_with_variant(val_cases_idx, correct_lists, cases, score_bm)
            test_ranks, test_vr = eval_by_scoring_with_variant(test_cases_idx, correct_lists, cases, score_bm)

            val_m = ranks_to_metrics(val_ranks)
            test_m = ranks_to_metrics(test_ranks)
            val_by_method.setdefault("baseline_mix", []).append(val_m)
            test_by_method.setdefault("baseline_mix", []).append(test_m)

            # Per-variant
            for vt, vr in test_vr.items():
                test_by_method_variant.setdefault("baseline_mix", {}).setdefault(vt, []).append(ranks_to_metrics(vr))

            seed_detail["results"]["baseline_mix"] = {"val": val_m, "test": test_m, "test_by_variant": {vt: ranks_to_metrics(vr) for vt, vr in test_vr.items()}}
            print(f"  baseline_mix  val_MRR={val_m['MRR']:.4f}  test_MRR={test_m['MRR']:.4f}  n_test={test_m['n']:.0f}")

        if "svd_cos" in methods:
            svd_dim = 256
            svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
            E_svd = svd.fit_transform(X_eq_tfidf)
            E_svd = E_svd / (np.linalg.norm(E_svd, axis=1, keepdims=True) + 1e-12)
            case_texts_io = [build_case_text(c, include_io=True) for c in cases]
            X_case_io = tfidf_eq.transform(case_texts_io)
            Q_svd = svd.transform(X_case_io)
            Q_svd = Q_svd / (np.linalg.norm(Q_svd, axis=1, keepdims=True) + 1e-12)

            def score_svd(i: int) -> np.ndarray:
                return (E_svd @ Q_svd[i]).astype(np.float32)

            val_ranks, val_vr = eval_by_scoring_with_variant(val_cases_idx, correct_lists, cases, score_svd)
            test_ranks, test_vr = eval_by_scoring_with_variant(test_cases_idx, correct_lists, cases, score_svd)

            val_m = ranks_to_metrics(val_ranks)
            test_m = ranks_to_metrics(test_ranks)
            val_by_method.setdefault("svd_cos", []).append(val_m)
            test_by_method.setdefault("svd_cos", []).append(test_m)

            for vt, vr in test_vr.items():
                test_by_method_variant.setdefault("svd_cos", {}).setdefault(vt, []).append(ranks_to_metrics(vr))

            seed_detail["results"]["svd_cos"] = {"val": val_m, "test": test_m, "test_by_variant": {vt: ranks_to_metrics(vr) for vt, vr in test_vr.items()}}
            print(f"  svd_cos       val_MRR={val_m['MRR']:.4f}  test_MRR={test_m['MRR']:.4f}  n_test={test_m['n']:.0f}")

        all_seed_details.append(seed_detail)

    # Aggregate
    test_agg = aggregate_multi_seed_results(test_by_method)
    val_agg = aggregate_multi_seed_results(val_by_method)

    # Per-variant aggregate
    variant_agg: dict[str, dict[str, dict[str, Any]]] = {}
    for method, vt_dict in test_by_method_variant.items():
        variant_agg[method] = {}
        for vt, metrics_list in vt_dict.items():
            if metrics_list:
                mrr_vals = [m["MRR"] for m in metrics_list]
                variant_agg[method][vt] = {
                    "n_seeds": len(mrr_vals),
                    "MRR_mean": round(float(np.mean(mrr_vals)), 4),
                    "MRR_std": round(float(np.std(mrr_vals, ddof=1)) if len(mrr_vals) > 1 else 0.0, 4),
                    "avg_n": round(float(np.mean([m["n"] for m in metrics_list])), 1),
                }

    # Save
    out_json = OUT_DIR / "retrieval_experiments_multiseed.json"
    payload = {
        "seeds": seeds,
        "methods": methods,
        "test_aggregate": test_agg,
        "val_aggregate": val_agg,
        "test_by_variant": variant_agg,
        "per_seed_details": all_seed_details,
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown summary
    out_md = OUT_DIR / "retrieval_experiments_multiseed.md"
    lines = []
    lines.append("# Multi-seed Retrieval Experiment Summary\n\n")
    lines.append(f"Seeds: {seeds}\n\n")

    lines.append("## Test Metrics (mean ± std, 95% CI)\n\n")
    lines.append("| Method | MRR | R@1 | R@5 | R@10 |\n")
    lines.append("|--------|-----|-----|-----|------|\n")
    for method, agg in sorted(test_agg.items()):
        mrr = f"{agg['MRR_mean']:.3f}±{agg['MRR_std']:.3f}"
        r1 = f"{agg['Recall@1_mean']:.3f}±{agg['Recall@1_std']:.3f}"
        r5 = f"{agg['Recall@5_mean']:.3f}±{agg['Recall@5_std']:.3f}"
        r10 = f"{agg['Recall@10_mean']:.3f}±{agg['Recall@10_std']:.3f}"
        ci = agg['MRR_ci95']
        mrr_full = f"{mrr} [{ci[0]:.3f}, {ci[1]:.3f}]"
        lines.append(f"| {method} | {mrr_full} | {r1} | {r5} | {r10} |\n")

    lines.append("\n## Test Metrics by Variant Type\n\n")
    lines.append("| Method | Variant | MRR (mean±std) | Avg n |\n")
    lines.append("|--------|---------|----------------|-------|\n")
    for method in sorted(variant_agg.keys()):
        for vt in sorted(variant_agg[method].keys()):
            va = variant_agg[method][vt]
            lines.append(f"| {method} | {vt} | {va['MRR_mean']:.3f}±{va['MRR_std']:.3f} | {va['avg_n']:.0f} |\n")

    lines.append("\n## Per-seed Details\n\n")
    for sd in all_seed_details:
        lines.append(f"### Seed {sd['seed']} (train={sd['n_train']}, val={sd['n_val']}, test={sd['n_test']})\n\n")
        for method, res in sd["results"].items():
            lines.append(f"- **{method}**: val MRR={res['val']['MRR']:.4f}, test MRR={res['test']['MRR']:.4f}\n")
            if res.get("test_by_variant"):
                for vt, vm in sorted(res["test_by_variant"].items()):
                    lines.append(f"  - {vt}: MRR={vm['MRR']:.4f} (n={vm['n']:.0f})\n")
        lines.append("\n")

    out_md.write_text("".join(lines), encoding="utf-8")

    # Print summary
    print("\n" + "=" * 60)
    print("MULTI-SEED SUMMARY")
    print("=" * 60)
    print(f"Seeds: {seeds}")
    for method, agg in sorted(test_agg.items()):
        ci = agg['MRR_ci95']
        print(f"\n{method}:")
        print(f"  Test MRR:    {agg['MRR_mean']:.4f} ± {agg['MRR_std']:.4f}  95%CI [{ci[0]:.4f}, {ci[1]:.4f}]")
        print(f"  Test R@1:    {agg['Recall@1_mean']:.4f} ± {agg['Recall@1_std']:.4f}")
        print(f"  Test R@5:    {agg['Recall@5_mean']:.4f} ± {agg['Recall@5_std']:.4f}")
        if method in variant_agg:
            print(f"  By variant:")
            for vt, va in sorted(variant_agg[method].items()):
                print(f"    {vt:25s}  MRR={va['MRR_mean']:.4f}±{va['MRR_std']:.4f}  (avg n={va['avg_n']:.0f})")

    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    import sys
    if "--seeds" in sys.argv or "--seed-list" in sys.argv or "--multi" in sys.argv:
        main_multi_seed()
    else:
        main()

