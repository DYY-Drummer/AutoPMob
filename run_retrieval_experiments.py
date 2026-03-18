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

Outputs:
  - experiments/retrieval_experiments.json
  - experiments/retrieval_experiments.md
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter
from dataclasses import dataclass, asdict
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
    edge_index = data.edge_index
    x_graph = data.x
    if x_graph.shape[0] != len(eq_keys):
        raise ValueError(f"Graph nodes ({x_graph.shape[0]}) != equations ({len(eq_keys)}).")

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
    x0 = F.normalize(x_graph, p=2, dim=-1).to(device)
    Q_t = torch.tensor(Q_svd, dtype=torch.float32).to(device)
    edge_index_dev = edge_index.to(device)

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
    # Method 4: gnn_refine_svd_residual (small grid)
    # -------------------
    E0_t = torch.tensor(E_svd, dtype=torch.float32).to(device)
    Qd_t = torch.tensor(Q_svd, dtype=torch.float32).to(device)

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
                E = gnn(E0_t, edge_index_dev)
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
            E = gnn(E0_t, edge_index_dev)
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


if __name__ == "__main__":
    main()

