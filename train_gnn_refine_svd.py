"""
GNN refinement step (paper-focused): refine equation embeddings with a GCN,
starting from a strong text baseline in TF-IDF+SVD space.

Why this script:
  - The SVD cosine baseline already works well when the query includes context + I/O vars.
  - We then ask: does graph structure (variable-sharing edges) improve retrieval further?

Method:
  - Build equation embeddings E0 in R^d via TF-IDF(eq_text)->SVD(d), L2-normalize.
  - Build query embeddings Q in same space via TF-IDF(case_text incl. I/O)->SVD(d), L2-normalize.
  - Learn a 2-layer GCN g(·) over the equation graph to produce refined embeddings E = g(E0).
  - Optimize sampled InfoNCE using training cases; evaluate Recall@K / MRR on held-out sources.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).parent
UNIFIED_JSON = ROOT / "unified_equations.json"
TRAINING_JSON = ROOT / "training_cases.json"
GRAPH_PT = ROOT / "equation_graph.pt"
OUT_JSON = ROOT / "gnn_refine_svd_report.json"


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


def build_case_text(c: dict[str, Any]) -> str:
    ins = " ".join([norm(v) for v in (c.get("input_variables") or []) if norm(v)])
    outs = " ".join([norm(v) for v in (c.get("output_variables") or []) if norm(v)])
    return " ".join([norm(c.get("context") or ""), f"INPUT {ins}" if ins else "", f"OUTPUT {outs}" if outs else ""]).strip()


class GCNRefiner(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(dim, dim)
        self.conv2 = GCNConv(dim, dim)
        self.dropout = dropout
        # Residual mixing. Start at 0 so initial behavior equals baseline E0.
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h, edge_index)
        # Residual: keep close to baseline embedding geometry
        a = torch.clamp(self.alpha, -1.0, 1.0)
        y = x + a * h
        return F.normalize(y, p=2, dim=-1)


@dataclass
class Report:
    config: dict[str, Any]
    data: dict[str, Any]
    train: dict[str, Any]
    eval: dict[str, Any]


def make_paper_split(source_ids: list[str], test_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    srcs = sorted(set(source_ids))
    rng = random.Random(seed)
    rng.shuffle(srcs)
    n_test = max(1, int(round(len(srcs) * test_ratio))) if srcs else 0
    return set(srcs[n_test:]), set(srcs[:n_test])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--svd-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--neg-per-case", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--test-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    equations = load_equations()
    cases = load_training_cases()

    eq_keys: list[str] = []
    eq_texts: list[str] = []
    eq_sources: list[str] = []
    for e in equations:
        k = equation_key(e)
        if not k:
            continue
        eq_keys.append(k)
        eq_texts.append(build_equation_text(e))
        eq_sources.append(get_source_id(e))

    data = torch.load(GRAPH_PT, weights_only=False)
    edge_index = data.edge_index
    if data.num_nodes != len(eq_keys):
        raise ValueError(f"Graph nodes ({data.num_nodes}) != equations ({len(eq_keys)}).")

    key_to_idx = {k: i for i, k in enumerate(eq_keys)}

    # Split cases by source_id
    case_sources = [case_source_id(c) for c in cases]
    train_src, test_src = make_paper_split(eq_sources, args.test_ratio, args.seed)
    train_case_idx = [i for i, s in enumerate(case_sources) if s in train_src]
    test_case_idx = [i for i, s in enumerate(case_sources) if s in test_src]

    def valid_pos_indices(i: int) -> list[int]:
        mids = [norm(m) for m in (cases[i].get("correct_model_ids") or [])]
        return [key_to_idx[m] for m in mids if m in key_to_idx]

    train_case_idx = [i for i in train_case_idx if valid_pos_indices(i)]
    test_case_idx = [i for i in test_case_idx if valid_pos_indices(i)]

    # TF-IDF/SVD space
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vectorizer = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = vectorizer.fit_transform(eq_texts)
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=args.seed)
    E0 = torch.tensor(svd.fit_transform(X_eq), dtype=torch.float32)  # [n_eq, d]
    E0 = F.normalize(E0, p=2, dim=-1)

    # Precompute Q
    X_q = svd.transform(vectorizer.transform([build_case_text(c) for c in cases]))
    Q = torch.tensor(X_q, dtype=torch.float32)
    Q = F.normalize(Q, p=2, dim=-1)

    E0_dev = E0.to(device)
    Q_dev = Q.to(device)
    edge_index_dev = edge_index.to(device)

    gnn = GCNRefiner(dim=args.svd_dim, dropout=0.1).to(device)
    opt = torch.optim.AdamW(gnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    rng = random.Random(args.seed)

    def sample_negs(exclude: set[int], n: int) -> list[int]:
        res = []
        while len(res) < n:
            j = rng.randrange(0, len(eq_keys))
            if j in exclude:
                continue
            res.append(j)
        return res

    total_loss = 0.0
    steps = 0
    for _ep in range(args.epochs):
        rng.shuffle(train_case_idx)
        for start in range(0, len(train_case_idx), args.batch_size):
            batch = train_case_idx[start : start + args.batch_size]
            if not batch:
                continue

            gnn.train()
            E = gnn(E0_dev, edge_index_dev)  # [n_eq, d]

            losses = []
            for i in batch:
                pos_list = valid_pos_indices(i)
                pos = rng.choice(pos_list)
                negs = sample_negs(exclude=set(pos_list), n=args.neg_per_case)
                cand = [pos] + negs
                q = Q_dev[i]  # [d]
                logits = (E[cand] @ q) / float(args.temperature)
                target = torch.tensor([0], dtype=torch.long, device=device)
                losses.append(F.cross_entropy(logits.unsqueeze(0), target))

            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            steps += 1

    avg_loss = total_loss / max(1, steps)

    # Eval
    gnn.eval()
    with torch.no_grad():
        E = gnn(E0_dev, edge_index_dev)
        ranks = []
        for i in test_case_idx:
            pos = set(valid_pos_indices(i))
            q = Q_dev[i]
            scores = (E @ q).detach().cpu()
            ranked = torch.argsort(scores, descending=True).tolist()
            best = None
            for r, j in enumerate(ranked, start=1):
                if j in pos:
                    best = r
                    break
            if best is not None:
                ranks.append(best)

    def r_at(k: int) -> float:
        return sum(1 for r in ranks if r <= k) / len(ranks) if ranks else 0.0

    rep = Report(
        config=vars(args) | {"device": device},
        data={
            "n_equations": len(eq_keys),
            "n_cases_total": len(cases),
            "n_cases_train": len(train_case_idx),
            "n_cases_test": len(test_case_idx),
            "n_sources_train": len(train_src),
            "n_sources_test": len(test_src),
        },
        train={"avg_loss": avg_loss, "steps": steps},
        eval={
            "n_evaluated": len(ranks),
            "Recall@1": r_at(1),
            "Recall@3": r_at(3),
            "Recall@5": r_at(5),
            "Recall@10": r_at(10),
            "MRR": (sum(1.0 / r for r in ranks) / len(ranks)) if ranks else 0.0,
        },
    )

    OUT_JSON.write_text(json.dumps(asdict(rep), indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== GNN refine (SVD space) ===")
    print(f"Device: {device}")
    print(f"Train cases: {len(train_case_idx)} | Test cases: {len(test_case_idx)}")
    print(f"Avg loss: {avg_loss:.4f} (steps={steps})")
    print(f"Eval n={rep.eval['n_evaluated']}  R@1={rep.eval['Recall@1']:.4f}  R@5={rep.eval['Recall@5']:.4f}  MRR={rep.eval['MRR']:.4f}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()

