"""
GNN-based retriever (1-epoch sanity training).

Model:
  - Equation encoder: 2-layer GCN over equation_graph.pt (initialized from x)
  - Query encoder: TF-IDF(context) -> TruncatedSVD -> MLP -> 768-dim
  - Score: dot(query_emb, eq_emb)

Loss:
  - sampled InfoNCE: for each case, sample 1 positive equation and N negatives

Eval:
  - Recall@{1,3,5,10} on held-out case sources (paper-level split)
"""

from __future__ import annotations

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
OUT_JSON = ROOT / "gnn_train_eval_report.json"


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


@dataclass
class TrainConfig:
    seed: int = 42
    test_ratio: float = 0.2
    svd_dim: int = 256
    neg_per_case: int = 64
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 1
    device: str = "cpu"


class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 768, out_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        from torch_geometric.nn import GCNConv

        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


class QueryEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        q = self.net(q)
        q = F.normalize(q, p=2, dim=-1)
        return q


class EqTextEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, out_dim),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        e = self.net(e)
        e = F.normalize(e, p=2, dim=-1)
        return e


def make_paper_split(source_ids: list[str], test_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    srcs = sorted(set(source_ids))
    rng = random.Random(seed)
    rng.shuffle(srcs)
    n_test = max(1, int(round(len(srcs) * test_ratio))) if srcs else 0
    test = set(srcs[:n_test])
    train = set(srcs[n_test:])
    return train, test


def recall_at_k(ranks: list[int], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r <= k) / len(ranks)


def main() -> None:
    cfg = TrainConfig()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)

    # device
    cfg.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    equations = load_equations()
    cases = load_training_cases()

    # map equation_key -> node index (assume order matches graph nodes)
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
    assert getattr(data, "edge_index", None) is not None
    edge_index = data.edge_index
    if data.num_nodes != len(eq_keys):
        raise ValueError(f"Graph nodes ({data.num_nodes}) != equations ({len(eq_keys)}). Rebuild graph or align ordering.")

    key_to_idx = {k: i for i, k in enumerate(eq_keys)}

    # split cases by source_id (paper-level)
    case_sources = [case_source_id(c) for c in cases]
    train_src, test_src = make_paper_split(eq_sources, cfg.test_ratio, cfg.seed)
    train_case_idx = [i for i, s in enumerate(case_sources) if s in train_src]
    test_case_idx = [i for i, s in enumerate(case_sources) if s in test_src]

    # Build text features: TF-IDF -> SVD (fit on equation texts to avoid using test case text for fitting)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    vectorizer = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = vectorizer.fit_transform(eq_texts)
    svd = TruncatedSVD(n_components=cfg.svd_dim, random_state=cfg.seed)
    X_eq_svd = svd.fit_transform(X_eq)
    eq_svd = torch.tensor(X_eq_svd, dtype=torch.float32)

    def case_vec(i: int) -> torch.Tensor:
        # Include variable symbols, since many cases are disambiguated primarily by I/O.
        ins = " ".join([norm(v) for v in (cases[i].get("input_variables") or []) if norm(v)])
        outs = " ".join([norm(v) for v in (cases[i].get("output_variables") or []) if norm(v)])
        q = " ".join(
            [
                norm(cases[i].get("context") or ""),
                f"INPUT {ins}" if ins else "",
                f"OUTPUT {outs}" if outs else "",
            ]
        ).strip()
        v = vectorizer.transform([q])
        z = svd.transform(v)[0]
        return torch.tensor(z, dtype=torch.float32)

    # Filter cases that have at least one valid correct_model_id
    def valid_pos_indices(i: int) -> list[int]:
        mids = [norm(m) for m in (cases[i].get("correct_model_ids") or [])]
        return [key_to_idx[m] for m in mids if m in key_to_idx]

    train_case_idx = [i for i in train_case_idx if valid_pos_indices(i)]
    test_case_idx = [i for i in test_case_idx if valid_pos_indices(i)]

    # Models
    # Start from TF-IDF+SVD cosine space (stable baseline). Optionally refine equation embeddings with a GCN in the same space.
    use_gnn = False
    eq_gnn = (GCNEncoder(in_dim=cfg.svd_dim, hidden_dim=cfg.svd_dim, out_dim=cfg.svd_dim, dropout=0.1).to(cfg.device) if use_gnn else None)

    params = (list(eq_gnn.parameters()) if eq_gnn is not None else [])
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay) if params else None

    edge_index_dev = edge_index.to(cfg.device)
    eq_svd_dev = eq_svd.to(cfg.device)

    def sample_negs(exclude: set[int], n: int) -> list[int]:
        res = []
        while len(res) < n:
            j = random.randrange(0, len(eq_keys))
            if j in exclude:
                continue
            res.append(j)
        return res

    # Train 1 epoch
    if eq_gnn is not None:
        eq_gnn.train()
    rng = random.Random(cfg.seed)
    rng.shuffle(train_case_idx)

    total_loss = 0.0
    n_steps = 0

    for start in range(0, len(train_case_idx), cfg.batch_size):
        batch = train_case_idx[start : start + cfg.batch_size]
        if not batch:
            continue

        z_eq = F.normalize(eq_svd_dev, p=2, dim=-1)
        if eq_gnn is not None:
            z_eq = eq_gnn(z_eq, edge_index_dev)

        losses = []
        for i in batch:
            pos_list = valid_pos_indices(i)
            pos = rng.choice(pos_list)
            negs = sample_negs(exclude=set(pos_list), n=cfg.neg_per_case)
            cand = [pos] + negs

            q = case_vec(i).to(cfg.device)
            qz = F.normalize(q, p=2, dim=-1)

            logits = (z_eq[cand] @ qz) / 0.07  # temperature
            target = torch.tensor([0], dtype=torch.long, device=cfg.device)
            loss = F.cross_entropy(logits.unsqueeze(0), target)
            losses.append(loss)

        loss_batch = torch.stack(losses).mean()
        if opt is not None:
            opt.zero_grad()
            loss_batch.backward()
            opt.step()

        total_loss += float(loss_batch.item())
        n_steps += 1

    avg_loss = total_loss / max(1, n_steps)

    # Eval
    if eq_gnn is not None:
        eq_gnn.eval()
    with torch.no_grad():
        z_eq = F.normalize(eq_svd_dev, p=2, dim=-1)
        if eq_gnn is not None:
            z_eq = eq_gnn(z_eq, edge_index_dev)
        ranks: list[int] = []
        for i in test_case_idx[:500]:  # cap for quick sanity
            pos_list = set(valid_pos_indices(i))
            q = case_vec(i).to(cfg.device)
            qz = F.normalize(q, p=2, dim=-1)
            scores = (z_eq @ qz).detach().cpu()
            ranked = torch.argsort(scores, descending=True)
            best_rank = None
            for r, idx in enumerate(ranked.tolist(), start=1):
                if idx in pos_list:
                    best_rank = r
                    break
            if best_rank is not None:
                ranks.append(best_rank)

    rep = {
        "config": asdict(cfg),
        "data": {
            "n_equations": len(eq_keys),
            "n_cases_total": len(cases),
            "n_cases_train": len(train_case_idx),
            "n_cases_test": len(test_case_idx),
            "n_sources_train": len(train_src),
            "n_sources_test": len(test_src),
        },
        "train": {"avg_loss": avg_loss, "steps": n_steps},
        "eval": {
            "n_evaluated": len(ranks),
            "Recall@1": recall_at_k(ranks, 1),
            "Recall@3": recall_at_k(ranks, 3),
            "Recall@5": recall_at_k(ranks, 5),
            "Recall@10": recall_at_k(ranks, 10),
            "MRR": sum(1.0 / r for r in ranks) / len(ranks) if ranks else 0.0,
        },
    }

    OUT_JSON.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== GNN retriever (sanity) ===")
    print(f"Device: {cfg.device}")
    print(f"Train cases: {len(train_case_idx)} | Test cases: {len(test_case_idx)}")
    print(f"Avg train loss: {avg_loss:.4f} (steps={n_steps})")
    print(f"Eval n={rep['eval']['n_evaluated']}  R@1={rep['eval']['Recall@1']:.4f}  R@5={rep['eval']['Recall@5']:.4f}  MRR={rep['eval']['MRR']:.4f}")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()

