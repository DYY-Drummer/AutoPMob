"""
Two-stage retrieval with query-conditioned graph features.

Key insight: Naive GNN features failed because they were query-independent.
This script adds graph features that are conditioned on each query's I/O variables:

  Feature 7: 2-hop reachability — what fraction of query I/O variables can reach
             the candidate equation through the bipartite graph (variable sharing)?
  Feature 8: Graph neighbor overlap — Jaccard between the candidate's graph-neighbor
             equations and the set of equations that contain query variables.
  Feature 9: Indirect variable bridge — number of "bridge" variables that connect
             the candidate equation to equations containing query I/O variables,
             normalized by candidate's total variables.

These features capture INDIRECT connections that simple Jaccard misses.
For example: query asks for variable A, candidate eq doesn't contain A directly,
but shares variable B with another equation that contains A → bridge signal.

Comparison:
  baseline    : fixed-weight TF-IDF + Jaccard (reference)
  reranker-7  : 7 base features (no graph)
  reranker-10 : 7 base + 3 query-conditioned graph features
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
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

def load_equations():
    with open(UNIFIED_JSON, encoding="utf-8") as f:
        raw = json.load(f)
    return raw if isinstance(raw, list) else raw.get("equations", raw.get("papers", raw))

def load_cases():
    with open(TRAINING_JSON, encoding="utf-8") as f:
        return json.load(f)

def norm(s): return (s or "").strip()
def get_src(e): return norm(e.get("source_id") or "unknown") or "unknown"

def eq_key(e):
    eid = norm(e.get("eq_id") or "")
    return eid if "__" in eid else f"{get_src(e)}__{eid}" if eid else None

def case_src(c):
    mids = [norm(m) for m in (c.get("correct_model_ids") or [])]
    pref = [m.split("__", 1)[0] for m in mids if "__" in m]
    return Counter(pref).most_common(1)[0][0] if pref else "unknown"

def eq_text(e):
    return "\n".join(p for p in [norm(e.get("context_text") or ""), norm(e.get("equation") or ""), norm(e.get("domain") or "")] if p)

def case_text(c, io=False):
    ctx = norm(c.get("context") or "")
    if not io: return ctx
    ins = " ".join(norm(v) for v in (c.get("input_variables") or []) if norm(v))
    outs = " ".join(norm(v) for v in (c.get("output_variables") or []) if norm(v))
    return " ".join(filter(None, [ctx, f"INPUT {ins}" if ins else "", f"OUTPUT {outs}" if outs else ""]))

def io_vars(c): return {norm(v) for v in (c.get("input_variables") or []) + (c.get("output_variables") or []) if norm(v)}
def in_vars(c): return {norm(v) for v in (c.get("input_variables") or []) if norm(v)}
def out_vars(c): return {norm(v) for v in (c.get("output_variables") or []) if norm(v)}

def eq_vars(e):
    vd = e.get("variables") or {}
    return {norm(k) for k in vd.keys() if norm(k)} if isinstance(vd, dict) else set()

def jaccard(a, b):
    u = len(a | b)
    return len(a & b) / u if u else 0.0

def src_split(srcs, seed=42):
    s = sorted(set(srcs))
    rng = random.Random(seed)
    rng.shuffle(s)
    nt = max(1, round(len(s) * 0.2))
    nv = max(1, round(len(s) * 0.2))
    return {"test": set(s[:nt]), "val": set(s[nt:nt+nv]), "train": set(s[nt+nv:])}

def r2m(ranks):
    if not ranks: return {"n":0.0,"MRR":0.0,"Recall@1":0.0,"Recall@3":0.0,"Recall@5":0.0,"Recall@10":0.0}
    n=len(ranks); mrr=sum(1/r for r in ranks)/n
    return {"n":float(n),"MRR":mrr,"Recall@1":sum(r<=1 for r in ranks)/n,"Recall@3":sum(r<=3 for r in ranks)/n,"Recall@5":sum(r<=5 for r in ranks)/n,"Recall@10":sum(r<=10 for r in ranks)/n}

def bci(vals, nb=2000, a=0.05):
    if len(vals)<2: return (vals[0] if vals else 0, vals[0] if vals else 0)
    rng=np.random.RandomState(0); arr=np.array(vals)
    b=np.array([arr[rng.randint(0,len(arr),len(arr))].mean() for _ in range(nb)])
    return (float(np.percentile(b,100*a/2)), float(np.percentile(b,100*(1-a/2))))


# ---------------------------------------------------------------------------
# Graph structure for query-conditioned features
# ---------------------------------------------------------------------------

class BipartiteGraph:
    """Precomputed bipartite graph structure for fast query-conditioned features."""

    def __init__(self, eq_vars_list: list[set[str]]):
        self.n_eq = len(eq_vars_list)
        self.eq_vars = eq_vars_list

        # Variable → set of equation indices
        self.var_to_eqs: dict[str, set[int]] = defaultdict(set)
        for i, vs in enumerate(eq_vars_list):
            for v in vs:
                self.var_to_eqs[v].add(i)

        # Equation → set of neighbor equations (share ≥1 variable)
        self.eq_neighbors: list[set[int]] = [set() for _ in range(self.n_eq)]
        for v, eqs in self.var_to_eqs.items():
            for i in eqs:
                self.eq_neighbors[i] |= eqs
        for i in range(self.n_eq):
            self.eq_neighbors[i].discard(i)

    def query_conditioned_features(
        self, cand_idx: int, query_vars: set[str],
    ) -> tuple[float, float, float]:
        """Compute 3 query-conditioned graph features for a (query, candidate) pair.

        Returns:
          f7: 2-hop reachability — fraction of query vars reachable from candidate in 2 hops
          f8: Graph neighbor overlap — Jaccard(candidate's graph-neighbors, equations containing query vars)
          f9: Indirect bridge ratio — fraction of candidate's vars that bridge to query-var equations
        """
        if not query_vars:
            return (0.0, 0.0, 0.0)

        cand_vars = self.eq_vars[cand_idx]

        # Equations that contain any query variable
        query_eqs: set[int] = set()
        for v in query_vars:
            query_eqs |= self.var_to_eqs.get(v, set())

        # f7: 2-hop reachability
        # Which query vars can reach the candidate through the bipartite graph?
        # A query var V reaches candidate eq_j if:
        #   - eq_j directly contains V (1-hop), OR
        #   - eq_j shares a variable with some equation that contains V (2-hop)
        reachable = 0
        cand_neighbor_eqs = self.eq_neighbors[cand_idx]
        for v in query_vars:
            if v in cand_vars:
                reachable += 1
            else:
                eqs_with_v = self.var_to_eqs.get(v, set())
                if eqs_with_v & cand_neighbor_eqs:
                    reachable += 1
        f7 = reachable / len(query_vars)

        # f8: Graph neighbor overlap
        # Jaccard between candidate's graph-neighbors and query-var-containing equations
        f8 = jaccard(cand_neighbor_eqs, query_eqs)

        # f9: Indirect bridge ratio
        # How many of candidate's variables serve as "bridges" to query-var equations?
        # A bridge variable = a variable in the candidate that is also in at least
        # one equation containing a query variable (but the variable itself is NOT a query var)
        bridges = 0
        for v in cand_vars:
            if v in query_vars:
                continue  # direct match, not a bridge
            eqs_with_v = self.var_to_eqs.get(v, set())
            if eqs_with_v & query_eqs:
                bridges += 1
        f9 = bridges / len(cand_vars) if cand_vars else 0.0

        return (f7, f8, f9)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Reranker(nn.Module):
    def __init__(self, n_features, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(
    cand_indices, text_sim, io_set, input_v, output_v, svd_sim,
    eq_vars_list, eq_domains, case_ctx_str,
    graph: BipartiteGraph | None = None,
    query_vars: set[str] | None = None,
) -> np.ndarray:
    n = len(cand_indices)
    use_graph = graph is not None and query_vars is not None
    n_feat = 10 if use_graph else 7
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

        if use_graph:
            f7, f8, f9 = graph.query_conditioned_features(j, query_vars)
            feats[k, 7] = f7
            feats[k, 8] = f8
            feats[k, 9] = f9

    return feats


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

def stage1(ci, X_ctx, X_eq, io_set, eq_vars_list, k=50):
    from sklearn.metrics.pairwise import cosine_similarity
    ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
    vs = np.array([jaccard(io_set, eq_vars_list[j]) for j in range(len(eq_vars_list))], dtype=np.float32)
    return np.argsort(-(0.7*ts + 0.3*vs))[:k].tolist()


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_mode(mode, seeds, cases, eq_keys_l, eq_texts_l, eq_vars_l, eq_domains_l, eq_sources_l,
             correct_lists, case_srcs, top_k, epochs, lr):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics.pairwise import cosine_similarity

    n_eq = len(eq_keys_l)
    n_feat = {"baseline": 0, "reranker-7": 7, "reranker-10": 10}[mode]

    test_mrrs, var_mrrs = [], {}
    details = []

    # Build graph structure once (same for all seeds)
    graph = BipartiteGraph(eq_vars_l) if mode == "reranker-10" else None

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

        if mode == "baseline":
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
            details.append({"seed": seed, "m": m, "vr": {vt: r2m(r) for vt, r in vr.items()}})
            print(f"  [seed={seed}] {mode:12s}  MRR={m['MRR']:.4f}  R@1={m['Recall@1']:.4f}  n={m['n']:.0f}")
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
        ranks, vr = [], {}
        with torch.no_grad():
            for ci in te:
                corr = set(correct_lists[ci])
                if not corr: continue
                cands = stage1(ci, X_ctx, X_eq, ios[ci], eq_vars_l, top_k)
                if not any(j in corr for j in cands):
                    ranks.append(top_k+1)
                    vt = norm(cases[ci].get("variant_type") or "?")
                    vr.setdefault(vt, []).append(top_k+1)
                    continue
                ts = cosine_similarity(X_ctx[ci], X_eq).ravel()
                feats = compute_features(
                    cands, ts, ios[ci], in_vars(cases[ci]), out_vars(cases[ci]),
                    svd_sim[ci], eq_vars_l, eq_domains_l, case_text(cases[ci]),
                    graph, ios[ci],
                )
                sc = model(torch.tensor(feats, dtype=torch.float32)).numpy()
                reranked = sorted(range(len(cands)), key=lambda k: -sc[k])
                for r, k in enumerate(reranked, 1):
                    if cands[k] in corr:
                        ranks.append(r); vt = norm(cases[ci].get("variant_type") or "?")
                        vr.setdefault(vt, []).append(r); break

        m = r2m(ranks); test_mrrs.append(m["MRR"])
        for vt, vranks in vr.items():
            var_mrrs.setdefault(vt, []).append(r2m(vranks)["MRR"])
        details.append({"seed": seed, "m": m, "vr": {vt: r2m(r) for vt, r in vr.items()}})
        print(f"  [seed={seed}] {mode:12s}  MRR={m['MRR']:.4f}  R@1={m['Recall@1']:.4f}  n={m['n']:.0f}")

    mn = float(np.mean(test_mrrs)); sd = float(np.std(test_mrrs, ddof=1)) if len(test_mrrs)>1 else 0
    ci = bci(test_mrrs)
    va = {vt: {"MRR_mean": round(float(np.mean(ms)),4), "MRR_std": round(float(np.std(ms,ddof=1)) if len(ms)>1 else 0,4)} for vt, ms in var_mrrs.items()}
    return {"mode": mode, "MRR_mean": round(mn,4), "MRR_std": round(sd,4), "MRR_ci95": [round(ci[0],4),round(ci[1],4)], "n_features": n_feat, "by_variant": va, "per_seed": details}


# ---------------------------------------------------------------------------

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
        ek.append(k); et.append(eq_text(e)); ev.append(eq_vars(e)); ed.append(norm(e.get("domain") or "")); es.append(get_src(e))

    ki = {k: i for i, k in enumerate(ek)}
    cl = [[ki[norm(m)] for m in (c.get("correct_model_ids") or []) if norm(m) in ki] for c in cases]
    cs = [case_src(c) for c in cases]

    results = {}
    for mode in ["baseline", "reranker-7", "reranker-10"]:
        print(f"\n{'='*60}\nMode: {mode}\n{'='*60}")
        results[mode] = run_mode(mode, seeds, cases, ek, et, ev, ed, es, cl, cs, args.top_k, args.epochs, args.lr)

    print(f"\n{'='*60}\nCOMPARISON (5-seed aggregate)\n{'='*60}")
    print(f"{'Method':15s} | {'MRR':>15s} | {'95% CI':>20s} | {'orig MRR':>12s}")
    print("-" * 70)
    for mode, r in results.items():
        ci = r["MRR_ci95"]
        orig = r["by_variant"].get("original", {}).get("MRR_mean", "—")
        print(f"{mode:15s} | {r['MRR_mean']:.4f} ± {r['MRR_std']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {orig if isinstance(orig,str) else f'{orig:.4f}'}")

    print("\nBy variant:")
    vts = set()
    for r in results.values(): vts |= set(r["by_variant"].keys())
    for vt in sorted(vts):
        line = f"  {vt:25s}"
        for mode, r in results.items():
            m = r["by_variant"].get(vt, {}).get("MRR_mean", 0)
            line += f"  {mode}={m:.4f}"
        print(line)

    out = OUT_DIR / "query_conditioned_comparison.json"
    out.write_text(json.dumps({"config": vars(args), "results": results}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
