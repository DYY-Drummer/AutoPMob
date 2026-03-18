"""
Baseline retrieval/evaluation for equation selection.

Task:
  Given a training case (context + input/output vars), rank candidate equations.

Baseline score:
  score = w_text * cosine(TF-IDF(context), TF-IDF(eq_text)) + w_var * Jaccard(io_vars, eq_vars)

Evaluation:
  - MRR
  - Recall@K for K in {1,3,5,10}

Splits:
  - default: random split on source_id (paper-level split) to reduce leakage
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).parent
UNIFIED_JSON = ROOT / "unified_equations.json"
TRAINING_JSON = ROOT / "training_cases.json"
OUT_JSON = ROOT / "baseline_eval_report.json"


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


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def safe_mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def build_equation_text(e: dict[str, Any]) -> str:
    # Context is most informative; include equation string and domain as weak signals.
    parts = [
        norm(e.get("context_text") or ""),
        norm(e.get("equation") or ""),
        norm(e.get("domain") or ""),
    ]
    return "\n".join([p for p in parts if p])


def build_case_text(c: dict[str, Any]) -> str:
    # Case context is the query text.
    parts = [norm(c.get("context") or "")]
    return "\n".join([p for p in parts if p])


def get_case_io_vars(c: dict[str, Any]) -> set[str]:
    ins = [norm(v) for v in (c.get("input_variables") or [])]
    outs = [norm(v) for v in (c.get("output_variables") or [])]
    return {v for v in (ins + outs) if v}


def get_eq_vars(e: dict[str, Any]) -> set[str]:
    vars_dict = e.get("variables") or {}
    if isinstance(vars_dict, dict):
        return {norm(k) for k in vars_dict.keys() if norm(k)}
    return set()


def get_source_id(e: dict[str, Any]) -> str:
    return norm(e.get("source_id") or "unknown") or "unknown"


def equation_key(e: dict[str, Any]) -> str | None:
    """
    Return globally-unique equation identifier.
    - If eq_id already looks global (contains '__'), keep it.
    - Else prefix with source_id: '{source_id}__{eq_id}'.
    """
    eq_id = norm(e.get("eq_id") or "")
    if not eq_id:
        return None
    if "__" in eq_id:
        return eq_id
    return f"{get_source_id(e)}__{eq_id}"


def make_paper_split(
    equations: list[dict[str, Any]],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """Return (train_source_ids, test_source_ids)."""
    srcs = sorted({get_source_id(e) for e in equations})
    rng = random.Random(seed)
    rng.shuffle(srcs)
    n_test = max(1, int(round(len(srcs) * test_ratio))) if srcs else 0
    test = set(srcs[:n_test])
    train = set(srcs[n_test:])
    return train, test


def case_source_id(c: dict[str, Any]) -> str:
    """
    Infer the paper/source_id for a case from correct_model_ids (<source_id>__<eq_id>).
    If mixed, return the most common prefix.
    """
    mids = [norm(m) for m in (c.get("correct_model_ids") or [])]
    prefixes = []
    for m in mids:
        if "__" in m:
            prefixes.append(m.split("__", 1)[0])
    if not prefixes:
        return "unknown"
    return Counter(prefixes).most_common(1)[0][0] or "unknown"


@dataclass
class EvalReport:
    n_equations: int
    n_cases_total: int
    n_cases_evaluable: int
    split: dict[str, Any]
    weights: dict[str, float]
    metrics: dict[str, float]
    recall_at_k: dict[str, float]
    per_variant_metrics: dict[str, dict[str, float]]
    notes: list[str]


def main() -> None:
    equations = load_equations()
    cases = load_training_cases()

    eq_id_to_index: dict[str, int] = {}
    eq_texts: list[str] = []
    eq_vars: list[set[str]] = []
    eq_sources: list[str] = []
    for i, e in enumerate(equations):
        eid = equation_key(e)
        if not eid:
            continue
        # ensure uniqueness
        if eid in eq_id_to_index:
            # If this happens, IDs are not globally unique; skip duplicates deterministically.
            continue
        eq_id_to_index[eid] = len(eq_texts)
        eq_texts.append(build_equation_text(e))
        eq_vars.append(get_eq_vars(e))
        eq_sources.append(get_source_id(e))

    # paper-level split: split CASES by source_id, evaluate on held-out sources.
    # Candidates remain all equations, which yields many evaluable test cases.
    train_src, test_src = make_paper_split(equations, test_ratio=0.2, seed=42)
    test_case_mask = [case_source_id(c) in test_src for c in cases]

    # Build TF-IDF on all equations (retrieval over full library).
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    cand_texts = eq_texts
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=50000,
        ngram_range=(1, 2),
        min_df=1,
    )
    X_eq = vectorizer.fit_transform(cand_texts)

    # weights
    w_text = 0.7
    w_var = 0.3

    ks = [1, 3, 5, 10]
    hits_at_k = Counter({k: 0 for k in ks})
    rr_list: list[float] = []

    # per-variant
    per_var_hits = defaultdict(lambda: Counter({k: 0 for k in ks}))
    per_var_rr: dict[str, list[float]] = defaultdict(list)

    n_eval = 0
    notes: list[str] = []

    # Precompute candidate vars/ids for speed (all equations)
    cand_vars = eq_vars
    inv_index = {idx: eid for eid, idx in eq_id_to_index.items()}
    cand_eq_ids = [inv_index[i] for i in range(len(eq_texts))]

    for idx_c, c in enumerate(cases):
        if not test_case_mask[idx_c]:
            continue
        cmids = [mid for mid in (c.get("correct_model_ids") or []) if mid in eq_id_to_index]
        if not cmids:
            continue

        correct_cand = set(cmids)

        q_text = build_case_text(c)
        if not q_text:
            continue

        q_vec = vectorizer.transform([q_text])
        text_sim = cosine_similarity(q_vec, X_eq).ravel()

        q_vars = get_case_io_vars(c)
        var_sim = [jaccard(q_vars, cand_vars[i]) for i in range(len(cand_vars))]

        scores = [w_text * float(text_sim[i]) + w_var * float(var_sim[i]) for i in range(len(cand_vars))]
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # find best rank among correct answers
        best_rank = None
        for r, idx in enumerate(ranked, start=1):
            if cand_eq_ids[idx] in correct_cand:
                best_rank = r
                break
        if best_rank is None:
            continue

        n_eval += 1
        rr = 1.0 / best_rank
        rr_list.append(rr)
        vt = norm(c.get("variant_type") or "unknown") or "unknown"
        per_var_rr[vt].append(rr)

        for k in ks:
            if best_rank <= k:
                hits_at_k[k] += 1
                per_var_hits[vt][k] += 1

    if n_eval == 0:
        notes.append("No evaluable cases after filtering (check ID alignment and split).")

    metrics = {
        "MRR": safe_mean(rr_list),
    }
    recall_at_k = {str(k): (hits_at_k[k] / n_eval if n_eval else 0.0) for k in ks}

    per_variant_metrics: dict[str, dict[str, float]] = {}
    for vt, rrs in per_var_rr.items():
        per_variant_metrics[vt] = {
            "MRR": safe_mean(rrs),
            **{f"Recall@{k}": (per_var_hits[vt][k] / len(rrs) if rrs else 0.0) for k in ks},
            "n": float(len(rrs)),
        }

    rep = EvalReport(
        n_equations=len(eq_texts),
        n_cases_total=len(cases),
        n_cases_evaluable=n_eval,
        split={
            "type": "paper_level_source_id_casesplit",
            "test_ratio": 0.2,
            "seed": 42,
            "n_source_ids_train": len(train_src),
            "n_source_ids_test": len(test_src),
            "n_candidate_equations": len(eq_texts),
            "n_test_cases": int(sum(1 for m in test_case_mask if m)),
        },
        weights={"w_text": w_text, "w_var": w_var},
        metrics=metrics,
        recall_at_k=recall_at_k,
        per_variant_metrics=per_variant_metrics,
        notes=notes,
    )

    OUT_JSON.write_text(json.dumps(asdict(rep), indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Baseline retrieval evaluation ===")
    print(f"Candidate equations: {rep.split['n_candidate_equations']}")
    print(f"Cases total: {rep.n_cases_total}")
    print(f"Cases evaluable: {rep.n_cases_evaluable}")
    print(f"MRR: {rep.metrics['MRR']:.4f}")
    for k in ks:
        print(f"Recall@{k}: {rep.recall_at_k[str(k)]:.4f}")
    print("")
    print("Per-variant (top):")
    for vt, m in sorted(per_variant_metrics.items(), key=lambda kv: (-kv[1].get("n", 0), kv[0]))[:10]:
        print(f"  {vt:<22} n={int(m['n'])}  MRR={m['MRR']:.4f}  R@1={m['Recall@1']:.4f}  R@5={m['Recall@5']:.4f}")
    print("")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()

