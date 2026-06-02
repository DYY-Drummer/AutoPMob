"""
Baseline failure analysis for equation retrieval.

Purpose:
  Identify WHERE and WHY the baseline_mix (0.7*TFIDF + 0.3*Jaccard) fails,
  to determine where GNN-based approaches could add value.

Failure pattern classification:
  (a) notation_mismatch: query I/O vars differ from equation variable names
  (b) context_divergence: low TF-IDF similarity between query context and correct equation text
  (c) multi_equation: correct answer requires multiple equations (harder retrieval)
  (d) cross_domain: correct equations come from a different domain than query implies

Outputs:
  - experiments/baseline_failure_analysis.json  (per-case RR + failure pattern)
  - figures/failure_analysis_summary.png        (visualization)
  - stdout summary
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent
UNIFIED_JSON = ROOT / "unified_equations.json"
TRAINING_JSON = ROOT / "training_cases.json"
OUT_DIR = ROOT / "experiments"
OUT_JSON = OUT_DIR / "baseline_failure_analysis.json"
FIG_DIR = ROOT / "figures"


# ---------------------------------------------------------------------------
# Shared helpers (from baseline_retrieval_eval.py)
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
    return norm(c.get("context") or "")


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


def make_source_split(
    source_ids: list[str], seed: int = 42, test_ratio: float = 0.2, val_ratio: float = 0.2,
) -> dict[str, set[str]]:
    srcs = sorted(set(source_ids))
    rng = random.Random(seed)
    rng.shuffle(srcs)
    n_test = max(1, int(round(len(srcs) * test_ratio))) if srcs else 0
    n_val = max(1, int(round(len(srcs) * val_ratio))) if srcs else 0
    test = set(srcs[:n_test])
    val = set(srcs[n_test : n_test + n_val])
    train = set(srcs[n_test + n_val :])
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Failure pattern classification
# ---------------------------------------------------------------------------

def classify_failure(
    case: dict[str, Any],
    correct_eq_indices: list[int],
    eq_vars_list: list[set[str]],
    eq_domains: list[str],
    text_sim_to_correct: list[float],
) -> list[str]:
    """Classify why the baseline might struggle on this case.

    Returns a list of applicable failure pattern tags.
    """
    patterns = []
    io_vars = get_case_io_vars(case)

    # (a) notation_mismatch: query I/O vars have low overlap with correct equation vars
    if io_vars:
        best_jaccard = max(jaccard(io_vars, eq_vars_list[j]) for j in correct_eq_indices)
        if best_jaccard < 0.3:
            patterns.append("notation_mismatch")

    # (b) context_divergence: low TF-IDF cosine between query context and correct equations
    if text_sim_to_correct:
        best_text_sim = max(text_sim_to_correct)
        if best_text_sim < 0.15:
            patterns.append("context_divergence")

    # (c) multi_equation: multiple correct equations needed
    if len(correct_eq_indices) >= 2:
        patterns.append("multi_equation")

    # (d) cross_domain: correct equations span multiple domains
    domains_of_correct = {eq_domains[j] for j in correct_eq_indices}
    case_ctx = build_case_text(case).lower()
    if domains_of_correct:
        domain_mentioned = any(d.lower() in case_ctx for d in domains_of_correct if d)
        if not domain_mentioned:
            patterns.append("cross_domain")

    if not patterns:
        patterns.append("other")

    return patterns


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    equations = load_equations()
    cases = load_training_cases()

    # Build equation index
    eq_id_to_index: dict[str, int] = {}
    eq_texts: list[str] = []
    eq_vars_list: list[set[str]] = []
    eq_domains: list[str] = []
    eq_sources: list[str] = []

    for e in equations:
        eid = equation_key(e)
        if not eid or eid in eq_id_to_index:
            continue
        eq_id_to_index[eid] = len(eq_texts)
        eq_texts.append(build_equation_text(e))
        eq_vars_list.append(get_eq_vars(e))
        eq_domains.append(norm(e.get("domain") or ""))
        eq_sources.append(get_source_id(e))

    n_eq = len(eq_texts)
    inv_index = {idx: eid for eid, idx in eq_id_to_index.items()}
    cand_eq_ids = [inv_index[i] for i in range(n_eq)]

    # Source-level split (same as run_retrieval_experiments.py)
    all_src = [get_source_id(e) for e in equations]
    split = make_source_split(all_src, seed=42)

    # TF-IDF
    vectorizer = TfidfVectorizer(lowercase=True, max_features=50000, ngram_range=(1, 2), min_df=1)
    X_eq = vectorizer.fit_transform(eq_texts)

    w_text, w_var = 0.7, 0.3

    # Analyze ALL cases (not just test), grouping by split
    results: list[dict[str, Any]] = []

    for idx_c, c in enumerate(cases):
        csrc = case_source_id(c)
        if csrc in split["train"]:
            split_label = "train"
        elif csrc in split["val"]:
            split_label = "val"
        elif csrc in split["test"]:
            split_label = "test"
        else:
            split_label = "unknown"

        cmids = [mid for mid in (c.get("correct_model_ids") or []) if mid in eq_id_to_index]
        if not cmids:
            continue

        correct_indices = [eq_id_to_index[mid] for mid in cmids]

        q_text = build_case_text(c)
        if not q_text:
            continue

        # Compute scores
        q_vec = vectorizer.transform([q_text])
        text_sim = cosine_similarity(q_vec, X_eq).ravel()
        io_vars = get_case_io_vars(c)
        var_sim = np.array([jaccard(io_vars, eq_vars_list[i]) for i in range(n_eq)])
        scores = w_text * text_sim + w_var * var_sim

        # Rank
        ranked = np.argsort(-scores)
        best_rank = None
        for r, idx in enumerate(ranked, start=1):
            if cand_eq_ids[int(idx)] in set(cmids):
                best_rank = r
                break

        if best_rank is None:
            continue

        rr = 1.0 / best_rank

        # Decompose score components for correct equations
        text_sims_correct = [float(text_sim[j]) for j in correct_indices]
        var_sims_correct = [float(var_sim[j]) for j in correct_indices]

        # Classify failure pattern
        failure_patterns = classify_failure(
            c, correct_indices, eq_vars_list, eq_domains, text_sims_correct,
        )

        # Top-3 distractors (higher-scored non-correct equations)
        distractors = []
        for idx in ranked[:min(best_rank - 1, 3)]:
            idx = int(idx)
            if cand_eq_ids[idx] not in set(cmids):
                distractors.append({
                    "eq_id": cand_eq_ids[idx],
                    "score": float(scores[idx]),
                    "text_sim": float(text_sim[idx]),
                    "var_sim": float(var_sim[idx]),
                    "domain": eq_domains[idx],
                })

        results.append({
            "case_id": c.get("case_id", f"case_{idx_c}"),
            "variant_type": norm(c.get("variant_type") or "unknown"),
            "split": split_label,
            "best_rank": best_rank,
            "reciprocal_rank": round(rr, 4),
            "n_correct_models": len(cmids),
            "correct_model_ids": cmids,
            "failure_patterns": failure_patterns,
            "best_text_sim_to_correct": round(max(text_sims_correct), 4),
            "best_var_sim_to_correct": round(max(var_sims_correct), 4),
            "best_score_to_correct": round(max(float(scores[j]) for j in correct_indices), 4),
            "top_distractors": distractors,
            "context_snippet": q_text[:200],
            "io_vars": sorted(io_vars),
        })

    # ---------------------------------------------------------------------------
    # Aggregate statistics
    # ---------------------------------------------------------------------------

    def compute_stats(subset: list[dict]) -> dict[str, Any]:
        if not subset:
            return {"n": 0, "MRR": 0.0}
        rrs = [r["reciprocal_rank"] for r in subset]
        ranks = [r["best_rank"] for r in subset]
        n = len(subset)
        return {
            "n": n,
            "MRR": round(sum(rrs) / n, 4),
            "Recall@1": round(sum(1 for r in ranks if r <= 1) / n, 4),
            "Recall@5": round(sum(1 for r in ranks if r <= 5) / n, 4),
            "Recall@10": round(sum(1 for r in ranks if r <= 10) / n, 4),
            "median_rank": int(np.median(ranks)),
            "mean_rank": round(float(np.mean(ranks)), 1),
        }

    # By split
    stats_by_split = {}
    for sl in ["train", "val", "test"]:
        subset = [r for r in results if r["split"] == sl]
        stats_by_split[sl] = compute_stats(subset)

    # By variant_type (test only)
    test_results = [r for r in results if r["split"] == "test"]
    stats_by_variant = {}
    for vt in sorted({r["variant_type"] for r in test_results}):
        subset = [r for r in test_results if r["variant_type"] == vt]
        stats_by_variant[vt] = compute_stats(subset)

    # By variant_type (all splits)
    stats_by_variant_all = {}
    for vt in sorted({r["variant_type"] for r in results}):
        subset = [r for r in results if r["variant_type"] == vt]
        stats_by_variant_all[vt] = compute_stats(subset)

    # Failure cases (RR < 0.5, i.e. best_rank >= 3)
    failures = [r for r in results if r["reciprocal_rank"] < 0.5]
    failures_test = [r for r in test_results if r["reciprocal_rank"] < 0.5]

    # Failure pattern distribution
    pattern_counts_all = Counter()
    pattern_counts_failures = Counter()
    for r in results:
        for p in r["failure_patterns"]:
            pattern_counts_all[p] += 1
    for r in failures:
        for p in r["failure_patterns"]:
            pattern_counts_failures[p] += 1

    summary = {
        "total_evaluable_cases": len(results),
        "total_failures_rr_lt_0.5": len(failures),
        "failure_rate": round(len(failures) / len(results), 4) if results else 0,
        "stats_by_split": stats_by_split,
        "stats_by_variant_type_test": stats_by_variant,
        "stats_by_variant_type_all": stats_by_variant_all,
        "failure_pattern_distribution_all": dict(pattern_counts_all.most_common()),
        "failure_pattern_distribution_failures_only": dict(pattern_counts_failures.most_common()),
        "test_failures": [
            {k: v for k, v in r.items() if k != "top_distractors"}
            for r in sorted(failures_test, key=lambda r: r["best_rank"], reverse=True)
        ],
    }

    # Save
    OUT_DIR.mkdir(exist_ok=True)
    report = {"summary": summary, "per_case_results": results}
    OUT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        FIG_DIR.mkdir(exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Baseline Failure Analysis", fontsize=14, fontweight="bold")

        # (1) RR distribution by variant_type
        ax = axes[0, 0]
        variant_types = sorted({r["variant_type"] for r in results})
        rr_by_vt = {vt: [r["reciprocal_rank"] for r in results if r["variant_type"] == vt] for vt in variant_types}
        ax.boxplot(rr_by_vt.values(), tick_labels=list(rr_by_vt.keys()), vert=True)
        ax.set_ylabel("Reciprocal Rank")
        ax.set_title("RR Distribution by Variant Type")
        ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="failure threshold")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)

        # (2) Failure pattern pie chart (failures only)
        ax = axes[0, 1]
        if pattern_counts_failures:
            labels = list(pattern_counts_failures.keys())
            sizes = list(pattern_counts_failures.values())
            ax.pie(sizes, labels=labels, autopct="%1.0f%%", startangle=140)
        ax.set_title(f"Failure Patterns (n={len(failures)} failures)")

        # (3) Text sim vs Var sim scatter (color by success/failure)
        ax = axes[1, 0]
        successes = [r for r in results if r["reciprocal_rank"] >= 0.5]
        if successes:
            ax.scatter(
                [r["best_text_sim_to_correct"] for r in successes],
                [r["best_var_sim_to_correct"] for r in successes],
                alpha=0.3, s=15, c="green", label=f"success (n={len(successes)})",
            )
        if failures:
            ax.scatter(
                [r["best_text_sim_to_correct"] for r in failures],
                [r["best_var_sim_to_correct"] for r in failures],
                alpha=0.6, s=25, c="red", marker="x", label=f"failure (n={len(failures)})",
            )
        ax.set_xlabel("Best TF-IDF cosine (query → correct eq)")
        ax.set_ylabel("Best Jaccard (I/O vars → correct eq vars)")
        ax.set_title("Score Components: Success vs Failure")
        ax.legend(fontsize=8)

        # (4) Rank distribution histogram
        ax = axes[1, 1]
        all_ranks = [r["best_rank"] for r in results]
        max_rank = min(max(all_ranks) if all_ranks else 50, 50)
        ax.hist(all_ranks, bins=range(1, max_rank + 2), edgecolor="black", alpha=0.7)
        ax.set_xlabel("Best Rank")
        ax.set_ylabel("Count")
        ax.set_title("Rank Distribution (all cases)")
        ax.axvline(x=3, color="red", linestyle="--", alpha=0.5, label="RR=0.5 threshold")
        ax.legend(fontsize=8)

        plt.tight_layout()
        fig_path = FIG_DIR / "failure_analysis_summary.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Saved figure: {fig_path}")
    except ImportError:
        print("matplotlib not available, skipping visualization.")

    # ---------------------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------------------
    print("\n=== Baseline Failure Analysis ===")
    print(f"Total evaluable cases: {summary['total_evaluable_cases']}")
    print(f"Total failures (RR < 0.5): {summary['total_failures_rr_lt_0.5']} "
          f"({summary['failure_rate']:.1%})")

    print("\n--- Stats by split ---")
    for sl, st in stats_by_split.items():
        print(f"  {sl:6s}  n={st['n']:4d}  MRR={st['MRR']:.4f}  R@1={st['Recall@1']:.4f}  "
              f"R@5={st['Recall@5']:.4f}  median_rank={st['median_rank']}")

    print("\n--- Stats by variant_type (ALL splits) ---")
    for vt, st in stats_by_variant_all.items():
        print(f"  {vt:25s}  n={st['n']:4d}  MRR={st['MRR']:.4f}  R@1={st['Recall@1']:.4f}")

    print("\n--- Failure pattern distribution (failures only) ---")
    for pat, cnt in pattern_counts_failures.most_common():
        print(f"  {pat:25s}  {cnt:4d}  ({cnt / len(failures):.0%})" if failures else "")

    print(f"\n--- Test set failures (n={len(failures_test)}) ---")
    for r in sorted(failures_test, key=lambda x: x["best_rank"], reverse=True)[:10]:
        print(f"  {r['case_id']:30s}  rank={r['best_rank']:3d}  "
              f"text_sim={r['best_text_sim_to_correct']:.3f}  "
              f"var_sim={r['best_var_sim_to_correct']:.3f}  "
              f"patterns={r['failure_patterns']}")

    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
