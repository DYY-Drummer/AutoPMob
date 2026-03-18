"""
Dataset health checks for AutoPMob.

Checks:
  - training_cases.json references valid equation IDs
  - input/output variables are at least consistent with equation variables
  - duplicates / near-duplicates (same signature) counts
  - variant_type distribution and required fields

Output:
  - prints a concise report to stdout
  - writes validate_dataset_report.json for downstream use
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent
UNIFIED_JSON = ROOT / "unified_equations.json"
TRAINING_JSON = ROOT / "training_cases.json"
OUT_JSON = ROOT / "validate_dataset_report.json"


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


def norm_var(v: str) -> str:
    return (v or "").strip()


def equation_key(e: dict[str, Any]) -> str | None:
    """
    Return globally-unique equation identifier.
    - If eq_id already looks global (contains '__'), keep it.
    - Else prefix with source_id: '{source_id}__{eq_id}'.
    """
    eq_id = (e.get("eq_id") or "").strip()
    if not eq_id:
        return None
    if "__" in eq_id:
        return eq_id
    source_id = (e.get("source_id") or "unknown").strip() or "unknown"
    return f"{source_id}__{eq_id}"


@dataclass
class HealthReport:
    n_equations: int
    n_cases: int

    # IDs
    missing_equation_ids_total: int
    cases_with_missing_equation_ids: int

    # required fields
    cases_missing_fields: dict[str, int]

    # variable consistency
    io_not_in_any_correct_model_vars_cases: int
    io_not_in_any_correct_model_vars_total_vars: int
    io_not_in_any_correct_model_vars_examples: list[dict[str, Any]]

    # duplicates
    duplicate_signature_groups: int
    duplicate_signature_cases: int

    # distributions
    variant_type_counts: dict[str, int]
    n_correct_models_hist: dict[str, int]
    n_input_vars_hist: dict[str, int]
    n_output_vars_hist: dict[str, int]


def main() -> None:
    equations = load_equations()
    cases = load_training_cases()

    eq_ids = {equation_key(e) for e in equations if equation_key(e)}
    id_to_vars: dict[str, set[str]] = {}
    id_to_source: dict[str, str] = {}
    for e in equations:
        eid = equation_key(e)
        if not eid:
            continue
        vars_dict = e.get("variables") or {}
        if isinstance(vars_dict, dict):
            id_to_vars[eid] = {norm_var(k) for k in vars_dict.keys() if norm_var(k)}
        else:
            id_to_vars[eid] = set()
        id_to_source[eid] = (e.get("source_id") or "unknown").strip()

    missing_fields = Counter()
    missing_eq_ids_total = 0
    cases_with_missing_eq_ids = 0

    io_not_in_any = 0
    io_not_in_any_total_vars = 0
    io_not_in_any_examples: list[dict[str, Any]] = []

    # duplicate signatures: same (original_core_id, variant_type, sorted in/out, sorted correct_model_ids, context)
    sig_counter: Counter[tuple] = Counter()

    variant_counts = Counter()
    hist_models = Counter()
    hist_in = Counter()
    hist_out = Counter()

    required = ["case_id", "original_core_id", "variant_type", "context", "input_variables", "output_variables", "correct_model_ids"]

    for c in cases:
        for k in required:
            if k not in c:
                missing_fields[k] += 1

        variant_counts[c.get("variant_type") or "unknown"] += 1

        cmids = list(c.get("correct_model_ids") or [])
        in_vars = [norm_var(v) for v in (c.get("input_variables") or [])]
        out_vars = [norm_var(v) for v in (c.get("output_variables") or [])]
        ctx = (c.get("context") or "").strip()

        hist_models[str(len(cmids))] += 1
        hist_in[str(len(in_vars))] += 1
        hist_out[str(len(out_vars))] += 1

        # missing equation ids
        missing = [mid for mid in cmids if mid not in eq_ids]
        if missing:
            cases_with_missing_eq_ids += 1
            missing_eq_ids_total += len(missing)

        # variable consistency: all io vars should appear in union of vars of correct models (soft check)
        union_vars: set[str] = set()
        for mid in cmids:
            union_vars |= id_to_vars.get(mid, set())

        io_vars = [v for v in (in_vars + out_vars) if v]
        bad = [v for v in io_vars if v not in union_vars] if union_vars else io_vars
        if bad:
            io_not_in_any += 1
            io_not_in_any_total_vars += len(bad)
            if len(io_not_in_any_examples) < 20:
                io_not_in_any_examples.append(
                    {
                        "case_id": c.get("case_id"),
                        "variant_type": c.get("variant_type"),
                        "missing_vars": bad[:10],
                        "n_missing_vars": len(bad),
                        "correct_model_ids": cmids[:5],
                    }
                )

        sig = (
            (c.get("original_core_id") or "").strip(),
            (c.get("variant_type") or "").strip(),
            tuple(sorted(set(in_vars))),
            tuple(sorted(set(out_vars))),
            tuple(sorted(set(cmids))),
            ctx,
        )
        sig_counter[sig] += 1

    dup_groups = sum(1 for _, n in sig_counter.items() if n >= 2)
    dup_cases = sum(n for _, n in sig_counter.items() if n >= 2)

    rep = HealthReport(
        n_equations=len(equations),
        n_cases=len(cases),
        missing_equation_ids_total=missing_eq_ids_total,
        cases_with_missing_equation_ids=cases_with_missing_eq_ids,
        cases_missing_fields=dict(missing_fields),
        io_not_in_any_correct_model_vars_cases=io_not_in_any,
        io_not_in_any_correct_model_vars_total_vars=io_not_in_any_total_vars,
        io_not_in_any_correct_model_vars_examples=io_not_in_any_examples,
        duplicate_signature_groups=dup_groups,
        duplicate_signature_cases=dup_cases,
        variant_type_counts=dict(variant_counts),
        n_correct_models_hist=dict(hist_models),
        n_input_vars_hist=dict(hist_in),
        n_output_vars_hist=dict(hist_out),
    )

    OUT_JSON.write_text(json.dumps(asdict(rep), indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== Dataset health check ===")
    print(f"Equations: {rep.n_equations}")
    print(f"Training cases: {rep.n_cases}")
    print("")
    print(f"Cases with missing correct_model_ids references: {rep.cases_with_missing_equation_ids} (missing IDs total: {rep.missing_equation_ids_total})")
    if rep.cases_missing_fields:
        print("Missing required fields (count):")
        for k, v in rep.cases_missing_fields.items():
            print(f"  {k}: {v}")
    print("")
    print("I/O vars not found in union(vars(correct_model_ids)) (soft consistency check):")
    print(f"  cases: {rep.io_not_in_any_correct_model_vars_cases}")
    print(f"  total missing vars (across those cases): {rep.io_not_in_any_correct_model_vars_total_vars}")
    if rep.io_not_in_any_correct_model_vars_examples:
        ex0 = rep.io_not_in_any_correct_model_vars_examples[0]
        print(f"  example: {ex0}")
    print("")
    print(f"Duplicate signatures: groups={rep.duplicate_signature_groups}, cases_in_groups={rep.duplicate_signature_cases}")
    print("")
    print("Variant type counts:")
    for k, v in sorted(rep.variant_type_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")
    print("")
    print(f"Wrote {OUT_JSON}")


if __name__ == "__main__":
    main()

