"""
Fix known data quality issues in training_cases.json and unified_equations.json.

Fixes:
  1. I/O variable mismatches: normalize case I/O variable names to match
     equation variable dict (strip functional notation, LaTeX artifacts).
  2. Add common implicit variables (s, t) to equation variable dicts
     where the equation LaTeX string contains them but they are not listed.
  3. Remove duplicate training cases (same signature = context+io+correct_model_ids).

Outputs:
  - training_cases.json (overwritten, backup saved as training_cases.bak.json)
  - unified_equations.json (overwritten, backup saved as unified_equations.bak.json)
  - fix_dataset_quality_report.json (summary of changes)
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
UNIFIED_JSON = ROOT / "unified_equations.json"
TRAINING_JSON = ROOT / "training_cases.json"
REPORT_JSON = ROOT / "fix_dataset_quality_report.json"


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def norm(s: str) -> str:
    return (s or "").strip()


def equation_key(e: dict[str, Any]) -> str | None:
    eq_id = norm(e.get("eq_id") or "")
    if not eq_id:
        return None
    if "__" in eq_id:
        return eq_id
    source_id = norm(e.get("source_id") or "unknown") or "unknown"
    return f"{source_id}__{eq_id}"


# ---------------------------------------------------------------------------
# Fix 1: Normalize I/O variable names in training cases
# ---------------------------------------------------------------------------

def normalize_io_var(v: str) -> str:
    """Strip functional notation and LaTeX artifacts from a variable name
    to match equation variable dict keys.

    Examples:
      C'_{A}(s) → C'_{A}
      C'_{A}(\\infty) → C'_{A}
      C'_{A}(t)|_{t=0} → C'_{A}
      C'_{A0}(s) → C'_{A0}
      G_1(s) → G_1
      G_2(s) → G_2
    """
    s = v.strip()
    # Remove trailing function argument: X(s), X(t), X(\infty), X(t)|_{...}
    s = re.sub(r'\((?:s|t|\\infty)\)(?:\|_\{[^}]*\})?$', '', s)
    return s.strip()


# ---------------------------------------------------------------------------
# Fix 2: Add implicit variables to equations
# ---------------------------------------------------------------------------

IMPLICIT_VARS = {
    "s": "Laplace variable (complex frequency domain)",
    "t": "Time variable",
}


def add_implicit_vars_to_equation(e: dict[str, Any], changes: list[str]) -> None:
    """If the equation LaTeX contains an implicit variable (s, t) that
    is not in the variables dict, add it."""
    eq_latex = norm(e.get("equation") or "")
    variables = e.get("variables") or {}
    if not isinstance(variables, dict):
        return

    for var, desc in IMPLICIT_VARS.items():
        if var in variables:
            continue
        # Check if var appears as a standalone symbol in the equation
        # Match: single letter not part of a longer identifier
        # For 's': match \frac{1}{s}, X(s), etc. but not "mass" or "cos"
        # For 't': match \frac{d}{dt}, x(t), etc. but not "not" or "let"
        if var == "s":
            # Laplace variable: appears as (s), /s, {s}, standalone s in equation context
            pattern = r'(?<![a-zA-Z\\])s(?![a-zA-Z_])'
        elif var == "t":
            # Time: appears as (t), /dt, {t}
            pattern = r'(?<![a-zA-Z\\])t(?![a-zA-Z_])'
        else:
            pattern = rf'(?<![a-zA-Z\\]){re.escape(var)}(?![a-zA-Z_])'

        if re.search(pattern, eq_latex):
            variables[var] = desc
            e["variables"] = variables
            eid = equation_key(e) or "?"
            changes.append(f"Added '{var}' to {eid}")


# ---------------------------------------------------------------------------
# Fix 3: Remove duplicate training cases
# ---------------------------------------------------------------------------

def case_signature(c: dict[str, Any]) -> str:
    """Signature for deduplication: context + sorted io + sorted correct_model_ids."""
    ctx = norm(c.get("context") or "")
    ins = sorted(norm(v) for v in (c.get("input_variables") or []))
    outs = sorted(norm(v) for v in (c.get("output_variables") or []))
    mids = sorted(norm(m) for m in (c.get("correct_model_ids") or []))
    return f"{ctx}|{ins}|{outs}|{mids}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    equations_raw = load_json(UNIFIED_JSON)
    cases = load_json(TRAINING_JSON)

    # Handle different JSON formats
    if isinstance(equations_raw, list):
        equations = equations_raw
    elif isinstance(equations_raw, dict) and "equations" in equations_raw:
        equations = equations_raw["equations"]
    else:
        equations = equations_raw

    # Build equation variable lookup
    eq_var_map: dict[str, set[str]] = {}
    for e in equations:
        eid = equation_key(e)
        if not eid:
            continue
        variables = e.get("variables") or {}
        if isinstance(variables, dict):
            eq_var_map[eid] = {norm(k) for k in variables.keys() if norm(k)}

    report: dict[str, Any] = {}

    # --- Fix 1: Normalize I/O variables in training cases ---
    io_fixes = []
    for c in cases:
        cmids = [norm(m) for m in (c.get("correct_model_ids") or []) if norm(m)]
        # Get union of all variables from correct equations
        correct_vars = set()
        for mid in cmids:
            if mid in eq_var_map:
                correct_vars |= eq_var_map[mid]

        for field in ("input_variables", "output_variables"):
            vars_list = c.get(field) or []
            new_vars = []
            for v in vars_list:
                v_norm = norm(v)
                if v_norm and v_norm not in correct_vars:
                    v_fixed = normalize_io_var(v_norm)
                    if v_fixed != v_norm and (v_fixed in correct_vars or v_fixed == v_norm):
                        io_fixes.append({"case_id": c.get("case_id"), "field": field,
                                         "old": v_norm, "new": v_fixed})
                        new_vars.append(v_fixed)
                    else:
                        new_vars.append(v_norm)
                else:
                    new_vars.append(v_norm)
            c[field] = new_vars

    report["io_var_normalizations"] = len(io_fixes)
    report["io_var_normalization_examples"] = io_fixes[:20]

    # --- Fix 2: Add implicit variables to equations ---
    implicit_changes: list[str] = []
    for e in equations:
        add_implicit_vars_to_equation(e, implicit_changes)

    report["implicit_vars_added"] = len(implicit_changes)
    report["implicit_vars_added_examples"] = implicit_changes[:20]

    # --- Fix 3: Remove duplicates ---
    seen: dict[str, int] = {}
    unique_cases: list[dict[str, Any]] = []
    removed: list[str] = []

    for c in cases:
        sig = case_signature(c)
        if sig in seen:
            removed.append(c.get("case_id", "?"))
        else:
            seen[sig] = len(unique_cases)
            unique_cases.append(c)

    report["duplicates_removed"] = len(removed)
    report["duplicates_removed_ids"] = removed[:20]
    report["cases_before"] = len(cases)
    report["cases_after"] = len(unique_cases)

    # --- Save ---
    # Backup originals
    shutil.copy2(TRAINING_JSON, ROOT / "training_cases.bak.json")
    shutil.copy2(UNIFIED_JSON, ROOT / "unified_equations.bak.json")

    save_json(TRAINING_JSON, unique_cases)

    if isinstance(equations_raw, list):
        save_json(UNIFIED_JSON, equations)
    elif isinstance(equations_raw, dict) and "equations" in equations_raw:
        equations_raw["equations"] = equations
        save_json(UNIFIED_JSON, equations_raw)
    else:
        save_json(UNIFIED_JSON, equations)

    save_json(REPORT_JSON, report)

    # --- Print summary ---
    print("=== Dataset Quality Fix Report ===")
    print(f"I/O variable normalizations: {report['io_var_normalizations']}")
    for fix in io_fixes[:5]:
        print(f"  {fix['case_id']}: {fix['old']} → {fix['new']}")
    print(f"Implicit variables added to equations: {report['implicit_vars_added']}")
    for chg in implicit_changes[:5]:
        print(f"  {chg}")
    print(f"Duplicate cases removed: {report['duplicates_removed']} ({len(cases)} → {len(unique_cases)})")
    print(f"\nBackups: training_cases.bak.json, unified_equations.bak.json")
    print(f"Wrote {REPORT_JSON}")


if __name__ == "__main__":
    main()
