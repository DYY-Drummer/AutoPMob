"""Count what the paper states about the equation database, and save the counts in a tracked file.

The database, unified_equations.json, is not tracked by git, so make_numbers.py cannot rely on it.
This script reads it once and writes paper/cace/data/db_counts.json, which is tracked; make_numbers.py
reads that file and stops when the database or the cases changed after the counts were taken.

    python3 paper/cace/db_counts.py        # run again whenever unified_equations.json or training_cases.json changes
"""
from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_FILE = ROOT / "unified_equations.json"
CASES_FILE = ROOT / "training_cases.json"
OUT = Path(__file__).resolve().parent / "data" / "db_counts.json"
REFERENCE_PREFIX = "handbook_"      # source identifiers of the generated reference equation sets


def fingerprint(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source(e: dict) -> str:
    return (e.get("source_id") or "unknown").strip()


def count_database(eqs: list) -> dict:
    """Sources, variable symbols (matched by exact spelling) and domain labels of the database."""
    src = Counter(_source(e) for e in eqs)
    ref = {s: n for s, n in src.items() if s.startswith(REFERENCE_PREFIX)}
    n_eq_of, defs = Counter(), defaultdict(set)
    for e in eqs:
        v = e.get("variables") or {}
        if isinstance(v, dict):
            for k, d in v.items():
                k = (k or "").strip()
                if k:
                    n_eq_of[k] += 1
                    defs[k].add(d.strip() if isinstance(d, str) else json.dumps(d, ensure_ascii=False))
    shared = [k for k, n in n_eq_of.items() if n >= 2]
    labels = Counter((e.get("domain") or "").strip() for e in eqs)
    return {
        "n_equations": len(eqs),
        "n_sources": len(src),
        "n_reference_sets": len(ref),
        "n_pdf": len(src) - len(ref),
        "n_equations_reference": sum(ref.values()),
        "n_equations_pdf": len(eqs) - sum(ref.values()),
        "n_symbols": len(n_eq_of),
        "n_symbols_shared": len(shared),                                        # symbols in two or more equations
        "n_symbols_conflict": sum(1 for k in shared if len(defs[k]) > 1),       # ... with differing definitions
        "n_labels": len(labels),
        "share_labels_once": sum(1 for n in labels.values() if n == 1) / len(labels),
    }


def synth_one_source(cases: list, eqs: list) -> int:
    """Synthesized cases (variant dae_*) whose ground-truth equations all come from one source."""
    src_of = {}
    for e in eqs:
        eid = (e.get("eq_id") or "").strip()
        src_of[eid if "__" in eid else f"{_source(e)}__{eid}"] = _source(e)
    one = 0
    for c in cases:
        if c.get("variant_type", "").startswith("dae_"):
            ids = c.get("correct_model_ids") or []
            ids = ast.literal_eval(ids) if isinstance(ids, str) else ids
            one += len({src_of[i.strip()] for i in ids if i.strip() in src_of}) == 1
    return one


def main() -> None:
    eqs = json.loads(DB_FILE.read_text(encoding="utf-8"))
    eqs = eqs if isinstance(eqs, list) else eqs.get("equations", [])
    cases = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    out = {"source": "unified_equations.json and training_cases.json (python3 paper/cace/db_counts.py)",
           "unified_equations_sha256": fingerprint(DB_FILE),
           "training_cases_sha256": fingerprint(CASES_FILE),
           **count_database(eqs),
           "n_synth_one_source": synth_one_source(cases, eqs)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}: {out['n_equations']} equations, {out['n_sources']} sources")


if __name__ == "__main__":
    main()
