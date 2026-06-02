"""Backfill missing summary keys (Recall@K_correct, Precision@C, multi_only__Recall@K_correct) for any set_aware-style result JSON whose per_seed entries already contain the metric values.

This is a no-op for files that don't have the per_seed shape and a no-op for files where the keys are already present in the summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
EXP_DIR = ROOT / "experiments"

KEYS_TO_BACKFILL = [
    "Recall@K_correct",
    "Precision@C",
    "multi_only__Recall@K_correct",
]


def backfill_file(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except Exception as e:  # noqa: BLE001
        return False
    results = data.get("results") if isinstance(data, dict) else None
    if not isinstance(results, dict):
        return False
    changed_any = False
    for mode, summary in results.items():
        if not isinstance(summary, dict):
            continue
        per_seed = summary.get("per_seed")
        if not isinstance(per_seed, list) or not per_seed:
            continue
        for key in KEYS_TO_BACKFILL:
            if key in summary:
                continue
            vals = [s.get(key) for s in per_seed if isinstance(s, dict) and s.get(key) is not None]
            if not vals:
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            summary[key] = {"mean": round(mean, 4), "std": round(std, 4)}
            changed_any = True
    if changed_any:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return changed_any


def main():
    files = sorted(EXP_DIR.glob("*.json"))
    n_changed = 0
    for f in files:
        if backfill_file(f):
            print(f"  ✓ patched: {f.name}")
            n_changed += 1
        else:
            print(f"    no-op:    {f.name}")
    print(f"\nDone. {n_changed} / {len(files)} files patched.")


if __name__ == "__main__":
    main()
