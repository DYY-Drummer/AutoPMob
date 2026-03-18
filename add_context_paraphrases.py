"""
Add context-only paraphrase variants to training_cases.json without calling an LLM.

It finds cases with variant_type == "original" and creates 3 additional cases where:
  - input_variables/output_variables/correct_model_ids are unchanged
  - context is paraphrased using simple templates
  - variant_type = "context_paraphrased"

Existing paraphrase cases (same original_core_id and same IO+model IDs+context) are not duplicated.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent
TRAINING_JSON = ROOT / "training_cases.json"


def paraphrase_context_variants(context: str) -> list[str]:
    context = (context or "").strip()
    return [
        f"This case describes the following physical model: {context}",
        f"In this scenario, the underlying process can be summarized as: {context}",
        f"From the perspective of process modeling, this case focuses on: {context}",
    ]


def norm_list(xs: list[str]) -> tuple[str, ...]:
    return tuple(sorted({(x or "").strip() for x in xs if (x or "").strip()}))


def make_sig(c: dict[str, Any]) -> tuple:
    return (
        (c.get("original_core_id") or "").strip(),
        (c.get("variant_type") or "").strip(),
        norm_list(c.get("input_variables") or []),
        norm_list(c.get("output_variables") or []),
        norm_list(c.get("correct_model_ids") or []),
        (c.get("context") or "").strip(),
    )


def main() -> None:
    if not TRAINING_JSON.is_file():
        raise FileNotFoundError(f"Not found: {TRAINING_JSON}")

    cases: list[dict[str, Any]] = json.loads(TRAINING_JSON.read_text(encoding="utf-8"))
    existing_case_ids = {c.get("case_id") for c in cases if c.get("case_id")}
    existing_sigs = {make_sig(c) for c in cases}

    added: list[dict[str, Any]] = []
    by_core_counter: Counter[str] = Counter()

    for c in cases:
        if (c.get("variant_type") or "") != "original":
            continue

        base_case_id = (c.get("case_id") or "").strip()
        if not base_case_id:
            continue

        for i, ctx in enumerate(paraphrase_context_variants(c.get("context") or ""), start=1):
            new_case = {
                "case_id": f"{base_case_id}_para{i}",
                "original_core_id": c.get("original_core_id"),
                "variant_type": "context_paraphrased",
                "context": ctx,
                "input_variables": list(c.get("input_variables") or []),
                "output_variables": list(c.get("output_variables") or []),
                "correct_model_ids": list(c.get("correct_model_ids") or []),
            }
            sig = make_sig(new_case)
            if sig in existing_sigs:
                continue
            # avoid ID collisions
            if new_case["case_id"] in existing_case_ids:
                # deterministic fallback
                suffix = 2
                while f"{new_case['case_id']}_{suffix}" in existing_case_ids:
                    suffix += 1
                new_case["case_id"] = f"{new_case['case_id']}_{suffix}"

            existing_case_ids.add(new_case["case_id"])
            existing_sigs.add(sig)
            added.append(new_case)
            by_core_counter[(c.get("original_core_id") or "unknown").strip()] += 1

    if added:
        cases_out = cases + added
        TRAINING_JSON.write_text(json.dumps(cases_out, indent=2, ensure_ascii=False), encoding="utf-8")

    vt = Counter((c.get("variant_type") or "unknown") for c in (cases + added))
    print(f"Added {len(added)} context_paraphrased cases.")
    print("Variant type counts:")
    for k, v in sorted(vt.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

