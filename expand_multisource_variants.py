"""multisource_original ケースから random_io バリアントを構成的に生成する.

expand_solvable_random_io.py と同じロジックを、マルチソースケースに限定して適用。
DoF=0 を構成上保証しながら、入出力のランダムな組合せを増やしてバリアント数を増やす。
"""
from __future__ import annotations

import json
import random
import shutil
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent
TRAINING_JSON = ROOT / "training_cases.json"
EQUATIONS_JSON = ROOT / "unified_equations.json"
OUT_JSON = TRAINING_JSON  # 直接上書き（バックアップ済）

TARGET_VARIANTS_PER_BASE = 6  # 各 base から 6 つの variant を生成


def main():
    cases = json.load(open(TRAINING_JSON, encoding="utf-8"))
    eqs = json.load(open(EQUATIONS_JSON, encoding="utf-8"))
    eq_db = {f"{e['source_id']}__{e['eq_id']}": e for e in eqs}

    # マルチソース base ケースを抽出
    bases = [c for c in cases if c.get("variant_type") == "multisource_original"]
    print(f"マルチソース base ケース: {len(bases)}")

    rng = random.Random(42)
    new_variants = []
    skipped = 0

    for base in bases:
        eq_keys = base["correct_model_ids"]
        # 全変数集合
        all_vars = set()
        for k in eq_keys:
            e = eq_db.get(k)
            if e is None:
                continue
            all_vars.update(e.get("variables", {}).keys())

        n_eq = len(eq_keys)
        # 出力数 = n_eq（DoF=0）
        # 全変数から n_eq 個選ぶ組合せを試す

        # 既に使用済みの (input, output) を排除
        existing_io = {(frozenset(base["input_variables"]), frozenset(base["output_variables"]))}
        # 既存ケース全部から、この base に対するものも除外
        for c in cases:
            if c.get("original_core_id") == base["case_id"]:
                existing_io.add((frozenset(c["input_variables"]), frozenset(c["output_variables"])))

        # 全 n_eq 個の output 選び方
        all_var_list = sorted(all_vars)
        if len(all_var_list) <= n_eq:
            # 出力候補が n_eq 個以下なら 1 通りしかない
            skipped += 1
            continue

        combos = list(combinations(all_var_list, n_eq))
        rng.shuffle(combos)

        added_for_this = 0
        for output_combo in combos:
            if added_for_this >= TARGET_VARIANTS_PER_BASE:
                break
            outputs = frozenset(output_combo)
            inputs = frozenset(all_var_list) - outputs

            if (inputs, outputs) in existing_io:
                continue

            # 構成上 DoF=0 を保証
            new_case = {
                "case_id": f"{base['case_id']}_rv{added_for_this+1}",
                "original_core_id": base["case_id"],
                "variant_type": "multisource_random_io",
                "context": base["context"],
                "input_variables": sorted(inputs),
                "output_variables": sorted(outputs),
                "correct_model_ids": eq_keys,
            }
            new_variants.append(new_case)
            existing_io.add((inputs, outputs))
            added_for_this += 1

        if added_for_this == 0:
            skipped += 1

    print(f"新規 variant 生成: {len(new_variants)} 件")
    print(f"variant 生成 skipped: {skipped}")

    # 統合
    merged = cases + new_variants
    print(f"\n統合: {len(cases)} → {len(merged)} ケース")

    # 検証用：DoF=0 を構成的に保証しているが、念のため
    def n_src(c):
        return len({m.split("__", 1)[0] for m in c.get("correct_model_ids", [])})

    from collections import Counter
    src_dist = Counter(n_src(c) for c in merged)
    print(f"\nソース横断度分布:")
    for k in sorted(src_dist):
        v = src_dist[k]
        print(f"  {k} ソース: {v:4d} ({100*v/len(merged):.1f}%)")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
