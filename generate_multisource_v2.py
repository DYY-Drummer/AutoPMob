"""アルゴリズム的な multi-source ケース生成 v2（新 DB 7,753 式対応）.

戦略：
1. 「複数ソースで変数を共有する式ペア」を全探索
2. ペアごとに DoF=0 を満たす (input, output) 分割を構成的に決定
3. context は自動生成（後で人手で改善可能）
4. 各ペアで複数の I/O 構成を生成 → 多様性確保

出力：multisource_cases_v2.json（マージ前のテンプレート）
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent
EQS_JSON = ROOT / "unified_equations.json"
OUT_JSON = ROOT / "multisource_cases_v2.json"


def safe_filename(s):
    return "".join(c if c.isalnum() else "_" for c in s)[:50]


def main():
    eqs = json.load(open(EQS_JSON))
    print(f"DB: {len(eqs)} 式")

    # source → eqs マップ
    src_eqs = defaultdict(list)
    for e in eqs:
        src_eqs[e['source_id']].append(e)

    # 既存ケースを読み込んで、組合せ重複を防ぐ
    cases = json.load(open(ROOT / "training_cases.json"))
    existing_keys = set()
    for c in cases:
        key = tuple(sorted(c.get('correct_model_ids', [])))
        existing_keys.add(key)
    print(f"既存ケースの正解集合: {len(existing_keys)} 通り")

    # 変数 -> その変数を含む (eq_key, eq) のリスト
    var_to_eqs = defaultdict(list)
    for e in eqs:
        eq_key = f"{e['source_id']}__{e['eq_id']}"
        for v in e.get('variables', {}).keys():
            var_to_eqs[v].append((eq_key, e))

    # 各変数についてソース横断ペアを探す
    rng = random.Random(42)
    candidates = []  # (eq_a, eq_b, shared_vars)

    print("変数共有のあるソース横断ペアを探索...")
    seen_pairs = set()
    for v, eq_list in var_to_eqs.items():
        if len(eq_list) < 2:
            continue
        # ソース別にグループ化
        by_src = defaultdict(list)
        for eq_key, eq in eq_list:
            by_src[eq['source_id']].append((eq_key, eq))
        srcs = list(by_src.keys())
        if len(srcs) < 2:
            continue
        # 各ソースペアで eq ペアを生成
        for s_a, s_b in combinations(srcs, 2):
            # 各ソースから 1 つずつ
            for eq_key_a, eq_a in by_src[s_a][:3]:  # 上位 3 式
                for eq_key_b, eq_b in by_src[s_b][:3]:
                    pair = tuple(sorted([eq_key_a, eq_key_b]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    vars_a = set(eq_a.get('variables', {}).keys())
                    vars_b = set(eq_b.get('variables', {}).keys())
                    shared = vars_a & vars_b
                    if len(shared) >= 1:
                        candidates.append((eq_a, eq_b, shared))

    print(f"候補ペア: {len(candidates)} 組")

    # ランダムに 1000 ペアサンプリング（全探索だと多すぎ）
    rng.shuffle(candidates)
    candidates = candidates[:1500]

    # 各ペアで DoF=0 を満たすケースを構築
    new_cases = []
    seen_io = set()
    for i, (eq_a, eq_b, shared) in enumerate(candidates):
        eq_key_a = f"{eq_a['source_id']}__{eq_a['eq_id']}"
        eq_key_b = f"{eq_b['source_id']}__{eq_b['eq_id']}"
        eq_keys = [eq_key_a, eq_key_b]
        eq_keys_sorted = tuple(sorted(eq_keys))

        if eq_keys_sorted in existing_keys:
            continue

        vars_a = set(eq_a.get('variables', {}).keys())
        vars_b = set(eq_b.get('variables', {}).keys())
        all_vars = vars_a | vars_b
        n_eq = 2

        # DoF=0 = |unknowns|=|n_eq|=2 ⇒ |inputs|=|all_vars|-2
        if len(all_vars) < 3:
            continue  # 出力 2 個取れない

        # 出力候補：n_eq=2 個の変数を選ぶ
        # 出力は両方の式の和集合から（できれば各式から 1 個ずつ）
        # 簡単のため：ランダムに 2 変数選ぶ
        all_vars_list = sorted(all_vars)
        # 試行：複数の I/O 構成を試す
        for trial in range(3):
            rng.shuffle(all_vars_list)
            for output_combo in combinations(all_vars_list, n_eq):
                outputs = set(output_combo)
                inputs = all_vars - outputs
                if len(inputs) == 0:
                    continue
                # 既出 I/O 重複避け
                io_key = (eq_keys_sorted, frozenset(inputs), frozenset(outputs))
                if io_key in seen_io:
                    continue
                seen_io.add(io_key)

                # 物理的に成立しやすい：出力に LHS 候補（記号が単純）を含める
                # （簡易ヒューリスティック：両式で同時に「入力にも出力にも」現れる変数を避ける）

                # context 自動生成
                domain_a = eq_a.get('domain', 'unknown')
                domain_b = eq_b.get('domain', 'unknown')
                context = (
                    f"A multi-source physical model combining {domain_a} from one literature source "
                    f"and {domain_b} from another. The shared variables {sorted(shared)[:3]} couple "
                    f"the two equations to determine {sorted(outputs)[:3]}."
                )

                case = {
                    "case_id": f"ms2_{i:04d}_{trial}",
                    "original_core_id": f"ms2_{i:04d}",
                    "variant_type": "multisource_v2",
                    "context": context,
                    "input_variables": sorted(inputs),
                    "output_variables": sorted(outputs),
                    "correct_model_ids": eq_keys,
                }
                new_cases.append(case)
                break  # 各 trial で 1 つ生成
            if len(new_cases) % 100 == 0 and len(new_cases) > 0:
                pass

    print(f"\n生成ケース数（未フィルタ）: {len(new_cases)}")

    # 上限を設けて保存
    rng.shuffle(new_cases)
    new_cases = new_cases[:500]
    print(f"サンプリング後: {len(new_cases)}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(new_cases, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
