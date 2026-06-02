"""自由度=0 を満たす random_io バリアントをアルゴリズム的に追加生成する.

目的:
  自由度解析フィルタで訓練ケースが減少した分を、物理的に可解な random_io
  バリアントで補充し、目標件数まで復元する。

アルゴリズム:
  各コアケースについて、正解式集合の変数 V と方程式数 n_eq に対して、
  V から n_eq 個の変数をランダムに選んで出力（未知）、残りを入力（既知）
  とする。自由度 = |V| - n_eq - |input| = 0 が構成上保証される。

  既存の (入力集合, 出力集合) と重複しないように生成。

使い方:
  python expand_solvable_random_io.py --target 1107        # 1107件まで増やす
  python expand_solvable_random_io.py --target 1107 --write
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).parent
TRAINING_JSON = ROOT / "training_cases.json"
UNIFIED_JSON = ROOT / "unified_equations.json"
BACKUP_SUFFIX = ".pre_expansion3.bak.json"


def build_eq_lookup(equations):
    lookup = {}
    for e in equations:
        sid = e.get("source_id")
        eid = e.get("eq_id")
        if sid is None or eid is None:
            continue
        mid = f"{sid}__{eid}"
        vars_dict = e.get("variables") or {}
        lookup[mid] = list(vars_dict.keys())
    return lookup


def next_variant_id(existing_ids, core):
    """core_XXX_vN 形式の次に使える番号を返す."""
    max_n = 0
    for cid in existing_ids:
        if not cid.startswith(core + "_v"):
            continue
        tail = cid[len(core) + 2:]
        # v番号は para サフィックス等がある場合もあるので数字部分のみ抽出
        num_str = ""
        for ch in tail:
            if ch.isdigit():
                num_str += ch
            else:
                break
        if num_str:
            max_n = max(max_n, int(num_str))
    return max_n + 1


def load_paraphrase_templates():
    """元の context_paraphrased で使われたテンプレート同等を用意（ランダム選択用）."""
    return [
        "This case describes the following physical model: {context}",
        "In this scenario, the underlying process can be summarized as: {context}",
        "Consider the physical setup below: {context}",
    ]


def generate_random_io_for_core(
    core_cases,
    eq_lookup,
    rng,
    n_needed,
):
    """1 コア分について、n_needed 件の新規 random_io バリアントを生成.

    Returns: list[dict] (新規 case 辞書)
    """
    # 元（original）を探す
    original = next((c for c in core_cases if c.get("variant_type") == "original"), None)
    if original is None:
        return []

    mids = original.get("correct_model_ids") or []
    # 全ての式が辞書にあることを確認
    if not all(m in eq_lookup for m in mids):
        return []

    # V_model を取得（順序保持用に list 化）
    V_seen = set()
    V_list = []
    for m in mids:
        for v in eq_lookup[m]:
            if v not in V_seen:
                V_seen.add(v)
                V_list.append(v)
    V = V_list
    n_eq = len(mids)

    if n_eq > len(V) or n_eq == 0:
        return []

    # 既存の (入力, 出力) を収集
    used = set()
    for c in core_cases:
        iv = frozenset(c.get("input_variables") or [])
        ov = frozenset(c.get("output_variables") or [])
        used.add((iv, ov))

    # 生成可能な (出力集合) の全候補を列挙（n_eq <= |V| なら C(|V|,n_eq) 通り）
    all_output_choices = [set(combo) for combo in combinations(V, n_eq)]
    rng.shuffle(all_output_choices)

    # 次に使える v 番号
    existing_ids = [c["case_id"] for c in core_cases]
    core = original["case_id"].rsplit("_v", 1)[0]
    start_v = next_variant_id(existing_ids, core)

    # context を元ケースから借りる
    base_context = original.get("context", "")

    new_cases = []
    v_counter = start_v
    for out_set in all_output_choices:
        if len(new_cases) >= n_needed:
            break
        out_list = sorted(out_set)
        in_list = sorted(v for v in V if v not in out_set)
        key = (frozenset(in_list), frozenset(out_list))
        if key in used:
            continue
        used.add(key)
        new_cases.append({
            "case_id": f"{core}_v{v_counter}",
            "variant_type": "random_io_from_models",
            "context": base_context,
            "input_variables": in_list,
            "output_variables": out_list,
            "correct_model_ids": list(mids),
        })
        v_counter += 1

    return new_cases


def main():
    parser = argparse.ArgumentParser(description="random_io を自由度=0制約のもとで拡張")
    parser.add_argument("--target", type=int, default=1107,
                        help="目標総ケース数（デフォルト: 1107）")
    parser.add_argument("--seed", type=int, default=42,
                        help="乱数シード（デフォルト: 42）")
    parser.add_argument("--write", action="store_true",
                        help="training_cases.json を上書き（バックアップ付）")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    with open(UNIFIED_JSON, encoding="utf-8") as f:
        equations = json.load(f)
    if not isinstance(equations, list):
        equations = equations.get("equations", [])
    eq_lookup = build_eq_lookup(equations)

    with open(TRAINING_JSON, encoding="utf-8") as f:
        cases = json.load(f)

    current_total = len(cases)
    n_needed = args.target - current_total
    if n_needed <= 0:
        print(f"既に {current_total} ≥ {args.target} のため生成不要")
        return

    print(f"現在: {current_total} → 目標: {args.target} → 追加: {n_needed}")
    print()

    # Group by core
    by_core = defaultdict(list)
    for c in cases:
        cid = c["case_id"]
        core = cid.rsplit("_v", 1)[0] if "_v" in cid else cid
        by_core[core].append(c)

    cores = list(by_core.keys())
    rng.shuffle(cores)

    # 各コアの生成上限（残席数）を計算
    core_headroom = {}
    for core in cores:
        core_cases = by_core[core]
        original = next((c for c in core_cases if c.get("variant_type") == "original"), None)
        if not original:
            core_headroom[core] = 0
            continue
        mids = original.get("correct_model_ids") or []
        V = set()
        for m in mids:
            V |= set(eq_lookup.get(m, []))
        n_eq = len(mids)
        # C(|V|, n_eq)
        from math import comb
        max_possible = comb(len(V), n_eq) if n_eq <= len(V) else 0
        used = len(set(
            (frozenset(c.get("input_variables") or []),
             frozenset(c.get("output_variables") or []))
            for c in core_cases
        ))
        core_headroom[core] = max(0, max_possible - used)

    total_headroom = sum(core_headroom.values())
    print(f"利用可能な総席数: {total_headroom}")
    if total_headroom < n_needed:
        print(f"  警告: 必要数 {n_needed} に対し {total_headroom} しか席が無い")

    # Distribute generation proportional to headroom with min 0
    # 実装: ラウンドロビンで各コアに 1 つずつ配分
    remaining = {c: core_headroom[c] for c in cores}
    allocations = {c: 0 for c in cores}
    still_needed = n_needed
    while still_needed > 0:
        made_progress = False
        for core in cores:
            if remaining[core] > 0 and still_needed > 0:
                allocations[core] += 1
                remaining[core] -= 1
                still_needed -= 1
                made_progress = True
        if not made_progress:
            break

    # Generate
    new_cases = []
    for core, n_alloc in allocations.items():
        if n_alloc <= 0:
            continue
        generated = generate_random_io_for_core(by_core[core], eq_lookup, rng, n_alloc)
        new_cases.extend(generated)

    print(f"新規生成: {len(new_cases)} 件")

    # Merge + summary
    expanded = cases + new_cases
    print(f"合計: {len(expanded)} 件")
    variant_counts = Counter(c.get("variant_type", "?") for c in expanded)
    for v, n in sorted(variant_counts.items()):
        print(f"  {v}: {n}")

    # Validate: confirm all new cases satisfy DoF <= 0
    print()
    print("=== 新規ケースの自由度検証（relaxed） ===")
    from filter_solvable_cases import analyze_case, build_eq_lookup as _bel
    _eq_lookup = _bel(equations)
    bad = 0
    for c in new_cases:
        v = analyze_case(c, _eq_lookup, relaxed=True)
        if not v.ok:
            bad += 1
            if bad <= 3:
                print(f"  NG: {c['case_id']} reason={v.reason}")
    print(f"新規ケースのうち不合格: {bad}/{len(new_cases)}")

    if args.write:
        bak = TRAINING_JSON.with_suffix(BACKUP_SUFFIX)
        shutil.copy(TRAINING_JSON, bak)
        print(f"\nバックアップ: {bak}")
        with open(TRAINING_JSON, "w", encoding="utf-8") as f:
            json.dump(expanded, f, ensure_ascii=False, indent=2)
        print(f"出力: {TRAINING_JSON}")


if __name__ == "__main__":
    main()
