"""訓練ケースの自由度解析フィルタ.

各ケースについて、正解式の変数集合とクエリの入出力変数を比較し、
物理的に解けないケース（DoF != 0 等）を特定・除外する。

判定条件（全て満たす場合のみ「可解」）:
  (1) correct_model_ids に含まれる全ての式が unified_equations.json に存在
  (2) output_variables が全て correct models の変数集合に含まれる
  (3) input_variables が全て correct models の変数集合に含まれる
  (4) output_variables ⊆ unknown（=式変数集合 − input_variables）
  (5) |unknown| == n_equations（自由度ゼロ：square system）

使い方:
  python filter_solvable_cases.py                                  # 統計のみ出力
  python filter_solvable_cases.py --write                          # クリーン版を書き出す
  python filter_solvable_cases.py --write --output PATH            # 出力先を指定
  python filter_solvable_cases.py --relaxed                        # DoF<=0 を許容（過剰条件OK）
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
UNIFIED_JSON = ROOT / "unified_equations.json"
TRAINING_JSON = ROOT / "training_cases.json"
DEFAULT_OUT = ROOT / "training_cases.solvable.json"
BACKUP_SUFFIX = ".pre_dof_filter.bak.json"


@dataclass
class CaseVerdict:
    case_id: str
    variant_type: str
    ok: bool
    reason: str
    n_model_vars: int = 0
    n_equations: int = 0
    n_known: int = 0
    n_unknown: int = 0
    dof: int = 0


def build_eq_lookup(equations: list[dict[str, Any]]) -> dict[str, set[str]]:
    """式 ID → 変数集合の辞書を構築."""
    lookup = {}
    for e in equations:
        sid = e.get("source_id")
        eid = e.get("eq_id")
        if sid is None or eid is None:
            continue
        mid = f"{sid}__{eid}"
        vars_dict = e.get("variables") or {}
        lookup[mid] = set(vars_dict.keys())
    return lookup


def analyze_case(
    case: dict[str, Any],
    eq_lookup: dict[str, set[str]],
    relaxed: bool = False,
) -> CaseVerdict:
    """単一ケースに自由度解析を適用.

    Args:
        case: training case dict
        eq_lookup: model_id -> variable set
        relaxed: True なら DoF < 0 も許容（過剰条件）
    """
    iv = set(case.get("input_variables") or [])
    ov = set(case.get("output_variables") or [])
    mids = case.get("correct_model_ids") or []
    verdict = CaseVerdict(
        case_id=case.get("case_id", "?"),
        variant_type=case.get("variant_type", "?"),
        ok=False,
        reason="",
    )

    # (1) 全ての正解式が存在するか
    all_vars: set[str] = set()
    missing = []
    for mid in mids:
        mv = eq_lookup.get(mid)
        if mv is None:
            missing.append(mid)
            continue
        all_vars |= mv
    if missing:
        verdict.reason = f"missing_models:{len(missing)}"
        return verdict

    n_eq = len(mids)
    verdict.n_model_vars = len(all_vars)
    verdict.n_equations = n_eq

    # (2) output_variables が全て式の変数集合に含まれる
    if not ov.issubset(all_vars):
        verdict.reason = "output_not_in_models"
        return verdict

    # (3) input_variables が全て式の変数集合に含まれる
    if not iv.issubset(all_vars):
        verdict.reason = "input_not_in_models"
        return verdict

    known = iv & all_vars
    unknown = all_vars - known
    verdict.n_known = len(known)
    verdict.n_unknown = len(unknown)
    verdict.dof = len(unknown) - n_eq

    # (4) output が unknown に含まれる
    if not ov.issubset(unknown):
        verdict.reason = "output_already_known"
        return verdict

    # (5) 自由度
    if relaxed:
        if verdict.dof > 0:
            verdict.reason = f"underdetermined_dof={verdict.dof:+d}"
            return verdict
    else:
        if verdict.dof != 0:
            verdict.reason = f"dof_nonzero={verdict.dof:+d}"
            return verdict

    verdict.ok = True
    verdict.reason = "ok"
    return verdict


def print_report(verdicts: list[CaseVerdict]) -> None:
    """統計をコンソール出力."""
    total = len(verdicts)
    by_variant: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"total": 0, "ok": 0, "reasons": Counter(), "dof": Counter()}
    )
    for v in verdicts:
        bv = by_variant[v.variant_type]
        bv["total"] += 1
        if v.ok:
            bv["ok"] += 1
        bv["reasons"][v.reason] += 1
        bv["dof"][v.dof] += 1

    print(f"=== 自由度解析結果 (総ケース数: {total}) ===\n")
    for vt in sorted(by_variant.keys()):
        b = by_variant[vt]
        ok_ratio = 100 * b["ok"] / b["total"] if b["total"] > 0 else 0
        print(f"[{vt}] {b['ok']}/{b['total']} ({ok_ratio:.1f}%) 可解")
        # Reason breakdown
        for r, n in sorted(b["reasons"].items(), key=lambda x: -x[1]):
            mark = "✓" if r == "ok" else "✗"
            print(f"    {mark} {r:<30s}: {n:>4}")
        print()
    all_ok = sum(v.ok for v in verdicts)
    print(f"=== 合計: {all_ok}/{total} ({100*all_ok/total:.1f}%) 可解 ===")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="自由度解析による訓練ケースのフィルタリング"
    )
    parser.add_argument(
        "--write", action="store_true",
        help="可解ケースのみを抽出したクリーン版を書き出す"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUT,
        help="--write 時の出力先 (デフォルト: training_cases.solvable.json)"
    )
    parser.add_argument(
        "--relaxed", action="store_true",
        help="DoF <= 0 を許容（過剰条件を許す）。デフォルトは DoF == 0 厳密のみ"
    )
    parser.add_argument(
        "--inplace", action="store_true",
        help="training_cases.json を直接置換（バックアップは .pre_dof_filter.bak.json に保存）"
    )
    args = parser.parse_args()

    # Load
    with open(UNIFIED_JSON, encoding="utf-8") as f:
        eqs_raw = json.load(f)
    equations = (
        eqs_raw if isinstance(eqs_raw, list)
        else eqs_raw.get("equations", eqs_raw.get("papers", []))
    )
    with open(TRAINING_JSON, encoding="utf-8") as f:
        cases = json.load(f)

    eq_lookup = build_eq_lookup(equations)
    print(f"式数: {len(eq_lookup)}, ケース数: {len(cases)}")
    print(f"判定モード: {'relaxed (DoF<=0許容)' if args.relaxed else 'strict (DoF==0のみ)'}")
    print()

    verdicts = [analyze_case(c, eq_lookup, relaxed=args.relaxed) for c in cases]
    print_report(verdicts)

    if args.write or args.inplace:
        solvable = [c for c, v in zip(cases, verdicts) if v.ok]
        print()
        print(f"可解ケース: {len(solvable)} 件を書き出します")

        if args.inplace:
            bak = TRAINING_JSON.with_suffix(BACKUP_SUFFIX)
            shutil.copy(TRAINING_JSON, bak)
            print(f"  バックアップ: {bak}")
            out_path = TRAINING_JSON
        else:
            out_path = args.output

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(solvable, f, ensure_ascii=False, indent=2)
        print(f"  出力: {out_path}")


if __name__ == "__main__":
    main()
